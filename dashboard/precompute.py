"""Scenario grid build + interpolation for the dashboard's Fast mode.

The grid is intentionally 3-axis so it stays small enough to build in well
under five minutes on a laptop:

    axis 1: ``basket_spot_shift``   in ``[-0.30, +0.30]`` step ``0.05``  (13 pts)
    axis 2: ``basket_vol_shift``    in ``[-0.10, +0.20]`` step ``0.05``  ( 7 pts)
    axis 3: ``corr_shift``          in ``[-0.10, +0.10]`` step ``0.05``  ( 5 pts)

Per-name independent moves are out of scope for the grid: with three
underlyings the per-name product set is ``13³ × 7³ × 5 ≈ 1.6 million`` cells,
which is unmanageable. When the sidebar is in *Independent* mode, the
P&L / Greeks panels fall back to live (Precise) MC. The basket grid still
projects the current state by averaging the per-name shifts back onto the
basket axes — good enough to keep the panels lit up while the user moves
sliders, and the "Precise" toggle gives them the exact answer.

Each cell stores only the base FCN price produced by a single MC pass under
CRN — this is enough for Fast-mode pricing (P&L panel's headline number
and the Price-curve panel's slice). Per-name Greeks are not stored in the
grid: they're served live by :func:`precise_greeks` whenever the Greeks or
Hedge tabs need them, which keeps grid build time roughly one minute.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.fcn_payoff import FCNProduct, ObservationGrid, payoff_per_path
from src.gbm_simulation import SimulationConfig, draw_normals, simulate_paths
from src.market_data import MarketData
from src.utils import is_psd, nearest_psd

from dashboard.state import MarketSnapshot


# ---------------------------------------------------------------------------
# Grid spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridAxes:
    """Define the 3 basket-axes the grid spans."""

    basket_spot_shift: np.ndarray   # relative, e.g. -0.30 = -30% on every spot
    basket_vol_shift: np.ndarray    # absolute vol points, e.g. +0.05 = +5 vol-pts on every vol
    corr_shift: np.ndarray          # absolute, added to every off-diagonal of corr

    @classmethod
    def default(cls) -> "GridAxes":
        return cls(
            basket_spot_shift=np.round(np.arange(-0.30, 0.30 + 1e-9, 0.05), 4),
            basket_vol_shift=np.round(np.arange(-0.10, 0.20 + 1e-9, 0.05), 4),
            corr_shift=np.round(np.arange(-0.10, 0.10 + 1e-9, 0.05), 4),
        )

    @classmethod
    def fast(cls) -> "GridAxes":
        """A smaller grid for the smoke-test / cold-start path.

        ``7 × 4 × 3 = 84`` cells was the original; this is the leaner
        ``5 × 3 × 3 = 45`` cell variant used when the shipped grid is
        missing or stale and we still want a sub-30 s cold start on
        Streamlit Cloud's shared CPU. Linear interpolation across this
        coarser grid is fine for the visualisation; "Precise" mode is
        still available for exact values.
        """
        return cls(
            basket_spot_shift=np.round(np.arange(-0.30, 0.30 + 1e-9, 0.15), 4),
            basket_vol_shift=np.array([-0.10, 0.0, 0.10]),
            corr_shift=np.array([-0.10, 0.0, 0.10]),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.basket_spot_shift),
            len(self.basket_vol_shift),
            len(self.corr_shift),
        )

    def total_cells(self) -> int:
        s = self.shape
        return s[0] * s[1] * s[2]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass
class ScenarioGrid:
    """Pre-computed price grid + lazy interpolator.

    Only the base price is stored. Greeks are served live by the engine.
    """

    axes: GridAxes
    initial: MarketSnapshot
    product: FCNProduct
    price: np.ndarray          # shape (S, V, C)
    diagnostics: dict = field(default_factory=dict)

    _price_interp: Optional[RegularGridInterpolator] = None

    def _ensure_interpolators(self) -> None:
        if self._price_interp is not None:
            return
        pts = (
            self.axes.basket_spot_shift,
            self.axes.basket_vol_shift,
            self.axes.corr_shift,
        )
        self._price_interp = RegularGridInterpolator(
            pts, self.price, method="linear", bounds_error=False, fill_value=None,
        )

    def project(self, current: MarketSnapshot) -> tuple[float, float, float]:
        """Map a current snapshot back onto the 3 basket axes.

        Returns ``(basket_spot_shift, basket_vol_shift, corr_shift)``. When the
        user has moved sliders independently per name, the shift is the simple
        cross-sectional mean — a sensible *visualisation* projection. Use the
        Precise mode for the exact value at independent inputs.
        """
        spot_rel = current.spots / self.initial.spots - 1.0
        vol_abs = current.vols - self.initial.vols
        d = self.initial.corr.shape[0]
        if d > 1:
            iu = np.triu_indices(d, k=1)
            corr_diff = current.corr[iu] - self.initial.corr[iu]
            corr_shift = float(np.mean(corr_diff))
        else:
            corr_shift = 0.0
        return (
            float(np.clip(spot_rel.mean(), self.axes.basket_spot_shift.min(),
                          self.axes.basket_spot_shift.max())),
            float(np.clip(vol_abs.mean(), self.axes.basket_vol_shift.min(),
                          self.axes.basket_vol_shift.max())),
            float(np.clip(corr_shift, self.axes.corr_shift.min(),
                          self.axes.corr_shift.max())),
        )

    def interpolate(self, current: MarketSnapshot) -> dict:
        """Return ``{price, projection}`` at the projection of ``current``.

        Linear interpolation across the 3 basket axes.
        """
        self._ensure_interpolators()
        pt = self.project(current)
        price = float(self._price_interp(np.array(pt)))
        return {"price": price, "projection": pt}


# ---------------------------------------------------------------------------
# Grid build — single CRN block, walk the 3 axes
# ---------------------------------------------------------------------------


def _bump_corr(corr: np.ndarray, shift: float) -> np.ndarray:
    """Add ``shift`` to every off-diagonal of ``corr``, clip, and PSD-fix."""
    out = corr.copy()
    d = out.shape[0]
    off = ~np.eye(d, dtype=bool)
    out[off] = np.clip(out[off] + shift, -0.999, 0.999)
    np.fill_diagonal(out, 1.0)
    if not is_psd(out):
        out = nearest_psd(out)
    return out


def _market_from_snapshot(snap: MarketSnapshot) -> MarketData:
    """Build a minimal MarketData from a MarketSnapshot for pricer plumbing.

    The pricer never inspects ``history``; a tiny placeholder DataFrame is OK.
    """
    import pandas as pd
    placeholder = pd.DataFrame(
        np.tile(snap.spots, (2, 1)),
        columns=list(snap.tickers),
        index=pd.to_datetime([snap.as_of, snap.as_of]),
    )
    return MarketData(
        tickers=list(snap.tickers),
        spots=np.asarray(snap.spots, dtype=float),
        vols=np.asarray(snap.vols, dtype=float),
        divs=np.asarray(snap.divs, dtype=float),
        corr=np.asarray(snap.corr, dtype=float),
        rate=float(snap.rate),
        history=placeholder,
        sources={"as_of": snap.as_of.isoformat()},
    )


def _price_once(
    *,
    snap: MarketSnapshot,
    product: FCNProduct,
    normals: np.ndarray,
    grid: ObservationGrid,
    spots: np.ndarray,
    vols: np.ndarray,
    corr: np.ndarray,
    initial_fixing: np.ndarray,
) -> tuple[float, np.ndarray]:
    """One CRN pricing call. Returns ``(mean_pv, per_path_pv)``.

    ``initial_fixing`` is held separate from ``spots``: spots drive the
    simulator start, ``initial_fixing`` is what the payoff normalises by
    (the desk's locked-in fixing at issue). See the spot-bump note in
    :func:`src.greeks._price_with_overrides`.
    """
    if not is_psd(corr):
        corr = nearest_psd(corr)
    cfg = SimulationConfig(
        spots=spots, vols=vols, divs=snap.divs, rate=snap.rate, corr=corr,
        T=grid.sim_T, n_steps=grid.sim_n_steps, n_paths=normals.shape[0],
        antithetic=True, seed=None,
    )
    paths = simulate_paths(cfg, normals=normals)
    pv = payoff_per_path(
        paths=paths, spots=initial_fixing, product=product,
        grid=grid, rate=snap.rate,
    )
    return float(pv.mean()), pv


def build_scenario_grid(
    initial: MarketSnapshot,
    product: FCNProduct,
    *,
    axes: Optional[GridAxes] = None,
    n_paths: int = 8_000,
    seed: int = 20260511,
    progress_cb=None,
) -> ScenarioGrid:
    """Build the price grid.

    Each cell is a single CRN-shared MC price call: 455 default cells × 1
    price call each = roughly 30–60 seconds total at ``n_paths=8_000``.
    Greeks are not pre-computed here — they're served live by
    :func:`precise_greeks`.
    """
    if axes is None:
        axes = GridAxes.default()

    d = len(initial.tickers)
    shape = axes.shape
    price = np.zeros(shape, dtype=float)

    day_count = 365.0
    obs_grid = ObservationGrid.from_product(product=product, day_count=day_count)

    # Single CRN block reused across every cell.
    rng = np.random.default_rng(seed)
    normals = draw_normals(
        n_paths=n_paths, n_steps=obs_grid.sim_n_steps, d=d, rng=rng,
    )

    total = axes.total_cells()
    done = 0
    initial_fixing = np.asarray(initial.spots, dtype=float)
    for si, s_shift in enumerate(axes.basket_spot_shift):
        for vi, v_shift in enumerate(axes.basket_vol_shift):
            for ci, c_shift in enumerate(axes.corr_shift):
                cell_spots = initial.spots * (1.0 + s_shift)
                cell_vols = np.clip(initial.vols + v_shift, 1e-3, None)
                cell_corr = _bump_corr(initial.corr, c_shift)
                base, _ = _price_once(
                    snap=initial, product=product, normals=normals,
                    grid=obs_grid,
                    spots=cell_spots, vols=cell_vols, corr=cell_corr,
                    initial_fixing=initial_fixing,
                )
                price[si, vi, ci] = base
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total)

    return ScenarioGrid(
        axes=axes,
        initial=initial,
        product=product,
        price=price,
        diagnostics={
            "n_paths_primal": n_paths,
            "n_paths_effective": 2 * n_paths,
            "shape": shape,
            "total_cells": total,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Live (Precise) mode — single MC pricing call
# ---------------------------------------------------------------------------


def precise_price(
    snap: MarketSnapshot,
    product: FCNProduct,
    *,
    n_paths: int = 30_000,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """One MC pricing call against ``snap``. Returns ``(price, std_error)``.

    Used by the dashboard's *Precise* compute mode — slower than the grid
    interpolation but uses every per-name spot/vol/corr value directly.
    """
    from src.mc_pricer import price_fcn
    market = _market_from_snapshot(snap)
    result = price_fcn(
        market=market, product=product, n_paths=n_paths,
        antithetic=True, seed=seed,
    )
    return result.price, result.standard_error


# ---------------------------------------------------------------------------
# Persistence — ship a pre-built grid in the repo to skip the cold-start build
# ---------------------------------------------------------------------------


def snapshot_fingerprint(snap: MarketSnapshot, product: FCNProduct) -> str:
    """Stable hex digest of the inputs that determine a grid's contents.

    Two grids with the same fingerprint are interchangeable; a mismatch
    means the shipped file no longer matches the live ``initial`` snapshot
    and we must rebuild.
    """
    h = hashlib.sha256()
    for arr in (snap.spots, snap.vols, snap.divs, snap.corr):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.float64(snap.rate).tobytes())
    h.update(",".join(snap.tickers).encode())
    # Product fields that change the payoff
    h.update(np.float64(product.notional).tobytes())
    h.update(np.float64(product.coupon_rate).tobytes())
    h.update(np.float64(product.autocall_barrier).tobytes())
    h.update(np.float64(product.strike).tobytes())
    h.update(str(product.issue_date).encode())
    h.update(",".join(str(d) for d in product.obs_dates).encode())
    return h.hexdigest()[:16]


def save_grid_npz(grid: ScenarioGrid, path: Path) -> None:
    """Persist a ScenarioGrid to disk.

    Only the price tensor + axes + a fingerprint of the inputs are saved.
    The MarketSnapshot itself isn't serialised — the live snapshot from
    ``state.py`` is the source of truth at load time.
    """
    fp = snapshot_fingerprint(grid.initial, grid.product)
    np.savez_compressed(
        path,
        price=grid.price,
        basket_spot_shift=grid.axes.basket_spot_shift,
        basket_vol_shift=grid.axes.basket_vol_shift,
        corr_shift=grid.axes.corr_shift,
        fingerprint=np.array(fp),
        diagnostics_keys=np.array(list(grid.diagnostics.keys()), dtype=object),
        diagnostics_vals=np.array(
            [str(v) for v in grid.diagnostics.values()], dtype=object
        ),
    )


def load_grid_npz(
    path: Path,
    *,
    initial: MarketSnapshot,
    product: FCNProduct,
) -> Optional[ScenarioGrid]:
    """Load a shipped grid if the fingerprint matches the current inputs.

    Returns ``None`` if the file is missing or stale — callers should fall
    back to building a fresh grid in that case.
    """
    if not path.exists():
        return None
    expected = snapshot_fingerprint(initial, product)
    with np.load(path, allow_pickle=True) as data:
        stored_fp = str(data["fingerprint"])
        if stored_fp != expected:
            return None
        axes = GridAxes(
            basket_spot_shift=np.array(data["basket_spot_shift"]),
            basket_vol_shift=np.array(data["basket_vol_shift"]),
            corr_shift=np.array(data["corr_shift"]),
        )
        price = np.array(data["price"])
        diag = {}
        if "diagnostics_keys" in data:
            keys = data["diagnostics_keys"]
            vals = data["diagnostics_vals"]
            diag = {str(k): str(v) for k, v in zip(keys, vals)}
        diag["loaded_from"] = str(path)
    return ScenarioGrid(
        axes=axes,
        initial=initial,
        product=product,
        price=price,
        diagnostics=diag,
    )


def precise_greeks(
    snap: MarketSnapshot,
    product: FCNProduct,
    *,
    n_paths: int = 20_000,
    seed: Optional[int] = 12345,
) -> dict:
    """Bump-and-revalue Greeks against ``snap`` under fresh CRN.

    Returns ``{price, delta, vega, cega_pair}`` matching the shape returned
    by :meth:`ScenarioGrid.interpolate` so the panels can swap engines
    without rewriting their downstream code.
    """
    from src.greeks import mc_greeks_bump
    market = _market_from_snapshot(snap)
    result = mc_greeks_bump(
        market=market, product=product, n_paths=n_paths,
        antithetic=True, seed=seed,
    )
    return {
        "price": result.price,
        "delta": np.asarray(result.delta),
        "vega": np.asarray(result.vega),
        "cega_pair": np.asarray(result.rho_pair) / 5.0,  # bump was 0.05 → per 0.01
        "projection": None,
        "std_errors": result.standard_errors,
    }
