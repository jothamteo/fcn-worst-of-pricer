r"""Greeks for the worst-of FCN.

Two engines, exercised the same way and reported side-by-side:

* **MC, bump-and-revalue with common random numbers (CRN).** Pre-draw one
  block of standard normals, then re-price the FCN with the parameter of
  interest bumped up and down, reusing the same normals each time. CRN
  removes the bulk of the sampling noise so the finite-difference signal
  is the parameter sensitivity, not Monte Carlo variance.

* **PDE, off-grid.** Δ and Γ are read straight off the 1D Crank–Nicolson
  value grid via central differences in $S$. Vega uses a small bump (no
  CRN needed — the PDE is noiseless). Correlation sensitivity is not
  defined in 1D and is MC-only.

Sensitivity conventions follow desk practice on FCNs:

* Δ_i = ∂V/∂S_i with $S_i$ in price units (so the unit is "value per
  dollar of spot move"). Reported here both raw and scaled to a 1% move.
* Γ_i = ∂²V/∂S_i² in price units; we also report the scaled per-1% form.
* Vega_i = ∂V/∂σ_i for a 1 vol-point bump (so vega units = value per
  1.00 vol; divide by 100 for value per 1 vol-pt).
* ρ_{ij} = ∂V/∂ρ_{ij} for a 0.01 bump in the off-diagonal correlation.

References:
* Glasserman (2003), Ch. 7 — sensitivity estimation, bump-and-revalue,
  pathwise method, and the noise-near-discontinuity issue.
* Wilmott (2006) — finite-difference Greeks off a PDE grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fcn_payoff import (
    FCNProduct,
    ObservationGrid,
    payoff_per_path,
)
from .gbm_simulation import SimulationConfig, draw_normals, simulate_paths
from .market_data import MarketData
from .pde_pricer import price_fcn_pde_1d
from .utils import is_psd, nearest_psd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreekResult:
    """Output of `mc_greeks_bump` / `pde_greeks_1d`.

    Per-underlying arrays are length `d` (1 for the PDE single-asset case).
    `rho_pair` is a `(d, d)` symmetric matrix, zero on the diagonal.

    Attributes
    ----------
    price : float
        Base-case price under the same engine that produced the Greeks.
    delta, gamma, vega : np.ndarray
        Per-underlying first and second spot derivatives, and vol derivative.
    delta_pct, gamma_pct, vega_per_volpt : np.ndarray
        Desk-style scaled views: Δ × spot / 100 (value per 1% spot move),
        Γ × spot² / 10_000 (value per 1% × 1% move), vega / 100 (per vol-pt).
    rho_pair : np.ndarray
        Pairwise correlation sensitivity (off-diagonal only; diagonals = 0).
        Bumped by `corr_bump` (default 0.05); reported per 0.01 bump.
    standard_errors : dict
        MC standard errors for each Greek (omitted for PDE results).
    diagnostics : dict
        Provenance: bump sizes, n_paths, seed, etc.
    """

    price: float
    delta: np.ndarray = field(repr=False)
    gamma: np.ndarray = field(repr=False)
    vega: np.ndarray = field(repr=False)
    delta_pct: np.ndarray = field(repr=False)
    gamma_pct: np.ndarray = field(repr=False)
    vega_per_volpt: np.ndarray = field(repr=False)
    rho_pair: np.ndarray = field(repr=False)
    standard_errors: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def summary(self, tickers: Optional[list[str]] = None) -> str:
        d = len(self.delta)
        names = tickers or [f"asset_{i}" for i in range(d)]
        lines = [f"Base price : {self.price:,.4f}", "", "Per-underlying Greeks (1% spot, 1 vol-pt):"]
        header = f"  {'name':<8} {'Δ_per_1%':>12} {'Γ_per_1%×1%':>14} {'vega_per_volpt':>16}"
        lines.append(header)
        for i, name in enumerate(names):
            lines.append(
                f"  {name:<8} {self.delta_pct[i]:>12,.4f} {self.gamma_pct[i]:>14,.4f}"
                f" {self.vega_per_volpt[i]:>16,.4f}"
            )
        if self.rho_pair.size and d >= 2:
            lines.append("")
            lines.append("Pairwise correlation sensitivity (per 0.01 bump):")
            for i in range(d):
                for j in range(i + 1, d):
                    lines.append(
                        f"  ρ_pair[{names[i]},{names[j]}] = {self.rho_pair[i, j] / 100:,.4f}"
                    )
        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MC: shared infrastructure (pre-draw normals, re-price with bumped params)
# ---------------------------------------------------------------------------


def _price_with_overrides(
    market: MarketData,
    product: FCNProduct,
    grid: ObservationGrid,
    normals: np.ndarray,
    *,
    spots: Optional[np.ndarray] = None,
    vols: Optional[np.ndarray] = None,
    corr: Optional[np.ndarray] = None,
    antithetic: bool = True,
) -> tuple[float, float, np.ndarray]:
    r"""Re-price the FCN reusing pre-drawn `normals` with optional bumped inputs.

    Returns `(mean_pv, std_err, pv_samples)`. The same `normals` array is used
    across calls — this is the CRN guarantee.

    **Spot-bump semantics (important).** The worst-of FCN payoff is invariant
    under a simultaneous scaling of the simulation start and the payoff's
    initial-fixing denominator. To get a non-zero Δ we adopt the desk
    convention: the **initial fixing** $S_i(0)$ stays frozen at
    `market.spots[i]` (it was locked in at trade date), and the bump moves
    only the simulation's starting point — i.e. *today's spot*. We therefore
    pass `spots=spots_use` into the simulator but always pass
    `market.spots` (un-bumped) into the payoff function.
    """
    spots_use = market.spots if spots is None else spots
    vols_use = market.vols if vols is None else vols
    corr_use = market.corr if corr is None else corr
    if not is_psd(corr_use):
        corr_use = nearest_psd(corr_use)

    cfg = SimulationConfig(
        spots=spots_use,
        vols=vols_use,
        divs=market.divs,
        rate=market.rate,
        corr=corr_use,
        T=grid.sim_T,
        n_steps=grid.sim_n_steps,
        n_paths=normals.shape[0],
        antithetic=antithetic,
        seed=None,
    )
    paths = simulate_paths(cfg, normals=normals)
    # NB: use the *un-bumped* initial fixing as the payoff's normalisation —
    # see the spot-bump semantics note above. Only the simulation start has
    # moved.
    pv = payoff_per_path(
        paths=paths, spots=np.asarray(market.spots, dtype=float),
        product=product, grid=grid, rate=market.rate,
    )
    n_total = pv.shape[0]
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n_total)), pv


def mc_greeks_bump(
    market: MarketData,
    product: FCNProduct,
    n_paths: int = 80_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
    spot_bump_rel: float = 0.01,
    vol_bump_abs: float = 0.01,
    corr_bump_abs: float = 0.05,
    day_count: float = 365.0,
) -> GreekResult:
    r"""Compute (Δ, Γ, vega, ρ_pair) by bump-and-revalue MC under CRN.

    Parameters
    ----------
    market, product :
        Market snapshot and product spec. As in `mc_pricer.price_fcn`.
    n_paths : int
        Primal path count. Effective path count is `2 × n_paths` with antithetic.
    antithetic : bool
        Use antithetic variates. Always on by default; CRN composes cleanly with
        the antithetic flip.
    seed : int, optional
        PRNG seed for the shared block of standard normals.
    spot_bump_rel : float
        Relative bump for Δ/Γ: $S_i^\pm = S_i \cdot (1 \pm \text{spot\_bump\_rel})$.
        Default 1%, which is the desk convention for an equity FCN.
    vol_bump_abs : float
        Absolute bump (in vol points) for vega. Default 0.01 (one vol pt).
    corr_bump_abs : float
        Absolute bump for ρ_pair. Default 0.05 (= 5 correlation points). We
        re-project the bumped correlation matrix onto the PSD cone if needed.
    day_count : float
        ACT/`day_count`.

    Returns
    -------
    GreekResult
    """
    grid = ObservationGrid.from_product(product=product, day_count=day_count)
    rng = np.random.default_rng(seed)
    normals = draw_normals(n_paths=n_paths, n_steps=grid.sim_n_steps, d=len(market.spots), rng=rng)

    base_price, base_se, base_pv = _price_with_overrides(
        market, product, grid, normals, antithetic=antithetic
    )

    d = len(market.spots)
    delta = np.zeros(d)
    gamma = np.zeros(d)
    vega = np.zeros(d)
    se_delta = np.zeros(d)
    se_gamma = np.zeros(d)
    se_vega = np.zeros(d)
    rho_pair = np.zeros((d, d))
    se_rho = np.zeros((d, d))

    spots = np.array(market.spots, dtype=float)
    vols = np.array(market.vols, dtype=float)

    # --- Δ, Γ: bump each spot independently, three-point central difference -----
    for i in range(d):
        eps = spot_bump_rel * spots[i]
        spots_up = spots.copy(); spots_up[i] += eps
        spots_dn = spots.copy(); spots_dn[i] -= eps
        p_up, se_up, pv_up = _price_with_overrides(
            market, product, grid, normals, spots=spots_up, antithetic=antithetic
        )
        p_dn, se_dn, pv_dn = _price_with_overrides(
            market, product, grid, normals, spots=spots_dn, antithetic=antithetic
        )
        delta[i] = (p_up - p_dn) / (2.0 * eps)
        gamma[i] = (p_up - 2.0 * base_price + p_dn) / (eps * eps)
        # SE via differences in the per-path estimates (CRN makes this the
        # natural estimator — Glasserman §7.1.2).
        diff_d = (pv_up - pv_dn) / (2.0 * eps)
        diff_g = (pv_up - 2.0 * base_pv + pv_dn) / (eps * eps)
        n_total = pv_up.shape[0]
        se_delta[i] = float(diff_d.std(ddof=1) / np.sqrt(n_total))
        se_gamma[i] = float(diff_g.std(ddof=1) / np.sqrt(n_total))

    # --- Vega: bump each vol independently -------------------------------------
    for i in range(d):
        vols_up = vols.copy(); vols_up[i] += vol_bump_abs
        vols_dn = vols.copy(); vols_dn[i] -= vol_bump_abs
        p_up, _, pv_up = _price_with_overrides(
            market, product, grid, normals, vols=vols_up, antithetic=antithetic
        )
        p_dn, _, pv_dn = _price_with_overrides(
            market, product, grid, normals, vols=vols_dn, antithetic=antithetic
        )
        vega[i] = (p_up - p_dn) / (2.0 * vol_bump_abs)
        diff_v = (pv_up - pv_dn) / (2.0 * vol_bump_abs)
        n_total = pv_up.shape[0]
        se_vega[i] = float(diff_v.std(ddof=1) / np.sqrt(n_total))

    # --- ρ_pair: bump each off-diagonal correlation, re-Cholesky ---------------
    for i in range(d):
        for j in range(i + 1, d):
            corr_up = np.array(market.corr, dtype=float)
            corr_dn = np.array(market.corr, dtype=float)
            corr_up[i, j] += corr_bump_abs; corr_up[j, i] += corr_bump_abs
            corr_dn[i, j] -= corr_bump_abs; corr_dn[j, i] -= corr_bump_abs
            corr_up = np.clip(corr_up, -0.999, 0.999); np.fill_diagonal(corr_up, 1.0)
            corr_dn = np.clip(corr_dn, -0.999, 0.999); np.fill_diagonal(corr_dn, 1.0)
            p_up, _, pv_up = _price_with_overrides(
                market, product, grid, normals, corr=corr_up, antithetic=antithetic
            )
            p_dn, _, pv_dn = _price_with_overrides(
                market, product, grid, normals, corr=corr_dn, antithetic=antithetic
            )
            rho_pair[i, j] = (p_up - p_dn) / (2.0 * corr_bump_abs)
            rho_pair[j, i] = rho_pair[i, j]
            diff_r = (pv_up - pv_dn) / (2.0 * corr_bump_abs)
            n_total = pv_up.shape[0]
            se_rho[i, j] = float(diff_r.std(ddof=1) / np.sqrt(n_total))
            se_rho[j, i] = se_rho[i, j]

    delta_pct = delta * spots / 100.0
    gamma_pct = gamma * spots * spots / 10_000.0
    vega_per_volpt = vega / 100.0

    return GreekResult(
        price=base_price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        delta_pct=delta_pct,
        gamma_pct=gamma_pct,
        vega_per_volpt=vega_per_volpt,
        rho_pair=rho_pair,
        standard_errors={
            "price": base_se,
            "delta": se_delta,
            "gamma": se_gamma,
            "vega": se_vega,
            "rho_pair": se_rho,
        },
        diagnostics={
            "engine": "mc_bump_crn",
            "tickers": list(market.tickers),
            "spots": spots.tolist(),
            "spot_bump_rel": spot_bump_rel,
            "vol_bump_abs": vol_bump_abs,
            "corr_bump_abs": corr_bump_abs,
            "n_paths_total": 2 * n_paths if antithetic else n_paths,
            "antithetic": antithetic,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# PDE Greeks (1D single-asset reduction)
# ---------------------------------------------------------------------------


def pde_greeks_1d(
    spot: float,
    vol: float,
    div: float,
    rate: float,
    product: FCNProduct,
    n_space: int = 1600,
    n_time_per_period: int = 160,
    x_range_sigma: float = 6.0,
    vol_bump_abs: float = 0.01,
    day_count: float = 365.0,
) -> GreekResult:
    r"""1D PDE Greeks (Δ, Γ, vega) for the single-asset FCN reduction.

    Δ and Γ are read off the value grid by central differences in $S$ at the
    grid point closest to $x = 0$ (i.e. $S = S_0$). Vega is a small bump-and-
    revalue (the PDE is deterministic, so this is just two PDE solves — no
    CRN needed and no Monte Carlo noise).

    `rho_pair` is identically zero in 1D and reported as an empty array.
    """
    base = price_fcn_pde_1d(
        spot=spot, vol=vol, div=div, rate=rate, product=product,
        n_space=n_space, n_time_per_period=n_time_per_period,
        x_range_sigma=x_range_sigma, day_count=day_count,
    )
    S_grid = base.S_grid
    V = base.V_at_issue
    # Locate the spot index (snap-to-zero grid puts S_0 exactly at the midpoint).
    i = int(np.argmin(np.abs(S_grid - spot)))
    # Central difference in S using the non-uniform price grid (log-spot is
    # uniform; S = S_0·exp(x) is not). Use the local 3-point formula:
    #   Δ ≈ (V[i+1] − V[i−1]) / (S[i+1] − S[i−1])
    #   Γ ≈ 2·[(S[i]−S[i−1])·V[i+1] − (S[i+1]−S[i−1])·V[i] + (S[i+1]−S[i])·V[i−1]]
    #         / [ (S[i+1]−S[i])·(S[i]−S[i−1])·(S[i+1]−S[i−1]) ]
    h_minus = S_grid[i] - S_grid[i - 1]
    h_plus = S_grid[i + 1] - S_grid[i]
    h_total = S_grid[i + 1] - S_grid[i - 1]
    delta = (V[i + 1] - V[i - 1]) / h_total
    gamma = 2.0 * (
        h_minus * V[i + 1] - h_total * V[i] + h_plus * V[i - 1]
    ) / (h_plus * h_minus * h_total)

    # Vega: bump-and-revalue.
    up = price_fcn_pde_1d(
        spot=spot, vol=vol + vol_bump_abs, div=div, rate=rate, product=product,
        n_space=n_space, n_time_per_period=n_time_per_period,
        x_range_sigma=x_range_sigma, day_count=day_count,
    )
    dn = price_fcn_pde_1d(
        spot=spot, vol=max(1e-6, vol - vol_bump_abs), div=div, rate=rate, product=product,
        n_space=n_space, n_time_per_period=n_time_per_period,
        x_range_sigma=x_range_sigma, day_count=day_count,
    )
    vega = (up.price - dn.price) / (2.0 * vol_bump_abs)

    return GreekResult(
        price=base.price,
        delta=np.array([delta]),
        gamma=np.array([gamma]),
        vega=np.array([vega]),
        delta_pct=np.array([delta * spot / 100.0]),
        gamma_pct=np.array([gamma * spot * spot / 10_000.0]),
        vega_per_volpt=np.array([vega / 100.0]),
        rho_pair=np.zeros((1, 1)),
        standard_errors={},
        diagnostics={
            "engine": "pde_offgrid",
            "spot": spot,
            "vol": vol,
            "div": div,
            "rate": rate,
            "n_space": n_space,
            "n_time_per_period": n_time_per_period,
            "x_range_sigma": x_range_sigma,
            "vol_bump_abs": vol_bump_abs,
            "i_at_spot": i,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: a barrier-distance scan that demonstrates the MC noise problem
# ---------------------------------------------------------------------------


def mc_delta_curve(
    market: MarketData,
    product: FCNProduct,
    spot_grid: np.ndarray,
    asset_index: int = 0,
    n_paths: int = 40_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
    spot_bump_rel: float = 0.01,
    day_count: float = 365.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sweep the MC Δ for one underlying across a range of *its* spot.

    Used in the "MC Greeks are noisy near barriers" comparison plot — we walk
    `S_i` over `spot_grid` keeping the other underlyings fixed, and report
    Δ_i + its MC standard error at each point.

    Returns `(delta_values, delta_se_values)`, each of shape `spot_grid.shape`.
    """
    grid = ObservationGrid.from_product(product=product, day_count=day_count)
    rng = np.random.default_rng(seed)
    normals = draw_normals(n_paths=n_paths, n_steps=grid.sim_n_steps, d=len(market.spots), rng=rng)

    deltas = np.zeros_like(spot_grid, dtype=float)
    delta_ses = np.zeros_like(spot_grid, dtype=float)
    for k, s_target in enumerate(spot_grid):
        spots_base = np.array(market.spots, dtype=float)
        spots_base[asset_index] = float(s_target)
        eps = spot_bump_rel * float(s_target)
        spots_up = spots_base.copy(); spots_up[asset_index] += eps
        spots_dn = spots_base.copy(); spots_dn[asset_index] -= eps
        _, _, pv_up = _price_with_overrides(
            market, product, grid, normals, spots=spots_up, antithetic=antithetic
        )
        _, _, pv_dn = _price_with_overrides(
            market, product, grid, normals, spots=spots_dn, antithetic=antithetic
        )
        diff = (pv_up - pv_dn) / (2.0 * eps)
        deltas[k] = float(diff.mean())
        delta_ses[k] = float(diff.std(ddof=1) / np.sqrt(diff.shape[0]))
    return deltas, delta_ses


def pde_delta_curve(
    vol: float,
    div: float,
    rate: float,
    product: FCNProduct,
    spot_grid: np.ndarray,
    n_space: int = 1600,
    n_time_per_period: int = 160,
    x_range_sigma: float = 6.0,
    day_count: float = 365.0,
) -> np.ndarray:
    r"""Sweep the PDE Δ across spot — the "smooth" reference for the MC plot.

    Returns Δ-values of shape `spot_grid.shape`. We re-grid for each spot to
    keep the grid centred on the new $S$.
    """
    out = np.zeros_like(spot_grid, dtype=float)
    for k, s in enumerate(spot_grid):
        g = pde_greeks_1d(
            spot=float(s), vol=vol, div=div, rate=rate, product=product,
            n_space=n_space, n_time_per_period=n_time_per_period,
            x_range_sigma=x_range_sigma, day_count=day_count,
        )
        out[k] = float(g.delta[0])
    return out
