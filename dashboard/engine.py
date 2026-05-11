"""Single repricing facade used by every panel.

Panels call :func:`get_pricing` / :func:`get_greeks` instead of touching
``src.mc_pricer`` / ``src.greeks`` directly. This keeps the Fast vs Precise
dispatch in one place and lets us swap in a richer interpolator later.

* **Fast** mode: read from the pre-computed scenario grid (built once on
  demand, then cached in ``st.session_state.scenario_grid``).
* **Precise** mode: run a reduced-path MC against the snapshot. Used when
  the user wants the exact value at independent (per-name) inputs.

The grid is large enough that it cannot be pickled cheaply across reruns,
so we keep it in ``st.session_state`` rather than in ``st.cache_*``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from dashboard.precompute import (
    GridAxes,
    ScenarioGrid,
    build_scenario_grid,
    precise_greeks,
    precise_price,
)
from dashboard.state import MarketSnapshot


# ---------------------------------------------------------------------------
# Grid lifecycle
# ---------------------------------------------------------------------------


def ensure_grid(*, force: bool = False, high_res: Optional[bool] = None) -> ScenarioGrid:
    """Return the cached grid, building it if missing.

    Parameters
    ----------
    force : bool
        Rebuild even if the cached grid is fresh.
    high_res : bool, optional
        If True, build the full default-axes grid (~100s, more accurate).
        If False or None, build the small ``GridAxes.fast()`` grid (~10s) —
        this is the first-load default so the page paints quickly. The
        sidebar's "Rebuild grid (high-res)" button promotes to the full grid.
    """
    cached = st.session_state.get("scenario_grid")
    if cached is not None and not force:
        return cached

    initial = st.session_state["initial"]
    product = st.session_state["product"]

    if high_res is None:
        high_res = bool(st.session_state.get("grid_high_res", False))

    axes = GridAxes.default() if high_res else GridAxes.fast()
    n_paths = 8_000 if high_res else 4_000

    progress = st.sidebar.progress(0.0, text="Building scenario grid…")

    def _cb(done: int, total_: int) -> None:
        progress.progress(done / max(total_, 1), text=f"Grid {done}/{total_}…")

    grid = build_scenario_grid(
        initial=initial, product=product, axes=axes,
        n_paths=n_paths, progress_cb=_cb,
    )
    progress.empty()
    st.session_state["scenario_grid"] = grid
    return grid


# ---------------------------------------------------------------------------
# Public API used by panels
# ---------------------------------------------------------------------------


def get_pricing(snapshot: MarketSnapshot, *, mode: Optional[str] = None) -> dict:
    """Return ``{price, source}`` for ``snapshot`` under the chosen mode.

    ``source`` is the string label used in tooltips: ``"grid"`` (interp from
    pre-computed grid) or ``"precise"`` (live MC).
    """
    if mode is None:
        mode = st.session_state.get("compute_mode", "Fast")
    if mode == "Precise":
        price, se = precise_price(snapshot, st.session_state["product"], n_paths=20_000)
        return {"price": price, "std_error": se, "source": "precise"}

    grid = ensure_grid()
    interp = grid.interpolate(snapshot)
    return {"price": interp["price"], "std_error": None, "source": "grid"}


def get_full_state(snapshot: MarketSnapshot, *, mode: Optional[str] = None) -> dict:
    """Return ``{price, delta, vega, cega_pair, source}`` for ``snapshot``.

    Always uses Precise live MC — Greeks are not pre-computed in the grid.
    The ``mode`` argument is kept for parity with :func:`get_pricing` but
    only affects which path-count is used: Fast→fewer paths→faster, Precise
    →more paths→more accurate.
    """
    if mode is None:
        mode = st.session_state.get("compute_mode", "Fast")
    n_paths = 10_000 if mode == "Fast" else 20_000
    out = precise_greeks(snapshot, st.session_state["product"], n_paths=n_paths)
    out["source"] = "precise" if mode == "Precise" else "precise-fast"
    return out


def initial_greeks() -> dict:
    """Return ``{price, delta, vega, cega_pair}`` at the *initial* snapshot.

    Cached in session_state because every panel needs it for attribution.
    Uses Precise MC at higher path count for accuracy at the linearisation
    point — this is run once on first load, so cost is one-shot.
    """
    cached = st.session_state.get("initial_full_state")
    if cached is not None:
        return cached
    initial = st.session_state["initial"]
    out = precise_greeks(initial, st.session_state["product"], n_paths=40_000, seed=20260101)
    st.session_state["initial_full_state"] = out
    return out
