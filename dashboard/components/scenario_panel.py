"""Scenarios panel — predefined one-click scenarios + custom builder.

The named scenarios use *historical* drawdowns for the AMZN / META / MU
basket rather than the brief's symmetric default. AMZN −50% / META −65% /
MU −45% are the actual 2022 peak-to-trough closes for these names —
both more dramatic and more honest than a flat −20% across the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import streamlit as st

from dashboard.state import MarketSnapshot


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    apply: Callable[[MarketSnapshot, MarketSnapshot], MarketSnapshot]


def _shift_snapshot(
    initial: MarketSnapshot, current: MarketSnapshot,
    *,
    spot_pct: Optional[dict] = None,
    spot_pct_basket: Optional[float] = None,
    vol_abs: Optional[dict] = None,
    vol_abs_basket: Optional[float] = None,
    corr_abs: Optional[float] = None,
) -> MarketSnapshot:
    """Build a new snapshot by applying named shocks to ``initial``.

    Every scenario is anchored on the *initial* (trade-at-issue) snapshot,
    not on the current one, so re-clicking a button is idempotent.
    """
    new_spots = np.asarray(initial.spots, dtype=float).copy()
    new_vols = np.asarray(initial.vols, dtype=float).copy()
    new_corr = np.asarray(initial.corr, dtype=float).copy()

    if spot_pct_basket is not None:
        new_spots = new_spots * (1.0 + spot_pct_basket)
    if spot_pct is not None:
        for i, t in enumerate(initial.tickers):
            if t in spot_pct:
                new_spots[i] = float(initial.spots[i]) * (1.0 + spot_pct[t])

    if vol_abs_basket is not None:
        new_vols = np.clip(new_vols + vol_abs_basket, 0.01, None)
    if vol_abs is not None:
        for i, t in enumerate(initial.tickers):
            if t in vol_abs:
                new_vols[i] = max(0.01, float(initial.vols[i]) + vol_abs[t])

    if corr_abs is not None:
        d = new_corr.shape[0]
        off = ~np.eye(d, dtype=bool)
        new_corr[off] = np.clip(new_corr[off] + corr_abs, -0.999, 0.999)
        np.fill_diagonal(new_corr, 1.0)

    return MarketSnapshot(
        tickers=initial.tickers,
        spots=new_spots,
        vols=new_vols,
        divs=current.divs.copy(),
        corr=new_corr,
        rate=current.rate,
        as_of=current.as_of,
        notional=current.notional,
    )


# ---------------------------------------------------------------------------
# Predefined scenarios
# ---------------------------------------------------------------------------


def _covid_2020(initial, current):
    # Mar-2020 style: deep basket sell-off, vol spike, correlation pinned to 1
    return _shift_snapshot(
        initial, current,
        spot_pct_basket=-0.30,
        vol_abs_basket=+0.20,
        corr_abs=+0.25,
    )


def _tech_rout_2022(initial, current):
    # Actual AMZN/META/MU 2022 peak-to-trough closes:
    #   AMZN: $187 → $84  (≈ −55%, dashboard uses round −50%)
    #   META: $384 → $88  (≈ −77%, dashboard uses round −65%)
    #   MU:   $96  → $48  (≈ −50%, dashboard uses round −45%)
    return _shift_snapshot(
        initial, current,
        spot_pct={"AMZN": -0.50, "META": -0.65, "MU": -0.45},
        vol_abs_basket=+0.10,
        corr_abs=+0.10,
    )


def _vol_spike(initial, current):
    return _shift_snapshot(
        initial, current,
        vol_abs_basket=+0.15,
    )


def _correlation_crisis(initial, current):
    return _shift_snapshot(
        initial, current,
        corr_abs=+0.20,
        spot_pct_basket=-0.05,
    )


def _mild_bull(initial, current):
    return _shift_snapshot(
        initial, current,
        spot_pct_basket=+0.10,
        vol_abs_basket=-0.05,
    )


SCENARIOS: list[Scenario] = [
    Scenario(
        "Covid 2020",
        "Spots −30%, vols +20 pts, correlation +0.25 — March 2020 sell-off shape.",
        _covid_2020,
    ),
    Scenario(
        "Tech Rout 2022",
        "Historical 2022 peak-to-trough closes: AMZN −50%, META −65%, MU −45%; vols +10 pts; ρ +0.10.",
        _tech_rout_2022,
    ),
    Scenario(
        "Vol Spike",
        "All vols +15 vol-pts, spots and correlation unchanged.",
        _vol_spike,
    ),
    Scenario(
        "Correlation Crisis",
        "Off-diagonal ρ +0.20, basket spot −5%.",
        _correlation_crisis,
    ),
    Scenario(
        "Mild Bull",
        "Basket +10%, vols −5 pts.",
        _mild_bull,
    ),
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render() -> None:
    st.subheader("Scenarios")
    st.caption(
        "One-click historical / stylised scenarios. Each shifts the **initial** "
        "snapshot (not the current one) and pushes the result into the live "
        "state — every other panel updates accordingly."
    )

    initial = st.session_state["initial"]
    current = st.session_state["current"]

    cols = st.columns(len(SCENARIOS))
    for col, scen in zip(cols, SCENARIOS):
        if col.button(scen.name, use_container_width=True, help=scen.description):
            new_snap = scen.apply(initial, current)
            st.session_state["current"] = new_snap
            st.session_state["current_pricing"] = None
            st.session_state["current_greeks"] = None
            st.toast(f"Applied scenario: {scen.name}")
            st.rerun()

    # Reset
    if st.button("↺ Reset to issue", help="Restore the current snapshot back to the trade-at-issue state."):
        from dashboard.state import reset_current_to_initial
        reset_current_to_initial()
        st.rerun()

    # -----------------------------------------------------------------------
    # Custom builder — collapsed so the named buttons stay visually primary.
    # -----------------------------------------------------------------------
    with st.expander("Custom shock builder", expanded=False):
        st.caption(
            "Dial in an arbitrary basket / vol / correlation shock. Applies on "
            "top of the **initial** snapshot — independent of the sidebar."
        )
        c_spot = st.slider(
            "Basket spot shift", -0.50, +0.50, 0.0, 0.01, format="%.0f%%",
            help="Same % applied to every spot.",
        )
        c_vol = st.slider(
            "Basket vol shift (abs vol pts)", -0.20, +0.30, 0.0, 0.01, format="%+.2f",
            help="Absolute change in vol points applied to every realised vol.",
        )
        c_corr = st.slider(
            "Off-diagonal corr shift", -0.30, +0.30, 0.0, 0.01, format="%+.2f",
            help="Added to every off-diagonal pairwise correlation; clipped & PSD-projected.",
        )
        if st.button("Apply custom shock", type="primary"):
            new_snap = _shift_snapshot(
                initial, current,
                spot_pct_basket=c_spot, vol_abs_basket=c_vol, corr_abs=c_corr,
            )
            st.session_state["current"] = new_snap
            st.session_state["current_pricing"] = None
            st.session_state["current_greeks"] = None
            st.rerun()

    # -----------------------------------------------------------------------
    # Current state at-a-glance, post-scenario
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("**Current snapshot vs issue:**")
    rows = []
    for i, t in enumerate(initial.tickers):
        rows.append(
            f"`{t}`  spot  ${initial.spots[i]:,.2f} → ${current.spots[i]:,.2f}  "
            f"({100 * (current.spots[i] / initial.spots[i] - 1):+.1f}%);  "
            f"vol  {100 * initial.vols[i]:.1f}% → {100 * current.vols[i]:.1f}%"
        )
    st.markdown("\n".join(f"- {r}" for r in rows))
