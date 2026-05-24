"""Greeks panel — per-ticker Δ/Vega + pairwise cega + headline aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.engine import get_full_state


def render() -> None:
    st.subheader("Greeks (investor view)")
    st.caption(
        "Per-underlying Δ (raw and per-1% spot move), Vega (per 1 vol-pt), "
        "and pairwise correlation sensitivity — all from the **investor's** "
        "perspective (the note holder is long the structure). These are the "
        "raw ∂V/∂input numbers from the pricing engine. The Hedging tab "
        "flips them to the dealer's side to show what the desk needs to do."
    )

    current = st.session_state["current"]
    tickers = current.tickers
    state = get_full_state(current)
    delta = np.asarray(state["delta"])
    vega = np.asarray(state["vega"])
    cega_pair = np.asarray(state["cega_pair"])
    source = state.get("source", "?")
    spots = np.asarray(current.spots, dtype=float)

    # -----------------------------------------------------------------------
    # Headline aggregates (investor side)
    # -----------------------------------------------------------------------
    total_delta_notional = float((delta * spots).sum())
    total_vega = float(vega.sum())

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Net Δ notional",
        f"${total_delta_notional:,.0f}",
        help=(
            "Σᵢ Δᵢ · Sᵢ — the basket-equivalent USD delta exposure for the "
            "investor (positive: investor is **long** the basket on a worst-of "
            "FCN). The dealer, short the structure, would carry the mirror "
            "of this number on their book and would hedge by buying this much "
            "long stock."
        ),
    )
    col2.metric(
        "Net Vega (per +1.0 vol)",
        f"${total_vega:,.0f}",
        help=(
            "Σᵢ Vegaᵢ for a parallel shift of every realised vol by +1.0 (investor "
            "side). The investor is **short** vol on a worst-of FCN → negative "
            "number → investor loses when vol rises. Divide by 100 to get vega "
            "per +1 vol-pt."
        ),
    )
    col3.metric(
        "Compute source",
        source,
        help="`grid` = interpolated from the pre-computed scenario grid; `precise` = live MC.",
    )

    # -----------------------------------------------------------------------
    # Per-ticker table
    # -----------------------------------------------------------------------
    st.markdown("### Per-ticker Greeks")
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker": t,
            "Spot": f"${spots[i]:,.2f}",
            "Δ (raw)": f"{delta[i]:,.4f}",
            "Δ per 1% spot": f"${delta[i] * spots[i] / 100:,.0f}",
            "Δ notional": f"${delta[i] * spots[i]:,.0f}",
            "Vega per vol-pt": f"${vega[i] / 100:,.0f}",
            "Vega per +1.0 vol": f"${vega[i]:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Cega (pairwise) — small d ⇒ flat table is fine
    # -----------------------------------------------------------------------
    if len(tickers) >= 2:
        st.markdown("### Pairwise correlation sensitivity (cega)")
        st.caption(
            "Investor P&L per +0.01 in the pairwise correlation between each leg. "
            "The investor is **long** correlation on a worst-of FCN — a positive "
            "number means the investor gains when ρ rises (the worst-of becomes "
            "less volatile as the names move together). The dealer carries the "
            "mirror."
        )
        crows = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                v = float(cega_pair[i, j])
                crows.append({
                    "Pair": f"{tickers[i]} × {tickers[j]}",
                    "ρ (current)": f"{float(current.corr[i, j]):+.3f}",
                    "cega (per +0.01)": f"${v:,.0f}",
                    "P&L per +5% in ρ": f"${5 * v:,.0f}",
                })
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)
