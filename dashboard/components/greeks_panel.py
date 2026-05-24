"""Greeks panel — per-ticker Δ/Vega + pairwise cega + headline aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.engine import get_full_state


def render() -> None:
    st.subheader("Greeks (dealer view)")
    st.caption(
        "Per-underlying Δ (raw and per-1% spot move), Vega (per 1 vol-pt), "
        "and pairwise correlation sensitivity, all from the dealer's perspective "
        "(dealer is short the FCN; signs are the mirror of the investor's). "
        "Sourced from the same engine that drives the P&L attribution."
    )

    current = st.session_state["current"]
    tickers = current.tickers
    state = get_full_state(current)
    # Engine returns holder-side Greeks; dealer is the mirror, so negate at the
    # display layer (no engine changes, consistent with the P&L panel).
    delta = -np.asarray(state["delta"])
    vega = -np.asarray(state["vega"])
    cega_pair = -np.asarray(state["cega_pair"])
    source = state.get("source", "?")
    spots = np.asarray(current.spots, dtype=float)

    # -----------------------------------------------------------------------
    # Headline aggregates (dealer side)
    # -----------------------------------------------------------------------
    total_delta_notional = float((delta * spots).sum())
    total_vega = float(vega.sum())

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Net Δ notional",
        f"${total_delta_notional:,.0f}",
        help=(
            "Σᵢ Δᵢ · Sᵢ from the dealer's side. The dealer is **short** the "
            "basket on a worst-of FCN — this number is negative, and its "
            "absolute value is how much long stock the dealer needs to hold "
            "to flatten the spot leg of the book."
        ),
    )
    col2.metric(
        "Net Vega (per +1.0 vol)",
        f"${total_vega:,.0f}",
        help=(
            "Σᵢ Vegaᵢ for a parallel shift of every realised vol by +1.0, dealer "
            "side. Dealer is **long** vol on a worst-of FCN → positive number → "
            "dealer gains when vol rises. Divide by 100 to get vega per +1 vol-pt."
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
            "Dealer P&L per +0.01 in the pairwise correlation between each leg. "
            "Dealer is **short** correlation on a worst-of FCN — a negative "
            "number means the dealer loses when ρ rises (equivalently, gains "
            "when ρ falls)."
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
