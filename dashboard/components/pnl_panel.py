"""P&L panel — headline numbers + Greek-based attribution waterfall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.engine import get_pricing, initial_greeks
from dashboard.pnl_attribution import PnLAttribution, attribute


def _format_pnl(x: float) -> str:
    sign = "+" if x >= 0 else "−"
    return f"{sign}${abs(x):,.0f}"


def render() -> None:
    st.subheader("P&L vs trade inception (dealer view)")

    initial = st.session_state["initial"]
    current = st.session_state["current"]
    notional = current.notional

    init_state = initial_greeks()
    cur_pricing = get_pricing(current)

    initial_price = float(init_state["price"])
    current_price = float(cur_pricing["price"])
    # Dealer is SHORT the FCN — dealer P&L is the mirror of the holder's PV move.
    # If MTM falls (structure cheaper to buy back), dealer gains.
    pnl = -(current_price - initial_price)

    days_since_issue = (current.as_of - initial.as_of).days

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "MTM price",
        f"${current_price:,.0f}",
        f"{100 * current_price / notional:.2f}% of notional",
        help=(
            "Mark-to-market value of the FCN at the current snapshot, in USD. "
            "This is the price of the structure itself (same number regardless "
            "of which side of the trade you're on). The dealer holds a short "
            "position in this — so a lower MTM is a gain on the dealer's book."
        ),
    )
    col2.metric(
        "Issue price",
        f"${initial_price:,.0f}",
        f"{100 * initial_price / notional:.2f}% of notional",
        help=(
            "Fair value at trade inception under the issue-date market state. "
            "The client paid par for this; the gap between par and Issue price "
            "is the desk's day-1 structuring margin."
        ),
    )
    col3.metric(
        "P&L (USD, dealer)",
        _format_pnl(pnl),
        f"{100 * pnl / max(abs(initial_price), 1.0):+.2f}% vs issue",
        help=(
            "Dealer P&L since inception = −(MTM − Issue). The dealer is short "
            "the FCN, so a lower MTM means the structure is cheaper to buy "
            "back and the dealer's book has gained. Decomposed by Greek "
            "bucket in the waterfall below."
        ),
    )
    col4.metric(
        "Days since issue",
        f"{days_since_issue} days",
        help="Calendar days between the issue date and the dashboard's as-of date.",
    )

    # -----------------------------------------------------------------------
    # Attribution — engine returns holder-side numbers; we flip signs for the
    # dealer view at the display layer (no engine changes).
    # -----------------------------------------------------------------------
    attr = attribute(
        initial=initial,
        current=current,
        initial_price=initial_price,
        current_price=current_price,
        initial_delta=np.asarray(init_state["delta"]),
        initial_vega=np.asarray(init_state["vega"]),
        initial_cega_pair=np.asarray(init_state["cega_pair"]),
    )

    st.markdown("### Attribution waterfall (dealer view)")
    st.caption(
        "Greek-by-Greek breakdown of dealer P&L. Bars sum to total; "
        "Residual captures Γ, cross-terms, and MC noise."
    )

    _render_waterfall(attr)
    _render_attribution_table(attr, initial, current)


def _render_waterfall(attr: PnLAttribution) -> None:
    labels = ["Spot (Δ·ΔS)", "Vol (Vega·Δσ)", "Corr (Cega·Δρ)", "Theta (proxy)", "Residual", "Total"]
    # Engine numbers are holder-side; dealer is the mirror, so negate every bar.
    values = [
        -attr.spot_pnl,
        -attr.vol_pnl,
        -attr.corr_pnl,
        -attr.theta_pnl,
        -attr.residual_pnl,
        -attr.total_pnl,
    ]
    measure = ["relative", "relative", "relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="P&L attribution",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        textposition="outside",
        text=[_format_pnl(v) for v in values],
        connector={"line": {"color": "rgb(120,120,120)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}},
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        yaxis_title="P&L (USD)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this", expanded=False):
        st.markdown(
            "All numbers are from the **dealer's** perspective (the dealer is "
            "short the FCN). Each Greek bar = − (Greek_holder × move) — the "
            "minus sign is what makes the dealer view the mirror of the holder.\n\n"
            "- **Spot** = `−Σᵢ Δᵢ(initial) · (Sᵢ_current − Sᵢ_initial)`. Dealer is short the basket; spot up → dealer loses.\n"
            "- **Vol** = `−Σᵢ Vegaᵢ(initial) · (σᵢ_current − σᵢ_initial)`. Dealer is long vol; vol up → dealer gains.\n"
            "- **Corr** = `−Σ_{i<j} cega_ij(initial) · Δρ_ij`. cega is per +0.01 in ρ. Dealer is short correlation; ρ up → dealer loses.\n"
            "- **Theta (proxy)** = `+r · V_initial · Δt`. Bond-leg approximation; full-reval theta would also reflect changes in the remaining observation tail. Treated as a sanity rail, not a P&L claim.\n"
            "- **Residual** = total dealer P&L minus the first-order sum. For large moves it picks up Γ and cross-terms; for small moves it should be close to zero modulo MC noise."
        )


def _render_attribution_table(attr: PnLAttribution, initial, current) -> None:
    st.markdown("### Per-name spot / vol contributions (dealer view)")
    rows = []
    for i, t in enumerate(current.tickers):
        rows.append({
            "Ticker": t,
            "Initial spot": f"${initial.spots[i]:,.2f}",
            "Current spot": f"${current.spots[i]:,.2f}",
            "ΔS (%)": f"{100 * (current.spots[i] / initial.spots[i] - 1):+.2f}%",
            "Spot P&L": _format_pnl(-float(attr.spot_pnl_by_name[i])),
            "Initial vol": f"{100 * initial.vols[i]:.1f}%",
            "Current vol": f"{100 * current.vols[i]:.1f}%",
            "Vol P&L": _format_pnl(-float(attr.vol_pnl_by_name[i])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
