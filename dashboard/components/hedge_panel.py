"""Hedging panel — delta hedge, vega hedge, correlation risk, residuals.

Issuer perspective: the bank has sold the FCN. ``src.greeks.mc_greeks_bump``
returns the holder's Greeks; the hedge translates those into the dealer
action (sign convention documented in :mod:`src.hedging`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.engine import get_full_state
from dashboard.pnl_attribution import attribute
from src.hedging import (
    black_scholes_call,
    compute_correlation_risk,
    compute_delta_hedge,
    compute_unhedgeable_risk_summary,
    compute_vega_hedge,
)


def _format_pnl(x: float) -> str:
    sign = "+" if x >= 0 else "−"
    return f"{sign}${abs(x):,.0f}"


def render() -> None:
    st.subheader("Hedging")
    st.caption(
        "Issuer-side hedge ratios derived from the current Greeks. Numbers "
        "assume the dealer is the seller of the FCN; sign conventions follow "
        "`src.hedging`."
    )

    current = st.session_state["current"]
    state = get_full_state(current)
    delta = np.asarray(state["delta"])
    vega = np.asarray(state["vega"])
    cega_pair = np.asarray(state["cega_pair"])
    tickers = list(current.tickers)
    spots = {t: float(current.spots[i]) for i, t in enumerate(tickers)}

    # -----------------------------------------------------------------------
    # Delta hedge
    # -----------------------------------------------------------------------
    st.markdown("### 1) Delta hedge")
    delta_rows = compute_delta_hedge(
        deltas={t: float(delta[i]) for i, t in enumerate(tickers)},
        notional=current.notional,
        spots=spots,
    )
    df_delta = pd.DataFrame([
        {
            "Ticker": r.ticker,
            "Δ (raw)": f"{r.delta:.4f}",
            "Spot": f"${r.spot:,.2f}",
            "Shares to buy": f"{r.shares_to_buy:+,d}",
            "Hedge notional": f"${r.hedge_notional:,.0f}",
            "Hedge / trade %": f"{r.hedge_pct_of_trade:+.2f}%",
        }
        for r in delta_rows.values()
    ])
    st.dataframe(df_delta, use_container_width=True, hide_index=True)
    st.caption(
        "Dealer action per ticker: positive `Shares to buy` = literally buy "
        "that many shares; negative = short-sell. The total hedge notional is "
        "what the dealer keeps on the spot leg of the book."
    )

    # -----------------------------------------------------------------------
    # Vega hedge — ATM 3M listed-call proxy via Black-Scholes
    # -----------------------------------------------------------------------
    st.markdown("### 2) Vega hedge (ATM 3M listed-call proxy)")
    T_hedge = 0.25
    listed_vegas = {}
    listed_prices = {}
    for i, t in enumerate(tickers):
        price, vega_pt = black_scholes_call(
            spot=float(current.spots[i]),
            strike=float(current.spots[i]),         # ATM
            T=T_hedge,
            vol=max(float(current.vols[i]), 0.05),
            rate=float(current.rate),
            div=float(current.divs[i]),
        )
        # `black_scholes_call` returns vega per +1 vol pt (i.e. +0.01).
        listed_vegas[t] = vega_pt * 100.0           # rescale to per +1.0 in vol
        listed_prices[t] = price
    # Dealer carries +ve vega from selling FCN → dealer_vega = +holder_vega.
    dealer_vegas = {t: float(vega[i]) for i, t in enumerate(tickers)}
    vega_rows = compute_vega_hedge(
        vegas=dealer_vegas,
        listed_option_vegas=listed_vegas,
        listed_option_prices=listed_prices,
    )
    df_vega = pd.DataFrame([
        {
            "Ticker": r.ticker,
            "FCN vega (dealer)": f"${r.fcn_vega:,.0f}",
            "Listed-call price": f"${r.listed_option_price:,.2f}",
            "Listed-call vega": f"${r.listed_option_vega:,.0f}",
            "# options (signed)": f"{r.n_options_to_buy:+,d}",
            "Total premium (signed)": (
                f"${r.total_premium:,.0f}" if not math.isnan(r.total_premium) else "—"
            ),
            "Residual vega": f"${r.residual_vega:,.0f}",
        }
        for r in vega_rows.values()
    ])
    st.dataframe(df_vega, use_container_width=True, hide_index=True)
    st.caption(
        "Negative `# options` means the dealer **sells** that many listed ATM "
        "3M calls per ticker to neutralise the FCN's vol exposure. Premium is "
        "signed (positive = dealer pays). The listed-call price/vega here is a "
        "Black-Scholes stand-in, not a live chain quote."
    )

    # -----------------------------------------------------------------------
    # Correlation risk
    # -----------------------------------------------------------------------
    st.markdown("### 3) Correlation risk")
    # Average pairwise cega → desk-level corr sensitivity.
    d = cega_pair.shape[0]
    cega_total = 0.0
    n_pairs = 0
    for i in range(d):
        for j in range(i + 1, d):
            cega_total += float(cega_pair[i, j])
            n_pairs += 1
    cega_avg = cega_total / max(n_pairs, 1)

    # Build a tiny price-at-corr-levels scan from the cega slope.
    base_price = float(state["price"])
    price_at_levels = {
        -0.10: base_price + (-0.10) * 100 * cega_avg,
        -0.05: base_price + (-0.05) * 100 * cega_avg,
         0.00: base_price,
         0.05: base_price + ( 0.05) * 100 * cega_avg,
         0.10: base_price + ( 0.10) * 100 * cega_avg,
    }
    report = compute_correlation_risk(price_at_corr_levels=price_at_levels)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "cega (per +0.01)",
        f"${report.cega:,.0f}",
        help="Average pairwise correlation sensitivity across the basket.",
    )
    c2.metric(
        "P&L per +5% in ρ",
        f"${report.correlation_pnl_per_5pct_move:,.0f}",
        help="Linear extrapolation of cega across a 5 percentage-point parallel ρ rise.",
    )
    c3.metric(
        "Hedgeable?",
        "No",
        help=(
            "There is no liquid single-stock correlation product for a generic "
            "3-name basket. See `HEDGING_NOTES.md` for the desk-level mitigants."
        ),
    )
    st.caption(report.narrative)

    # -----------------------------------------------------------------------
    # What-if hedged P&L
    # -----------------------------------------------------------------------
    st.markdown("### 4) What-if hedged P&L")
    st.caption(
        "Side-by-side: the unhedged FCN P&L (from the P&L panel) vs the same "
        "P&L net of a static delta hedge sized at issue. The point of the "
        "comparison is that the spot bucket should largely cancel."
    )
    initial = st.session_state["initial"]
    init_state = st.session_state.get("initial_full_state")
    if init_state is None:
        st.info("Initial Greeks not yet computed — visit the P&L tab once.")
        return

    pnl = attribute(
        initial=initial,
        current=current,
        initial_price=float(init_state["price"]),
        current_price=float(state["price"]),
        initial_delta=np.asarray(init_state["delta"]),
        initial_vega=np.asarray(init_state["vega"]),
        initial_cega_pair=np.asarray(init_state["cega_pair"]),
    )
    # Static day-1 delta hedge: dealer holds Δ(initial) shares per name.
    # Hedge P&L = Σ Δᵢ(initial) · (S_i_current − S_i_initial)   on the dealer side.
    # From holder's perspective this *adds* the same amount, so the residual
    # spot exposure is zero modulo Γ.
    spot_hedge_pnl = -float(pnl.spot_pnl)        # dealer pockets the holder's spot move
    hedged_total = pnl.total_pnl + spot_hedge_pnl

    cu1, cu2, cu3 = st.columns(3)
    cu1.metric(
        "Unhedged P&L",
        _format_pnl(pnl.total_pnl),
        help="Total MTM move since issue (from the P&L tab).",
    )
    cu2.metric(
        "Static delta hedge P&L",
        _format_pnl(spot_hedge_pnl),
        help="P&L of holding Δ(initial) shares per name from issue to today.",
    )
    cu3.metric(
        "Net after hedge",
        _format_pnl(hedged_total),
        help=(
            "Sum of the two above. Should be close to the FCN's residual "
            "(Γ + θ + vol + corr) — i.e. the un-hedgeable bucket."
        ),
    )

    # -----------------------------------------------------------------------
    # Unhedgeable risks summary table
    # -----------------------------------------------------------------------
    st.markdown("### 5) Un-hedgeable risks")
    summary = compute_unhedgeable_risk_summary(
        greeks={"delta": delta, "vega": vega},
        scenario_pnls=None,                       # filled by the Scenario tab work
        correlation_report=report,
        notional=current.notional,
    )
    if summary.rows:
        df = pd.DataFrame([
            {
                "Risk": r["risk"],
                "Metric": r["metric"],
                "Value": (f"${r['value']:,.2f}" if abs(r['value']) < 1
                          else f"${r['value']:,.0f}"),
                "P&L per 5% move": (
                    "—" if math.isnan(r["p_and_l_5pct"])
                    else f"${r['p_and_l_5pct']:,.0f}"
                ),
                "% of notional": (
                    "—" if math.isnan(r["pct_of_notional"])
                    else f"{r['pct_of_notional']:+.3f}%"
                ),
                "Notes": r["notes"],
            }
            for r in summary.rows
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No un-hedgeable rows for the current snapshot.")
