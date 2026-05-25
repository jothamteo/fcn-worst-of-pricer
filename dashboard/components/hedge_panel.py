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

from dashboard.engine import get_full_state, get_pricing
from dashboard.pnl_attribution import attribute
from src.hedging import (
    CorrelationRiskReport,
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
    st.subheader("Hedging (dealer view)")
    st.caption(
        "**Dealer-side hedge ratios.** This panel flips the investor-side "
        "Greeks shown elsewhere on the dashboard to the dealer's mirror — "
        "the dealer sold the FCN to the client, is short the structure, and "
        "needs to hedge the opposite of every exposure shown in the P&L and "
        "Greeks tabs. Sign conventions follow `src.hedging`."
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
    # Dealer is the *mirror* of the holder: dealer_vega = -holder_vega.
    # Holder is short vol (negative vega) → dealer is long vol (positive
    # vega) → to neutralise, dealer SELLS listed options. The previous
    # version mistakenly fed +holder_vega here, producing a buy-side
    # recommendation. Matches notebook 09's hedge construction.
    dealer_vegas = {t: -float(vega[i]) for i, t in enumerate(tickers)}
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
    # Correlation risk (dealer view)
    # -----------------------------------------------------------------------
    st.markdown("### 3) Correlation risk (dealer view)")
    # Basket-wide pairwise cega: a +1% parallel shift in every pairwise ρ
    # changes the FCN's MTM by Σᵢ<ⱼ cega_ij. The previous version averaged
    # the pairwise cegas, which understated the parallel-shift sensitivity
    # by a factor of n_pairs (3× for a 3-name basket). The P&L tab's
    # waterfall correctly sums; this panel must match.
    d = cega_pair.shape[0]
    cega_sum_investor = 0.0
    n_pairs = 0
    for i in range(d):
        for j in range(i + 1, d):
            cega_sum_investor += float(cega_pair[i, j])
            n_pairs += 1

    # Build the price scan around the basket-wide sum so the finite-
    # difference inside compute_correlation_risk recovers the basket
    # sensitivity, not the per-pair average. compute_correlation_risk
    # reports investor-side numbers; we display the dealer's mirror.
    base_price = float(state["price"])
    price_at_levels = {
        -0.10: base_price + (-0.10) * 100 * cega_sum_investor,
        -0.05: base_price + (-0.05) * 100 * cega_sum_investor,
         0.00: base_price,
         0.05: base_price + ( 0.05) * 100 * cega_sum_investor,
         0.10: base_price + ( 0.10) * 100 * cega_sum_investor,
    }
    report_investor = compute_correlation_risk(price_at_corr_levels=price_at_levels)
    cega_dealer = -float(report_investor.cega)
    pnl_5pct_dealer = -float(report_investor.correlation_pnl_per_5pct_move)
    direction = "long" if cega_dealer > 0 else "short"
    outcome = "gain" if pnl_5pct_dealer > 0 else "loss"
    dealer_narrative = (
        f"The dealer is {direction} correlation. A +5 percentage-point "
        f"parallel rise in pairwise ρ across the basket would be a "
        f"{outcome} of ${abs(pnl_5pct_dealer):,.0f} (≈ basket cega × 5). "
        f"No liquid single-stock correlation product covers a generic 3-"
        f"name basket, so the desk reserves capital against this risk."
    )
    # Dealer-mirrored report flows into the §5 un-hedgeable risks table so
    # the whole Hedging tab presents one convention.
    report = CorrelationRiskReport(
        cega=cega_dealer,
        correlation_pnl_per_5pct_move=pnl_5pct_dealer,
        base_price=report_investor.base_price,
        hedgeable=report_investor.hedgeable,
        narrative=dealer_narrative,
        samples=report_investor.samples,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "cega (per +0.01, dealer)",
        f"${cega_dealer:,.0f}",
        help=(
            f"Basket-wide pairwise correlation sensitivity from the "
            f"dealer's side — sum of {n_pairs} pairwise cegas, sign-"
            f"flipped from the holder."
        ),
    )
    c2.metric(
        "P&L per +5% in ρ (dealer)",
        f"${pnl_5pct_dealer:,.0f}",
        help=(
            "Linear extrapolation of basket cega across a 5 percentage-"
            "point parallel ρ rise — dealer side."
        ),
    )
    c3.metric(
        "Hedgeable?",
        "No",
        help=(
            "There is no liquid single-stock correlation product for a generic "
            "3-name basket. See `HEDGING_NOTES.md` for the desk-level mitigants."
        ),
    )
    st.caption(dealer_narrative)

    # -----------------------------------------------------------------------
    # What-if hedged P&L (dealer side)
    # -----------------------------------------------------------------------
    st.markdown("### 4) What-if hedged P&L (dealer view)")
    st.caption(
        "Dealer-side P&L since issue: the unhedged FCN leg (mirror of the "
        "investor's MTM from the P&L tab) vs the same P&L net of a static "
        "delta hedge sized at issue. The spot bucket should largely cancel — "
        "the residual is the un-hedgeable bucket the desk runs against."
    )
    initial = st.session_state["initial"]
    init_state = st.session_state.get("initial_full_state")
    if init_state is None:
        st.info("Initial Greeks not yet computed — visit the P&L tab once.")
        return

    # Use ``get_pricing`` (grid in Fast mode, MC in Precise) so the §4 P&L
    # numbers reconcile to the P&L tab's waterfall. ``state["price"]`` is a
    # live precise-MC reval that disagrees with the grid by ~$300 of MC noise
    # — that noise would show up here as an "unexplained" P&L that doesn't
    # match any waterfall bar.
    current_price_for_attr = float(get_pricing(current)["price"])
    pnl = attribute(
        initial=initial,
        current=current,
        initial_price=float(init_state["price"]),
        current_price=current_price_for_attr,
        initial_delta=np.asarray(init_state["delta"]),
        initial_vega=np.asarray(init_state["vega"]),
        initial_cega_pair=np.asarray(init_state["cega_pair"]),
    )
    # Static day-1 delta hedge from the dealer's side. The dealer is short
    # the FCN to the client and holds Δᵢ(initial) shares per name against
    # it. When spot moves by ΔSᵢ:
    #     FCN leg    = -investor.total_pnl   (mirror of holder's MTM)
    #     Hedge leg  = +Σ Δᵢ(initial)·ΔSᵢ  =  +investor.spot_pnl
    # Net residual is the un-hedgeable bucket (Γ + θ + vol + corr), dealer side.
    unhedged_dealer = -float(pnl.total_pnl)
    spot_hedge_pnl = float(pnl.spot_pnl)
    hedged_total = unhedged_dealer + spot_hedge_pnl

    cu1, cu2, cu3 = st.columns(3)
    cu1.metric(
        "Unhedged P&L (dealer)",
        _format_pnl(unhedged_dealer),
        help=(
            "Dealer's mirror of the FCN's MTM move since issue — sign-flipped "
            "from the investor P&L shown in the P&L tab."
        ),
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
            "Sum of the two above — dealer's residual (Γ + θ + vol + corr). "
            "This is the un-hedgeable bucket the desk runs against."
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
