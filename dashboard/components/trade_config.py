"""Collapsible "Trade structure" panel — FCN spec at a glance.

Phase 1: read-only summary of the default product. In later phases this
becomes an expander with editable fields for the FCN structure
(notional, coupon, barriers, observation grid).
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Draw the trade-structure summary at the top of the main pane."""
    product = st.session_state["product"]
    initial = st.session_state["initial"]

    with st.expander("Trade structure (FCN spec)", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Notional", f"${product.notional:,.0f}")
        col_a.metric("Tenor", f"{(product.pay_dates[-1] - product.issue_date).days} days")
        col_b.metric("Coupon (per period)", f"{100 * product.coupon_rate:.3f}%")
        col_b.metric("# observations", f"{product.n_obs}  ({product.n_autocall_obs} AC + 1 final)")
        col_c.metric("Autocall barrier", f"{100 * product.autocall_barrier:.0f}%")
        col_c.metric("Knock-in (strike)", f"{100 * product.strike:.0f}%")

        st.markdown(
            f"**Underlyings:** `{', '.join(initial.tickers)}` &nbsp;|&nbsp; "
            f"**Issue date:** {product.issue_date.isoformat()} &nbsp;|&nbsp; "
            f"**Maturity payment:** {product.pay_dates[-1].isoformat()}"
        )
        st.caption(
            "Observation dates: "
            + ", ".join(d.isoformat() for d in product.obs_dates)
        )
        st.caption(
            "Editable trade-structure fields ship in a later phase. The Phase 1 "
            "skeleton uses the JT-spec'd 1Y worst-of FCN with 8% p.a. coupon."
        )
