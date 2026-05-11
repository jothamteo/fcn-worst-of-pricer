"""Page CSS and Plotly theme nudges for the FCN dashboard.

Kept minimal on purpose: tightens the default Streamlit spacing a touch so
the four-panel layout fits comfortably without scrolling on a 13" laptop,
and adds a Plotly theme that uses the same accent colour as the page chrome.
"""

from __future__ import annotations

import plotly.io as pio
import streamlit as st


_PAGE_CSS = """
<style>
section.main > div { padding-top: 1rem; }
.stMetric > div { gap: 0.25rem; }
div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.0rem; padding: 0.6rem 1.2rem; }
section[data-testid="stSidebar"] .stCaption { font-size: 0.78rem; color: #666; }
</style>
"""


def inject_css() -> None:
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)


def install_plotly_theme() -> None:
    """Register a tidy Plotly template and make it the default.

    Idempotent — calling more than once is harmless.
    """
    tpl_name = "fcn_dashboard"
    if tpl_name in pio.templates:
        pio.templates.default = tpl_name
        return
    base = pio.templates["plotly_white"]
    tpl = base.layout.update(
        font=dict(family="Inter, sans-serif", size=12),
        title=dict(font=dict(size=14)),
        margin=dict(l=10, r=10, t=30, b=10),
        colorway=["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf"],
    )
    pio.templates[tpl_name] = pio.templates["plotly_white"]
    pio.templates[tpl_name].layout = tpl
    pio.templates.default = tpl_name


# ---------------------------------------------------------------------------
# "What is an FCN?" expander — beginner-friendly, opt-in by clicking.
# ---------------------------------------------------------------------------


def render_what_is_an_fcn() -> None:
    """Collapsed top-of-page explainer.

    Kept collapsed by default so the dashboard reads like a desk tool to
    anyone who already knows what an FCN is, and remains discoverable for
    everyone else.
    """
    with st.expander("What is an FCN?", expanded=False):
        st.markdown(
            "A **Fixed Coupon Note (FCN)** is a structured yield product. "
            "The investor lends the issuer a notional amount and in return "
            "earns a stream of periodic coupons. The risk is in how the note "
            "redeems at the end:\n\n"
            "- **Autocall.** At each observation date (here, monthly), if the "
            "worst-performing underlying is back at or above its issue level "
            "(the *autocall barrier* — 100% here), the note redeems early at "
            "par. The investor keeps coupons received to date and walks away.\n"
            "- **Knock-in / strike.** If the note runs to maturity *and* the "
            "worst-performing underlying has finished below the *strike* (70% "
            "here), the investor takes a 1-for-1 loss on the basket: redemption "
            "= notional × worst-of performance, capped at par.\n"
            "- **Coupons** are paid every period until autocall (in this build, "
            "guaranteed — no coupon barrier).\n\n"
            "From the **issuer's** perspective the trade is the mirror image: "
            "they're short the optionality, long correlation, short vol, and "
            "carry barrier-driven Γ that's hard to hedge perfectly. The "
            "panels on this page show what that risk looks like and what the "
            "static hedge would be."
        )
