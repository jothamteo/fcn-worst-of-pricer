"""Streamlit dashboard for the worst-of FCN pricer.

Run from the repo root::

    streamlit run app.py

Architecture
------------
The dashboard is a thin UI layer on top of ``src.*``. It never owns pricing
logic — every value displayed is produced by the existing pricer
modules. The package layout is::

    dashboard/
      state.py             session-state init (initial vs current snapshots)
      precompute.py        scenario grid build + interpolation (Phase 2)
      pnl_attribution.py   first-order P&L decomposition  (Phase 4)
      styles.py            page CSS / Plotly theme        (Phase 9)
      components/          one module per panel

Compute modes (Phase 3 onwards)
-------------------------------
* **Fast** (default) — every panel reads from a pre-computed scenario grid
  and interpolates. Slider response is instant.
* **Precise** — every panel re-runs a reduced-path MC. Slower but exact.

This file (`app.py`) is intentionally short: it sets the page config,
initialises session state, draws the sidebar, and dispatches to the four
content tabs.
"""

from __future__ import annotations

import streamlit as st

from dashboard.state import init_session_state
from dashboard.styles import (
    inject_css,
    install_plotly_theme,
    render_what_is_an_fcn,
)
from dashboard.components import (
    greeks_panel,
    hedge_panel,
    market_inputs,
    pnl_panel,
    price_curve_panel,
    scenario_panel,
    trade_config,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FCN Worst-of Pricer — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Interactive dashboard for the worst-of Fixed-Coupon Note pricer "
            "at github.com/jothamteo/fcn-worst-of-pricer. "
            "Built on top of the project's MC + PDE engines."
        ),
    },
)


# ---------------------------------------------------------------------------
# State + sidebar
# ---------------------------------------------------------------------------

init_session_state()
inject_css()
install_plotly_theme()

market_inputs.render()


# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------

st.title("Worst-of FCN — Live Pricing & Risk Dashboard")
st.caption(
    "Demo dashboard for the open-source FCN pricer. "
    "Move the market sliders in the sidebar; the panels below reprice the "
    "trade through the same engine used in the project's notebooks."
)

render_what_is_an_fcn()
trade_config.render()

tab_pnl, tab_greeks, tab_scenarios, tab_hedge, tab_curve = st.tabs(
    ["P&L", "Greeks", "Scenarios", "Hedging", "Price curve"]
)

with tab_pnl:
    pnl_panel.render()

with tab_greeks:
    greeks_panel.render()

with tab_scenarios:
    scenario_panel.render()

with tab_hedge:
    hedge_panel.render()

with tab_curve:
    price_curve_panel.render()

# Footer
st.markdown("---")
st.caption(st.session_state.get("last_status", ""))
