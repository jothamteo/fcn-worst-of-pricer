"""Streamlit dashboard for the worst-of FCN pricer.

This package is a thin UI layer on top of `src.*`. It never re-implements
pricing logic — every value displayed in the dashboard is produced by the
existing pricer modules, optionally read from a pre-computed scenario grid
for interactive responsiveness.

Layout (one module per concern):

* ``state.py`` — initialises ``st.session_state`` with the "trade at issue"
  snapshot and the live "current market" snapshot.
* ``precompute.py`` — builds and queries the pre-computed scenario grid.
* ``pnl_attribution.py`` — first-order P&L decomposition.
* ``styles.py`` — page-wide CSS and Plotly theme.
* ``components/`` — one Streamlit component per panel.
"""
