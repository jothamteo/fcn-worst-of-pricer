"""Streamlit panel components for the FCN dashboard.

Each module exports a single ``render(...)`` function that draws its section
of the page from ``st.session_state``. Components do not own state — they
read from session_state and write back via the sidebar/event handlers only.
"""
