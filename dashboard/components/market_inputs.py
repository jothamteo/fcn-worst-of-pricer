"""Sidebar component — "Live Market State".

Spot / vol / correlation / rate inputs, in two modes:

* **Basket-together** (default) — one slider moves all three spots by the
  same %; one slider moves all three vols by the same absolute vol-pt; one
  slider sets the off-diagonal correlation shift. Cheapest input mode and
  the only one the pre-computed grid is exact for.
* **Independent** — three spot sliders, three vol sliders, three pairwise
  correlation sliders. Falls back to Precise (live MC) for pricing because
  the grid is basket-axes only — see ``dashboard.precompute``.

Every control carries a Streamlit ``help=`` tooltip so a recruiter who
doesn't know the Greeks can still navigate the page without a walkthrough.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from dashboard.state import MarketSnapshot, reset_current_to_initial


_SPOT_RANGE = (-0.30, 0.30)         # basket spot shift bounds
_VOL_RANGE = (-0.10, 0.20)          # basket vol shift bounds (absolute, vol points)
_CORR_RANGE = (-0.30, 0.30)         # basket off-diagonal correlation shift bounds
_RATE_RANGE = (0.0, 0.10)


def render() -> None:
    """Draw the sidebar and write user changes to ``st.session_state.current``."""
    st.sidebar.markdown("## Live Market State")
    st.sidebar.caption(
        "Move the sliders to walk the market away from the trade-at-issue "
        "snapshot. Every panel reprices off these inputs."
    )

    initial = st.session_state["initial"]
    current: MarketSnapshot = st.session_state["current"]
    tickers = initial.tickers
    d = len(tickers)

    # -----------------------------------------------------------------------
    # As-of date
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**As-of date**")
    new_as_of = st.sidebar.date_input(
        "As-of date",
        value=current.as_of,
        min_value=initial.as_of,
        max_value=date(2027, 2, 11),
        help=(
            "The valuation date the dashboard prices to. Moving this forward "
            "exercises the time-decay (theta) component of the P&L."
        ),
        label_visibility="collapsed",
        key="as_of_input",
    )

    # -----------------------------------------------------------------------
    # Input mode: basket-together vs independent (per-name)
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**Input mode**")
    basket_mode = st.sidebar.toggle(
        "Move basket-together",
        value=st.session_state.get("basket_mode", True),
        key="basket_mode",
        help=(
            "On: one slider per axis (spot %, vol pts, corr shift) applies "
            "the same change to every underlying.\n\n"
            "Off: three independent sliders per axis. Pricing falls back to "
            "the Precise (live MC) engine because the pre-computed grid is "
            "basket-axes only."
        ),
    )

    # When in independent mode, force Precise so the panels don't show
    # interpolated values that ignore the cross-sectional dispersion.
    if not basket_mode and st.session_state.get("compute_mode") == "Fast":
        st.session_state["compute_mode"] = "Precise"
        st.sidebar.caption(
            "_Independent-mode auto-switched compute to Precise._"
        )

    # -----------------------------------------------------------------------
    # Spots
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**Spots**")
    new_spots = np.asarray(initial.spots, dtype=float).copy()
    if basket_mode:
        basket_spot_shift = st.sidebar.slider(
            "Basket spot shift",
            min_value=_SPOT_RANGE[0], max_value=_SPOT_RANGE[1],
            value=float(np.mean(current.spots / initial.spots - 1.0)),
            step=0.01, format="%.0f%%",
            help=(
                "Single % move applied to every underlying. Slide left for "
                "a basket sell-off, right for a basket rally."
            ),
            key="basket_spot_shift",
        )
        new_spots = initial.spots * (1.0 + basket_spot_shift)
        for i, t in enumerate(tickers):
            st.sidebar.caption(
                f"`{t}` ${initial.spots[i]:,.2f} → ${new_spots[i]:,.2f} "
                f"({100 * basket_spot_shift:+.1f}%)"
            )
    else:
        for i, t in enumerate(tickers):
            new_spots[i] = st.sidebar.slider(
                f"{t} spot",
                min_value=float(initial.spots[i] * (1.0 + _SPOT_RANGE[0])),
                max_value=float(initial.spots[i] * (1.0 + _SPOT_RANGE[1])),
                value=float(current.spots[i]),
                step=float(initial.spots[i] * 0.005),
                format="$%.2f",
                help=f"Independent {t} spot. Issue price was ${initial.spots[i]:,.2f}.",
                key=f"spot_{t}",
            )

    # -----------------------------------------------------------------------
    # Vols
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**Implied vols**")
    new_vols = np.asarray(initial.vols, dtype=float).copy()
    if basket_mode:
        basket_vol_shift = st.sidebar.slider(
            "Basket vol shift (absolute, vol-pts)",
            min_value=_VOL_RANGE[0], max_value=_VOL_RANGE[1],
            value=float(np.mean(current.vols - initial.vols)),
            step=0.01, format="%+.2f",
            help=(
                "Absolute change in implied vol, applied to every underlying. "
                "+0.10 = +10 vol-points. The grid axes go from −10 to +20 "
                "vol-pts; outside that range the panel switches to Precise."
            ),
            key="basket_vol_shift",
        )
        new_vols = np.clip(initial.vols + basket_vol_shift, 0.01, None)
        for i, t in enumerate(tickers):
            st.sidebar.caption(
                f"`{t}` {100 * initial.vols[i]:.1f}% → {100 * new_vols[i]:.1f}%"
            )
    else:
        for i, t in enumerate(tickers):
            new_vols[i] = st.sidebar.slider(
                f"{t} vol",
                min_value=0.05, max_value=1.00,
                value=float(current.vols[i]),
                step=0.005, format="%.3f",
                help=f"Implied vol for {t}. Issue level was {100 * initial.vols[i]:.1f}%.",
                key=f"vol_{t}",
            )

    # -----------------------------------------------------------------------
    # Correlation
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**Correlation**")
    new_corr = np.asarray(initial.corr, dtype=float).copy()
    if basket_mode:
        # Current shift = mean off-diagonal diff from initial.
        iu = np.triu_indices(d, k=1)
        current_corr_shift = float(np.mean(current.corr[iu] - initial.corr[iu])) if d > 1 else 0.0
        corr_shift = st.sidebar.slider(
            "Off-diagonal corr shift",
            min_value=_CORR_RANGE[0], max_value=_CORR_RANGE[1],
            value=current_corr_shift,
            step=0.01, format="%+.2f",
            help=(
                "Absolute shift applied to every off-diagonal pairwise "
                "correlation. Positive = pairs move together more; negative "
                "= pairs decouple. For a worst-of FCN the issuer is short "
                "correlation (price rises when ρ rises)."
            ),
            key="corr_shift",
        )
        if d > 1:
            off = ~np.eye(d, dtype=bool)
            new_corr = initial.corr.copy()
            new_corr[off] = np.clip(initial.corr[off] + corr_shift, -0.999, 0.999)
            np.fill_diagonal(new_corr, 1.0)
        # Display the resulting matrix concisely.
        for i in range(d):
            for j in range(i + 1, d):
                st.sidebar.caption(
                    f"ρ({tickers[i]},{tickers[j]}) "
                    f"{initial.corr[i, j]:+.3f} → {new_corr[i, j]:+.3f}"
                )
    else:
        new_corr = initial.corr.copy()
        for i in range(d):
            for j in range(i + 1, d):
                new_corr[i, j] = st.sidebar.slider(
                    f"ρ({tickers[i]},{tickers[j]})",
                    min_value=-0.95, max_value=0.99,
                    value=float(current.corr[i, j]),
                    step=0.01, format="%+.2f",
                    help=f"Pairwise correlation between {tickers[i]} and {tickers[j]} log-returns.",
                    key=f"corr_{tickers[i]}_{tickers[j]}",
                )
                new_corr[j, i] = new_corr[i, j]

    # -----------------------------------------------------------------------
    # Rate
    # -----------------------------------------------------------------------
    st.sidebar.markdown("**Risk-free rate**")
    new_rate = st.sidebar.slider(
        "Risk-free rate",
        min_value=_RATE_RANGE[0], max_value=_RATE_RANGE[1],
        value=float(current.rate), step=0.0005, format="%.4f",
        help=(
            "Continuous-compounding 1Y rate used for discounting and as the "
            "GBM drift. Driving this moves the discount factor on the bond "
            "leg of the FCN; θ ≈ −r·V uses this as a proxy for time decay."
        ),
        key="rate_input",
    )

    # -----------------------------------------------------------------------
    # Compute mode + action buttons
    # -----------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Compute mode**")
    st.sidebar.radio(
        label="Compute mode",
        options=("Fast", "Precise"),
        key="compute_mode",
        label_visibility="collapsed",
        help=(
            "Fast: interpolate from a pre-computed basket-axes grid "
            "(near-instant). The grid is exact only along the basket axes.\n\n"
            "Precise: run a reduced-path MC against the current inputs. "
            "Slower (~2–5 s), exact for any independent move."
        ),
    )

    col_reset, col_rebuild = st.sidebar.columns(2)
    if col_reset.button("Reset to issue", use_container_width=True,
                        help="Restore every market input to its trade-at-issue level."):
        reset_current_to_initial()
        st.rerun()
    if col_rebuild.button("Rebuild grid (high-res)", use_container_width=True,
                          help=(
                              "Force-rebuild the scenario grid at full resolution "
                              "(~100s, 13×7×5 cells, ~800 MB peak memory). The "
                              "default first-load grid is a coarser 7×4×3 build "
                              "(~10s). On the free Streamlit Cloud tier (1 GB RAM) "
                              "this can OOM — run locally for the high-res grid."
                          )):
        st.session_state["scenario_grid"] = None
        st.session_state["grid_high_res"] = True
        st.rerun()

    # -----------------------------------------------------------------------
    # Commit the new snapshot back into session_state.
    # -----------------------------------------------------------------------
    st.session_state["current"] = MarketSnapshot(
        tickers=tickers,
        spots=new_spots,
        vols=new_vols,
        divs=current.divs.copy(),
        corr=new_corr,
        rate=float(new_rate),
        as_of=new_as_of,
        notional=current.notional,
    )
    # Pricing caches that depend on `current` are invalidated.
    st.session_state["current_pricing"] = None
    st.session_state["current_greeks"] = None
