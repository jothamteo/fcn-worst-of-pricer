"""Session-state initialisation for the FCN dashboard.

The dashboard distinguishes two snapshots:

* ``initial`` — the trade-at-issue snapshot. Frozen on first load; this is
  what the P&L panel measures against.
* ``current`` — the live market state the user is editing. Sliders write
  here; every panel reads from here.

Both snapshots carry the same fields (spots, vols, divs, corr, rate, as_of)
so the same pricer call signature works on either. ``product`` is shared.

The defaults below are AMZN/META/MU at the levels used in notebook 07
(the project's headline summary), shifted to a 2026-02-11 issue so the
trade is in-flight as of today (2026-05-11). They are deliberately
hard-coded rather than fetched on app start — the page must render in well
under 10 seconds on cold start. A "Refresh from yfinance" button on the
sidebar (Phase 3) overwrites ``current`` with a real yfinance pull.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import streamlit as st

from src.fcn_payoff import FCNProduct


# ---------------------------------------------------------------------------
# Defaults — AMZN/META/MU, 1Y FCN issued 2026-02-11
# (matches notebook 07's basket; shifted forward so the trade is live today)
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: tuple[str, ...] = ("AMZN", "META", "MU")

# Spot/vol/corr levels follow notebook 07's 2025-10-17 snapshot — the
# project's canonical numbers. Dividends and rate match the same snapshot.
_DEFAULT_ISSUE_SPOTS = np.array([213.04, 715.72, 202.21], dtype=float)
_DEFAULT_ISSUE_VOLS = np.array([0.3502, 0.4397, 0.4713], dtype=float)
_DEFAULT_ISSUE_DIVS = np.array([0.0, 0.0029, 0.00227], dtype=float)
_DEFAULT_ISSUE_CORR = np.array(
    [
        [1.0000, 0.6139, 0.4422],
        [0.6139, 1.0000, 0.4176],
        [0.4422, 0.4176, 1.0000],
    ],
    dtype=float,
)
_DEFAULT_ISSUE_RATE = 0.03819

# Today's snapshot starts equal to issue — the user moves sliders to
# create a P&L story. (Pre-filling a non-zero P&L would obscure the
# attribution mechanics the panel is supposed to demonstrate.)
_DEFAULT_CURRENT_SPOTS = _DEFAULT_ISSUE_SPOTS.copy()
_DEFAULT_CURRENT_VOLS = _DEFAULT_ISSUE_VOLS.copy()
_DEFAULT_CURRENT_DIVS = _DEFAULT_ISSUE_DIVS.copy()
_DEFAULT_CURRENT_CORR = _DEFAULT_ISSUE_CORR.copy()
_DEFAULT_CURRENT_RATE = _DEFAULT_ISSUE_RATE

DEFAULT_ISSUE_DATE = date(2026, 5, 21)
DEFAULT_AS_OF_DATE = date(2026, 5, 21)
DEFAULT_NOTIONAL = 1_000_000.0

# ---------------------------------------------------------------------------
# Standard FCN structure — shared by dashboard and notebooks.
# Changing any of these requires updating notebooks/_build_*.py to match.
# `tests/test_dashboard_consistency.py` enforces structural equality.
# ---------------------------------------------------------------------------

STANDARD_COUPON_RATE_PER_PERIOD = 0.01     # 1.0% per monthly observation = 12.0% p.a.
STANDARD_AUTOCALL_BARRIER = 1.00
STANDARD_STRIKE = 0.70
STANDARD_N_AC_OBS = 5
STANDARD_N_OBS_TOTAL = 6
STANDARD_PHYSICAL_DELIVERY = True
STANDARD_COUPON_BARRIER = None
STANDARD_CONTINUOUS_KI = False
STANDARD_TENOR_MONTHS = 6
STANDARD_OBS_INTERVAL_MONTHS = 1
STANDARD_PAY_DELAY_DAYS = 2                # T+2 calendar days


def _default_product() -> FCNProduct:
    """The standard 6-month worst-of FCN on the default basket.

    Structure matches the notebook product exactly (5 AC obs + maturity
    check, 100% AC, 70% KI, physical delivery, 1.0%/period flat coupon).
    Only the calendar dates differ — the dashboard places the trade
    forward-looking from today so all observations are ahead; the
    notebooks place the trade in the past for historical replay.
    """
    return FCNProduct(
        notional=DEFAULT_NOTIONAL,
        coupon_rate=STANDARD_COUPON_RATE_PER_PERIOD,
        obs_dates=(
            date(2026, 6, 21),
            date(2026, 7, 21),
            date(2026, 8, 21),
            date(2026, 9, 21),
            date(2026, 10, 21),
            date(2026, 11, 21),              # final valuation (KI check)
        ),
        pay_dates=(
            date(2026, 6, 23),
            date(2026, 7, 23),
            date(2026, 8, 23),
            date(2026, 9, 23),
            date(2026, 10, 23),
            date(2026, 11, 23),              # maturity payment (T+2)
        ),
        issue_date=DEFAULT_ISSUE_DATE,
        autocall_barrier=STANDARD_AUTOCALL_BARRIER,
        strike=STANDARD_STRIKE,
        n_autocall_obs=STANDARD_N_AC_OBS,
        coupon_barrier=STANDARD_COUPON_BARRIER,
        physical_delivery=STANDARD_PHYSICAL_DELIVERY,
        continuous_ki=STANDARD_CONTINUOUS_KI,
    )


# ---------------------------------------------------------------------------
# Snapshot dataclass — used identically for "initial" and "current"
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    """A single market state — used for both ``initial`` and ``current``.

    Mirrors the shape of ``src.market_data.MarketData`` so the pricer-facing
    helpers (Phase 2 onwards) can construct a ``MarketData`` from one of
    these without copying anything.

    Attributes
    ----------
    tickers : tuple of str
    spots, vols, divs : np.ndarray
        Per-name arrays, in ``tickers`` order.
    corr : np.ndarray
        (d, d) symmetric correlation matrix.
    rate : float
        Continuous risk-free rate.
    as_of : date
        Calendar date the snapshot represents — issue date for ``initial``,
        today's date for ``current``.
    notional : float
        Trade size in USD; lives at the snapshot level for convenience even
        though it's an attribute of the product. Kept in sync.
    """

    tickers: tuple[str, ...]
    spots: np.ndarray
    vols: np.ndarray
    divs: np.ndarray
    corr: np.ndarray
    rate: float
    as_of: date
    notional: float = DEFAULT_NOTIONAL

    def copy(self) -> "MarketSnapshot":
        return MarketSnapshot(
            tickers=tuple(self.tickers),
            spots=np.array(self.spots, copy=True),
            vols=np.array(self.vols, copy=True),
            divs=np.array(self.divs, copy=True),
            corr=np.array(self.corr, copy=True),
            rate=float(self.rate),
            as_of=self.as_of,
            notional=float(self.notional),
        )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


SESSION_VERSION = 1


def init_session_state() -> None:
    """Populate ``st.session_state`` with default snapshots if missing.

    Called once at the top of ``app.py``. Safe to call multiple times — it
    is a no-op when state is already set up.

    ``session_state.initial`` is intended to stay constant for the life of
    the browser session; only the "Reset trade" / "Reload defaults" actions
    overwrite it. Sliders write to ``session_state.current``.
    """
    if st.session_state.get("_dashboard_state_version") == SESSION_VERSION:
        return

    st.session_state["_dashboard_state_version"] = SESSION_VERSION
    st.session_state["tickers"] = DEFAULT_TICKERS
    st.session_state["product"] = _default_product()
    st.session_state["initial"] = MarketSnapshot(
        tickers=DEFAULT_TICKERS,
        spots=_DEFAULT_ISSUE_SPOTS.copy(),
        vols=_DEFAULT_ISSUE_VOLS.copy(),
        divs=_DEFAULT_ISSUE_DIVS.copy(),
        corr=_DEFAULT_ISSUE_CORR.copy(),
        rate=_DEFAULT_ISSUE_RATE,
        as_of=DEFAULT_ISSUE_DATE,
        notional=DEFAULT_NOTIONAL,
    )
    st.session_state["current"] = MarketSnapshot(
        tickers=DEFAULT_TICKERS,
        spots=_DEFAULT_CURRENT_SPOTS.copy(),
        vols=_DEFAULT_CURRENT_VOLS.copy(),
        divs=_DEFAULT_CURRENT_DIVS.copy(),
        corr=_DEFAULT_CURRENT_CORR.copy(),
        rate=_DEFAULT_CURRENT_RATE,
        as_of=DEFAULT_AS_OF_DATE,
        notional=DEFAULT_NOTIONAL,
    )
    st.session_state["compute_mode"] = "Fast"            # 'Fast' or 'Precise'
    st.session_state["saved_scenarios"] = {}             # name → MarketSnapshot
    # Cache slots populated by later phases. None means "not computed yet".
    st.session_state["initial_pricing"] = None           # PricingResult
    st.session_state["initial_greeks"] = None            # GreekResult
    st.session_state["current_pricing"] = None
    st.session_state["current_greeks"] = None
    st.session_state["scenario_grid"] = None             # pandas.DataFrame
    st.session_state["last_status"] = "Skeleton loaded — components fill in across phases."


def reset_current_to_initial() -> None:
    """Restore ``current`` to a fresh copy of ``initial``. Used by the
    sidebar's "Reset to issue" button."""
    init_session_state()
    st.session_state["current"] = st.session_state["initial"].copy()
    st.session_state["current_pricing"] = None
    st.session_state["current_greeks"] = None
