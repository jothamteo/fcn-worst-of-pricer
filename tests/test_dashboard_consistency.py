"""Regression tests catching dashboard/notebook drift.

Pins the FCN structural spec (coupon, barriers, settlement, observation
schedule shape) so the dashboard and the notebooks can't silently
diverge. Catalyst: a prior commit had the dashboard at coupon=0.08/6
while the notebooks were at 0.01535 — different prices in the two
surfaces, no test caught it.

Also asserts that the shipped ``dashboard/grid_initial.npz`` fingerprint
matches the live state.py defaults — if the defaults move without
``scripts/build_initial_grid.py`` being re-run, the dashboard falls back
to a slow MC rebuild on first load instead of the fast cached path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# The standard FCN structure shared by dashboard and notebooks. The dashboard
# exposes these as STANDARD_* constants in dashboard/state.py; this test
# pins them so any future divergence breaks loudly.
EXPECTED_COUPON_RATE_PER_PERIOD = 0.01      # 1.0% per monthly observation = 12% p.a.
EXPECTED_AUTOCALL_BARRIER = 1.00
EXPECTED_STRIKE = 0.70
EXPECTED_N_AC_OBS = 5
EXPECTED_N_OBS_TOTAL = 6
EXPECTED_PHYSICAL_DELIVERY = True
EXPECTED_COUPON_BARRIER = None
EXPECTED_CONTINUOUS_KI = False
EXPECTED_TENOR_DAYS = 184                   # 6 months ± a couple of days for date arithmetic
EXPECTED_TENOR_DAYS_TOLERANCE = 4           # accommodate weekend rolls


def test_dashboard_default_product_matches_standard_spec():
    """dashboard._default_product() must match every structural field.

    Tenor is checked in days with a small tolerance so the test isn't
    brittle against weekend-roll adjustments of the pay date.
    """
    from dashboard.state import _default_product
    product = _default_product()

    np.testing.assert_allclose(product.coupon_rate, EXPECTED_COUPON_RATE_PER_PERIOD, rtol=1e-12)
    np.testing.assert_allclose(product.autocall_barrier, EXPECTED_AUTOCALL_BARRIER, rtol=1e-12)
    np.testing.assert_allclose(product.strike, EXPECTED_STRIKE, rtol=1e-12)
    assert product.n_autocall_obs == EXPECTED_N_AC_OBS
    assert product.n_obs == EXPECTED_N_OBS_TOTAL
    assert product.physical_delivery is EXPECTED_PHYSICAL_DELIVERY
    assert product.coupon_barrier == EXPECTED_COUPON_BARRIER
    assert product.continuous_ki is EXPECTED_CONTINUOUS_KI

    tenor_days = (product.pay_dates[-1] - product.issue_date).days
    assert abs(tenor_days - EXPECTED_TENOR_DAYS) <= EXPECTED_TENOR_DAYS_TOLERANCE, (
        f"Dashboard product tenor = {tenor_days} days, expected "
        f"{EXPECTED_TENOR_DAYS} ± {EXPECTED_TENOR_DAYS_TOLERANCE}"
    )


def test_dashboard_state_exposes_standard_constants():
    """STANDARD_* constants in state.py match the expected spec."""
    from dashboard import state
    assert state.STANDARD_COUPON_RATE_PER_PERIOD == EXPECTED_COUPON_RATE_PER_PERIOD
    assert state.STANDARD_AUTOCALL_BARRIER == EXPECTED_AUTOCALL_BARRIER
    assert state.STANDARD_STRIKE == EXPECTED_STRIKE
    assert state.STANDARD_N_AC_OBS == EXPECTED_N_AC_OBS
    assert state.STANDARD_N_OBS_TOTAL == EXPECTED_N_OBS_TOTAL
    assert state.STANDARD_PHYSICAL_DELIVERY is EXPECTED_PHYSICAL_DELIVERY
    assert state.STANDARD_COUPON_BARRIER == EXPECTED_COUPON_BARRIER
    assert state.STANDARD_CONTINUOUS_KI is EXPECTED_CONTINUOUS_KI


def test_shipped_grid_fingerprint_matches_state_defaults():
    """grid_initial.npz must match the live state.py defaults.

    On cold start the dashboard checks the shipped grid's fingerprint
    against the snapshot built from state.py. A match skips MC entirely
    (~3s cold start). A mismatch triggers a fast-grid rebuild (~10s) —
    still usable but slower, and an indicator that someone moved a default
    without re-running ``scripts/build_initial_grid.py``.
    """
    from dashboard.state import (
        DEFAULT_ISSUE_DATE,
        DEFAULT_NOTIONAL,
        DEFAULT_TICKERS,
        MarketSnapshot,
        _DEFAULT_ISSUE_CORR,
        _DEFAULT_ISSUE_DIVS,
        _DEFAULT_ISSUE_RATE,
        _DEFAULT_ISSUE_SPOTS,
        _DEFAULT_ISSUE_VOLS,
        _default_product,
    )
    from dashboard.precompute import snapshot_fingerprint

    grid_path = Path(__file__).parent.parent / "dashboard" / "grid_initial.npz"
    if not grid_path.exists():
        pytest.skip(f"grid_initial.npz not committed at {grid_path}")

    product = _default_product()
    initial = MarketSnapshot(
        tickers=DEFAULT_TICKERS,
        spots=_DEFAULT_ISSUE_SPOTS.copy(),
        vols=_DEFAULT_ISSUE_VOLS.copy(),
        divs=_DEFAULT_ISSUE_DIVS.copy(),
        corr=_DEFAULT_ISSUE_CORR.copy(),
        rate=_DEFAULT_ISSUE_RATE,
        as_of=DEFAULT_ISSUE_DATE,
        notional=DEFAULT_NOTIONAL,
    )
    expected_fp = snapshot_fingerprint(initial, product)
    shipped_fp = str(np.load(grid_path)["fingerprint"])

    assert expected_fp == shipped_fp, (
        f"grid_initial.npz fingerprint ({shipped_fp}) does not match "
        f"state.py defaults ({expected_fp}). Re-run "
        f"`python scripts/build_initial_grid.py` after changing any "
        f"_default_product or _DEFAULT_ISSUE_* values."
    )
