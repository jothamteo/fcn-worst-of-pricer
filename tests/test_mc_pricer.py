"""Tests for the Monte Carlo wrapper and the ex-post realised replay.

The hard correctness checks live in `test_gbm.py` (GBM engine) and
`test_fcn_payoff.py` (payoff logic). This file checks the *composition*:
  - the wrapper plumbs `MarketData` and `FCNProduct` through correctly,
  - antithetic actually reduces MC standard error,
  - seed reproducibility holds end-to-end,
  - the zero-vol limit collapses to a deterministic discounted-cashflow sum
    that matches `realised_payoff` exactly.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fcn_payoff import FCNProduct
from src.market_data import MarketData
from src.mc_pricer import (
    _worst_of_european_put_expectation,
    _worst_of_european_put_pv,
    price_fcn,
    realised_payoff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _product(**overrides) -> FCNProduct:
    base = dict(
        notional=50_000.0,
        coupon_rate=0.01535,
        obs_dates=(
            date(2025, 12, 1),
            date(2025, 12, 31),
            date(2026, 2, 2),
            date(2026, 3, 2),
            date(2026, 3, 31),
            date(2026, 4, 30),
        ),
        pay_dates=(
            date(2025, 12, 3),
            date(2026, 1, 5),
            date(2026, 2, 4),
            date(2026, 3, 4),
            date(2026, 4, 2),
            date(2026, 5, 4),
        ),
        issue_date=date(2025, 10, 17),
        autocall_barrier=1.00,
        strike=0.70,
        n_autocall_obs=5,
        geared_downside=False,
        continuous_ki=False,
    )
    base.update(overrides)
    return FCNProduct(**base)


def _market(spots=(100.0, 100.0, 100.0), vols=(0.30, 0.40, 0.55), rate=0.04, **overrides) -> MarketData:
    tickers = ("AMZN", "META", "MU")
    base = dict(
        tickers=list(tickers),
        spots=np.asarray(spots, dtype=float),
        vols=np.asarray(vols, dtype=float),
        divs=np.zeros(3),
        corr=np.array([
            [1.00, 0.55, 0.45],
            [0.55, 1.00, 0.40],
            [0.45, 0.40, 1.00],
        ]),
        rate=rate,
        history=pd.DataFrame(),
        sources={"as_of": "test"},
    )
    base.update(overrides)
    return MarketData(**base)


# ---------------------------------------------------------------------------
# Sanity / plumbing
# ---------------------------------------------------------------------------


def test_pricer_returns_finite_price_and_positive_se():
    res = price_fcn(_market(), _product(), n_paths=2_000, seed=1)
    assert np.isfinite(res.price)
    assert res.standard_error > 0
    assert 0 < res.price_pct_of_notional < 2.0  # bound: < 2x notional is sanity


def test_pricer_seed_reproducibility():
    a = price_fcn(_market(), _product(), n_paths=2_000, seed=42)
    b = price_fcn(_market(), _product(), n_paths=2_000, seed=42)
    assert a.price == b.price
    np.testing.assert_array_equal(a.pv_samples, b.pv_samples)


def test_antithetic_reduces_standard_error():
    p, m = _product(), _market()
    no = price_fcn(m, p, n_paths=4_000, antithetic=False, seed=7)
    yes = price_fcn(m, p, n_paths=2_000, antithetic=True, seed=7)
    # Same draw count (n_paths_total = 4_000), antithetic should not be worse —
    # for an FCN it's usually a meaningful improvement.
    assert no.n_paths_total == yes.n_paths_total
    assert yes.standard_error <= no.standard_error


def test_probability_decomposition_partitions_paths():
    res = price_fcn(_market(), _product(), n_paths=5_000, seed=11)
    total = (
        res.probability.p_autocall_total
        + res.probability.p_ki_at_maturity
        + res.probability.p_alive_no_ki
    )
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Zero-vol limit: every path is identical to deterministic forward.
# ---------------------------------------------------------------------------


def test_zero_vol_price_matches_deterministic_cashflows():
    """With zero vols and zero divs, S_i(t) = S_i(0) * exp(r t). The worst-of
    normalised perf at each obs is exp(r * obs_t), identical across underlyings.
    The note autocalls at the first obs where exp(r t) >= 1 — which is t = 0+
    for r > 0. So we expect autocall in period 1 with one coupon.
    """
    p = _product()
    m = _market(vols=(0.0, 0.0, 0.0), rate=0.04)
    res = price_fcn(m, p, n_paths=64, seed=0)

    c = p.coupon_rate * p.notional
    df_1 = float(np.exp(-m.rate * (p.pay_dates[0] - p.issue_date).days / 365.0))
    expected = (p.notional + c) * df_1
    np.testing.assert_allclose(res.price, expected, rtol=1e-10)
    np.testing.assert_allclose(res.probability.p_autocall_total, 1.0, atol=1e-12)


def test_zero_vol_zero_rate_no_autocall_no_ki():
    """With zero vols, zero rate, zero divs: S(t) = S(0) constantly. Worst-of
    perf = 1.0 at every obs >= autocall_barrier => autocalls in period 1.
    """
    p = _product()
    m = _market(vols=(0.0, 0.0, 0.0), rate=0.0)
    res = price_fcn(m, p, n_paths=32, seed=0)

    c = p.coupon_rate * p.notional
    expected = p.notional + c
    np.testing.assert_allclose(res.price, expected, rtol=1e-10)


def test_zero_vol_extreme_negative_drift_triggers_ki():
    """Force the path deep below the strike via large negative net drift
    (rate=0, divs=1.5/yr). Worst-of stays below strike — non-geared payoff."""
    p = _product()
    m = _market(
        vols=(0.0, 0.0, 0.0),
        divs=np.array([1.5, 1.5, 1.5]),
        rate=0.0,
    )
    res = price_fcn(m, p, n_paths=16, seed=0)

    # Six coupons (flat) + maturity = N * W(T) where W(T) = exp(-1.5 * 195/365).
    obs_yf_final = (p.obs_dates[-1] - p.issue_date).days / 365.0
    W_T = float(np.exp(-1.5 * obs_yf_final))
    coupons = 6 * p.coupon_rate * p.notional
    expected = coupons + p.notional * W_T
    np.testing.assert_allclose(res.price, expected, rtol=1e-10)
    np.testing.assert_allclose(res.probability.p_ki_at_maturity, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Ex-post realised replay
# ---------------------------------------------------------------------------


def _flat_history(perf_by_date: dict, spots=(100.0, 100.0, 100.0)) -> pd.DataFrame:
    """Build a daily DataFrame from issue_date through maturity. `perf_by_date`
    maps a calendar date to a 3-tuple of normalised performances; intermediate
    days inherit the last set values."""
    issue = date(2025, 10, 17)
    maturity = date(2026, 5, 4)
    idx = pd.date_range(start=pd.Timestamp(issue), end=pd.Timestamp(maturity), freq="D")
    values = np.tile(np.asarray(spots, dtype=float), (len(idx), 1))
    for d, perfs in sorted(perf_by_date.items()):
        ts = pd.Timestamp(d)
        # Apply from this date forward.
        mask = idx >= ts
        values[mask] = np.asarray(spots, dtype=float) * np.asarray(perfs, dtype=float)
    return pd.DataFrame(values, index=idx, columns=["AMZN", "META", "MU"])


def test_realised_payoff_autocall_period_1():
    p = _product()
    hist = _flat_history({date(2025, 11, 1): (1.10, 1.20, 1.30)})  # all up before obs 1
    out = realised_payoff(history=hist, product=p, rate=0.04)

    assert out["outcome"] == "autocalled_period_1"
    # Two cashflows: coupon_1 + autocall_redemption_1 on the same pay date.
    assert len(out["cashflows"]) == 2
    pay1 = p.pay_dates[0]
    expected_total = p.notional + p.coupon_rate * p.notional
    np.testing.assert_allclose(out["total_paid"], expected_total, rtol=1e-12)

    df_1 = float(np.exp(-0.04 * (pay1 - p.issue_date).days / 365.0))
    np.testing.assert_allclose(out["pv_at_issue"], expected_total * df_1, rtol=1e-12)


def test_realised_payoff_matured_ki_non_geared():
    p = _product()
    # Hover above autocall for early obs? No — keep below 1.00 and dip at the end.
    hist = _flat_history(
        {
            date(2025, 11, 1): (0.95, 0.92, 0.97),
            date(2026, 4, 25): (0.50, 0.90, 0.95),  # MU still highest? Worst is AMZN at 0.50.
        }
    )
    out = realised_payoff(history=hist, product=p, rate=None)
    assert out["outcome"] == "matured_ki"
    # 6 coupons paid (flat) + maturity = N * 0.50
    coupons = 6 * p.coupon_rate * p.notional
    expected = coupons + p.notional * 0.50
    np.testing.assert_allclose(out["total_paid"], expected, rtol=1e-12)


def test_realised_payoff_matured_par():
    p = _product()
    hist = _flat_history({date(2025, 11, 1): (0.85, 0.88, 0.95)})
    out = realised_payoff(history=hist, product=p, rate=None)
    assert out["outcome"] == "matured_par"
    coupons = 6 * p.coupon_rate * p.notional
    expected = coupons + p.notional
    np.testing.assert_allclose(out["total_paid"], expected, rtol=1e-12)


def test_realised_payoff_requires_dataframe():
    with pytest.raises(TypeError):
        realised_payoff(history="not a dataframe", product=_product())


# ---------------------------------------------------------------------------
# Control variate (worst-of European put)
# ---------------------------------------------------------------------------


def test_cv_path_runs_and_reduces_variance():
    """With realistic vols, the worst-of put CV should reduce variance —
    typically by a factor in the 2–10× range for FCNs of this shape. We
    require ≥ 1.3× as a generous floor (catches both wiring bugs and a
    pathological β̂ estimate).
    """
    res = price_fcn(
        _market(), _product(), n_paths=15_000, seed=20260601,
        control_variate="worst_of_put", cv_n_paths_for_ey=60_000,
    )
    assert res.cv_diagnostics is not None
    cv = res.cv_diagnostics
    assert cv["var_reduction_ratio"] >= 1.3, (
        f"CV did not meaningfully reduce variance: ratio = {cv['var_reduction_ratio']:.3f}"
    )
    # The CV correction should be modest in size (β̂ × (Ȳ − Ê[Y]) of order an SE).
    assert abs(cv["price_cv"] - cv["price_no_cv"]) < 5 * cv["se_no_cv"]


def test_cv_correlation_is_negative():
    """The worst-of put pays *more* when the FCN PV is *less* (KI breached →
    FCN loses, put gains). β̂ should be negative."""
    res = price_fcn(
        _market(), _product(), n_paths=10_000, seed=99,
        control_variate="worst_of_put", cv_n_paths_for_ey=40_000,
    )
    cv = res.cv_diagnostics
    assert cv["rho_XY"] < 0.0, f"corr(FCN PV, worst-of put PV) should be < 0, got {cv['rho_XY']:.4f}"
    assert cv["beta_hat"] < 0.0, f"β̂ should be negative, got {cv['beta_hat']:.4f}"


def test_cv_price_within_no_cv_band():
    """The CV price is unbiased (in expectation), so for any seed the
    `price_cv` should land within a few SE of `price_no_cv` and of an
    independent high-N reference price."""
    p, m = _product(), _market()
    cv_res = price_fcn(
        m, p, n_paths=10_000, seed=20260602,
        control_variate="worst_of_put", cv_n_paths_for_ey=40_000,
    )
    ref = price_fcn(m, p, n_paths=80_000, seed=20260603)  # high-N reference (no CV)
    band = 4.0 * (cv_res.standard_error + ref.standard_error)
    assert abs(cv_res.price - ref.price) < band, (
        f"CV price {cv_res.price:.2f} vs reference {ref.price:.2f} differ by "
        f"{abs(cv_res.price - ref.price):.2f} > 4·(SE+SE_ref) = {band:.2f}"
    )


def test_worst_of_put_pv_intrinsic_at_zero_vol():
    """With zero vols and a non-trivial strike, the worst-of put PV reduces
    to the deterministic intrinsic, discounted from the pay date."""
    p = _product()
    m = _market(vols=(0.0, 0.0, 0.0), rate=0.05)
    # At zero vol, S_i(T)/S_i(0) = exp(rate · T). With strike=0.70 and exp(rT) > 0.70,
    # the put expires worthless.
    res = price_fcn(m, p, n_paths=64, seed=0,
                    control_variate="worst_of_put", cv_n_paths_for_ey=64)
    cv = res.cv_diagnostics
    assert cv["EY_hat"] < 1e-6, f"E[Y] should be ~0 at zero vol with positive rate, got {cv['EY_hat']}"


def test_cv_invalid_kind_raises():
    with pytest.raises(ValueError):
        price_fcn(_market(), _product(), n_paths=64, seed=0, control_variate="not_a_thing")
