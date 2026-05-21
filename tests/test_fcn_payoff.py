"""Unit tests for the worst-of FCN payoff.

Tests are deterministic — hand-crafted paths drive specific code paths
(autocall in period 1/2/.../last, KI at maturity, par at maturity, flat vs
conditional coupon, physical-delivery vs cash-settled downside, continuous KI).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.fcn_payoff import (
    FCNProduct,
    ObservationGrid,
    payoff_per_path,
    probability_decomposition,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _amzn_meta_mu_product(**overrides) -> FCNProduct:
    """JT-spec'd worst-of FCN on AMZN/META/MU."""
    base = dict(
        notional=50_000.0,
        coupon_rate=0.01535,
        obs_dates=(
            date(2025, 12, 1),
            date(2025, 12, 31),
            date(2026, 2, 2),
            date(2026, 3, 2),
            date(2026, 3, 31),
            date(2026, 4, 30),  # final valuation
        ),
        pay_dates=(
            date(2025, 12, 3),
            date(2026, 1, 5),
            date(2026, 2, 4),
            date(2026, 3, 4),
            date(2026, 4, 2),
            date(2026, 5, 4),   # maturity
        ),
        issue_date=date(2025, 10, 17),
        autocall_barrier=1.00,
        strike=0.70,
        n_autocall_obs=5,
        coupon_barrier=None,
        physical_delivery=False,
        continuous_ki=False,
    )
    base.update(overrides)
    return FCNProduct(**base)


def _build_constant_perf_paths(
    product: FCNProduct,
    grid: ObservationGrid,
    perf_by_period: np.ndarray,
    d: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build paths whose worst-of at each obs date equals `perf_by_period[j]`.

    Asset 0 carries the worst-of perf; assets 1..d-1 are pinned far above the
    barriers so they never bind. `perf_by_period` is (M,).
    """
    spots = np.array([100.0] * d, dtype=float)
    n_paths = 1
    n_grid = grid.sim_n_steps + 1
    paths = np.empty((n_paths, n_grid, d), dtype=float)
    paths[:, :, :] = spots  # default: every grid point at the initial spot
    # Set the worst-of asset to the requested perf at each obs date.
    for j, idx in enumerate(grid.obs_indices):
        paths[0, idx, 0] = spots[0] * perf_by_period[j]
        # Other assets stay well above all barriers (worst-of binds to asset 0).
        paths[0, idx, 1:] = spots[1:] * 5.0
    # Between observation dates, fill with a level safely above all barriers
    # (only the obs indices are read by the European-KI payoff path).
    return paths, spots


# ---------------------------------------------------------------------------
# Product spec validation
# ---------------------------------------------------------------------------


def test_product_rejects_obs_pay_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        FCNProduct(
            notional=100.0,
            coupon_rate=0.01,
            obs_dates=(date(2025, 1, 1), date(2025, 2, 1)),
            pay_dates=(date(2025, 1, 3),),
            issue_date=date(2024, 12, 31),
        )


def test_product_rejects_non_increasing_obs_dates():
    with pytest.raises(ValueError, match="strictly increasing"):
        FCNProduct(
            notional=100.0,
            coupon_rate=0.01,
            obs_dates=(date(2025, 2, 1), date(2025, 1, 1)),
            pay_dates=(date(2025, 2, 3), date(2025, 1, 3)),
            issue_date=date(2024, 12, 31),
        )


def test_product_rejects_pay_before_obs():
    with pytest.raises(ValueError, match="before obs_dates"):
        FCNProduct(
            notional=100.0,
            coupon_rate=0.01,
            obs_dates=(date(2025, 2, 1),),
            pay_dates=(date(2025, 1, 31),),
            issue_date=date(2024, 12, 31),
        )


def test_product_rejects_strike_above_autocall():
    with pytest.raises(ValueError, match="strike"):
        FCNProduct(
            notional=100.0,
            coupon_rate=0.01,
            obs_dates=(date(2025, 2, 1),),
            pay_dates=(date(2025, 2, 3),),
            issue_date=date(2024, 12, 31),
            autocall_barrier=1.0,
            strike=1.5,
        )


def test_product_default_n_autocall_obs_is_M_minus_1():
    p = _amzn_meta_mu_product()
    assert p.n_autocall_obs == 5
    assert p.n_obs == 6


def test_product_year_fractions_match_calendar():
    p = _amzn_meta_mu_product()
    obs_yf = p.obs_year_fractions(day_count=365.0)
    # First obs: 17 Oct 2025 -> 1 Dec 2025 = 45 days; 45/365 = 0.12329...
    np.testing.assert_allclose(obs_yf[0], 45.0 / 365.0, rtol=1e-12)
    # Final obs: 30 Apr 2026 = 195 days from 17 Oct 2025.
    np.testing.assert_allclose(obs_yf[-1], 195.0 / 365.0, rtol=1e-12)
    pay_yf = p.pay_year_fractions(day_count=365.0)
    # Maturity: 4 May 2026 = 199 days.
    np.testing.assert_allclose(pay_yf[-1], 199.0 / 365.0, rtol=1e-12)


def test_observation_grid_from_product_lands_on_calendar_days():
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    assert grid.sim_n_steps == 199  # days from 17 Oct 2025 to 4 May 2026
    expected_indices = np.array([45, 75, 108, 136, 165, 195], dtype=int)
    np.testing.assert_array_equal(grid.obs_indices, expected_indices)


# ---------------------------------------------------------------------------
# Payoff: autocall in each period (with rate=0 so we can ignore discounting).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("autocall_period", [0, 1, 2, 3, 4])
def test_autocall_at_period_pays_par_plus_running_coupons(autocall_period):
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    perf = np.full(p.n_obs, 0.85)  # below autocall, above strike
    perf[autocall_period:] = 1.10   # all autocall periods >= here would trigger
    perf[autocall_period] = 1.01    # the trigger
    if autocall_period > 0:
        perf[:autocall_period] = 0.85  # below 1.00 to avoid earlier triggers
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    expected = p.notional + p.coupon_rate * p.notional * (autocall_period + 1)
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_no_autocall_par_at_maturity_returns_notional_plus_six_coupons():
    """Worst-of stays in (strike, autocall) every period — no autocall, no KI."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81])  # all in (0.70, 1.00)
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    expected = p.notional + 6 * p.coupon_rate * p.notional
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_no_autocall_ki_pays_worst_perf_times_notional_cash_settled():
    """Cash-settled knock-in at maturity: redemption = N * W(T), not N * W(T) / strike."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    W_T = 0.50
    perf = np.array([0.85, 0.80, 0.75, 0.72, 0.71, W_T])  # final dips below strike
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    expected = (p.notional * W_T) + 6 * p.coupon_rate * p.notional
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_physical_delivery_scales_by_strike():
    p = _amzn_meta_mu_product(physical_delivery=True)
    grid = ObservationGrid.from_product(p)
    W_T = 0.50
    perf = np.array([0.85, 0.80, 0.75, 0.72, 0.71, W_T])
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    expected = (p.notional * W_T / p.strike) + 6 * p.coupon_rate * p.notional
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_settlement_method_formula_equivalence_at_w_below_strike():
    """At W(T)=0.5, K=0.7: physical → N·W/K = 0.714N, cash → N·W = 0.5N.

    Both branches must match the respective formulas exactly and differ from
    each other by the 1/K factor that is the *whole* economic distinction
    between the two settlement methods.
    """
    W_T = 0.50
    perf = np.array([0.85, 0.80, 0.75, 0.72, 0.71, W_T])

    p_phys = _amzn_meta_mu_product(physical_delivery=True)
    p_cash = _amzn_meta_mu_product(physical_delivery=False)
    assert p_phys.strike == 0.70 == p_cash.strike  # sanity

    grid_phys = ObservationGrid.from_product(p_phys)
    grid_cash = ObservationGrid.from_product(p_cash)
    paths_phys, spots_phys = _build_constant_perf_paths(p_phys, grid_phys, perf)
    paths_cash, spots_cash = _build_constant_perf_paths(p_cash, grid_cash, perf)

    coupons = 6 * p_phys.coupon_rate * p_phys.notional

    pv_phys = payoff_per_path(paths=paths_phys, spots=spots_phys, product=p_phys, grid=grid_phys, rate=0.0)
    pv_cash = payoff_per_path(paths=paths_cash, spots=spots_cash, product=p_cash, grid=grid_cash, rate=0.0)

    expected_phys_redemption = p_phys.notional * W_T / p_phys.strike  # 0.714 N
    expected_cash_redemption = p_cash.notional * W_T                    # 0.500 N
    np.testing.assert_allclose(pv_phys[0], expected_phys_redemption + coupons, rtol=1e-12)
    np.testing.assert_allclose(pv_cash[0], expected_cash_redemption + coupons, rtol=1e-12)

    # The 1/K relationship is the entire economic difference between the two.
    redemption_phys = pv_phys[0] - coupons
    redemption_cash = pv_cash[0] - coupons
    np.testing.assert_allclose(redemption_phys, redemption_cash / p_phys.strike, rtol=1e-12)


def test_conditional_coupon_skipped_when_below_barrier():
    """Conditional-coupon mode: coupon paid only when W ≥ coupon_barrier."""
    p = _amzn_meta_mu_product(coupon_barrier=0.75)
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.70, 0.78, 0.65, 0.80, 0.90])  # period 2/4 skipped
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    coupons_paid = 4 * p.coupon_rate * p.notional  # periods 1,3,5,6
    expected = p.notional + coupons_paid           # maturity at par (W_T=0.90 > strike)
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_flat_coupon_paid_every_period_until_autocall():
    """Even when W < strike at an interim obs, the flat coupon is still paid."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.50, 0.55, 0.40, 0.65, 0.60, 0.45])  # all below strike, no AC
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    coupons = 6 * p.coupon_rate * p.notional
    expected = p.notional * 0.45 + coupons
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


def test_continuous_ki_triggers_off_obs_dates():
    """Touching strike between obs dates triggers KI even if W(T) ≥ strike."""
    p = _amzn_meta_mu_product(continuous_ki=True)
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81])  # all above strike at obs
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    # Dip asset 0 to 0.50 of spot between obs 3 and 4 (grid index ~125).
    paths[0, 125, 0] = spots[0] * 0.50

    pv = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)
    # Continuous KI fires because path_min < strike; redemption = N * W(T) = N * 0.81.
    coupons = 6 * p.coupon_rate * p.notional
    expected = p.notional * 0.81 + coupons
    np.testing.assert_allclose(pv[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------


def test_discounting_with_positive_rate_reduces_pv():
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81])
    paths, spots = _build_constant_perf_paths(p, grid, perf)

    pv_zero = payoff_per_path(paths, spots, p, grid, rate=0.0)[0]
    pv_pos = payoff_per_path(paths, spots, p, grid, rate=0.05)[0]
    assert pv_pos < pv_zero
    # Manual check: PV = sum(c * df_j) + N * df_M
    c = p.coupon_rate * p.notional
    pay_yf = p.pay_year_fractions()
    expected = sum(c * np.exp(-0.05 * yf) for yf in pay_yf)
    expected += p.notional * np.exp(-0.05 * pay_yf[-1])
    np.testing.assert_allclose(pv_pos, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Probability decomposition
# ---------------------------------------------------------------------------


def test_probability_decomposition_sums_to_one():
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    # Mix of three deterministic paths: autocall period 1, KI at maturity, par at maturity.
    perfs = np.array([
        [1.05, 1.05, 1.05, 1.05, 1.05, 1.05],  # autocall period 0
        [0.80, 0.75, 0.72, 0.68, 0.65, 0.50],  # KI at maturity
        [0.85, 0.80, 0.82, 0.78, 0.84, 0.81],  # par at maturity
    ])
    n_paths = perfs.shape[0]
    n_grid = grid.sim_n_steps + 1
    d = 3
    spots = np.array([100.0] * d, dtype=float)
    paths = np.full((n_paths, n_grid, d), spots[0], dtype=float)
    for i in range(n_paths):
        for j, idx in enumerate(grid.obs_indices):
            paths[i, idx, 0] = spots[0] * perfs[i, j]
            paths[i, idx, 1:] = spots[1:] * 5.0

    dec = probability_decomposition(paths=paths, spots=spots, product=p, grid=grid)
    np.testing.assert_allclose(dec.p_autocall_total, 1.0 / 3.0, rtol=1e-12)
    np.testing.assert_allclose(dec.p_ki_at_maturity, 1.0 / 3.0, rtol=1e-12)
    np.testing.assert_allclose(dec.p_alive_no_ki, 1.0 / 3.0, rtol=1e-12)
    total = dec.p_autocall_total + dec.p_ki_at_maturity + dec.p_alive_no_ki
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)
    np.testing.assert_allclose(dec.autocall_by_period[0], 1.0 / 3.0, rtol=1e-12)
    for j in range(1, 5):
        np.testing.assert_allclose(dec.autocall_by_period[j], 0.0, atol=0)
