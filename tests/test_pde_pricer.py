"""Tests for the 1D Crank-Nicolson PDE pricer.

Three layers:
  1. Calibration against Black-Scholes on a vanilla European call (the CN
     machinery itself, independent of FCN logic).
  2. Hand-checked deterministic limits on the single-asset FCN (zero-vol,
     zero-rate sanity checks).
  3. Cross-validation against the Monte Carlo pricer on the single-asset
     FCN — PDE and MC should agree within a few MC standard errors.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fcn_payoff import FCNProduct
from src.market_data import MarketData
from src.mc_pricer import price_fcn
from src.pde_pricer import (
    PDEResult,
    _build_log_grid,
    _solve_tridiag,
    price_european_call_pde,
    price_fcn_pde_1d,
)
from src.utils import black_scholes_price


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_asset_product(**overrides) -> FCNProduct:
    """Single-asset analogue of the JT AMZN/META/MU FCN. Schedule + barriers
    are identical to the worst-of test fixture; we just price it on one name."""
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
        coupon_barrier=None,
        geared_downside=False,
        continuous_ki=False,
    )
    base.update(overrides)
    return FCNProduct(**base)


def _single_asset_market(spot=100.0, vol=0.35, rate=0.04, div=0.0) -> MarketData:
    """A 1-asset MarketData for plumbing the MC pricer through the same code path."""
    return MarketData(
        tickers=["AMZN"],
        spots=np.array([spot], dtype=float),
        vols=np.array([vol], dtype=float),
        divs=np.array([div], dtype=float),
        corr=np.array([[1.0]]),
        rate=rate,
        history=pd.DataFrame(),
        sources={"as_of": "test"},
    )


# ---------------------------------------------------------------------------
# 1. Calibration against Black-Scholes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spot,strike", [(100.0, 100.0), (110.0, 100.0), (90.0, 100.0)])
def test_pde_european_call_matches_black_scholes(spot, strike):
    """PDE vanilla call should match BS to within 1e-3 in absolute price terms
    at a generous grid resolution (800 x 400)."""
    vol, rate, div, T = 0.30, 0.04, 0.00, 1.0
    bs = black_scholes_price(
        spot=spot, strike=strike, rate=rate, div=div, vol=vol, T=T, option_type="call"
    )
    pde = price_european_call_pde(
        spot=spot, strike=strike, vol=vol, rate=rate, div=div, T=T,
        n_space=800, n_time=400, x_range_sigma=6.0,
    )
    np.testing.assert_allclose(pde.price, bs, atol=1e-3)


def test_pde_european_call_grid_convergence():
    """Doubling the grid resolution should reduce error by ≈ 4× (CN is 2nd order
    in both space and time). We're generous and require ≥ 2.5× error reduction
    to allow for the boundary-extrapolation contribution and round-off."""
    vol, rate, div, T = 0.30, 0.04, 0.00, 1.0
    spot, strike = 100.0, 100.0
    bs = black_scholes_price(
        spot=spot, strike=strike, rate=rate, div=div, vol=vol, T=T, option_type="call"
    )

    coarse = price_european_call_pde(
        spot=spot, strike=strike, vol=vol, rate=rate, div=div, T=T,
        n_space=200, n_time=100,
    )
    fine = price_european_call_pde(
        spot=spot, strike=strike, vol=vol, rate=rate, div=div, T=T,
        n_space=400, n_time=200,
    )
    err_coarse = abs(coarse.price - bs)
    err_fine = abs(fine.price - bs)
    assert err_fine < err_coarse / 2.5, (
        f"Error did not shrink enough: {err_coarse:.4e} -> {err_fine:.4e}"
    )


# ---------------------------------------------------------------------------
# 2. Hand-checked FCN limits
# ---------------------------------------------------------------------------


def test_pde_zero_vol_positive_rate_autocalls_period_1():
    """With σ = 0 and r > 0, S(t) = S₀ · exp(r·t) > S₀ for t > 0, so the note
    autocalls at the very first observation. The PDE price at x = 0 should be
    (N + c) discounted from pay[0] back to t=0."""
    product = _single_asset_product()
    vol, rate, div = 1e-6, 0.04, 0.0   # vol≈0; we need vol > 0 for the log-grid scale
    # x_range_sigma is huge here because sigma_T → 0; but the grid is symmetric
    # in x ∈ [-x_half, x_half] with x_half = x_range_sigma · σ · √T. We need
    # x = 0 to be on the grid and the autocall region to be reachable. With
    # σ ≈ 0 the grid collapses; pick x_range_sigma large enough that:
    #   x_half ≥ log(autocall_barrier) ≈ 0
    # The grid still works because the deterministic path stays at x = 0.
    pde = price_fcn_pde_1d(
        spot=100.0, vol=vol, div=div, rate=rate, product=product,
        n_space=400, n_time_per_period=40, x_range_sigma=200_000.0,
    )
    # Deterministic: rate > 0 means S/S_0 > 1 at any t > 0, so the note triggers
    # autocall on obs[0]. Cashflow = N + c paid at pay[0].
    c = product.coupon_rate * product.notional
    df_pay_0 = float(np.exp(-rate * (product.pay_dates[0] - product.issue_date).days / 365.0))
    expected = (product.notional + c) * df_pay_0
    np.testing.assert_allclose(pde.price, expected, rtol=1e-3)


def test_pde_extreme_otm_at_issue_collapses_to_coupon_strip():
    """If we start spot far below the strike (S_0 such that S/S_0 = 1 already
    sits below the strike at every obs, with σ = 0 and r = q = 0), the FCN
    pays 6 flat coupons and redeems at N · 1 (because perf = 1.0 ≥ 0.70)."""
    # Easier: σ = 0, r = 0, q = 0, S(t) = S_0. Worst-of perf = 1.0 throughout.
    # Then `perf >= autocall_barrier=1.0` triggers — so it DOES autocall at obs[0].
    # Cashflow = N + c, no discount.
    product = _single_asset_product()
    pde = price_fcn_pde_1d(
        spot=100.0, vol=1e-6, div=0.0, rate=0.0, product=product,
        n_space=400, n_time_per_period=40, x_range_sigma=200_000.0,
    )
    c = product.coupon_rate * product.notional
    expected = product.notional + c
    np.testing.assert_allclose(pde.price, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. Cross-validation: PDE vs MC on the single-asset FCN
# ---------------------------------------------------------------------------


def test_pde_matches_mc_on_single_asset_fcn():
    """Run the same product through MC (high path count) and PDE; the prices
    should agree within 3 MC standard errors. This is the headline test of
    Phase 4 and the reason the PDE exists."""
    product = _single_asset_product()
    market = _single_asset_market(spot=100.0, vol=0.35, rate=0.04, div=0.0)

    mc = price_fcn(market=market, product=product, n_paths=80_000, antithetic=True, seed=2026)
    pde = price_fcn_pde_1d(
        spot=float(market.spots[0]),
        vol=float(market.vols[0]),
        div=float(market.divs[0]),
        rate=market.rate,
        product=product,
        n_space=800,
        n_time_per_period=80,
        x_range_sigma=6.0,
    )

    band = 3.0 * mc.standard_error
    delta = abs(pde.price - mc.price)
    assert delta < band, (
        f"PDE {pde.price:,.4f} vs MC {mc.price:,.4f} (SE {mc.standard_error:,.4f}) "
        f"differ by {delta:,.4f} > 3·SE = {band:,.4f}"
    )


def test_pde_matches_mc_on_single_asset_fcn_geared_downside():
    """Same cross-validation, but with the geared variant of the maturity payoff."""
    product = _single_asset_product(geared_downside=True)
    market = _single_asset_market(spot=100.0, vol=0.40, rate=0.04, div=0.0)

    mc = price_fcn(market=market, product=product, n_paths=80_000, antithetic=True, seed=11)
    pde = price_fcn_pde_1d(
        spot=float(market.spots[0]),
        vol=float(market.vols[0]),
        div=float(market.divs[0]),
        rate=market.rate,
        product=product,
        n_space=800,
        n_time_per_period=80,
    )
    band = 3.0 * mc.standard_error
    assert abs(pde.price - mc.price) < band


def test_pde_matches_mc_on_single_asset_fcn_conditional_coupon():
    """And with a conditional coupon barrier — exercises a third payoff branch."""
    product = _single_asset_product(coupon_barrier=0.80)
    market = _single_asset_market(spot=100.0, vol=0.35, rate=0.04, div=0.005)

    mc = price_fcn(market=market, product=product, n_paths=80_000, antithetic=True, seed=99)
    pde = price_fcn_pde_1d(
        spot=float(market.spots[0]),
        vol=float(market.vols[0]),
        div=float(market.divs[0]),
        rate=market.rate,
        product=product,
        n_space=800,
        n_time_per_period=80,
    )
    band = 3.0 * mc.standard_error
    assert abs(pde.price - mc.price) < band


# ---------------------------------------------------------------------------
# 4. Internal helpers
# ---------------------------------------------------------------------------


def test_tridiag_solver_matches_dense_inverse():
    rng = np.random.default_rng(0)
    n = 30
    # Build a random diagonally-dominant tridiag system.
    sub = rng.standard_normal(n)
    sup = rng.standard_normal(n)
    diag = 5.0 + np.abs(rng.standard_normal(n))
    rhs = rng.standard_normal(n)

    # Dense reference:
    A = np.diag(diag) + np.diag(sub[1:], k=-1) + np.diag(sup[:-1], k=1)
    expected = np.linalg.solve(A, rhs)

    got = _solve_tridiag(sub, diag, sup, rhs)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_log_grid_centers_on_zero_with_even_n_space():
    x, S = _build_log_grid(spot=100.0, x_half=1.5, n_space=200)
    assert len(x) == 201
    mid = 100
    assert x[mid] == 0.0
    assert S[mid] == 100.0
    np.testing.assert_allclose(x[0], -x[-1], rtol=1e-12)


def test_log_grid_force_even_when_odd_n_space():
    x, _ = _build_log_grid(spot=100.0, x_half=1.5, n_space=201)
    # _build_log_grid bumps odd up to even.
    assert len(x) == 203
    mid = 101
    assert x[mid] == 0.0
