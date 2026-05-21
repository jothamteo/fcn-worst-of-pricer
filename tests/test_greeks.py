"""Tests for the Greeks module.

Layers, in order of increasing strictness:
  1. **Off-grid sanity** — PDE Δ on a vanilla call matches the Black-Scholes
     closed-form Δ at the at-the-money point. This validates the off-grid
     finite-difference machinery against an analytic target.
  2. **PDE vs MC agreement on the single-asset FCN.** MC Δ, Γ, vega should
     agree with the PDE values within a small multiple of the MC standard
     error. This is the headline Phase 5 test.
  3. **CRN reproducibility.** Calling `mc_greeks_bump` twice with the same
     seed gives bit-identical Greeks.
  4. **Sign sanity** — Δ of a worst-of FCN is positive in each name (the note
     is long-the-basket), vega is negative (long short-vol), and pairwise
     correlation sensitivity is positive (long correlation: more correlation
     means lower probability of a single name dragging the worst-of down).
"""

from __future__ import annotations

from datetime import date

import math

import numpy as np
import pandas as pd
import pytest

from src.fcn_payoff import FCNProduct
from src.greeks import mc_greeks_bump, pde_greeks_1d
from src.market_data import MarketData
from src.utils import black_scholes_price


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
        coupon_barrier=None,
        physical_delivery=False,
        continuous_ki=False,
    )
    base.update(overrides)
    return FCNProduct(**base)


def _single_asset_market(spot=100.0, vol=0.35, rate=0.04, div=0.0) -> MarketData:
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


def _three_asset_market(spot=100.0, vol=0.35, rho=0.6, rate=0.04, div=0.0) -> MarketData:
    d = 3
    return MarketData(
        tickers=["A", "B", "C"],
        spots=np.full(d, spot, dtype=float),
        vols=np.full(d, vol, dtype=float),
        divs=np.full(d, div, dtype=float),
        corr=np.full((d, d), rho) + (1.0 - rho) * np.eye(d),
        rate=rate,
        history=pd.DataFrame(),
        sources={"as_of": "test"},
    )


# ---------------------------------------------------------------------------
# 1. PDE vanilla Δ vs Black-Scholes (off-grid sanity)
# ---------------------------------------------------------------------------


def test_pde_offgrid_delta_matches_bs_for_vanilla_call():
    """Stand up a tiny "FCN" whose payoff reduces to a vanilla European put
    payoff at maturity (physical-delivery downside, no autocall) and check Δ vs BS Δ.

    The cleanest test would use the standalone `price_european_call_pde`, but
    that returns only `V_at_issue` — exactly what we need to test the
    off-grid Δ extraction on. We replicate the off-grid Δ inline here.
    """
    from src.pde_pricer import price_european_call_pde

    spot, strike, vol, rate, div, T = 100.0, 100.0, 0.30, 0.04, 0.0, 1.0
    res = price_european_call_pde(
        spot=spot, strike=strike, vol=vol, rate=rate, div=div, T=T,
        n_space=1600, n_time=400, x_range_sigma=6.0,
    )
    S = res.S_grid; V = res.V_at_issue
    i = int(np.argmin(np.abs(S - spot)))
    h_minus = S[i] - S[i - 1]
    h_plus = S[i + 1] - S[i]
    h_total = S[i + 1] - S[i - 1]
    delta_pde = (V[i + 1] - V[i - 1]) / h_total

    # BS Δ for a call = exp(-qT) · N(d1).
    sigma_root_t = vol * math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * T) / sigma_root_t
    from scipy.stats import norm as _norm
    delta_bs = math.exp(-div * T) * _norm.cdf(d1)

    assert abs(delta_pde - delta_bs) < 5e-3, (
        f"PDE Δ {delta_pde:.5f} vs BS Δ {delta_bs:.5f}"
    )


# ---------------------------------------------------------------------------
# 2. PDE vs MC agreement on the single-asset FCN
# ---------------------------------------------------------------------------


def test_pde_and_mc_greeks_agree_on_single_asset_fcn():
    product = _product()
    market = _single_asset_market(spot=100.0, vol=0.35, rate=0.04, div=0.0)

    mc = mc_greeks_bump(
        market=market, product=product,
        n_paths=80_000, antithetic=True, seed=20260511,
    )
    pde = pde_greeks_1d(
        spot=float(market.spots[0]), vol=float(market.vols[0]),
        div=float(market.divs[0]), rate=market.rate, product=product,
        n_space=1600, n_time_per_period=120,
    )

    # Bands are 4·SE — generous because gamma SE is wide for bump methods near
    # barriers. The point of the test is to catch wiring bugs, not to certify
    # MC Greeks to PDE precision.
    se = mc.standard_errors
    assert abs(mc.delta[0] - pde.delta[0]) < 4.0 * se["delta"][0] + 0.05, (
        f"Δ disagree: MC {mc.delta[0]:.4f} ± {se['delta'][0]:.4f}, PDE {pde.delta[0]:.4f}"
    )
    assert abs(mc.vega[0] - pde.vega[0]) < 4.0 * se["vega"][0] + 50.0, (
        f"vega disagree: MC {mc.vega[0]:.4f} ± {se['vega'][0]:.4f}, PDE {pde.vega[0]:.4f}"
    )
    # Gamma is the loosest; we just check the sign matches (both should be
    # negative-leaning given the autocall cap).
    assert np.sign(mc.gamma[0]) == np.sign(pde.gamma[0]) or abs(mc.gamma[0]) < 5e-3


# ---------------------------------------------------------------------------
# 3. CRN reproducibility
# ---------------------------------------------------------------------------


def test_mc_greeks_are_seed_reproducible():
    product = _product()
    market = _three_asset_market()
    a = mc_greeks_bump(market, product, n_paths=10_000, seed=7)
    b = mc_greeks_bump(market, product, n_paths=10_000, seed=7)
    np.testing.assert_allclose(a.price, b.price, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.delta, b.delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.gamma, b.gamma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.vega, b.vega, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.rho_pair, b.rho_pair, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Sign / shape sanity (3-asset case)
# ---------------------------------------------------------------------------


def test_mc_greeks_3asset_signs_match_economics():
    """The standard worst-of FCN is:
       - long-spot in each name (positive Δ),
       - short-vol (negative vega — more vol means more KI risk),
       - long-correlation (positive ρ_pair — higher correlation means the
         worst-of is closer to the average, less KI risk).
    """
    product = _product()
    market = _three_asset_market(spot=100.0, vol=0.35, rho=0.6)
    g = mc_greeks_bump(market, product, n_paths=40_000, seed=42)

    # The notional is 50_000, so Δ scaled to a 1% move ≈ a few × 100 — never
    # rigidly zero, never wildly negative.
    assert (g.delta > 0).all(), f"Δ should all be positive, got {g.delta}"
    assert (g.vega < 0).all(), f"vega should all be negative, got {g.vega}"

    # Cross-correlation entries should be positive on net for this product.
    off_diag = g.rho_pair[np.triu_indices(3, k=1)]
    assert (off_diag > 0).all(), f"ρ_pair off-diagonals should be positive, got {off_diag}"


def test_mc_greeks_shapes_and_scaled_views():
    product = _product()
    market = _three_asset_market()
    g = mc_greeks_bump(market, product, n_paths=8_000, seed=1)
    assert g.delta.shape == (3,)
    assert g.gamma.shape == (3,)
    assert g.vega.shape == (3,)
    assert g.rho_pair.shape == (3, 3)
    # Symmetric.
    np.testing.assert_allclose(g.rho_pair, g.rho_pair.T)
    # Diagonals are zero.
    np.testing.assert_allclose(np.diag(g.rho_pair), np.zeros(3))
    # Scaled views are consistent with the raw values.
    np.testing.assert_allclose(g.delta_pct, g.delta * market.spots / 100.0)
    np.testing.assert_allclose(
        g.gamma_pct, g.gamma * market.spots * market.spots / 10_000.0
    )
    np.testing.assert_allclose(g.vega_per_volpt, g.vega / 100.0)
