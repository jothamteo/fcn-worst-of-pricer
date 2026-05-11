"""Tests for the hedging ratios module.

The hedging module is a thin composition layer on top of the Greeks the
pricer has already computed. The tests verify that the arithmetic is right
on hand-checked examples and that the sign conventions match the desk
convention documented in :mod:`src.hedging`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.hedging import (
    CorrelationRiskReport,
    black_scholes_call,
    compute_correlation_risk,
    compute_delta_hedge,
    compute_unhedgeable_risk_summary,
    compute_vega_hedge,
)


# ---------------------------------------------------------------------------
# Delta hedge — hand-checked arithmetic and sign convention
# ---------------------------------------------------------------------------


def test_delta_hedge_simple_hand_check():
    """delta = 0.5, notional = 1M, spot = 100  →  shares = 5,000 to buy."""
    deltas = {"AAPL": 5_000.0}            # raw delta (USD per USD spot)
    spots = {"AAPL": 100.0}
    out = compute_delta_hedge(deltas, notional=1_000_000.0, spots=spots)
    row = out["AAPL"]
    assert row.shares_to_buy == 5_000
    assert row.hedge_notional == 500_000.0
    assert row.hedge_pct_of_trade == pytest.approx(50.0)
    # spec-named alias is the negation
    assert row.shares_to_short == -5_000


def test_delta_hedge_negative_delta_sells_shares():
    """Negative FCN delta → dealer holds a short stock position (shares_to_buy < 0)."""
    deltas = {"AAPL": -300.0}
    spots = {"AAPL": 100.0}
    out = compute_delta_hedge(deltas, notional=1_000_000.0, spots=spots)
    row = out["AAPL"]
    assert row.shares_to_buy == -300
    assert row.hedge_notional == -30_000.0


def test_delta_hedge_rescales_from_pricer_reference_notional():
    """Pricer ran at N=50,000; trade is $1M — deltas should scale linearly."""
    deltas = {"AAPL": 50.0}                # delta at N_ref = 50,000
    spots = {"AAPL": 100.0}
    out = compute_delta_hedge(
        deltas, notional=1_000_000.0, spots=spots,
        pricer_reference_notional=50_000.0,
    )
    # scale = 1_000_000 / 50_000 = 20  →  delta_at_trade = 1,000
    assert out["AAPL"].shares_to_buy == 1_000


def test_delta_hedge_multi_name_covers_all_tickers():
    deltas = {"NVDA": 200.0, "AMD": -100.0, "TSM": 50.0}
    spots = {"NVDA": 150.0, "AMD": 200.0, "TSM": 180.0}
    out = compute_delta_hedge(deltas, notional=1_000_000.0, spots=spots)
    assert set(out.keys()) == {"NVDA", "AMD", "TSM"}
    assert out["NVDA"].shares_to_buy == 200
    assert out["AMD"].shares_to_buy == -100
    assert out["TSM"].shares_to_buy == 50


def test_delta_hedge_validates_inputs():
    with pytest.raises(ValueError, match="notional must be > 0"):
        compute_delta_hedge({"X": 1.0}, notional=0.0, spots={"X": 1.0})
    with pytest.raises(ValueError, match="must cover the same tickers"):
        compute_delta_hedge({"X": 1.0}, notional=1.0, spots={"Y": 1.0})


# ---------------------------------------------------------------------------
# Vega hedge — hand-checked arithmetic and sign convention
# ---------------------------------------------------------------------------


def test_vega_hedge_simple_hand_check():
    """Dealer +$10k vega; listed call vega +$200  →  short 50 options."""
    vegas = {"NVDA": 10_000.0}             # dealer view
    listed_vegas = {"NVDA": 200.0}         # listed ATM call vega per +1 vol pt
    listed_prices = {"NVDA": 12.50}        # premium per option
    out = compute_vega_hedge(vegas, listed_vegas, listed_prices)
    row = out["NVDA"]
    # n_options_to_buy = -10000 / 200 = -50  (short 50 options)
    assert row.n_options_to_buy == -50
    # total_premium = -50 × 12.50 = -625 (dealer receives premium)
    assert row.total_premium == pytest.approx(-625.0)
    # residual vega = 10_000 + (-50)*200 = 0
    assert row.residual_vega == pytest.approx(0.0)


def test_vega_hedge_opposite_sign_buys_options():
    """Dealer -$10k vega → BUY +50 options to neutralise."""
    vegas = {"NVDA": -10_000.0}
    listed_vegas = {"NVDA": 200.0}
    out = compute_vega_hedge(vegas, listed_vegas)
    row = out["NVDA"]
    assert row.n_options_to_buy == 50
    assert math.isnan(row.total_premium)        # no prices supplied
    assert row.residual_vega == pytest.approx(0.0)


def test_vega_hedge_residual_picks_up_rounding():
    """Non-integer ratio → residual vega is non-zero (rounding to whole options)."""
    vegas = {"NVDA": 10_050.0}
    listed_vegas = {"NVDA": 200.0}
    out = compute_vega_hedge(vegas, listed_vegas)
    # 10_050 / 200 = 50.25 → round → -50; residual = 10_050 + (-50)*200 = 50
    assert out["NVDA"].n_options_to_buy == -50
    assert out["NVDA"].residual_vega == pytest.approx(50.0)


def test_vega_hedge_rejects_zero_listed_vega():
    with pytest.raises(ValueError, match="non-zero"):
        compute_vega_hedge({"X": 1.0}, {"X": 0.0})


def test_black_scholes_call_matches_known_value():
    """Standard textbook: S=K=100, T=0.25, σ=0.20, r=0.05  →  call ≈ 4.6149."""
    price, vega = black_scholes_call(
        spot=100.0, strike=100.0, T=0.25, vol=0.20, rate=0.05, div=0.0,
    )
    assert price == pytest.approx(4.6149, abs=1e-3)
    # Vega is reported per +1 vol pt (per 0.01): textbook S·√T·φ(d1) ≈ 19.74
    # for the same params, divided by 100 → 0.1974
    assert vega == pytest.approx(0.1974, abs=1e-3)


# ---------------------------------------------------------------------------
# Correlation risk
# ---------------------------------------------------------------------------


def test_correlation_risk_central_difference_matches_hand_calc():
    """ρ ∈ {-0.10, 0.0, +0.10} with prices that are linear in ρ
    →  cega should equal the slope × 0.01."""
    # P(ρ) = 50_000 + 10_000 × ρ  (linear, slope = 10_000 per unit ρ)
    price_at = {-0.10: 49_000.0, 0.0: 50_000.0, +0.10: 51_000.0}
    rep = compute_correlation_risk(price_at)
    # cega per +0.01 = slope × 0.01 = 100
    assert rep.cega == pytest.approx(100.0)
    # P&L per +5 pp = cega × 5 = 500
    assert rep.correlation_pnl_per_5pct_move == pytest.approx(500.0)
    assert rep.hedgeable is False
    assert "long correlation" in rep.narrative


def test_correlation_risk_uses_closest_symmetric_pair():
    """Multi-shift dict: function picks the smallest +/- pair available."""
    price_at = {
        -0.10: 48_000.0,
        -0.05: 49_000.0,
        0.0:   50_000.0,
        +0.05: 51_000.0,
        +0.10: 52_000.0,
    }
    rep = compute_correlation_risk(price_at)
    # Uses ±0.05: (51_000 - 49_000) / 0.10 × 0.01 = 200
    assert rep.cega == pytest.approx(200.0)


def test_correlation_risk_requires_both_sides():
    with pytest.raises(ValueError, match="positive and one negative"):
        compute_correlation_risk({-0.05: 49_000.0, 0.0: 50_000.0})


# ---------------------------------------------------------------------------
# Unhedgeable risk summary — composition
# ---------------------------------------------------------------------------


def test_unhedgeable_risk_summary_composes_all_three_sources():
    greeks = {"delta": np.array([1.0, 2.0, 3.0])}
    corr_report = compute_correlation_risk(
        {-0.05: 49_000.0, 0.0: 50_000.0, +0.05: 51_000.0}
    )
    summary = compute_unhedgeable_risk_summary(
        greeks=greeks,
        scenario_pnls={"overnight_gap_to_ki_worst": -42_000.0},
        correlation_report=corr_report,
        notional=1_000_000.0,
        vol_skew_residual_vega={"NVDA": 1_500.0, "AMD": 800.0},
    )
    # 1 correlation + 1 scenario + 2 skew residuals = 4 rows
    assert len(summary.rows) == 4
    risks = [r["risk"] for r in summary.rows]
    assert "Correlation" in risks
    assert any("Gap risk" in r for r in risks)
    assert sum(1 for r in risks if "Vol-of-vol" in r) == 2
    # All four are not hedgeable
    assert all(r["hedgeable"] is False for r in summary.rows)
    # Percent-of-notional populated everywhere
    for r in summary.rows:
        if r["risk"] != "Correlation":
            assert math.isfinite(r["pct_of_notional"])


def test_unhedgeable_risk_summary_handles_empty_inputs():
    summary = compute_unhedgeable_risk_summary(greeks={}, notional=1_000_000.0)
    assert summary.rows == []
    assert summary.notional == 1_000_000.0
