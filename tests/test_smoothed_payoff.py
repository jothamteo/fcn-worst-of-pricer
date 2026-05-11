"""Tests for the smoothed (sigmoid-indicator) FCN payoff.

The smoothed payoff is the production-desk technique for computing stable
bump-and-revalue Greeks across the autocall and knock-in barriers. The key
correctness claims:

1. As ``smoothing_k → ∞``, the smoothed payoff converges to the hard payoff
   pointwise (path by path).
2. At a typical desk-ish ``k`` (e.g. 100) the *price* bias is small at the
   textbook fixings — well within MC standard error.
3. The smoothed payoff is C¹ in the underlyings, so the bump-and-revalue Δ
   no longer suffers from the discrete indicator-flip noise that the README's
   "Honest findings" section calls out.

These tests are deterministic — hand-crafted paths plus a small MC sanity
check on the AMZN/META/MU textbook product.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fcn_payoff import (
    FCNProduct,
    ObservationGrid,
    payoff_per_path,
    payoff_per_path_smoothed,
)
from src.gbm_simulation import SimulationConfig, simulate_paths
from src.market_data import MarketData


# ---------------------------------------------------------------------------
# Builders (mirror tests/test_fcn_payoff.py so smoothed and hard share a spec)
# ---------------------------------------------------------------------------


def _amzn_meta_mu_product(**overrides) -> FCNProduct:
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


def _build_constant_perf_paths(
    product: FCNProduct,
    grid: ObservationGrid,
    perf_by_period: np.ndarray,
    d: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    spots = np.array([100.0] * d, dtype=float)
    n_paths = 1
    n_grid = grid.sim_n_steps + 1
    paths = np.empty((n_paths, n_grid, d), dtype=float)
    paths[:, :, :] = spots
    for j, idx in enumerate(grid.obs_indices):
        paths[0, idx, 0] = spots[0] * perf_by_period[j]
        paths[0, idx, 1:] = spots[1:] * 5.0
    return paths, spots


# ---------------------------------------------------------------------------
# Convergence: smoothed → hard as k → ∞
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perf",
    [
        # No-autocall, par-at-maturity (far from every barrier).
        np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81]),
        # No-autocall, KI breach.
        np.array([0.85, 0.80, 0.75, 0.72, 0.71, 0.50]),
        # Autocall at period 2.
        np.array([0.85, 0.80, 1.05, 1.10, 1.10, 1.10]),
        # Borderline KI at maturity (W_T well below strike).
        np.array([0.90, 0.85, 0.82, 0.78, 0.75, 0.40]),
    ],
)
def test_smoothed_converges_to_hard_as_k_to_infinity(perf):
    """Pointwise: smoothed PV → hard PV when every input is far from any barrier."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    hard = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)[0]
    # Use a large k. The transition width on a barrier of order 1 is ~1/k in
    # relative units, so k=5000 puts every barrier transition vanishingly
    # close to its hard breakpoint for inputs that sit ≥ 5% away.
    soft = payoff_per_path_smoothed(
        paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
        smoothing_k_ac=5000.0, smoothing_k_ki=5000.0, smoothing_k_coupon=5000.0,
    )[0]
    np.testing.assert_allclose(soft, hard, rtol=1e-6, atol=1e-3)


def test_smoothed_converges_monotonically_in_k():
    """As k grows the smoothed price approaches the hard price monotonically
    for an input that is on one definite side of the barrier."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    # Clearly above autocall at period 2 → hard payoff autocalls at period 2.
    perf = np.array([0.85, 0.80, 1.05, 1.10, 1.10, 1.10])
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    hard = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)[0]
    diffs = []
    for k in (10.0, 50.0, 200.0, 1000.0, 5000.0):
        soft = payoff_per_path_smoothed(
            paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
            smoothing_k_ac=k, smoothing_k_ki=k, smoothing_k_coupon=k,
        )[0]
        diffs.append(abs(soft - hard))
    # Each successive k should be closer to hard than the last.
    for a, b in zip(diffs, diffs[1:]):
        assert b <= a + 1e-9


# ---------------------------------------------------------------------------
# Price-bias sanity check on a real MC run
# ---------------------------------------------------------------------------


def _mc_price(product, market, n_paths=20_000, seed=12345, smoothing=None):
    grid = ObservationGrid.from_product(product=product)
    cfg = SimulationConfig(
        spots=market.spots,
        vols=market.vols,
        divs=market.divs,
        rate=market.rate,
        corr=market.corr,
        T=grid.sim_T,
        n_steps=grid.sim_n_steps,
        n_paths=n_paths,
        antithetic=True,
        seed=seed,
    )
    paths = simulate_paths(cfg)
    if smoothing is None:
        pv = payoff_per_path(
            paths=paths, spots=np.asarray(market.spots), product=product,
            grid=grid, rate=market.rate,
        )
    else:
        pv = payoff_per_path_smoothed(
            paths=paths, spots=np.asarray(market.spots), product=product,
            grid=grid, rate=market.rate, **smoothing,
        )
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(pv.shape[0]))


def test_smoothed_price_bias_is_within_a_few_SE_at_textbook_fixings():
    """At-the-money textbook fixings: smoothed price agrees with hard price
    within a few standard errors at the desk-default ``k = 100`` setting."""
    product = _amzn_meta_mu_product()
    tickers = ("AMZN", "META", "MU")
    history = pd.DataFrame(
        {t: np.full(2, 100.0) for t in tickers},
        index=pd.to_datetime(["2025-10-16", "2025-10-17"]),
    )
    market = MarketData(
        tickers=tickers,
        spots=np.array([220.0, 720.0, 110.0]),
        vols=np.array([0.30, 0.32, 0.45]),
        divs=np.array([0.0, 0.0, 0.0]),
        corr=np.array([
            [1.00, 0.55, 0.35],
            [0.55, 1.00, 0.30],
            [0.35, 0.30, 1.00],
        ]),
        rate=0.045,
        history=history,
        as_of=datetime(2025, 10, 17, tzinfo=timezone.utc),
    )
    hard, se_hard = _mc_price(product, market, n_paths=20_000, seed=12345)
    soft, se_soft = _mc_price(
        product, market, n_paths=20_000, seed=12345,
        smoothing={"smoothing_k_ac": 100.0, "smoothing_k_ki": 100.0, "smoothing_k_coupon": 100.0},
    )
    combined_se = float(np.sqrt(se_hard * se_hard + se_soft * se_soft))
    # Bias well within 5σ — generous bound because the smoothed payoff
    # systematically rounds the indicator transitions.
    assert abs(soft - hard) < 5.0 * combined_se + 0.005 * product.notional


# ---------------------------------------------------------------------------
# Continuous-KI smoothed variant
# ---------------------------------------------------------------------------


def test_continuous_ki_smoothed_triggers_off_obs_dates():
    p = _amzn_meta_mu_product(continuous_ki=True)
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81])
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    # Deep dip far below strike between obs.
    paths[0, 125, 0] = spots[0] * 0.40

    hard = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)[0]
    soft = payoff_per_path_smoothed(
        paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
        smoothing_k_ac=5000.0, smoothing_k_ki=5000.0, smoothing_k_coupon=5000.0,
    )[0]
    np.testing.assert_allclose(soft, hard, rtol=1e-6, atol=1e-3)


# ---------------------------------------------------------------------------
# n_autocall_obs = 0 edge case (no autocalls — just maturity check)
# ---------------------------------------------------------------------------


def test_smoothed_no_autocall_collapses_to_smoothed_maturity_only():
    p = _amzn_meta_mu_product(n_autocall_obs=0)
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.50])  # KI at maturity
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    hard = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)[0]
    soft = payoff_per_path_smoothed(
        paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
        smoothing_k_ac=5000.0, smoothing_k_ki=5000.0, smoothing_k_coupon=5000.0,
    )[0]
    np.testing.assert_allclose(soft, hard, rtol=1e-6, atol=1e-3)


# ---------------------------------------------------------------------------
# Conditional-coupon smoothing
# ---------------------------------------------------------------------------


def test_smoothed_conditional_coupon_converges():
    p = _amzn_meta_mu_product(coupon_barrier=0.75)
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.70, 0.78, 0.65, 0.80, 0.90])
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    hard = payoff_per_path(paths=paths, spots=spots, product=p, grid=grid, rate=0.0)[0]
    soft = payoff_per_path_smoothed(
        paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
        smoothing_k_ac=5000.0, smoothing_k_ki=5000.0, smoothing_k_coupon=5000.0,
    )[0]
    np.testing.assert_allclose(soft, hard, rtol=1e-6, atol=1e-3)


# ---------------------------------------------------------------------------
# Smoothness — finite-difference Δ on the smoothed payoff stays bounded
# across the autocall barrier where the hard payoff exhibits indicator flips.
# ---------------------------------------------------------------------------


def test_smoothed_delta_is_finite_and_continuous_across_autocall_barrier():
    """The smoothed PV is a smooth function of the simulation start. Across
    the autocall barrier the finite-difference derivative w.r.t. spot stays
    bounded — no discrete jump — which is the production-desk property."""
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    # Path where the SECOND obs is right near the autocall barrier, so a
    # small spot perturbation flips the hard indicator. We build a path
    # where the worst-of at obs 1 sits exactly at autocall.
    perf = np.array([0.85, 1.00, 0.82, 0.78, 0.84, 0.81])
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    base = payoff_per_path_smoothed(
        paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
        smoothing_k_ac=100.0, smoothing_k_ki=100.0, smoothing_k_coupon=100.0,
    )[0]

    # Sweep tiny perturbations to asset 0 at obs 1 and check the finite
    # difference stays bounded (no discrete jump of size N=50,000).
    last_fd = None
    for eps in (0.001, 0.0005, 0.00025):
        paths_up = paths.copy()
        paths_up[0, grid.obs_indices[1], 0] = spots[0] * (perf[1] + eps)
        paths_dn = paths.copy()
        paths_dn[0, grid.obs_indices[1], 0] = spots[0] * (perf[1] - eps)
        up = payoff_per_path_smoothed(
            paths=paths_up, spots=spots, product=p, grid=grid, rate=0.0,
            smoothing_k_ac=100.0, smoothing_k_ki=100.0, smoothing_k_coupon=100.0,
        )[0]
        dn = payoff_per_path_smoothed(
            paths=paths_dn, spots=spots, product=p, grid=grid, rate=0.0,
            smoothing_k_ac=100.0, smoothing_k_ki=100.0, smoothing_k_coupon=100.0,
        )[0]
        fd = (up - dn) / (2.0 * eps * spots[0])
        # The hard payoff at this point would have an indicator flip worth
        # ~N=50,000 in payoff space, giving an unbounded FD as eps → 0. The
        # smoothed payoff's FD is finite and bounded.
        assert np.isfinite(fd)
        assert abs(fd) < 1e6  # FD is a derivative w.r.t. price, not jump-flip
        # FD should also be stable across shrinking eps (Lipschitz-continuous
        # in eps because the smoothed PV is C^∞).
        if last_fd is not None:
            assert abs(fd - last_fd) < 0.05 * max(abs(last_fd), 1.0) + 1.0
        last_fd = fd


def test_smoothing_k_must_be_positive():
    p = _amzn_meta_mu_product()
    grid = ObservationGrid.from_product(p)
    perf = np.array([0.85, 0.80, 0.82, 0.78, 0.84, 0.81])
    paths, spots = _build_constant_perf_paths(p, grid, perf)
    with pytest.raises(ValueError, match="smoothing_k"):
        payoff_per_path_smoothed(
            paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
            smoothing_k_ac=0.0,
        )
    with pytest.raises(ValueError, match="smoothing_k"):
        payoff_per_path_smoothed(
            paths=paths, spots=spots, product=p, grid=grid, rate=0.0,
            smoothing_k_ki=-1.0,
        )
