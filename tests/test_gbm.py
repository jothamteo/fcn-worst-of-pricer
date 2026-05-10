"""Statistical sanity tests for the GBM simulator.

These check that:
  - Marginal moments of S_T match the closed-form GBM distribution.
  - Pairwise correlation of log-returns recovers the input correlation.
  - Antithetic variates preserve the mean and zero out odd functionals.
  - Same seed -> same paths (reproducibility).
  - Pre-drawn normals route works (CRN plumbing).
  - n_paths * n_steps small enough to stay fast (< 1s per test).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gbm_simulation import (
    SimulationConfig,
    cholesky_lower,
    draw_normals,
    observation_indices,
    simulate_paths,
)
from src.utils import black_scholes_price


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _three_name_config(**overrides):
    base = dict(
        spots=np.array([215.20, 455.19, 411.68]),
        vols=np.array([0.449, 0.659, 0.468]),
        divs=np.array([0.0002, 0.0, 0.0085]),
        rate=0.0366,
        corr=np.array(
            [
                [1.000, 0.676, 0.688],
                [0.676, 1.000, 0.612],
                [0.688, 0.612, 1.000],
            ]
        ),
        T=1.0,
        n_steps=252,
        n_paths=20_000,
        antithetic=False,
        seed=42,
    )
    base.update(overrides)
    return SimulationConfig(**base)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_config_rejects_non_psd_corr():
    bad_corr = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(ValueError, match="PSD"):
        _three_name_config(corr=bad_corr)


def test_config_rejects_off_unit_diagonal():
    corr = np.eye(3)
    corr[0, 0] = 1.01
    with pytest.raises(ValueError, match="unit diagonal"):
        _three_name_config(corr=corr)


def test_config_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="vols shape"):
        _three_name_config(vols=np.array([0.4, 0.5]))


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_marginal_mean_under_risk_neutral_measure():
    """E[S_i(T) / S_i(0)] = exp((r - q_i) T)."""
    cfg = _three_name_config(n_paths=50_000)
    paths = simulate_paths(cfg)
    terminal = paths[:, -1, :] / cfg.spots
    expected = np.exp((cfg.rate - cfg.divs) * cfg.T)
    realised = terminal.mean(axis=0)
    # 3-sigma band: std of S_T/S_0 ~ sqrt(exp(sigma^2 T) - 1) * E[S_T/S_0]
    std_ratio = np.sqrt(np.exp(cfg.vols ** 2 * cfg.T) - 1.0) * expected
    se = std_ratio / np.sqrt(cfg.n_paths)
    assert np.all(np.abs(realised - expected) < 4 * se)


def test_marginal_log_variance():
    """Var(log S_i(T) / S_i(0)) = sigma_i^2 T."""
    cfg = _three_name_config(n_paths=50_000)
    paths = simulate_paths(cfg)
    log_ret = np.log(paths[:, -1, :] / cfg.spots)
    expected = cfg.vols ** 2 * cfg.T
    realised = log_ret.var(axis=0, ddof=1)
    # Sample variance has rel SE ~ sqrt(2 / (n - 1)).
    rel_se = np.sqrt(2.0 / (cfg.n_paths - 1))
    assert np.all(np.abs(realised - expected) / expected < 5 * rel_se)


def test_pairwise_correlation_recovered():
    """Corr(log S_i(T)/S_i(0), log S_j(T)/S_j(0)) = rho_{ij}."""
    cfg = _three_name_config(n_paths=80_000)
    paths = simulate_paths(cfg)
    log_ret = np.log(paths[:, -1, :] / cfg.spots)
    realised_corr = np.corrcoef(log_ret, rowvar=False)
    # Fisher-z 3-sigma band; for n=80k, |z_hat - z| < ~0.011.
    assert np.allclose(realised_corr, cfg.corr, atol=0.02)


def test_zero_vol_is_deterministic_drift():
    cfg = _three_name_config(
        vols=np.zeros(3), divs=np.zeros(3), n_paths=10, n_steps=12
    )
    paths = simulate_paths(cfg)
    expected = np.broadcast_to(cfg.spots * np.exp(cfg.rate * cfg.T), paths[:, -1, :].shape)
    np.testing.assert_allclose(paths[:, -1, :], expected, rtol=1e-12)


def test_zero_drift_with_zero_vol_holds_spot_constant():
    cfg = _three_name_config(
        rate=0.0, vols=np.zeros(3), divs=np.zeros(3), n_paths=5, n_steps=8
    )
    paths = simulate_paths(cfg)
    np.testing.assert_allclose(paths, np.broadcast_to(cfg.spots, paths.shape))


# ---------------------------------------------------------------------------
# Antithetic
# ---------------------------------------------------------------------------

def test_antithetic_doubles_path_count_and_centres_log_returns():
    cfg = _three_name_config(n_paths=5_000, antithetic=True)
    paths = simulate_paths(cfg)
    assert paths.shape == (2 * cfg.n_paths, cfg.n_steps + 1, cfg.d)

    log_ret = np.log(paths[:, -1, :] / cfg.spots)
    primal, mirror = log_ret[: cfg.n_paths], log_ret[cfg.n_paths :]
    # primal + mirror = 2 * drift (independent of eta).
    drift_term = np.broadcast_to(
        2.0 * (cfg.rate - cfg.divs - 0.5 * cfg.vols ** 2) * cfg.T,
        primal.shape,
    )
    np.testing.assert_allclose(primal + mirror, drift_term, atol=1e-12)


def test_antithetic_reduces_variance_for_linear_payoff():
    """A linear-in-S_T payoff has its noise component perfectly cancelled by
    antithetic — variance of the pair-averaged terminal price should be
    dramatically smaller than the iid case at the same draw count.
    """
    cfg_no = _three_name_config(n_paths=4_000, antithetic=False, seed=7)
    cfg_yes = _three_name_config(n_paths=2_000, antithetic=True, seed=7)
    pno = simulate_paths(cfg_no)[:, -1, 0]
    pyes = simulate_paths(cfg_yes)[:, -1, 0]
    pyes_paired = 0.5 * (pyes[: cfg_yes.n_paths] + pyes[cfg_yes.n_paths :])
    # log S_T is exactly antithetic; S_T = exp(log S_T) is convex but the
    # variance reduction should still be material (>5x in this regime).
    assert pyes_paired.var(ddof=1) < pno.var(ddof=1) / 5.0


# ---------------------------------------------------------------------------
# Reproducibility & CRN
# ---------------------------------------------------------------------------

def test_same_seed_same_paths():
    a = simulate_paths(_three_name_config(n_paths=512, seed=123))
    b = simulate_paths(_three_name_config(n_paths=512, seed=123))
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_paths():
    a = simulate_paths(_three_name_config(n_paths=512, seed=1))
    b = simulate_paths(_three_name_config(n_paths=512, seed=2))
    assert not np.allclose(a, b)


def test_pre_drawn_normals_route():
    cfg = _three_name_config(n_paths=512)
    rng = np.random.default_rng(99)
    eta = draw_normals(cfg.n_paths, cfg.n_steps, cfg.d, rng)
    a = simulate_paths(cfg, normals=eta)
    b = simulate_paths(cfg, normals=eta.copy())
    np.testing.assert_array_equal(a, b)


def test_pre_drawn_normals_shape_validation():
    cfg = _three_name_config(n_paths=10)
    bad = np.zeros((10, cfg.n_steps, cfg.d + 1))
    with pytest.raises(ValueError, match="normals shape"):
        simulate_paths(cfg, normals=bad)


def test_crn_kills_bump_noise_vs_iid():
    """Using CRN across a small spot bump should give Delta estimates with
    much lower variance than redrawing normals for each bump.
    """
    cfg_base = _three_name_config(n_paths=2_000, seed=11)
    rng = np.random.default_rng(11)
    eta = draw_normals(cfg_base.n_paths, cfg_base.n_steps, cfg_base.d, rng)

    bump = 1e-3 * cfg_base.spots[0]
    spots_up = cfg_base.spots.copy()
    spots_up[0] += bump
    spots_dn = cfg_base.spots.copy()
    spots_dn[0] -= bump

    cfg_up = _three_name_config(n_paths=cfg_base.n_paths, spots=spots_up)
    cfg_dn = _three_name_config(n_paths=cfg_base.n_paths, spots=spots_dn)

    # Per-path delta under CRN: fluctuation around true delta should be tiny.
    p_up_crn = simulate_paths(cfg_up, normals=eta)[:, -1, 0]
    p_dn_crn = simulate_paths(cfg_dn, normals=eta)[:, -1, 0]
    delta_crn = (p_up_crn - p_dn_crn) / (2 * bump)

    # Independent draws for the bumps.
    p_up_iid = simulate_paths(_three_name_config(n_paths=cfg_base.n_paths, spots=spots_up, seed=21))[:, -1, 0]
    p_dn_iid = simulate_paths(_three_name_config(n_paths=cfg_base.n_paths, spots=spots_dn, seed=22))[:, -1, 0]
    delta_iid = (p_up_iid - p_dn_iid) / (2 * bump)

    assert delta_crn.std(ddof=1) * 100 < delta_iid.std(ddof=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_cholesky_lower_recovers_corr():
    corr = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0]])
    L = cholesky_lower(corr)
    np.testing.assert_allclose(L @ L.T, corr, atol=1e-12)


def test_observation_indices_quarterly_on_daily_grid():
    idx = observation_indices(n_steps=252, n_obs=4)
    np.testing.assert_array_equal(idx, np.array([63, 126, 189, 252]))


def test_observation_indices_requires_multiple():
    with pytest.raises(ValueError, match="multiple"):
        observation_indices(n_steps=250, n_obs=4)


# ---------------------------------------------------------------------------
# Cross-check against Black-Scholes for the d=1 European call.
# This is not a GBM test per se, but it's a useful end-to-end pulse check.
# ---------------------------------------------------------------------------

def test_european_call_matches_black_scholes_1d():
    spot, vol, div, rate, T, strike = 100.0, 0.20, 0.0, 0.03, 1.0, 110.0
    cfg = SimulationConfig(
        spots=np.array([spot]),
        vols=np.array([vol]),
        divs=np.array([div]),
        rate=rate,
        corr=np.array([[1.0]]),
        T=T,
        n_steps=50,
        n_paths=100_000,
        antithetic=True,
        seed=2026,
    )
    paths = simulate_paths(cfg)
    payoff = np.maximum(paths[:, -1, 0] - strike, 0.0)
    mc_price = np.exp(-rate * T) * payoff.mean()
    bs_price = black_scholes_price(spot, strike, rate, div, vol, T, "call")
    # 4-sigma MC band given antithetic; loose tolerance is intentional.
    se = np.exp(-rate * T) * payoff.std(ddof=1) / np.sqrt(payoff.size)
    assert abs(mc_price - bs_price) < 4 * se
