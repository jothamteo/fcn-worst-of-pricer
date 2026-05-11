"""Worst-of FCN payoff — vectorised across Monte Carlo paths.

Notation matches `METHODOLOGY.md` §3, generalised to:

  - configurable, irregularly-spaced observation dates,
  - separate payment dates per observation (T+settlement);
  - flat (guaranteed) OR conditional-on-barrier coupons;
  - geared OR non-geared maturity downside;
  - European OR continuous knock-in monitoring;
  - any number of underlyings ≥ 1.

The product spec is held in `FCNProduct`. The payoff function takes simulated
paths (from `gbm_simulation.simulate_paths`) plus an `ObservationGrid` that maps
the product's calendar dates onto the simulation's time grid, and returns
present value per path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------------------
# Product spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FCNProduct:
    """Worst-of FCN product spec.

    Parameters
    ----------
    notional : float
        Issue size per note (e.g. 50_000).
    coupon_rate : float
        Coupon per period, as a decimal fraction of notional
        (e.g. 0.01535 = 1.535% per month).
    obs_dates : tuple of date
        Observation dates: the first `n_autocall_obs` of them are autocall fixing
        dates; the last one is the final valuation date (where the maturity
        strike / knock-in check is applied). Must equal `pay_dates` in length
        and be strictly increasing.
    pay_dates : tuple of date
        Payment date for each observation period. The j-th coupon (and the j-th
        autocall redemption, if it occurs) is paid on `pay_dates[j]`. Maturity
        payoff is paid on `pay_dates[-1]`. Must be ≥ the matching `obs_dates[j]`.
    issue_date : date
        Trade / fixing date (initial valuation). All `obs_dates` and `pay_dates`
        are measured forward from here. (We do not use `issue_date` itself as
        an observation point.)
    autocall_barrier : float
        Worst-of normalised price level at or above which the note autocalls.
        Standard worst-of FCN = 1.00 (i.e. 100% of initial).
    strike : float
        Worst-of normalised level at maturity below which the downside payoff
        kicks in. Standard = 0.70 (70% of initial). For non-geared FCNs this
        is also the knock-in barrier — there is no separate KI level.
    n_autocall_obs : int
        Number of leading observations that are autocall fixing dates. The
        remaining `len(obs_dates) - n_autocall_obs` observations (typically
        just one, the final valuation) are maturity-only checks. Must satisfy
        `0 ≤ n_autocall_obs ≤ len(obs_dates) - 1`.
    coupon_barrier : float, optional
        If supplied, coupon for period j is paid only when
        `W(obs_dates[j]) ≥ coupon_barrier`. If None (default), the coupon is
        flat / guaranteed each period until autocall.
    geared_downside : bool
        If True, redemption when knocked in is `N · W(T)/strike` (the standard
        "geared put" payoff with break-even at strike). If False (default for
        this JT-spec'd structure), redemption is `N · W(T)` (1:1 with worst-of
        performance from initial, capped at par).
    continuous_ki : bool
        If True, knock-in is triggered when the worst-of touches the strike
        on ANY simulated grid point during the life of the note. If False
        (default), knock-in is checked only at the final observation.
    """

    notional: float
    coupon_rate: float
    obs_dates: Sequence[date]
    pay_dates: Sequence[date]
    issue_date: date
    autocall_barrier: float = 1.00
    strike: float = 0.70
    n_autocall_obs: Optional[int] = None
    coupon_barrier: Optional[float] = None
    geared_downside: bool = False
    continuous_ki: bool = False

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError(f"notional must be > 0, got {self.notional}")
        if self.coupon_rate < 0:
            raise ValueError(f"coupon_rate must be ≥ 0, got {self.coupon_rate}")
        obs = tuple(self.obs_dates)
        pay = tuple(self.pay_dates)
        if len(obs) != len(pay):
            raise ValueError(
                f"obs_dates/pay_dates length mismatch: {len(obs)} vs {len(pay)}"
            )
        if len(obs) < 1:
            raise ValueError("need at least one observation date")
        for i in range(len(obs)):
            if pay[i] < obs[i]:
                raise ValueError(
                    f"pay_dates[{i}]={pay[i]} is before obs_dates[{i}]={obs[i]}"
                )
        for i in range(1, len(obs)):
            if obs[i] <= obs[i - 1]:
                raise ValueError("obs_dates must be strictly increasing")
            if pay[i] <= pay[i - 1]:
                raise ValueError("pay_dates must be strictly increasing")
        if obs[0] <= self.issue_date:
            raise ValueError("obs_dates must start strictly after issue_date")
        if not 0.0 < self.autocall_barrier:
            raise ValueError(f"autocall_barrier must be > 0, got {self.autocall_barrier}")
        if not 0.0 < self.strike <= self.autocall_barrier:
            raise ValueError(
                f"strike must satisfy 0 < strike ≤ autocall_barrier, got "
                f"strike={self.strike}, autocall_barrier={self.autocall_barrier}"
            )
        n_ac = self.n_autocall_obs if self.n_autocall_obs is not None else len(obs) - 1
        if not 0 <= n_ac <= len(obs) - 1:
            raise ValueError(
                f"n_autocall_obs must be in [0, {len(obs) - 1}], got {n_ac}"
            )
        if self.coupon_barrier is not None and self.coupon_barrier <= 0:
            raise ValueError(
                f"coupon_barrier must be > 0 when set, got {self.coupon_barrier}"
            )

        object.__setattr__(self, "obs_dates", obs)
        object.__setattr__(self, "pay_dates", pay)
        object.__setattr__(self, "n_autocall_obs", n_ac)

    @property
    def n_obs(self) -> int:
        return len(self.obs_dates)

    def obs_year_fractions(self, day_count: float = 365.0) -> np.ndarray:
        """Year fractions from `issue_date` to each observation date (ACT/`day_count`)."""
        return np.array(
            [(d - self.issue_date).days / day_count for d in self.obs_dates],
            dtype=float,
        )

    def pay_year_fractions(self, day_count: float = 365.0) -> np.ndarray:
        """Year fractions from `issue_date` to each payment date (ACT/`day_count`)."""
        return np.array(
            [(d - self.issue_date).days / day_count for d in self.pay_dates],
            dtype=float,
        )

    def maturity_year_fraction(self, day_count: float = 365.0) -> float:
        """Year fraction from `issue_date` to the maturity payment date."""
        return (self.pay_dates[-1] - self.issue_date).days / day_count


# --------------------------------------------------------------------------------------
# Observation-grid mapping
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationGrid:
    """Maps a product's observation/payment dates onto a simulation time grid.

    Use `ObservationGrid.from_product` to build one from an `FCNProduct` and a
    `SimulationConfig`-style (T, n_steps) grid spec — that helper handles the
    rounding to the nearest grid step.
    """

    obs_indices: np.ndarray             # (M,) int indices into the path's time axis
    pay_year_fractions: np.ndarray      # (M,) floats, used purely for discounting
    sim_T: float                        # total simulation horizon, years
    sim_n_steps: int                    # number of GBM steps from 0 to sim_T
    day_count: float                    # day-count basis used for year fractions

    @classmethod
    def from_product(
        cls,
        product: "FCNProduct",
        day_count: float = 365.0,
        sim_n_steps: Optional[int] = None,
    ) -> "ObservationGrid":
        """Build a daily-resolution simulation grid covering issue→maturity.

        If `sim_n_steps` is None (default), one GBM step per calendar day is
        used, so each observation date lands exactly on a grid point. Overriding
        `sim_n_steps` is useful for sensitivity tests but introduces rounding.
        """
        obs_days = np.array(
            [(d - product.issue_date).days for d in product.obs_dates], dtype=int
        )
        if np.any(obs_days <= 0):
            raise ValueError("obs_dates must be strictly after issue_date")
        total_days = int((product.pay_dates[-1] - product.issue_date).days)
        if total_days < int(obs_days[-1]):
            raise ValueError("maturity payment date is before final observation")
        sim_T = total_days / day_count
        if sim_n_steps is None:
            sim_n_steps = total_days
            obs_indices = obs_days.astype(int)
        else:
            obs_indices = np.rint(
                obs_days * (sim_n_steps / total_days)
            ).astype(int)
        if np.any(obs_indices < 1) or np.any(obs_indices > sim_n_steps):
            raise ValueError("rounded obs_indices fall outside the simulation grid")
        if len(np.unique(obs_indices)) != len(obs_indices):
            raise ValueError(
                "obs_indices collide on the simulation grid — increase sim_n_steps"
            )
        return cls(
            obs_indices=obs_indices,
            pay_year_fractions=product.pay_year_fractions(day_count=day_count),
            sim_T=sim_T,
            sim_n_steps=int(sim_n_steps),
            day_count=day_count,
        )


# --------------------------------------------------------------------------------------
# Payoff
# --------------------------------------------------------------------------------------


def payoff_per_path(
    paths: np.ndarray,
    spots: np.ndarray,
    product: FCNProduct,
    grid: ObservationGrid,
    rate: float,
) -> np.ndarray:
    """Present value of the FCN per simulated path.

    Parameters
    ----------
    paths : (n_paths, sim_n_steps + 1, d) array
        From `gbm_simulation.simulate_paths`. `paths[:, 0, :] == spots`.
    spots : (d,) array
        Initial fixings, in the same order as `product`'s underlyings.
    product : FCNProduct
    grid : ObservationGrid
        Must have been built from the same `product` (or be compatible).
    rate : float
        Constant continuous-compounding discount rate.

    Returns
    -------
    pv : (n_paths,) array
        Discounted total cashflow per path, expressed in the same currency
        as `notional`.
    """
    if paths.ndim != 3:
        raise ValueError(f"paths must be 3D, got shape {paths.shape}")
    n_paths, n_grid, d = paths.shape
    if n_grid != grid.sim_n_steps + 1:
        raise ValueError(
            f"paths has {n_grid} grid points; expected {grid.sim_n_steps + 1}"
        )
    if spots.shape != (d,):
        raise ValueError(f"spots shape {spots.shape} != ({d},)")

    M = product.n_obs
    n_ac = int(product.n_autocall_obs)
    N = float(product.notional)
    c = float(product.coupon_rate) * N

    obs_prices = paths[:, grid.obs_indices, :]          # (n_paths, M, d)
    perf = obs_prices / spots                            # (n_paths, M, d)
    W = perf.min(axis=2)                                 # worst-of (n_paths, M)

    df = np.exp(-rate * grid.pay_year_fractions)         # (M,)

    # ------------------------------------------------------------------
    # Autocall logic — first j < n_ac where W[:, j] >= autocall_barrier.
    # ------------------------------------------------------------------
    if n_ac > 0:
        is_ac = W[:, :n_ac] >= product.autocall_barrier  # (n_paths, n_ac)
        has_ac = is_ac.any(axis=1)
        first_ac = is_ac.argmax(axis=1)                   # 0 if !has_ac (unused)
    else:
        has_ac = np.zeros(n_paths, dtype=bool)
        first_ac = np.zeros(n_paths, dtype=int)

    last_period = np.where(has_ac, first_ac, M - 1)      # (n_paths,) index into 0..M-1
    j_idx = np.arange(M)
    paid_mask = j_idx[None, :] <= last_period[:, None]   # (n_paths, M)

    # ------------------------------------------------------------------
    # Coupon cashflows.
    # ------------------------------------------------------------------
    if product.coupon_barrier is None:
        coupon_cf = c * paid_mask.astype(float)
    else:
        coupon_cf = c * paid_mask.astype(float) * (W >= product.coupon_barrier).astype(float)

    # ------------------------------------------------------------------
    # Maturity redemption (for paths not autocalled).
    # ------------------------------------------------------------------
    W_final = W[:, -1]
    if product.continuous_ki:
        # Continuous KI looks at every simulated grid point — not just obs.
        # Cheapest is min over the path (cumulative min suffices).
        path_min_perf = (paths / spots).min(axis=2).min(axis=1)  # (n_paths,)
        ki_triggered = path_min_perf < product.strike
    else:
        ki_triggered = W_final < product.strike

    if product.geared_downside:
        downside_payoff = N * W_final / product.strike
    else:
        downside_payoff = N * W_final

    mat_redemption = np.where(ki_triggered, downside_payoff, N)

    redemption_amount = np.where(has_ac, N, mat_redemption)
    redemption_col = last_period
    redemption = np.zeros((n_paths, M))
    rows = np.arange(n_paths)
    redemption[rows, redemption_col] = redemption_amount

    cf = coupon_cf + redemption                          # (n_paths, M)
    pv = (cf * df[None, :]).sum(axis=1)                  # (n_paths,)
    return pv


# --------------------------------------------------------------------------------------
# Smoothed payoff (sigmoid replacements for the autocall / KI / coupon indicators)
# --------------------------------------------------------------------------------------


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable σ(x) = 1 / (1 + exp(-x))."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def payoff_per_path_smoothed(
    paths: np.ndarray,
    spots: np.ndarray,
    product: FCNProduct,
    grid: ObservationGrid,
    rate: float,
    smoothing_k_ac: float = 100.0,
    smoothing_k_ki: float = 100.0,
    smoothing_k_coupon: float = 100.0,
) -> np.ndarray:
    r"""Smoothed-indicator variant of :func:`payoff_per_path` for stable Greeks.

    Every hard indicator in the FCN payoff is replaced by a logistic sigmoid::

        1{W ≥ B}  →  σ(k · (W − B) / B)
        1{W < B}  →  σ(k · (B − W) / B)

    The barrier is normalised by ``B`` so the same ``k`` corresponds to roughly
    the same *relative* transition width regardless of barrier level.

    The autocall is treated as a *soft* event: at observation ``j`` the path
    autocalls with probability ``p_j = σ(k_ac · (W_j − B_ac) / B_ac)``, and the
    survival probability through obs ``j`` propagates multiplicatively across
    observation dates. Cashflows are then probability-weighted:

    * Coupon at obs ``j`` = ``c · b_j · s_{j−1}``  (``s_{j−1}`` = survival into
      obs ``j``; ``b_j`` = smoothed coupon-barrier indicator, or 1 if no coupon
      barrier is set).
    * Autocall redemption at obs ``j`` = ``N · p_j · s_{j−1}``.
    * Maturity redemption at obs ``M−1`` =
      ``[N · (1 − π_ki) + downside · π_ki] · s_{M−1}``,
      with ``π_ki`` the smoothed KI-breach probability.

    As ``smoothing_k → ∞`` the smoothed payoff converges pointwise to the hard
    payoff (this is verified in :mod:`tests.test_smoothed_payoff`). At finite
    ``k`` the *price* picks up a small O(1/k) bias, but the gradient w.r.t.
    spots/vols/correlation becomes continuous — which is what makes
    bump-and-revalue Greeks stable across the barrier regions where the hard
    payoff's discontinuities create the noise visible in notebook 05.

    Parameters
    ----------
    paths, spots, product, grid, rate :
        As in :func:`payoff_per_path`.
    smoothing_k_ac, smoothing_k_ki, smoothing_k_coupon : float
        Steepness of the sigmoid for the autocall, knock-in, and (if applicable)
        coupon-barrier indicators. Larger ``k`` → tighter transition → smaller
        price bias but more curvature, i.e. larger Γ noise away from the
        barrier; smaller ``k`` → smoother gradients but larger price bias.
        Defaults of 100 give a transition width of ~1% of barrier on each side
        and a price bias well within MC standard error at the textbook fixings.

    Returns
    -------
    pv : (n_paths,) array
        Discounted total cashflow per path (smoothed).
    """
    if paths.ndim != 3:
        raise ValueError(f"paths must be 3D, got shape {paths.shape}")
    n_paths, n_grid, d = paths.shape
    if n_grid != grid.sim_n_steps + 1:
        raise ValueError(
            f"paths has {n_grid} grid points; expected {grid.sim_n_steps + 1}"
        )
    if spots.shape != (d,):
        raise ValueError(f"spots shape {spots.shape} != ({d},)")
    if smoothing_k_ac <= 0 or smoothing_k_ki <= 0 or smoothing_k_coupon <= 0:
        raise ValueError("smoothing_k_* must be strictly positive")

    M = product.n_obs
    n_ac = int(product.n_autocall_obs)
    N = float(product.notional)
    c = float(product.coupon_rate) * N

    obs_prices = paths[:, grid.obs_indices, :]
    perf = obs_prices / spots
    W = perf.min(axis=2)                                  # (n_paths, M)
    df = np.exp(-rate * grid.pay_year_fractions)          # (M,)

    B_ac = float(product.autocall_barrier)

    # Soft autocall probabilities at every autocallable obs.
    if n_ac > 0:
        p_ac = _stable_sigmoid(smoothing_k_ac * (W[:, :n_ac] - B_ac) / B_ac)
    else:
        p_ac = np.zeros((n_paths, 0))

    # Survival into obs j: s_at_start[:, 0] = 1; s_at_start[:, j] propagates
    # via the autocallable obs and stays flat thereafter (only the last obs
    # is then the maturity check).
    s_at_start = np.ones((n_paths, M))
    for j in range(1, M):
        if j - 1 < n_ac:
            s_at_start[:, j] = s_at_start[:, j - 1] * (1.0 - p_ac[:, j - 1])
        else:
            s_at_start[:, j] = s_at_start[:, j - 1]

    # Coupon CFs: at every obs j, c · (smoothed coupon-barrier ind) · s_{j-1}.
    if product.coupon_barrier is None:
        coupon_b = np.ones((n_paths, M))
    else:
        Bc = float(product.coupon_barrier)
        coupon_b = _stable_sigmoid(smoothing_k_coupon * (W - Bc) / Bc)
    coupon_cf = c * coupon_b * s_at_start                  # (n_paths, M)

    # Autocall redemption CFs at each autocallable obs.
    redemption_cf = np.zeros((n_paths, M))
    if n_ac > 0:
        redemption_cf[:, :n_ac] = N * s_at_start[:, :n_ac] * p_ac

    # Maturity redemption at obs M-1 (smoothed KI vs no-KI mixture).
    if M - 1 >= n_ac:
        if product.continuous_ki:
            path_min_perf = (paths / spots).min(axis=2).min(axis=1)
            ki_prob = _stable_sigmoid(
                smoothing_k_ki * (product.strike - path_min_perf) / product.strike
            )
        else:
            ki_prob = _stable_sigmoid(
                smoothing_k_ki * (product.strike - W[:, -1]) / product.strike
            )
        W_final = W[:, -1]
        if product.geared_downside:
            downside_payoff = N * W_final / product.strike
        else:
            downside_payoff = N * W_final
        mat_redemption = (1.0 - ki_prob) * N + ki_prob * downside_payoff
        redemption_cf[:, -1] = redemption_cf[:, -1] + mat_redemption * s_at_start[:, -1]

    cf = coupon_cf + redemption_cf
    pv = (cf * df[None, :]).sum(axis=1)
    return pv


# --------------------------------------------------------------------------------------
# Probability decomposition (diagnostics)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityDecomposition:
    """Breakdown of how each simulated path resolves.

    All values are probabilities estimated as path fractions.
    """

    autocall_by_period: np.ndarray   # (n_autocall_obs,) prob of first autocall in period j
    p_autocall_total: float
    p_ki_at_maturity: float
    p_alive_no_ki: float

    def __str__(self) -> str:
        lines = ["Probability decomposition:"]
        for j, p in enumerate(self.autocall_by_period):
            lines.append(f"  P(first autocall = period {j+1}) = {p:.4f}")
        lines.append(f"  P(autocall total)             = {self.p_autocall_total:.4f}")
        lines.append(f"  P(KI at maturity, no AC)      = {self.p_ki_at_maturity:.4f}")
        lines.append(f"  P(alive, no KI, no AC)        = {self.p_alive_no_ki:.4f}")
        return "\n".join(lines)


def probability_decomposition(
    paths: np.ndarray,
    spots: np.ndarray,
    product: FCNProduct,
    grid: ObservationGrid,
) -> ProbabilityDecomposition:
    """Compute the autocall / KI / par-at-maturity probabilities."""
    n_paths = paths.shape[0]
    obs_prices = paths[:, grid.obs_indices, :]
    W = (obs_prices / spots).min(axis=2)
    n_ac = int(product.n_autocall_obs)

    if n_ac > 0:
        is_ac = W[:, :n_ac] >= product.autocall_barrier
        has_ac = is_ac.any(axis=1)
        first_ac = np.where(has_ac, is_ac.argmax(axis=1), -1)
        by_period = np.array(
            [(first_ac == j).mean() for j in range(n_ac)], dtype=float
        )
        p_ac_total = float(has_ac.mean())
    else:
        has_ac = np.zeros(n_paths, dtype=bool)
        by_period = np.array([], dtype=float)
        p_ac_total = 0.0

    if product.continuous_ki:
        path_min_perf = (paths / spots).min(axis=2).min(axis=1)
        ki = path_min_perf < product.strike
    else:
        ki = W[:, -1] < product.strike
    ki_no_ac = ki & ~has_ac
    alive_no_ki = ~ki & ~has_ac

    return ProbabilityDecomposition(
        autocall_by_period=by_period,
        p_autocall_total=p_ac_total,
        p_ki_at_maturity=float(ki_no_ac.mean()),
        p_alive_no_ki=float(alive_no_ki.mean()),
    )
