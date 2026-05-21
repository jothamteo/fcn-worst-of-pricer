r"""1D Crank-Nicolson PDE pricer for the single-asset FCN reduction.

Used to cross-validate the Monte Carlo engine on the single-underlying
reduction of the worst-of FCN. The 3-asset worst-of is a 3D PDE problem with
discrete-observation features — out of scope for this repo. Dropping to one
underlying removes the worst-of operator while keeping every other product
feature: autocall ladder, conditional/flat coupons, European knock-in,
physical-delivery (or cash-settled) maturity payoff.

We solve in log-spot $x = \log(S/S_0)$, where the PDE has constant
coefficients,

    ∂V/∂t + ½σ² ∂²V/∂x² + (r − q − ½σ²) ∂V/∂x − r V = 0,

discretised with central differences in x and Crank–Nicolson (θ = ½) in t.
Boundary conditions are linear-extrapolation in V(x) (i.e. $V_{xx} = 0$) at
both ends of the truncated log-spot domain. Discrete observation events
(autocall + coupon decisions at fixing dates, maturity payoff at the final
valuation) are applied between PDE sub-steps.

Note on the grid choice. METHODOLOGY.md mentions a sinh-stretched grid
concentrated near the barriers; we use a uniform log-spot grid instead
because it (i) makes the PDE coefficients constant in x (cleaner stability /
truncation analysis) and (ii) naturally puts more S-space density in the
low-S region where the strike / KI sit. Either choice gives second-order
convergence in space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import solve_banded

from .fcn_payoff import FCNProduct


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDEResult:
    """Output of `price_fcn_pde_1d` (and the standalone vanilla-call helper).

    Attributes
    ----------
    price : float
        Value at $S = S_0$ (the at-the-money interpolation from the grid).
    price_pct_of_notional : float
        `price / notional` for the FCN pricer; equal to `price` for the
        vanilla helper (no notional concept).
    n_space : int
        Final number of grid intervals in x.
    n_time : int
        Total number of CN sub-steps used across all segments.
    runtime_sec : float
        Wall-clock seconds.
    x_grid, S_grid : ndarray
        Log-spot and price grids (length `n_space + 1`).
    V_at_issue : ndarray
        Value $V(x, t=0)$ across the full grid — useful for plotting,
        finite-difference Δ/Γ in later phases.
    diagnostics : dict
        Provenance for the notebook.
    """

    price: float
    price_pct_of_notional: float
    n_space: int
    n_time: int
    runtime_sec: float
    x_grid: np.ndarray = field(repr=False)
    S_grid: np.ndarray = field(repr=False)
    V_at_issue: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"PDE price : {self.price:,.4f}  (= {100 * self.price_pct_of_notional:.4f}% of notional)",
            f"Grid      : {self.n_space} space × {self.n_time} time steps",
            f"Runtime   : {self.runtime_sec * 1000:.1f} ms",
        ]
        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core: tridiagonal solver, grid, Crank-Nicolson step
# ---------------------------------------------------------------------------


def _solve_tridiag(
    sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal linear system via scipy's banded solver.

    `sub[0]` and `sup[-1]` are ignored (no sub-diag below row 0, no super-diag
    above the last row).
    """
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def _build_log_grid(
    spot: float, x_half: float, n_space: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(x_grid, S_grid)` with `n_space + 1` equally-spaced points.

    The grid spans $[-x_{\\text{half}}, +x_{\\text{half}}]$ in $x$. If
    `n_space` is odd, we add one so $x = 0$ lies exactly on the grid (the
    natural lookup point for at-the-money pricing).
    """
    if n_space % 2 == 1:
        n_space += 1
    x_grid = np.linspace(-x_half, x_half, n_space + 1)
    # Snap the midpoint to exactly 0 for numerical cleanliness.
    mid = (n_space + 1) // 2
    x_grid = x_grid - x_grid[mid]
    S_grid = spot * np.exp(x_grid)
    return x_grid, S_grid


def _cn_step(
    V_in: np.ndarray,
    dt: float,
    dx: float,
    vol: float,
    rate: float,
    div: float,
) -> np.ndarray:
    """One Crank-Nicolson backward step on a uniform log-spot grid.

    Boundary handling: linear extrapolation $V_{xx} = 0$ at both ends. This
    is implemented by substituting $V_{\\text{new}}[0] = 2 V_{\\text{new}}[1] -
    V_{\\text{new}}[2]$ into the row of the tridiagonal system that would
    otherwise reference $V_{\\text{new}}[0]$, and mirroring at the right end.

    Parameters
    ----------
    V_in : (n_pts,) ndarray
        Values at the later time.
    dt : float
        Positive step size (we are walking backward; `dt > 0`).
    dx, vol, rate, div :
        Grid spacing and PDE coefficients.
    """
    mu = rate - div - 0.5 * vol * vol
    sigma2 = vol * vol
    a = 0.5 * sigma2 / (dx * dx)
    b = mu / (2.0 * dx)
    c0 = -rate

    # Stencil for L V = a V_xx + b V_x + c0 V using central differences.
    L_sub = a - b      # coefficient of V_{i-1}
    L_dia = -2.0 * a + c0
    L_sup = a + b      # coefficient of V_{i+1}

    theta = 0.5
    n_pts = len(V_in)
    n_int = n_pts - 2

    # Interior system rows for V_new[1 .. n_pts-2].
    sub = np.full(n_int, -theta * dt * L_sub)
    dia = np.full(n_int, 1.0 - theta * dt * L_dia)
    sup = np.full(n_int, -theta * dt * L_sup)

    rhs = (
        ((1.0 - theta) * dt * L_sub) * V_in[:-2]
        + (1.0 + (1.0 - theta) * dt * L_dia) * V_in[1:-1]
        + ((1.0 - theta) * dt * L_sup) * V_in[2:]
    )

    # Left BC: V_new[0] = 2 V_new[1] − V_new[2]. The row 0 of the interior
    # system would multiply V_new[0] by `sub[0]`; substituting gives:
    #   sub[0] · V_new[0] = 2 sub[0] · V_new[1] − sub[0] · V_new[2]
    # which folds into the existing diagonal and super-diagonal entries.
    dia[0] += 2.0 * sub[0]
    sup[0] -= sub[0]
    sub[0] = 0.0

    # Right BC: V_new[-1] = 2 V_new[-2] − V_new[-3]. Symmetric.
    dia[-1] += 2.0 * sup[-1]
    sub[-1] -= sup[-1]
    sup[-1] = 0.0

    V_int_new = _solve_tridiag(sub, dia, sup, rhs)
    V_out = np.empty(n_pts)
    V_out[1:-1] = V_int_new
    V_out[0] = 2.0 * V_out[1] - V_out[2]
    V_out[-1] = 2.0 * V_out[-2] - V_out[-3]
    return V_out


def _cn_evolve(
    V: np.ndarray,
    t_end: float,
    t_start: float,
    n_steps: int,
    dx: float,
    vol: float,
    rate: float,
    div: float,
) -> np.ndarray:
    """Evolve V backward from `t_end` to `t_start` with `n_steps` CN steps."""
    if t_end < t_start - 1e-15:
        raise ValueError(
            f"t_end ({t_end}) must be ≥ t_start ({t_start}) for a backward walk."
        )
    if t_end - t_start <= 0.0:
        return V
    n_steps = max(1, int(n_steps))
    dt = (t_end - t_start) / n_steps
    for _ in range(n_steps):
        V = _cn_step(V, dt, dx, vol, rate, div)
    return V


# ---------------------------------------------------------------------------
# Vanilla European pricer — used to calibrate the CN engine vs Black-Scholes.
# ---------------------------------------------------------------------------


def price_european_call_pde(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    div: float,
    T: float,
    n_space: int = 800,
    n_time: int = 400,
    x_range_sigma: float = 6.0,
) -> PDEResult:
    """Standalone Crank-Nicolson pricer for a vanilla European call.

    No event handling, no notional, no FCN logic — just terminal payoff
    `max(S - K, 0)` and one continuous PDE sweep back to `t = 0`. Used in
    `tests/test_pde_pricer.py` to validate the CN machinery against the
    Black-Scholes closed form.
    """
    t0 = time.perf_counter()
    sigma_T = vol * np.sqrt(T)
    x_grid, S_grid = _build_log_grid(spot, x_range_sigma * sigma_T, n_space)
    n_space_actual = len(x_grid) - 1
    dx = x_grid[1] - x_grid[0]

    V = np.maximum(S_grid - strike, 0.0)
    V = _cn_evolve(V, T, 0.0, n_time, dx, vol, rate, div)
    price = float(np.interp(0.0, x_grid, V))

    return PDEResult(
        price=price,
        price_pct_of_notional=price,
        n_space=n_space_actual,
        n_time=n_time,
        runtime_sec=time.perf_counter() - t0,
        x_grid=x_grid,
        S_grid=S_grid,
        V_at_issue=V,
        diagnostics={
            "product": "vanilla_call",
            "strike": strike,
            "T": T,
            "vol": vol,
            "rate": rate,
            "div": div,
            "x_range_sigma": x_range_sigma,
        },
    )


# ---------------------------------------------------------------------------
# FCN pricer
# ---------------------------------------------------------------------------


def price_fcn_pde_1d(
    spot: float,
    vol: float,
    div: float,
    rate: float,
    product: FCNProduct,
    n_space: int = 800,
    n_time_per_period: int = 60,
    x_range_sigma: float = 6.0,
    day_count: float = 365.0,
) -> PDEResult:
    """Price the single-asset reduction of `product` by Crank-Nicolson.

    The "single-asset reduction" keeps `product`'s coupon schedule, barriers,
    settlement method (physical delivery vs cash), and KI mode (European only — `continuous_ki=True`
    is not supported here) but applies them to one underlying instead of the
    worst-of operator. So `W(t) = S(t)/S_0`.

    Parameters
    ----------
    spot : float
        Initial spot $S_0$.
    vol : float
        Flat Black-Scholes volatility (annualised).
    div : float
        Continuous dividend yield $q$.
    rate : float
        Continuous risk-free rate $r$.
    product : FCNProduct
        Product spec; the underlyings list is ignored.
    n_space : int
        Number of grid intervals in $x = \\log(S/S_0)$. Rounded up to the
        next even number so $x = 0$ lies on the grid.
    n_time_per_period : int
        Number of CN sub-steps between consecutive observation events.
    x_range_sigma : float
        Grid half-width in standard deviations of $\\log(S_T/S_0)$.
        Default 6.0 — already deep into the tails for typical equity vols.
    day_count : float
        ACT/`day_count` basis for year fractions.

    Returns
    -------
    PDEResult
    """
    if product.continuous_ki:
        raise NotImplementedError(
            "PDE engine supports European knock-in only — set `continuous_ki=False` "
            "on the product spec, or stick with the Monte Carlo pricer for continuous KI."
        )

    t0 = time.perf_counter()

    obs_yf = product.obs_year_fractions(day_count=day_count)
    pay_yf = product.pay_year_fractions(day_count=day_count)
    n_ac = int(product.n_autocall_obs)
    N = float(product.notional)
    c = float(product.coupon_rate) * N
    cb = product.coupon_barrier

    sigma_T = vol * np.sqrt(pay_yf[-1])
    x_grid, S_grid = _build_log_grid(spot, x_range_sigma * sigma_T, n_space)
    n_space_actual = len(x_grid) - 1
    dx = x_grid[1] - x_grid[0]
    perf = S_grid / spot

    # ---- Terminal condition at t = obs_yf[-1] (final valuation) ----
    # Cashflow at maturity pay-date is determined by S(obs_yf[-1]); we set the
    # PDE terminal at obs_yf[-1] equal to that cashflow discounted from the
    # pay-date back to the obs-date (a deterministic discount, since the
    # cashflow is locked in by the obs-date fixing).
    if product.physical_delivery:
        redemption = np.where(perf >= product.strike, N, N * perf / product.strike)
    else:
        redemption = np.where(perf >= product.strike, N, N * perf)
    if cb is None:
        final_coupon = c * np.ones_like(perf)
    else:
        final_coupon = c * (perf >= cb).astype(float)
    df_pay_to_obs_M = float(np.exp(-rate * (pay_yf[-1] - obs_yf[-1])))
    V = (redemption + final_coupon) * df_pay_to_obs_M

    # ---- Backward time integration with event processing ----
    # Boundaries (descending in t): obs[-1] → obs[n_ac-1] → ... → obs[0] → 0.
    autocall_times = obs_yf[:n_ac].tolist()
    breakpoints = [obs_yf[-1]] + list(reversed(autocall_times)) + [0.0]

    n_time_total = 0
    for k in range(len(breakpoints) - 1):
        t_high = float(breakpoints[k])
        t_low = float(breakpoints[k + 1])
        if t_high - t_low <= 0.0:
            continue

        V = _cn_evolve(
            V, t_end=t_high, t_start=t_low,
            n_steps=n_time_per_period, dx=dx, vol=vol, rate=rate, div=div,
        )
        n_time_total += n_time_per_period

        # If the next boundary is an autocall obs date, apply the event.
        if k < len(breakpoints) - 2:
            j_match = int(np.argmin(np.abs(obs_yf[:n_ac] - t_low)))
            pay_t = float(pay_yf[j_match])
            df_pay_to_obs = float(np.exp(-rate * (pay_t - t_low)))

            # Inside autocall region: V(S, obs[j]^-) = (N + coupon_j) · df.
            # The coupon is paid on autocall whenever its own barrier passes
            # (or always when `cb is None`).
            if cb is None:
                V_ac = np.full_like(perf, (N + c) * df_pay_to_obs)
            else:
                V_ac = (N + c * (perf >= cb).astype(float)) * df_pay_to_obs

            # Outside autocall region: keep the continuation value, add coupon.
            if cb is None:
                V_cpn_add = np.full_like(perf, c * df_pay_to_obs)
            else:
                V_cpn_add = c * df_pay_to_obs * (perf >= cb).astype(float)

            in_ac = perf >= product.autocall_barrier
            V = np.where(in_ac, V_ac, V + V_cpn_add)

    price = float(np.interp(0.0, x_grid, V))
    return PDEResult(
        price=price,
        price_pct_of_notional=price / N,
        n_space=n_space_actual,
        n_time=n_time_total,
        runtime_sec=time.perf_counter() - t0,
        x_grid=x_grid,
        S_grid=S_grid,
        V_at_issue=V,
        diagnostics={
            "spot": spot,
            "vol": vol,
            "div": div,
            "rate": rate,
            "x_range_sigma": x_range_sigma,
            "n_time_per_period": n_time_per_period,
            "physical_delivery": product.physical_delivery,
            "coupon_barrier": product.coupon_barrier,
            "autocall_barrier": product.autocall_barrier,
            "strike": product.strike,
            "n_autocall_obs": n_ac,
        },
    )
