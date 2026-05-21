"""Monte Carlo pricer for the worst-of FCN.

Composes `MarketData` (from `market_data.py`), an `FCNProduct` (from
`fcn_payoff.py`), and the GBM engine (`gbm_simulation.py`) into a single
`price_fcn(...)` call. Returns a `PricingResult` with the discounted MC mean,
its standard error, and a probability decomposition over how paths resolved.

Variance reduction:
* **Antithetic variates** — always on by default; doubles the effective path
  count for free.
* **Worst-of European put control variate** — opt-in via
  `control_variate='worst_of_put'`. The control variate is a European put
  on the worst-of basket struck at the FCN's KI level. Because the FCN's
  downside risk comes from exactly this event (KI breached at maturity),
  the CV is strongly correlated with the FCN PV in the loss tail — which
  is where the FCN's variance lives. See METHODOLOGY §2.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fcn_payoff import (
    FCNProduct,
    ObservationGrid,
    ProbabilityDecomposition,
    payoff_per_path,
    probability_decomposition,
)
from .gbm_simulation import SimulationConfig, simulate_paths
from .market_data import MarketData


@dataclass(frozen=True)
class PricingResult:
    """Output of `price_fcn`.

    Attributes
    ----------
    price : float
        MC estimate of present value, in the same currency as `product.notional`.
    standard_error : float
        Sample standard error of the MC estimate (sample_std / sqrt(n_paths_total)).
    price_pct_of_notional : float
        `price / notional` — the natural way to quote an FCN to a desk.
    n_paths_total : int
        Total number of MC paths used (= n_paths × 2 if antithetic, else n_paths).
    probability : ProbabilityDecomposition
        How paths resolved (autocall-by-period, KI, par-at-maturity).
    pv_samples : np.ndarray, optional
        Per-path PVs. Kept around so the notebook can plot the distribution
        and compute control-variate diagnostics in later phases.
    diagnostics : dict
        Free-form provenance for the notebook to print.
    """

    price: float
    standard_error: float
    price_pct_of_notional: float
    n_paths_total: int
    probability: ProbabilityDecomposition
    pv_samples: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict)
    cv_diagnostics: Optional[dict] = None

    def summary(self) -> str:
        lines = [
            f"FCN price : {self.price:,.4f}  (= {100 * self.price_pct_of_notional:.4f}% of notional)",
            f"MC SE     : {self.standard_error:,.4f}  ({100 * self.standard_error / max(abs(self.price), 1e-12):.3f}% of price)",
            f"Paths     : {self.n_paths_total:,}",
            "",
            str(self.probability),
        ]
        if self.cv_diagnostics is not None:
            cv = self.cv_diagnostics
            lines.append("")
            lines.append("Control variate (worst-of European put):")
            lines.append(
                f"  β̂           : {cv['beta_hat']:,.4f}   "
                f"(corr(X, Y) = {cv['rho_XY']:+.4f})"
            )
            lines.append(
                f"  Ê[Y]        : {cv['EY_hat']:,.4f} ± {cv['EY_se']:.4f}  "
                f"(pre-pass n = {cv['n_paths_for_ey']:,})"
            )
            lines.append(
                f"  Price w/o CV: {cv['price_no_cv']:,.4f} ± {cv['se_no_cv']:.4f}"
            )
            lines.append(
                f"  Price w/  CV: {cv['price_cv']:,.4f} ± {cv['se_cv']:.4f}"
            )
            lines.append(
                f"  Var(X) / Var(X_cv): {cv['var_reduction_ratio']:.2f}× "
                f"  (SE ratio: {np.sqrt(cv['var_reduction_ratio']):.2f}×)"
            )
        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Worst-of European put — used as the control variate
# --------------------------------------------------------------------------------------


def _worst_of_european_put_pv(
    paths: np.ndarray,
    spots: np.ndarray,
    strike_perf: float,
    rate: float,
    notional: float,
    T_pay: float,
) -> np.ndarray:
    r"""Per-path discounted payoff of a worst-of European put.

    Payoff: $N \cdot \max(K - W(T), 0)$ where $W(T) = \min_i S_i(T)/S_i(0)$
    and $K$ is the put strike *in performance units* (e.g. 0.70). Notional
    matches the FCN's notional so the CV scales naturally. Discounted from
    the payment date `T_pay` back to issue.
    """
    W_T = (paths[:, -1, :] / spots).min(axis=1)         # (n_paths,)
    intrinsic = np.maximum(strike_perf - W_T, 0.0)
    return notional * intrinsic * float(np.exp(-rate * T_pay))


def _worst_of_european_put_expectation(
    market: MarketData,
    product: FCNProduct,
    n_paths: int,
    antithetic: bool,
    seed: int,
    day_count: float,
) -> tuple[float, float]:
    """Pre-pass MC estimate of $\\mathbb{E}[Y]$ for the CV.

    Returns `(mean, std_error)`. Run with a separate seed (and typically more
    paths) so the result is independent of the main pricing sample.
    """
    grid = ObservationGrid.from_product(product=product, day_count=day_count)
    cfg = SimulationConfig(
        spots=market.spots, vols=market.vols, divs=market.divs, rate=market.rate,
        corr=market.corr, T=grid.sim_T, n_steps=grid.sim_n_steps, n_paths=n_paths,
        antithetic=antithetic, seed=seed,
    )
    paths = simulate_paths(cfg)
    y = _worst_of_european_put_pv(
        paths=paths, spots=cfg.spots, strike_perf=product.strike,
        rate=market.rate, notional=product.notional,
        T_pay=float(grid.pay_year_fractions[-1]),
    )
    n_total = y.shape[0]
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(n_total))


def price_fcn(
    market: MarketData,
    product: FCNProduct,
    n_paths: int = 50_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
    sim_n_steps: Optional[int] = None,
    day_count: float = 365.0,
    control_variate: Optional[str] = None,
    cv_n_paths_for_ey: int = 200_000,
    cv_seed_offset: int = 7919,
) -> PricingResult:
    """Price a worst-of FCN by Monte Carlo on correlated GBM.

    Parameters
    ----------
    market : MarketData
        Spots, vols, divs, rate, correlation matrix. Ticker order in `market`
        is the order used through the simulation.
    product : FCNProduct
        Product spec. `product.obs_dates` and `product.pay_dates` are mapped
        onto a daily simulation grid by default (see `ObservationGrid`).
    n_paths : int
        Number of independent path draws. With `antithetic=True` the effective
        path count is `2 * n_paths`.
    antithetic : bool
        Toggle antithetic variates. Free variance reduction; on by default.
    seed : int, optional
        PRNG seed. None = OS entropy.
    sim_n_steps : int, optional
        Override the number of GBM steps. Defaults to one step per calendar
        day so every observation date lands exactly on a grid point.
    day_count : float
        ACT/`day_count` day-count basis for year fractions. Default 365.
    control_variate : {'worst_of_put', None}, optional
        If `'worst_of_put'`, the pricer also computes the worst-of
        European put struck at `product.strike` (the KI level) on each
        path, estimates β̂ = Cov(X,Y)/Var(Y) in-sample, and reports the
        CV-adjusted price + variance reduction ratio in
        `result.cv_diagnostics`. $\\mathbb{E}[Y]$ is estimated by an
        independent pre-pass with `cv_n_paths_for_ey` paths.
    cv_n_paths_for_ey : int
        Path count for the $\\mathbb{E}[Y]$ pre-pass (ignored unless
        `control_variate` is set). Larger = less Monte Carlo error in
        $\\mathbb{E}[Y]$ → cleaner CV correction.
    cv_seed_offset : int
        Added to `seed` (when set) for the pre-pass to keep it independent
        of the main pricing sample.

    Returns
    -------
    PricingResult
    """
    grid = ObservationGrid.from_product(
        product=product, day_count=day_count, sim_n_steps=sim_n_steps
    )

    cfg = SimulationConfig(
        spots=market.spots,
        vols=market.vols,
        divs=market.divs,
        rate=market.rate,
        corr=market.corr,
        T=grid.sim_T,
        n_steps=grid.sim_n_steps,
        n_paths=n_paths,
        antithetic=antithetic,
        seed=seed,
    )
    paths = simulate_paths(cfg)

    pv = payoff_per_path(
        paths=paths, spots=cfg.spots, product=product, grid=grid, rate=market.rate
    )
    prob = probability_decomposition(
        paths=paths, spots=cfg.spots, product=product, grid=grid
    )

    n_total = pv.shape[0]
    price = float(pv.mean())
    se = float(pv.std(ddof=1) / np.sqrt(n_total))

    diagnostics = {
        "tickers": list(market.tickers),
        "as_of": market.sources.get("as_of", "live"),
        "rate": f"{market.rate:.5f} ({market.sources.get('rate', '?')})",
        "vols": dict(zip(market.tickers, np.round(market.vols, 4).tolist())),
        "divs": dict(zip(market.tickers, np.round(market.divs, 5).tolist())),
        "spots": dict(zip(market.tickers, np.round(market.spots, 4).tolist())),
        "sim_n_steps": grid.sim_n_steps,
        "sim_T_years": round(grid.sim_T, 6),
        "antithetic": antithetic,
        "n_paths_drawn": n_paths,
        "seed": seed,
    }

    cv_diag: Optional[dict] = None
    if control_variate is not None:
        if control_variate != "worst_of_put":
            raise ValueError(
                f"control_variate must be 'worst_of_put' or None, got {control_variate!r}"
            )
        # --- Same-sample worst-of put PVs (correlated with FCN PVs by construction) ---
        y_main = _worst_of_european_put_pv(
            paths=paths, spots=cfg.spots, strike_perf=product.strike,
            rate=market.rate, notional=product.notional,
            T_pay=float(grid.pay_year_fractions[-1]),
        )
        # Independent pre-pass for E[Y]. We choose a separate seed so the
        # pre-pass and the main sample are truly independent — otherwise the
        # CV correction would have zero expectation by construction.
        pre_seed = (seed + cv_seed_offset) if seed is not None else cv_seed_offset
        ey_hat, ey_se = _worst_of_european_put_expectation(
            market=market, product=product,
            n_paths=cv_n_paths_for_ey, antithetic=antithetic,
            seed=pre_seed, day_count=day_count,
        )
        var_y = float(y_main.var(ddof=1))
        cov_xy = float(np.cov(pv, y_main, ddof=1)[0, 1])
        beta_hat = cov_xy / var_y if var_y > 0.0 else 0.0
        rho_xy = cov_xy / np.sqrt(max(var_y, 1e-30) * max(float(pv.var(ddof=1)), 1e-30))

        pv_cv = pv - beta_hat * (y_main - ey_hat)
        price_cv = float(pv_cv.mean())
        se_cv = float(pv_cv.std(ddof=1) / np.sqrt(n_total))
        var_reduction_ratio = (
            float(pv.var(ddof=1)) / max(float(pv_cv.var(ddof=1)), 1e-30)
        )

        cv_diag = {
            "kind": "worst_of_put",
            "strike_perf": product.strike,
            "beta_hat": beta_hat,
            "rho_XY": rho_xy,
            "EY_hat": ey_hat,
            "EY_se": ey_se,
            "n_paths_for_ey": cv_n_paths_for_ey * (2 if antithetic else 1),
            "price_no_cv": price,
            "se_no_cv": se,
            "price_cv": price_cv,
            "se_cv": se_cv,
            "var_reduction_ratio": var_reduction_ratio,
        }
        # Overwrite the headline price/SE with the CV-adjusted values when CV is on.
        price = price_cv
        se = se_cv

    return PricingResult(
        price=price,
        standard_error=se,
        price_pct_of_notional=price / product.notional,
        n_paths_total=n_total,
        probability=prob,
        pv_samples=pv,
        diagnostics=diagnostics,
        cv_diagnostics=cv_diag,
    )


# --------------------------------------------------------------------------------------
# Ex-post realised payoff
# --------------------------------------------------------------------------------------


def realised_payoff(
    history: "object",  # pd.DataFrame to avoid the import dance — typed below
    product: FCNProduct,
    rate: Optional[float] = None,
) -> dict:
    """Replay the FCN payoff on actual historical prices.

    Walks `history` (a wide DataFrame indexed by date, one column per ticker
    in the order of `product`'s underlyings) over the product's observation
    dates and computes the realised cashflow schedule. If `rate` is supplied,
    cashflows are discounted back to `issue_date` as well.

    Returns
    -------
    dict with keys:
      `cashflows`    : list of (pay_date, label, amount) tuples
      `total_paid`   : sum of all cashflows
      `pv_at_issue` : present value at `issue_date` (None if rate not supplied)
      `outcome`      : 'autocalled_period_<j>' or 'matured_par' or 'matured_ki'
      `worst_path`  : list of (obs_date, worst_perf) tuples for inspection
    """
    import pandas as pd

    if not isinstance(history, pd.DataFrame):
        raise TypeError(f"history must be a pandas DataFrame, got {type(history)}")

    cols = list(history.columns)
    needed_dates = [product.issue_date] + list(product.obs_dates)
    closes_per_obs = []
    for d in needed_dates:
        ts = pd.Timestamp(d)
        if history.index.tz is not None:
            ts = ts.tz_localize(history.index.tz)
        # Last available close on or before `d` — handles weekends / market closures.
        slice_ = history.loc[history.index <= ts]
        if slice_.empty:
            raise ValueError(f"no historical data on or before {d}")
        closes_per_obs.append(slice_.iloc[-1])

    initial = closes_per_obs[0].to_numpy(dtype=float)
    obs_closes = np.array([row.to_numpy(dtype=float) for row in closes_per_obs[1:]])
    perf = obs_closes / initial                              # (M, d)
    W = perf.min(axis=1)                                     # (M,)
    n_ac = int(product.n_autocall_obs)

    N = float(product.notional)
    c = float(product.coupon_rate) * N

    cashflows: list[tuple] = []
    worst_path = [(product.obs_dates[j], float(W[j])) for j in range(product.n_obs)]
    outcome: str
    autocalled_at: Optional[int] = None
    for j in range(n_ac):
        if (product.coupon_barrier is None) or (W[j] >= product.coupon_barrier):
            cashflows.append((product.pay_dates[j], f"coupon_{j+1}", c))
        else:
            cashflows.append((product.pay_dates[j], f"coupon_{j+1}_skipped", 0.0))
        if W[j] >= product.autocall_barrier:
            cashflows.append((product.pay_dates[j], f"autocall_redemption_{j+1}", N))
            autocalled_at = j
            outcome = f"autocalled_period_{j+1}"
            break

    if autocalled_at is None:
        # Walk remaining periods (the final-valuation tail) with no autocall.
        for j in range(n_ac, product.n_obs - 1):
            if (product.coupon_barrier is None) or (W[j] >= product.coupon_barrier):
                cashflows.append((product.pay_dates[j], f"coupon_{j+1}", c))
            else:
                cashflows.append((product.pay_dates[j], f"coupon_{j+1}_skipped", 0.0))

        # Maturity (j = n_obs - 1)
        j = product.n_obs - 1
        W_final = float(W[j])
        if (product.coupon_barrier is None) or (W_final >= product.coupon_barrier):
            cashflows.append((product.pay_dates[j], f"coupon_{j+1}_final", c))
        else:
            cashflows.append((product.pay_dates[j], f"coupon_{j+1}_final_skipped", 0.0))

        if W_final >= product.strike:
            cashflows.append((product.pay_dates[j], "maturity_redemption_par", N))
            outcome = "matured_par"
        else:
            if product.physical_delivery:
                redemption = N * W_final / product.strike
            else:
                redemption = N * W_final
            cashflows.append((product.pay_dates[j], "maturity_redemption_ki", redemption))
            outcome = "matured_ki"

    total = float(sum(cf[2] for cf in cashflows))

    pv: Optional[float] = None
    if rate is not None:
        pv = 0.0
        for pay_dt, _, amt in cashflows:
            yf_ = (pay_dt - product.issue_date).days / 365.0
            pv += amt * float(np.exp(-rate * yf_))

    return {
        "cashflows": cashflows,
        "total_paid": total,
        "pv_at_issue": pv,
        "outcome": outcome,
        "worst_path": worst_path,
    }
