"""Monte Carlo pricer for the worst-of FCN.

Composes `MarketData` (from `market_data.py`), an `FCNProduct` (from
`fcn_payoff.py`), and the GBM engine (`gbm_simulation.py`) into a single
`price_fcn(...)` call. Returns a `PricingResult` with the discounted MC mean,
its standard error, and a probability decomposition over how paths resolved.
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

    def summary(self) -> str:
        lines = [
            f"FCN price : {self.price:,.4f}  (= {100 * self.price_pct_of_notional:.4f}% of notional)",
            f"MC SE     : {self.standard_error:,.4f}  ({100 * self.standard_error / max(abs(self.price), 1e-12):.3f}% of price)",
            f"Paths     : {self.n_paths_total:,}",
            "",
            str(self.probability),
        ]
        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for k, v in self.diagnostics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def price_fcn(
    market: MarketData,
    product: FCNProduct,
    n_paths: int = 50_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
    sim_n_steps: Optional[int] = None,
    day_count: float = 365.0,
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

    return PricingResult(
        price=price,
        standard_error=se,
        price_pct_of_notional=price / product.notional,
        n_paths_total=n_total,
        probability=prob,
        pv_samples=pv,
        diagnostics=diagnostics,
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
            if product.geared_downside:
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
