"""Assemble notebooks/03_fcn_pricing.ipynb.

Kept as a script (rather than hand-editing JSON) so the notebook layout is
reproducible. Re-run with `python notebooks/_build_03.py` after edits.
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

NB = nbf.v4.new_notebook()


def md(text: str) -> None:
    NB.cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    NB.cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------

md(
    """# 03 — Worst-of FCN Pricing on AMZN / META / MU (Oct 2025 → May 2026)

This notebook prices a **6-month, monthly-observed worst-of FCN** on AMZN, META, MU
issued at **17 Oct 2025**, then replays the *actual* payoff on the realised price
path through maturity at **4 May 2026** to compare model vs. reality.

**Product spec (JT-supplied)**

| Field | Value |
|---|---|
| Underlyings | AMZN, META, MU |
| Notional | USD 50,000 |
| Issue date | 31 Oct 2025 |
| Initial valuation | 17 Oct 2025 |
| Final valuation | 30 Apr 2026 |
| Maturity | 4 May 2026 |
| Coupon | 1.535% per period, **flat** (18.42% p.a., monthly) |
| Autocall barrier | 100% of initial, worst-of basis |
| Strike (= KI) | 70% of initial, **physical delivery** at strike (shares of worst-performer) |
| KI observation | European (at final valuation only) |
| Autocall fixings | 1 Dec '25, 31 Dec '25, 2 Feb '26, 2 Mar '26, 31 Mar '26 |
| Payment dates | 3 Dec '25, 5 Jan '26, 4 Feb '26, 4 Mar '26, 2 Apr '26, 4 May '26 |

**What "use historical prices" means here**: market inputs are taken **as-of 17 Oct 2025**
(no look-ahead — realised vol/corr from a 5Y window ending 17 Oct 2025, Treasury yields
from that date, spots = 17 Oct 2025 close). The note has already matured, so once we
have a model price we also pull the actual prices through 4 May 2026 and replay the
realised cashflow schedule.
"""
)

code(
    """import sys, os
# Ensure the repo root is on sys.path regardless of where Jupyter was launched.
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.market_data import load_market_data, _download_history
from src.fcn_payoff import FCNProduct, ObservationGrid, probability_decomposition
from src.mc_pricer import price_fcn, realised_payoff

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
np.set_printoptions(suppress=True, precision=4)
"""
)

# ---------------------------------------------------------------------------
md(
    r"""### Settlement method: physical delivery vs cash settlement

This FCN settles the downside (KI-triggered) leg by **physical delivery**: if
the worst-performing underlying finishes below the strike at maturity, the
issuer delivers approximately $N / (K \cdot S_{\text{worst}}(0))$ shares of
that underlying to the client, with fractional shares settled in cash. The
client ends up holding a concentrated long position in the worst-performer at
the strike-level purchase price.

The pricer values this at the cash-equivalent fair value $N \cdot W(T) / K$ —
the fractional-share rounding residual is negligible relative to MC standard
error, so the payoff is reported as a continuous function of $W(T)$ rather
than an integer share count. Break-even is exactly at $W(T) = K$, so the
payoff is *continuous* at the strike (no cliff).

The library also supports **cash settlement** (`physical_delivery=False`),
where the downside leg is $N \cdot W(T)$ — discontinuous at the strike and
*harsher* than physical delivery by a factor of $1/K$ in the KI region. Cash
settlement is non-standard for retail FCNs in Asia; the JT-spec'd structure
here is physical delivery.

Neither is "geared" in the structured-products sense — gearing implies
amplified, super-linear losses, which neither mechanism has.
"""
)

# ---------------------------------------------------------------------------
md(
    """## 1. Market data as-of 17 Oct 2025

We restrict yfinance to a 5Y window ending 17 Oct 2025, compute realised vol/corr on
that window, take the close on 17 Oct 2025 as $S_0$, and pull the Treasury curve on
that date interpolated to the 0.5Y tenor (matching the note's tenor)."""
)
code(
    """ISSUE_DATE = date(2025, 10, 17)
TICKERS = ("AMZN", "META", "MU")

market = load_market_data(
    tickers=TICKERS,
    lookback_years=5,
    target_T=0.5,            # 6M Treasury for a 6M note
    as_of=ISSUE_DATE,
)
print(market.summary())
"""
)

# ---------------------------------------------------------------------------
md(
    """## 2. Product spec — the JT-supplied parameters

`FCNProduct` validates the date schedule and the barrier ordering, and exposes
year fractions used by the simulation grid."""
)
code(
    """product = FCNProduct(
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
    issue_date=ISSUE_DATE,
    autocall_barrier=1.00,
    strike=0.70,
    n_autocall_obs=5,
    coupon_barrier=None,        # flat coupons, no barrier
    physical_delivery=True,     # shares of worst-performer at strike; cash-equivalent = N * W(T) / strike
    continuous_ki=False,        # KI checked at final valuation only
)

print('Obs year fractions :', np.round(product.obs_year_fractions(), 4))
print('Pay year fractions :', np.round(product.pay_year_fractions(), 4))
print('Sim grid (daily)   :', ObservationGrid.from_product(product).obs_indices)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 3. Monte Carlo price

Daily simulation grid (199 steps = calendar days from 17 Oct 2025 to 4 May 2026),
antithetic variates on, 50,000 primal paths = 100,000 total."""
)
code(
    """result = price_fcn(
    market=market,
    product=product,
    n_paths=50_000,
    antithetic=True,
    seed=20260511,
)
print(result.summary())
"""
)

md(
    """### Interpretation

For an FCN to be issued at par (100% of notional), the model price should be **at or
above par** — anything below means the structurer is underpaying for the embedded
short worst-of put / long autocall. Read the % of notional and the probability
breakdown together: a high P(autocall in period 1) implies the issuer expects
the note to die fast at par + one coupon, so the model price hugs par from above."""
)

# ---------------------------------------------------------------------------
md(
    """## 4. Distribution of present values

Per-path discounted PVs from the MC. The vertical line is the MC mean."""
)
code(
    """fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(result.pv_samples, bins=80, color='steelblue', alpha=0.85)
ax.axvline(result.price, color='black', lw=1.5, label=f'MC mean = {result.price:,.0f}')
ax.axvline(product.notional, color='crimson', lw=1.0, ls='--', label=f'Notional = {product.notional:,.0f}')
ax.set_xlabel('PV per path (USD)')
ax.set_ylabel('Count')
ax.set_title('FCN PV distribution — 100k antithetic paths')
ax.legend()
fig.tight_layout(); plt.show()
"""
)

# ---------------------------------------------------------------------------
md(
    """## 5. Worst-of trajectory chart vs barriers

Twenty randomly-chosen paths of the worst-of normalised price, against the
autocall (100%) and strike (70%) lines and the six observation dates."""
)
code(
    """from src.gbm_simulation import SimulationConfig, simulate_paths

grid = ObservationGrid.from_product(product)
cfg = SimulationConfig(
    spots=market.spots, vols=market.vols, divs=market.divs, rate=market.rate, corr=market.corr,
    T=grid.sim_T, n_steps=grid.sim_n_steps, n_paths=500, antithetic=False, seed=42,
)
paths = simulate_paths(cfg)
worst = (paths / market.spots).min(axis=2)  # (n_paths, n_steps+1)

t_grid = np.linspace(0, grid.sim_T, grid.sim_n_steps + 1)
obs_yf = product.obs_year_fractions()

fig, ax = plt.subplots(figsize=(11, 5))
sample = np.random.default_rng(0).choice(paths.shape[0], size=20, replace=False)
for i in sample:
    ax.plot(t_grid, worst[i], color='steelblue', alpha=0.35, lw=0.9)
ax.axhline(product.autocall_barrier, color='black', lw=1.0, ls='--', label='Autocall = 100%')
ax.axhline(product.strike, color='crimson', lw=1.0, ls='--', label='Strike = 70%')
for j, t in enumerate(obs_yf):
    label = 'Obs date' if j == 0 else None
    ax.axvline(t, color='grey', alpha=0.45, lw=0.6, label=label)
ax.set_xlabel('Years from 17 Oct 2025')
ax.set_ylabel('Worst-of perf (normalised)')
ax.set_title('20 simulated worst-of paths vs FCN barriers')
ax.legend(loc='upper left')
fig.tight_layout(); plt.show()
"""
)

# ---------------------------------------------------------------------------
md(
    """## 6. Ex-post replay — what actually happened?

The note matured on 4 May 2026 (before today). We now pull the *actual* AMZN /
META / MU closes over the life of the note and run the deterministic payoff."""
)
code(
    """post_history = _download_history(
    TICKERS,
    lookback_years=1,                      # 1Y window comfortably covers Oct 25 -> May 26
    end=date(2026, 5, 5),
)
# Restrict to the life of the note for display.
life_mask = (post_history.index >= pd.Timestamp(ISSUE_DATE)) & (post_history.index <= pd.Timestamp(2026, 5, 4))
life = post_history.loc[life_mask, list(TICKERS)]
display(life.head())
display(life.tail())
"""
)

code(
    """realised = realised_payoff(history=post_history[list(TICKERS)], product=product, rate=market.rate)

print(f"Outcome             : {realised['outcome']}")
print(f"Total cash paid     : {realised['total_paid']:,.2f}")
print(f"PV at issue (model r): {realised['pv_at_issue']:,.2f}")
print()
print("Worst-of trajectory at each obs date:")
for d, w in realised['worst_path']:
    print(f"  {d}: {w:.4f}")
print()
print("Cashflow schedule:")
for pay_dt, label, amt in realised['cashflows']:
    print(f"  {pay_dt}: {label:<35s} {amt:>12,.2f}")
"""
)

# ---------------------------------------------------------------------------
md(
    """## 7. Model price vs realised payoff

The model gives an *expected* PV across the risk-neutral distribution; the
realised replay gives a *single sample* from the real-world path. The two
should be loosely consistent — and we don't expect equality, because:

1. **Risk-neutral vs real-world drift.** GBM under $\\mathbb{Q}$ uses $r-q$ as
   drift; the realised path used the actual physical drift, which for AMZN /
   META / MU over a 6M window in late 2025 / early 2026 may be very different.
2. **One realisation vs an expectation.** The realised PV is a single draw;
   the MC 1-sigma band gives the right scale of "how unlikely is this draw."

What we *do* expect to match exactly is the deterministic discounted-cashflow
calculus once a path is fixed — verified in `tests/test_mc_pricer.py`."""
)
code(
    """realised_pv = realised['pv_at_issue']
delta = realised_pv - result.price
pv = result.pv_samples
path_std = pv.std(ddof=1)
percentile = float((pv < realised_pv).mean()) * 100.0
print(f"Realised PV (at model r)       : {realised_pv:>12,.2f}")
print(f"Model MC price (mean)          : {result.price:>12,.2f}")
print(f"Δ (realised − model)           : {delta:>+12,.2f}")
print()
print(f"MC SE (precision of the mean)  : {result.standard_error:>12,.2f}")
print(f"Per-path PV stdev (dispersion) : {path_std:>12,.2f}")
print(f"Δ in per-path stdev units      : {delta / path_std:>+12,.3f} (≈ z-score in path distribution)")
print(f"Percentile of realised PV in MC: {percentile:>12.2f}%")
"""
)

# ---------------------------------------------------------------------------
md(
    """## 8. Realised worst-of trajectory overlay

Same axes as §5, but with the *actual* worst-of path superimposed in black
on the simulated cone."""
)
code(
    """initial = life.iloc[0].to_numpy(dtype=float)
realised_perf = life.to_numpy(dtype=float) / initial
realised_worst = realised_perf.min(axis=1)
realised_t = (life.index - pd.Timestamp(ISSUE_DATE)).days / 365.0

fig, ax = plt.subplots(figsize=(11, 5))
for i in sample:
    ax.plot(t_grid, worst[i], color='steelblue', alpha=0.20, lw=0.8)
ax.plot(realised_t, realised_worst, color='black', lw=1.8, label='Realised worst-of')
ax.axhline(product.autocall_barrier, color='black', lw=1.0, ls='--', label='Autocall = 100%')
ax.axhline(product.strike, color='crimson', lw=1.0, ls='--', label='Strike = 70%')
for t in obs_yf:
    ax.axvline(t, color='grey', alpha=0.45, lw=0.6)
ax.set_xlabel('Years from 17 Oct 2025')
ax.set_ylabel('Worst-of perf (normalised)')
ax.set_title('Realised worst-of (black) vs 20 simulated paths')
ax.legend(loc='upper left')
fig.tight_layout(); plt.show()
"""
)

# ---------------------------------------------------------------------------
md(
    """## 9. Headline takeaways

- Model MC price, %-of-notional, MC SE in §3.
- P(autocall total), P(KI at maturity), P(par at maturity) in §3 → tells us the
  shape of the structurer's expected outcome distribution.
- Realised outcome vs model expectation in §7 → the path-specific result vs
  the risk-neutral mean.

**Phase 3 deliverable complete.** Next phases:
- Phase 4: 1D Crank–Nicolson PDE to cross-validate MC on the single-asset
  reduction.
- Phase 5: Greeks (Δ, Γ, vega, ρ_pair) by bump-and-revalue with CRN.
- Phase 6: worst-of put control variate to compress MC SE further.
- Phase 7: write-up."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "03_fcn_pricing.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
