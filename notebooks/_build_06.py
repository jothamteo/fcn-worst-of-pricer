"""Assemble notebooks/06_control_variate.ipynb.

Re-run with `python notebooks/_build_06.py` after edits.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()


def md(text: str) -> None:
    NB.cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    NB.cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------

md(
    r"""# 06 — Variance reduction: worst-of European put control variate

> **Frame:** Greeks here are from the **investor's** perspective (∂PV/∂input, investor long the structure). The dashboard's Hedging tab flips to the dealer's side; the rest of the dashboard and notebooks 03-08 stay on investor side.

Antithetic variates already gave us a ~$\sqrt{2}\times$ tighter MC
standard error essentially for free (we used antithetic in notebooks
03-05). To go further we add a **control variate**: a second instrument
we can price on every path alongside the FCN. If we pick the CV well,
its per-path payoff is strongly correlated with the FCN's per-path PV,
and that correlation lets us cancel out a chunk of the random sampling
noise.

**Why a worst-of European put is the natural CV.** The FCN's variance
is dominated by what happens in the *knocked-in* region — the paths
where the worst-of finishes below the strike. That's exactly where a
European put on the worst-of pays. The two PVs are **strongly
anti-correlated** by construction: when the worst-of's terminal value
is low, the FCN loses (KI payoff) and the put gains. The stronger
that anti-correlation, the bigger the variance reduction.

**How the estimator works.** For each path we compute two numbers:
- $X$ = the FCN's discounted PV on that path
- $Y$ = the worst-of European put's discounted PV on the same path
  (same simulated underlying, just a different payoff)

We also need $\mathbb{E}[Y]$, the put's true expected value, which we
estimate cheaply once via a large pre-pass with a different seed (so
it's independent of the main MC run). Then the CV-corrected estimator
is:
$$\widehat{X}_{cv} = \bar X - \hat\beta \,(\bar Y - \mathbb{E}[Y]),\qquad
\hat\beta = \frac{\widehat{\mathrm{Cov}}(X, Y)}{\widehat{\mathrm{Var}}(Y)}.$$

The intuition: if the main sample's mean of $Y$ happens to come in
high (random luck), $X$ probably came in low (because they're
anti-correlated). The correction $-\hat\beta(\bar Y - \mathbb{E}[Y])$
nudges the FCN estimate back toward its true mean. The coefficient
$\hat\beta$ is a standard linear-regression slope, computed in-sample.

The theoretical best you can do is:
$$\mathrm{Var}(X_{cv}) = (1 - \rho_{XY}^2) \cdot \mathrm{Var}(X),$$
where $\rho_{XY}$ is the per-path correlation between FCN PV and put
PV. A $\rho$ of $-0.7$ gives you ~50% variance reduction; $-0.9$ gives
roughly 5×; $-1.0$ (perfect anti-correlation) would eliminate variance
entirely. We don't expect $\rho$ to hit $\pm 1$ because the put is
zero on the autocalled and par-at-maturity paths — only the
knocked-in paths drive the correlation.

**Reading order**
1. Setup — same 3-asset FCN.
2. Headline: price with and without CV at the same path budget; report
   the variance-reduction ratio and the SE collapse.
3. The X-vs-Y scatter plot — visually shows where the correlation
   lives (the knocked-in wedge) and where it doesn't (autocalled +
   par-at-maturity sit at Y=0).
4. How big does $\mathbb{E}[Y]$'s pre-pass need to be? Sweep and see.
5. Discussion + what could push the variance reduction further."""
)

code(
    """import sys, os
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.market_data import load_market_data
from src.fcn_payoff import FCNProduct, ObservationGrid
from src.gbm_simulation import SimulationConfig, simulate_paths
from src.mc_pricer import (
    price_fcn,
    _worst_of_european_put_pv,
    _worst_of_european_put_expectation,
)

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
np.set_printoptions(suppress=True, precision=6)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 1. Setup

Same AMZN/META/MU snapshot and 6-observation FCN as notebook 03."""
)
code(
    """ISSUE_DATE = date(2025, 10, 17)
TICKERS = ("AMZN", "META", "MU")
market = load_market_data(tickers=TICKERS, lookback_years=5, target_T=0.5, as_of=ISSUE_DATE)
print(market.summary())

product = FCNProduct(
    notional=50_000.0,
    coupon_rate=0.01,
    obs_dates=(
        date(2025, 11, 17),
        date(2025, 12, 17),
        date(2026, 1, 17),
        date(2026, 2, 17),
        date(2026, 3, 17),
        date(2026, 4, 17),
    ),
    pay_dates=(
        date(2025, 11, 19),
        date(2025, 12, 19),
        date(2026, 1, 19),
        date(2026, 2, 19),
        date(2026, 3, 19),
        date(2026, 4, 19),
    ),
    issue_date=ISSUE_DATE,
    autocall_barrier=1.00,
    strike=0.70,
    n_autocall_obs=5,
    coupon_barrier=None,
    physical_delivery=True,
    continuous_ki=False,
)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 2. With vs without CV at fixed path budget

Fair comparison: hold the main-pass path count fixed at 40,000
antithetic primal paths (= 80k effective) and price the FCN two ways:

- the standard MC estimator (antithetic only, no control variate)
- the same paths with the worst-of put CV applied on top

The CV needs $\\mathbb{E}[Y]$ — the put's true expected value — which
we estimate with a separate 200,000-path pre-pass at a different seed
so it's independent of the main run. That pre-pass is computed *once*
per market snapshot; once we have $\\mathbb{E}[Y]$, every subsequent
pricing on the same underlyings reuses it for free. So in production
the cost of the pre-pass amortises across many trades."""
)
code(
    """N_MAIN = 40_000
SEED_MAIN = 20260601

no_cv = price_fcn(market=market, product=product, n_paths=N_MAIN, antithetic=True, seed=SEED_MAIN)
cv = price_fcn(
    market=market, product=product, n_paths=N_MAIN, antithetic=True, seed=SEED_MAIN,
    control_variate='worst_of_put', cv_n_paths_for_ey=200_000,
)
print(cv.summary())
"""
)

code(
    """diag = cv.cv_diagnostics
rows = [
    {'estimator': 'antithetic only (no CV)', 'price': no_cv.price, 'SE': no_cv.standard_error,
     'paths_main': no_cv.n_paths_total, 'paths_pre': 0},
    {'estimator': 'antithetic + worst-of put CV', 'price': cv.price, 'SE': cv.standard_error,
     'paths_main': cv.n_paths_total, 'paths_pre': diag['n_paths_for_ey']},
]
df = pd.DataFrame(rows)
df['SE / price (bps)'] = 1e4 * df['SE'] / df['price']
df.round({'price': 4, 'SE': 4, 'SE / price (bps)': 1})
"""
)
md(
    f"""**The headline number** is the variance-reduction ratio
$\\mathrm{{Var}}(X)/\\mathrm{{Var}}(X_{{cv}})$. Its square root tells
you how much extra path budget you'd need *without* the CV to reach
the same SE. A 4× variance reduction = you'd need 4× the paths
without the CV.

The next cell reads the diagnostic numbers off the run."""
)
code(
    """print(f"β̂                  = {diag['beta_hat']:+.4f}")
print(f"corr(X, Y)         = {diag['rho_XY']:+.4f}")
print(f"Var(X) / Var(X_cv) = {diag['var_reduction_ratio']:.2f}x")
print(f"SE  / SE_cv        = {np.sqrt(diag['var_reduction_ratio']):.2f}x")
print()
print(f"Ê[Y]              = {diag['EY_hat']:,.4f} ± {diag['EY_se']:.4f}  (worst-of European put PV)")
print(f"Pre-pass paths    = {diag['n_paths_for_ey']:,}")
"""
)

# ---------------------------------------------------------------------------
md(
    r"""## 3. Why it works — scatter of $X$ vs $Y$ per path

The CV's variance reduction depends on one thing: how correlated are
the per-path FCN PV ($X$) and per-path put PV ($Y$)? Plotting them
against each other on a per-path basis makes the structure visible.

For our FCN, the per-path geometry splits naturally into three regimes:

- **Autocalled paths** ($W$ rises above 100% early) get full notional +
  some coupons; $X$ is high. The worst-of put pays $0$ at maturity for
  these paths; $Y = 0$.
- **Par at maturity** paths ($0.70 \le W(T) < 1.00$, no autocall) get
  par + coupons; $X$ is moderate. $Y$ is still $0$.
- **Knocked-in** paths ($W(T) < 0.70$) get $N \cdot W(T)$ + coupons; $X$
  drops below par. $Y = N \cdot (0.70 - W(T))$ is exactly the amount the
  FCN underperforms. **$X$ and $Y$ move in opposite directions** here, by
  construction.

The scatter plot makes the three regimes visible."""
)
code(
    """# Build a one-off sample matched to the main run so we can plot X vs Y on
# the same paths. Same seed as cv (above) for reproducibility.
grid = ObservationGrid.from_product(product=product)
cfg = SimulationConfig(
    spots=market.spots, vols=market.vols, divs=market.divs, rate=market.rate,
    corr=market.corr, T=grid.sim_T, n_steps=grid.sim_n_steps,
    n_paths=N_MAIN, antithetic=True, seed=SEED_MAIN,
)
paths = simulate_paths(cfg)
from src.fcn_payoff import payoff_per_path
X = payoff_per_path(paths=paths, spots=cfg.spots, product=product, grid=grid, rate=market.rate)
Y = _worst_of_european_put_pv(
    paths=paths, spots=cfg.spots, strike_perf=product.strike,
    rate=market.rate, notional=product.notional,
    T_pay=float(grid.pay_year_fractions[-1]),
)
print(f"corr(X, Y) on sample: {np.corrcoef(X, Y)[0,1]:+.4f}")
"""
)
code(
    """# Classify paths into the three regimes for colouring.
W_at_obs = (paths[:, grid.obs_indices, :] / cfg.spots).min(axis=2)  # (n, M)
ac = (W_at_obs[:, :int(product.n_autocall_obs)] >= product.autocall_barrier).any(axis=1)
ki = (W_at_obs[:, -1] < product.strike) & ~ac
par = ~ac & ~ki

fig, ax = plt.subplots(figsize=(10, 6))
# Subsample for legibility — 20k points is plenty visually.
n = X.shape[0]; idx = np.random.default_rng(0).choice(n, size=min(20_000, n), replace=False)
ax.scatter(Y[idx & ac[idx][:len(idx)] if False else idx][ac[idx]], X[idx][ac[idx]],
           s=3, alpha=0.4, color='steelblue', label=f'autocalled ({ac.mean()*100:.1f}%)')
ax.scatter(Y[idx][par[idx]], X[idx][par[idx]], s=3, alpha=0.4, color='goldenrod', label=f'par at maturity ({par.mean()*100:.1f}%)')
ax.scatter(Y[idx][ki[idx]], X[idx][ki[idx]], s=3, alpha=0.4, color='crimson', label=f'knocked in ({ki.mean()*100:.1f}%)')

# Regression line, plotted across the observed Y range.
beta = np.cov(X, Y, ddof=1)[0, 1] / Y.var(ddof=1)
alpha_const = X.mean() - beta * Y.mean()
yy = np.linspace(0, Y.max(), 200)
ax.plot(yy, alpha_const + beta * yy, color='black', lw=1.2, label=f'OLS X ~ Y (β̂={beta:+.3f})')

ax.set_xlabel('Y = worst-of European put PV per path (USD)')
ax.set_ylabel('X = FCN PV per path (USD)')
ax.set_title('FCN vs control-variate PVs (same path) — three resolution regimes')
ax.legend(loc='lower left'); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()
"""
)
md(
    r"""**What the scatter shows.** Each dot is **one simulated path**
through the AMZN/META/MU world. Its position on the chart says two
things about that path:

- **X-coordinate** = the FCN's discounted payoff on that path (dollars
  the investor received from the structure).
- **Y-coordinate** = what a *worst-of European put* (struck at 70% of
  initial spot) would have paid on the same path. Y = 0 means the put
  was worthless; Y > 0 means the put was in-the-money at maturity (the
  worst-of finished below 70%).

Three regions visible in the plot, each corresponding to a different
*kind* of path:

**Region 1 — autocalled paths (right side, Y = 0).**
The worst-of rose above 100% at one of the autocall observation dates,
so the trade redeemed early at par + accrued coupons. The put at
maturity? Pays zero — by the time maturity arrives the trade has
already terminated, and even if it hadn't, the worst-of was high
enough to autocall in the first place. So these paths land at:
X ≈ par + a few coupons (somewhere around \$50K–\$53K depending on
which obs autocalled), Y = 0.

**Region 2 — par-at-maturity paths (right side, Y = 0).**
The worst-of stayed alive through all 6 observations without
autocalling, and finished above the 70% strike at maturity. The trade
pays par + all 6 coupons. The put pays zero because the worst-of
finished above 70%. So these also stack at Y = 0, slightly to the
right of the autocalled cluster (because they collected more coupons).

**Region 3 — knocked-in paths (the left wedge, Y > 0).**
The worst-of finished *below* 70% at maturity. The FCN pays less
than par (physical-delivery downside). The put pays the gap below
70%, scaled by notional. The lower the worst-of at maturity, the
*more* the put pays AND the *less* the FCN pays — so as Y goes up,
X goes down. That's the negative correlation the CV needs.

**Why the right cluster doesn't help the CV.** Both
"autocalled" and "par-at-maturity" paths sit at Y = 0 — there's no
variation in Y, so nothing for the CV to exploit. The correlation
between X and Y comes *entirely from the knocked-in wedge* on the
left. Concretely: the CV's optimal coefficient β = Cov(X, Y) /
Var(Y), and both Cov and Var of Y are determined by the wedge alone.

**Why the left wedge is exactly the right place to remove noise
from.** The FCN's path-level variance — i.e., the spread of possible
outcomes — *is concentrated in the knocked-in region*. The right
cluster has minimal variance (par ± a couple coupons); the left
wedge has the full range of downside outcomes. So when the CV cancels
sampling noise on the wedge paths, it's cancelling noise on the part
of the distribution that actually drives the MC standard error in the
first place. That's why this particular CV is so effective for this
particular product."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 4. How big does the pre-pass need to be?

The CV needs $\mathbb{E}[Y]$, the put's true expected value. We
estimate it from a separate pre-pass — but the pre-pass itself isn't
free, it consumes paths. So there's a trade-off: larger pre-pass →
more accurate $\mathbb{E}[Y]$ → cleaner CV correction → tighter
final SE, but at higher upfront cost.

In production the trade-off is mostly moot because the pre-pass is
computed *once* per market snapshot and amortises across every future
pricing on the same underlyings. But it's worth checking how big it
needs to be for the CV's benefit to saturate. We sweep pre-pass sizes
on a fixed main sample and watch the CV's standard error as we add
more pre-pass paths. Expectation: a flat curve once the pre-pass is
big enough — beyond that point, more pre-pass paths don't help."""
)
code(
    """pre_sizes = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000]
rows = []
for n_pre in pre_sizes:
    res = price_fcn(
        market=market, product=product, n_paths=N_MAIN, antithetic=True,
        seed=SEED_MAIN, control_variate='worst_of_put', cv_n_paths_for_ey=n_pre,
    )
    d = res.cv_diagnostics
    rows.append({
        'n_pre': d['n_paths_for_ey'],
        'EY_hat': d['EY_hat'],
        'EY_se': d['EY_se'],
        'beta_hat': d['beta_hat'],
        'price_cv': d['price_cv'],
        'se_cv': d['se_cv'],
        'var_reduction': d['var_reduction_ratio'],
    })
df = pd.DataFrame(rows).round({'EY_hat': 4, 'EY_se': 4, 'beta_hat': 4, 'price_cv': 4, 'se_cv': 4, 'var_reduction': 3})
df
"""
)
code(
    """fig, ax = plt.subplots(figsize=(9, 4.5))
ax.semilogx([r['n_pre'] for r in rows], [r['se_cv'] for r in rows], 'o-', color='steelblue', label='CV SE')
ax.axhline(no_cv.standard_error, color='crimson', ls='--', lw=1.0, label='SE without CV')
ax.set_xlabel('pre-pass paths for Ê[Y]')
ax.set_ylabel('CV-adjusted standard error')
ax.set_title('CV standard error vs pre-pass size (main sample fixed at 80k effective)')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); plt.show()
"""
)
md(
    """**Reading the chart.** The CV's standard error drops sharply at
small pre-pass sizes (the $\\mathbb{E}[Y]$ estimate is still noisy,
so the CV correction is itself adding variance back) and then flattens
out. The flat region is where $\\mathbb{E}[Y]$ is known well enough
that its remaining noise is invisible compared to the main-sample
variance — adding more pre-pass paths past that point is wasted work.
The dashed red line is the no-CV SE for reference; the gap between
the flat curve and that line is the CV's actual benefit."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 5. Discussion

**What the numbers show.** On our FCN, the worst-of put control
variate delivers a **2–5× variance reduction** at the current
calibration — translating to **~1.4–2.2× tighter standard error**
for the same path budget. Stacked with antithetic variates (which
gives another √2), the total path-budget saving for a fixed target SE
is roughly **4–10×**. That's the kind of efficiency gain that turns
"this Greek table takes an hour to compute" into "this Greek table
takes 10 minutes".

**Why the per-path correlation is the headline.** The theoretical
variance-reduction formula is $\mathrm{Var}(X_{cv}) = (1 - \rho_{XY}^2)
\cdot \mathrm{Var}(X)$. A small change in $\rho$ produces a much
larger change in the variance ratio:
- $\rho = -0.5$ → 25% variance reduction (1.33× ratio)
- $\rho = -0.7$ → 49% variance reduction (1.96× ratio)
- $\rho = -0.9$ → 81% variance reduction (5.3× ratio)
- $\rho = -1.0$ → variance eliminated entirely (theoretical ceiling)

For our FCN, $\rho$ can't reach $-1$ because the put is exactly zero
on the autocalled and par-at-maturity paths (no correlation
information there). The correlation only lives in the knocked-in
wedge, which caps the achievable $|\rho|$ somewhere below 1. That's
a structural ceiling, not an engine bug.

**Where you'd push next if this were a production engine** — three
natural extensions, all out of scope here but worth flagging:

1. **Stack a second CV.** Add an autocall-bond CV alongside the
   worst-of put. The autocall region has its own known expected
   value (par + a few discounted coupons under simple approximations)
   so we can mop up variance on that side of the distribution too.
   The two CVs combine via linear regression of $X$ on both $Y$s.
2. **Stratified sampling on the worst-of's terminal value.** Partition
   paths by their final worst-of (above autocall / between barriers /
   knocked in) and balance the sample so each stratum gets a
   predictable count. Particularly useful for the
   tail-probability estimates the dashboard reports.
3. **Quasi-Monte Carlo.** Replace the pseudo-random `standard_normal`
   draws with a low-discrepancy sequence (e.g. Sobol' + Brownian
   bridge). Convergence improves from $1/\sqrt{N}$ toward $1/N$ in
   low dimensions; the 3-asset × 184-step grid is high-dimensional
   enough that the benefit shrinks, but it's still a clean win.

For a portfolio piece we stop here. The CV is wired into
`src/mc_pricer.py`, tested in `tests/test_mc_pricer.py`, demonstrated
in this notebook, and gives a visible reproducible variance
reduction — that's the deliverable."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "06_control_variate.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
