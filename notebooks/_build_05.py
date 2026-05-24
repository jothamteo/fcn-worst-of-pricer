"""Assemble notebooks/05_greeks.ipynb.

Re-run with `python notebooks/_build_05.py` after edits.
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
    r"""# 05 — Greeks: bump-and-revalue MC vs off-grid PDE

> **Frame:** Greeks here are from the **investor's** perspective (∂PV/∂input, investor long the structure). The dashboard's Hedging tab flips to the dealer's side; the rest of the dashboard and notebooks 03-08 stay on investor side.

This notebook computes the FCN's Greeks — Δ (spot sensitivity), Γ
(curvature), vega (vol sensitivity), and pairwise cega (correlation
sensitivity) — using two engines, and checks they agree.

**The two engines**

1. **MC, bump-and-revalue with common random numbers.** The standard
   trick: pre-draw a block of random numbers, then price the FCN twice
   — once at the current input, once with one parameter bumped up by a
   tiny amount — using the **same** noise both times. The price
   difference is the Greek. Reusing the noise means the random-sampling
   error largely cancels between the two runs, so the signal we measure
   is the parameter effect itself.

2. **PDE, off-grid finite differences.** The Crank–Nicolson PDE from
   notebook 04 gives us the FCN's value across a whole range of spot
   levels at $t=0$. Δ and Γ at today's spot fall straight out of a
   3-point central difference on that curve. Vega is a small bump at
   the PDE level — deterministic, no random noise.

**Headline plot.** Δ as a function of spot, near both the autocall and
KI barriers, with MC's noise band overlaid on PDE's smooth reference.
This shows the well-known weakness of bump-and-revalue MC Greeks —
they get noisy near a barrier where a tiny bump can flip a path's
outcome — and is the motivation for the smoothed-payoff work in
notebook 08.

**Reading order**
1. Setup — same 3-asset FCN as in notebook 03.
2. The 3-asset Greeks table (MC only — PDE only handles single-asset).
3. Single-asset cross-validation: MC vs PDE Greeks should agree away
   from the barriers; Γ is the loosest because it's the noisiest
   estimator.
4. Δ-vs-spot scan: MC noise spikes near the KI and AC barriers; PDE stays smooth.
5. Discussion.
"""
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

from src.market_data import load_market_data, MarketData
from src.fcn_payoff import FCNProduct
from src.mc_pricer import price_fcn
from src.pde_pricer import price_fcn_pde_1d
from src.greeks import mc_greeks_bump, pde_greeks_1d, mc_delta_curve, pde_delta_curve

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
np.set_printoptions(suppress=True, precision=6)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 1. Setup

Same AMZN/META/MU snapshot and 6-observation FCN from notebook 03."""
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
    """## 2. 3-asset MC Greeks (the headline)

The standard desk risk table — Δ, Γ, vega per name plus pairwise
correlation sensitivities. 80,000 antithetic primal paths (= 160k
effective), seed pinned.

The "raw" Greeks are mathematical derivatives (∂V/∂S, ∂²V/∂S², ∂V/∂σ).
Those numbers are awkward to interpret — derivatives are per *unit* of
input, but a "unit" of spot is $1 of share price, which isn't how
anyone thinks about moves. The desk-scaled columns are the more
useful ones:

- **Δ_per_1%**: dollar value change from a 1% relative spot move (= Δ · S₀ / 100).
- **Γ_per_1%×1%**: the convexity contribution from a 1% × 1% spot move (= Γ · S₀² / 10,000).
- **vega_per_volpt**: dollar value change from a +1 vol-point bump (= vega / 100).

Use these when reading the table."""
)
code(
    """mc_greeks = mc_greeks_bump(
    market=market, product=product,
    n_paths=80_000, antithetic=True, seed=20260511,
    spot_bump_rel=0.01, vol_bump_abs=0.01, corr_bump_abs=0.05,
)
print(mc_greeks.summary(tickers=list(market.tickers)))
"""
)

code(
    """se = mc_greeks.standard_errors
rows = []
spots = np.array(market.spots)
for i, name in enumerate(market.tickers):
    rows.append({
        'asset': name,
        'spot': spots[i],
        'Δ (raw)': mc_greeks.delta[i],
        'Δ SE': se['delta'][i],
        'Δ_per_1%': mc_greeks.delta_pct[i],
        'Γ_per_1%×1%': mc_greeks.gamma_pct[i],
        'vega/volpt': mc_greeks.vega_per_volpt[i],
        'vega SE / volpt': se['vega'][i] / 100.0,
    })
pd.DataFrame(rows).round(4)
"""
)

code(
    """# Pairwise correlation sensitivity (off-diagonals).
d = len(market.tickers)
corr_rows = []
for i in range(d):
    for j in range(i + 1, d):
        corr_rows.append({
            'pair': f"{market.tickers[i]}–{market.tickers[j]}",
            'ρ_base': market.corr[i, j],
            '∂V/∂ρ per 0.01': mc_greeks.rho_pair[i, j] / 100.0,
            'SE per 0.01': se['rho_pair'][i, j] / 100.0,
        })
pd.DataFrame(corr_rows).round(4)
"""
)
md(
    r"""**Reading the table.** Three things the signs should pass before
you trust the numbers — these are what a desk would eyeball first:

1. **Δ > 0 on every name.** The investor is *long the basket*. Each
   name moving up reduces KI risk (worst-of stays away from 70%) and
   pulls the worst-of toward the autocall (which terminates with par +
   coupon). Both effects lift the FCN's value.
2. **vega < 0 on every name.** The investor is *short vol*. Higher
   vol on any one name fattens its loss-tail, raises KI probability,
   and hurts the structure. Classic short-vol exposure — same shape
   as any short-put position.
3. **Pairwise correlation sensitivity is positive.** Slightly
   counter-intuitive: usually correlation is "bad" for option holders.
   But here the investor is *short* a worst-of put. When names
   decorrelate, the worst-of's distribution gets wider (one name can
   crash while others rally), which raises KI probability — bad for
   the short-put-side investor. So the investor *wants* the names to
   move together — positive cega is the right sign."""
)

# ---------------------------------------------------------------------------
md(
    """## 3. Single-asset cross-validation — MC vs PDE

Same trick as notebook 04's cross-validation: drop to the AMZN-only
reduction (worst-of operator becomes identity for one asset), compute
the Greeks both ways, check the numbers agree.

Expect Δ and vega from MC to land within a few MC standard errors of
the PDE benchmark. Γ is the loosest — bump-and-revalue Γ is a
notoriously noisy estimator (the next cell explains why), so the gap
will be wider on that one.
"""
)
code(
    """amzn_market = MarketData(
    tickers=['AMZN'],
    spots=np.array([market.spots[0]], dtype=float),
    vols=np.array([market.vols[0]], dtype=float),
    divs=np.array([market.divs[0]], dtype=float),
    corr=np.array([[1.0]]),
    rate=market.rate,
    history=pd.DataFrame(),
    sources={'as_of': 'cross-val'},
)

mc_amzn = mc_greeks_bump(
    market=amzn_market, product=product,
    n_paths=80_000, antithetic=True, seed=2026, spot_bump_rel=0.01,
)
pde_amzn = pde_greeks_1d(
    spot=float(amzn_market.spots[0]),
    vol=float(amzn_market.vols[0]),
    div=float(amzn_market.divs[0]),
    rate=amzn_market.rate,
    product=product,
    n_space=1600, n_time_per_period=160,
)

se = mc_amzn.standard_errors
rows = [
    {'greek': 'Δ_per_1%', 'MC': mc_amzn.delta_pct[0], 'MC SE': se['delta'][0] * amzn_market.spots[0] / 100,
     'PDE': pde_amzn.delta_pct[0], '|Δ| / SE': abs(mc_amzn.delta[0] - pde_amzn.delta[0]) / max(se['delta'][0], 1e-12)},
    {'greek': 'Γ_per_1%×1%', 'MC': mc_amzn.gamma_pct[0], 'MC SE': se['gamma'][0] * amzn_market.spots[0]**2 / 10_000,
     'PDE': pde_amzn.gamma_pct[0], '|Δ| / SE': abs(mc_amzn.gamma[0] - pde_amzn.gamma[0]) / max(se['gamma'][0], 1e-12)},
    {'greek': 'vega/volpt', 'MC': mc_amzn.vega_per_volpt[0], 'MC SE': se['vega'][0] / 100,
     'PDE': pde_amzn.vega_per_volpt[0], '|Δ| / SE': abs(mc_amzn.vega[0] - pde_amzn.vega[0]) / max(se['vega'][0], 1e-12)},
]
pd.DataFrame(rows).round(4)
"""
)
md(
    r"""**Δ and vega** land well inside a few standard errors of the PDE
benchmark — exactly what you'd want from two independent engines.

**Γ is structurally harder to estimate.** Γ is the *second* derivative
of price with respect to spot, computed by a 3-point central
difference:
$$\Gamma \approx \frac{V_+ - 2V_0 + V_-}{\epsilon^2}.$$
The numerator is a *small* number (two big prices nearly cancelling)
divided by a *tiny* number ($\epsilon^2$, where $\epsilon$ is the bump
size, e.g. 1% of spot). Any sampling noise in the prices gets amplified
by $1/\epsilon^2$ in the ratio — so even small per-path noise produces
a noticeable wobble in the Γ estimate. CRN helps a lot (the cancellation
between $V_+$ and $V_-$ is much tighter when they share noise), but
can't eliminate the structural variance from the second-derivative
geometry. So a wider Γ gap to the PDE is the expected outcome, not a
bug."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 4. The headline plot — Δ vs spot

Sweep AMZN's spot from well below the KI strike (50%) up to well above
the autocall barrier (130%), and at each spot level compute Δ both
ways — MC and PDE. Plot them together with the MC's error band shown
as a ribbon.

**What you should see** — and what motivates the smoothed-payoff work
in notebook 08:

- The PDE Δ is a **smooth curve**: deterministic, well-behaved
  everywhere.
- The MC Δ tracks PDE closely in the **bulk** of the distribution
  (where the worst-of is comfortably between the two barriers).
- The MC ribbon **widens dramatically** at two specific spot levels:
  `S/S₀ = 1.00` (the autocall barrier) and `S/S₀ = 0.70` (the strike /
  KI barrier).

**Why the MC noise spikes at the barriers:** at those spot levels, a
±ε bump in spot can flip a path's outcome between two completely
different regimes — "autocalled at obs 1 with par + small coupon" vs
"alive, continues to obs 2"; or "knocked in, takes the worst-of
downside" vs "above strike, gets full par". The per-path payoff
*difference* between the bumped and unbumped runs jumps from a small
number (sensitivity) to a huge number (regime switch), and the
estimator's variance blows up.

This is the textbook "discontinuous payoff" failure mode of
bump-and-revalue MC Greeks (Glasserman 2003, §7.2). The three ways
out: (a) throw more paths at it (expensive, slow convergence), (b)
use a wider bump (biases the Greek), or (c) replace the hard payoff
with a smoothed version — option (c) is what notebook 08 does."""
)
code(
    """# Sweep the AMZN spot only — keep the others at base.
spot_anchor = float(amzn_market.spots[0])
# Cover from well below KI (0.5) up through autocall+ (1.2) to expose the barriers.
spot_grid = np.linspace(0.4 * spot_anchor, 1.3 * spot_anchor, 41)

mc_delta, mc_delta_se = mc_delta_curve(
    market=amzn_market, product=product,
    spot_grid=spot_grid, asset_index=0,
    n_paths=20_000, antithetic=True, seed=20260512, spot_bump_rel=0.01,
)
pde_delta = pde_delta_curve(
    vol=float(amzn_market.vols[0]), div=float(amzn_market.divs[0]), rate=amzn_market.rate,
    product=product, spot_grid=spot_grid,
    n_space=1200, n_time_per_period=120, x_range_sigma=6.0,
)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(11, 5))
perf = spot_grid / spot_anchor
ax.fill_between(perf, mc_delta - mc_delta_se, mc_delta + mc_delta_se,
                alpha=0.25, color='steelblue', label='MC ± 1σ')
ax.plot(perf, mc_delta, color='steelblue', lw=1.4, label='MC Δ (20k antithetic, CRN)')
ax.plot(perf, pde_delta, color='crimson', lw=2.0, label='PDE Δ (off-grid)')
ax.axvline(product.autocall_barrier, color='black', lw=0.8, ls='--', alpha=0.6, label='Autocall (S/S₀=1.00)')
ax.axvline(product.strike, color='crimson', lw=0.8, ls='--', alpha=0.6, label='Strike / KI (S/S₀=0.70)')
ax.set_xlabel('S / S₀ at spot scan'); ax.set_ylabel('Δ (dV/dS) for AMZN reduction')
ax.set_title('Δ vs spot — MC bump-and-revalue vs PDE off-grid')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()
"""
)
md(
    r"""**Two things worth pointing at in the plot:**

1. **The MC ribbon widens near both barriers.** Near `S/S₀ = 1.00`
   the ±1% spot bump occasionally flips a path between "autocalls at
   obs 1, redeems at par + first coupon" and "doesn't autocall,
   continues to obs 2" — a jump of roughly notional - one coupon in
   the per-path payoff. Near `S/S₀ = 0.70` the same thing happens at
   maturity with the KI check. The per-path differences become
   bimodal (small for paths that don't flip, huge for paths that do),
   which is exactly the recipe for high MC variance.

2. **The PDE curve doesn't have this problem.** It computes the
   option's value as a function of spot on a continuous grid, applying
   the discrete events (autocall, KI) analytically at every grid node.
   There's no "path" to flip — the barriers are baked into the maths,
   not sampled. That's why the PDE was worth building: it gives a
   reference Δ surface that MC can only approximate noisily."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 5. Discussion

**What we got out of this notebook:**

- **3-asset Greeks** that agree with desk intuition: positive Δ on
  every name (investor is long the basket), negative vega (short
  vol), positive pair-cega (long correlation). Reproducible to bit
  precision under common-random-numbers.
- **Single-asset cross-validation** placed MC Δ and vega inside a few
  standard errors of the PDE benchmark — both engines agree on the
  numbers they can both compute.
- **The Δ-vs-spot plot** showed honestly where bump-and-revalue MC
  Greeks break down: at the two barrier levels. Bias is bounded; the
  bulk of the spot range is fine. The barriers are the exception, not
  the rule.

**What this notebook deliberately doesn't do (and where to look
next):**

1. **Pathwise / likelihood-ratio Greeks.** These are lower-variance
   estimators that use the chain rule along the *smooth* parts of the
   payoff instead of bumping. They give cleaner Greeks where they
   apply but they don't work across discontinuities like the autocall
   or KI — those still need a smoothed payoff or a bigger bump. We
   stick with bump-and-revalue throughout this notebook so the
   limitation is visible; a production pricer would use both, picking
   the right tool per payoff region.
2. **A worst-of European put control variate.** That's notebook 06.
   It tightens the MC standard error on the *price* (and indirectly,
   on Greeks bumped from that price), particularly on knocked-in
   paths where the FCN's variance is dominated by the worst-of-put
   leg.

**What a desk would actually ship.** Daily Greeks under CRN-controlled
bumps (we have that), a per-path Greek aggregator for risk reports
(we expose this via `pv_samples`), and a smoothed-payoff version of
the autocall and KI indicators so Γ doesn't spike at the barriers
(notebook 08 builds this — the plot above is exactly why)."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "05_greeks.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
