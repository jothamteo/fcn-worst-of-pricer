"""Assemble notebooks/07_summary.ipynb — the portfolio-piece headline notebook.

A single-screen pricing dashboard for a hiring manager: best-effort price
with all variance reduction stacked, Greeks table, single-asset PDE
cross-validation row, and the two "honest findings" plots.

Re-run with `python notebooks/_build_07.py` after edits.
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
    r"""# 07 — Summary

This is the headline notebook — a single-screen view of what the
pricer produces for a hypothetical worst-of FCN on AMZN, META and MU.
No exploration; no build-up. Just the things a structured-products
desk would ask to see on the trade ticket:

1. **Final price** — best-effort MC price with antithetic + worst-of
   put control variate stacked, alongside the single-asset PDE
   cross-validation that says the engine is wired up correctly.
2. **Greeks table** — Δ, Γ, vega, and pairwise correlation
   sensitivity, computed via bump-and-revalue under common random
   numbers.
3. **Honest findings** — two plots showing where the implementation
   works well (the bulk of the spot range) and where it gets noisy
   (near the autocall and KI barriers).
4. **Limitations** — what the model deliberately doesn't capture.
   Flat vol, constant correlation, European KI, etc.

This notebook is meant to be read top-to-bottom in one sitting.
The detailed work lives in notebooks 01–06."""
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
"""
)

# ---------------------------------------------------------------------------
md(
    """## Setup

AMZN/META/MU FCN, **exactly 6-month tenor** (issue 17 Oct 2025 →
final valuation 17 Apr 2026 = 182 days; maturity payment 19 Apr 2026 T+2),
**12.0% p.a. coupon** (1.0% per monthly observation, flat), autocall at 100%
(5 observation dates), European KI / strike at 70%, settled by physical
delivery on KI."""
)
code(
    """ISSUE_DATE = date(2025, 10, 17)
TICKERS = ("AMZN", "META", "MU")
market = load_market_data(tickers=TICKERS, lookback_years=5, target_T=0.5, as_of=ISSUE_DATE)
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
print(market.summary())
"""
)

# ---------------------------------------------------------------------------
md(
    """## 1. Price

The headline number. We stack both variance-reduction techniques —
antithetic variates *and* the worst-of put control variate — on top
of each other to get the tightest possible standard error for a given
path budget.

Main MC run: 80,000 antithetic primal paths (= 160k effective), seed
pinned for reproducibility. The control variate needs $\\mathbb{E}[Y]$
(the worst-of put's true expected value) — we pre-pass that with
400,000 paths at a different seed (independent of the main run) so the
CV correction is unbiased."""
)
code(
    """final_mc = price_fcn(
    market=market, product=product, n_paths=80_000, antithetic=True,
    seed=20260601, control_variate='worst_of_put', cv_n_paths_for_ey=400_000,
)
print(final_mc.summary())
"""
)

md(
    """### Single-asset PDE cross-validation

For each of the three names, we price a 1-asset version of the same
FCN (worst-of operator dropped because there's only one underlying)
using the deterministic Crank–Nicolson PDE from notebook 04, and
compare against the MC pricer running the same single-asset
reduction. All three names land **inside a few MC standard errors**
of the PDE — that's independent confirmation that the path simulator,
the payoff arithmetic, the observation-grid plumbing, and the
discounting are all consistent with a completely different solver."""
)
code(
    """rows = []
for i, name in enumerate(TICKERS):
    mkt_i = MarketData(
        tickers=[name],
        spots=np.array([market.spots[i]], dtype=float),
        vols=np.array([market.vols[i]], dtype=float),
        divs=np.array([market.divs[i]], dtype=float),
        corr=np.array([[1.0]]), rate=market.rate,
        history=pd.DataFrame(), sources={'as_of': 'reduction'},
    )
    mc_i = price_fcn(market=mkt_i, product=product, n_paths=80_000, antithetic=True, seed=20260511)
    pde_i = price_fcn_pde_1d(
        spot=float(mkt_i.spots[0]), vol=float(mkt_i.vols[0]),
        div=float(mkt_i.divs[0]), rate=mkt_i.rate, product=product,
        n_space=1600, n_time_per_period=160,
    )
    rows.append({
        'reduction': name,
        'spot': float(mkt_i.spots[0]),
        'vol': float(mkt_i.vols[0]),
        'MC price': mc_i.price, 'MC SE': mc_i.standard_error,
        'PDE price': pde_i.price,
        '|Δ| / SE': abs(pde_i.price - mc_i.price) / max(mc_i.standard_error, 1e-12),
    })
pd.DataFrame(rows).round(4)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 2. Greeks

The standard desk risk table — Δ (spot), Γ (curvature), vega (vol),
and pairwise cega (correlation), all per-name.

Bump-and-revalue with common random numbers: 80k antithetic paths,
±1% spot bump, ±1 vol-point bump, ±0.05 correlation bump (reported
per +0.01 in ρ, the desk convention). See notebook 05 for the full
discussion of how these Greeks are estimated and why some are noisier
than others."""
)
code(
    """greeks = mc_greeks_bump(
    market=market, product=product, n_paths=80_000, antithetic=True,
    seed=20260512, spot_bump_rel=0.01, vol_bump_abs=0.01, corr_bump_abs=0.05,
)
se = greeks.standard_errors
rows = []
spots = np.array(market.spots)
for i, name in enumerate(TICKERS):
    rows.append({
        'asset': name, 'spot': spots[i],
        'Δ_per_1%': greeks.delta_pct[i], 'Δ SE_per_1%': se['delta'][i] * spots[i] / 100,
        'Γ_per_1%×1%': greeks.gamma_pct[i],
        'vega/volpt': greeks.vega_per_volpt[i], 'vega SE/volpt': se['vega'][i] / 100,
    })
pd.DataFrame(rows).round(4)
"""
)
code(
    """rho_rows = []
for i in range(3):
    for j in range(i+1, 3):
        rho_rows.append({
            'pair': f"{TICKERS[i]}–{TICKERS[j]}",
            'ρ_base': market.corr[i, j],
            '∂V/∂ρ per 0.01': greeks.rho_pair[i, j] / 100,
            'SE per 0.01': se['rho_pair'][i, j] / 100,
        })
pd.DataFrame(rho_rows).round(4)
"""
)

# ---------------------------------------------------------------------------
md(
    r"""## 3. Honest findings

Two plots that show what works and what doesn't. Neither tries to
sell you on the model.

### 3.1 Δ near the barriers — MC noise vs the PDE smooth reference

Bump-and-revalue MC Greeks have a known weakness: they get noisy near
discrete-observation barriers. The mechanism is straightforward — at
`S/S₀ = 1.00` (autocall) or `S/S₀ = 0.70` (KI), a small spot bump can
flip a path's outcome between two completely different regimes:
"autocalls at obs 1, redeems early at par + one coupon" vs "doesn't
autocall, continues to maturity"; or "knocked in, takes the worst-of
downside" vs "above strike, gets full par". The per-path payoff
difference jumps from a small sensitivity number to a regime-switch
discontinuity, and the variance estimator can't smooth over it.

The plot below sweeps AMZN's spot from below the KI strike up through
the autocall barrier, computing Δ both ways. The PDE curve is smooth.
The MC ribbon (its ±1σ noise band) widens visibly at exactly
`S/S₀ = 0.70` and `S/S₀ = 1.00`. That's not a bug — it's a structural
feature of bump-and-revalue MC on barrier products, documented in
Glasserman (2003) §7.2. Notebook 08 fixes it with a smoothed-payoff
variant; here we leave it visible because honesty about the failure
mode is part of the deliverable."""
)
code(
    """amzn_market = MarketData(
    tickers=['AMZN'],
    spots=np.array([market.spots[0]], dtype=float),
    vols=np.array([market.vols[0]], dtype=float),
    divs=np.array([market.divs[0]], dtype=float),
    corr=np.array([[1.0]]), rate=market.rate,
    history=pd.DataFrame(), sources={'as_of': 'plot'},
)
spot_anchor = float(amzn_market.spots[0])
spot_grid = np.linspace(0.4 * spot_anchor, 1.3 * spot_anchor, 41)
mc_delta, mc_delta_se = mc_delta_curve(
    market=amzn_market, product=product, spot_grid=spot_grid, asset_index=0,
    n_paths=20_000, antithetic=True, seed=20260513, spot_bump_rel=0.01,
)
pde_delta = pde_delta_curve(
    vol=float(amzn_market.vols[0]), div=float(amzn_market.divs[0]), rate=amzn_market.rate,
    product=product, spot_grid=spot_grid,
    n_space=1200, n_time_per_period=120,
)
"""
)
code(
    """fig, ax = plt.subplots(figsize=(10, 5))
perf = spot_grid / spot_anchor
ax.fill_between(perf, mc_delta - mc_delta_se, mc_delta + mc_delta_se, alpha=0.25, color='steelblue', label='MC Δ ± 1σ (20k antithetic, CRN)')
ax.plot(perf, mc_delta, color='steelblue', lw=1.2)
ax.plot(perf, pde_delta, color='crimson', lw=2.0, label='PDE Δ (off-grid)')
ax.axvline(product.autocall_barrier, color='black', lw=0.8, ls='--', alpha=0.6, label='Autocall (S/S₀=1.00)')
ax.axvline(product.strike, color='crimson', lw=0.8, ls='--', alpha=0.6, label='KI (S/S₀=0.70)')
ax.set_xlabel('S / S₀'); ax.set_ylabel('Δ = dV/dS (AMZN reduction)')
ax.set_title('Δ vs spot: MC bump-and-revalue noise blows up near the barriers')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()
"""
)

md(
    r"""### 3.2 Variance reduction stack

How much do antithetic variates and the worst-of put control variate
actually buy us in MC efficiency? The chart below plots MC standard
error against the total path budget for three configurations:

- **No variance reduction** — vanilla MC at $N$ paths.
- **Antithetic only** — pair each draw $\eta$ with $-\eta$; cuts the
  linear-in-noise component of the payoff variance for free
  (~$\sqrt{2}$ tighter SE).
- **Antithetic + worst-of put CV** — adds the control variate layer
  on top; further $\sqrt{2}$–$\sqrt{5}$ tighter SE depending on
  $\rho_{XY}$.

All three lines should converge as $1/\sqrt{N}$ at high path counts;
the variance-reduction techniques effectively shift the line *down*
(same slope, lower intercept). The further below the no-VR line a
configuration sits, the less budget you need to reach a given target
SE — which is the real-world payoff of variance reduction."""
)
code(
    """ns = [5_000, 10_000, 20_000, 40_000, 80_000]
rows = []
for n in ns:
    no = price_fcn(market=market, product=product, n_paths=n, antithetic=False, seed=20260601)
    anti = price_fcn(market=market, product=product, n_paths=n // 2, antithetic=True, seed=20260601)
    full = price_fcn(market=market, product=product, n_paths=n // 2, antithetic=True, seed=20260601,
                     control_variate='worst_of_put', cv_n_paths_for_ey=200_000)
    rows.append({'n_total': n, 'no_VR': no.standard_error, 'antithetic': anti.standard_error,
                 'anti + CV': full.standard_error,
                 'ratio_anti': no.standard_error / max(anti.standard_error, 1e-12),
                 'ratio_full': no.standard_error / max(full.standard_error, 1e-12)})
vr_df = pd.DataFrame(rows)
vr_df.round({'no_VR': 3, 'antithetic': 3, 'anti + CV': 3, 'ratio_anti': 3, 'ratio_full': 3})
"""
)
code(
    """fig, ax = plt.subplots(figsize=(10, 5))
ax.loglog(vr_df['n_total'], vr_df['no_VR'], 'o-', color='black', label='no VR')
ax.loglog(vr_df['n_total'], vr_df['antithetic'], 'o-', color='steelblue', label='antithetic')
ax.loglog(vr_df['n_total'], vr_df['anti + CV'], 'o-', color='crimson', label='antithetic + CV')
# Reference: 1/sqrt(N) lines anchored to the leftmost no-VR point.
ns_arr = vr_df['n_total'].to_numpy(dtype=float)
anchor = vr_df['no_VR'].iloc[0]
ax.loglog(ns_arr, anchor * np.sqrt(ns_arr[0] / ns_arr), 'k:', alpha=0.4, label=r'$\\propto 1/\\sqrt{N}$')
ax.set_xlabel('total effective paths'); ax.set_ylabel('MC standard error (USD)')
ax.set_title('Variance reduction stack: antithetic + worst-of put CV')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); plt.show()
"""
)

# ---------------------------------------------------------------------------
md(
    r"""## 4. Limitations

This is the textbook model. The things below are deliberately
**not** in it — each is a place a real desk would layer extra
modelling on top:

1. **Flat per-name vol.** Every name carries one constant vol number
   over the life of the trade (5Y realised in this build). No
   smile (downside puts trade at a different implied vol than
   ATM calls — relevant because the KI is a 70%-strike put), no
   term structure (vols vary by maturity), no stochastic vol
   (Heston-style dynamics). A production system would calibrate a
   full vol surface per name and feed it into a local-vol or
   stochastic-vol simulator.
2. **Constant correlation matrix.** Realised correlations *drift*,
   and they rise in stress (the "correlation 1 in a crisis"
   phenomenon). A Brownian-correlation or DCC-style dynamic model
   would change the cega numbers materially. The Greeks table
   reports the *sensitivity* to correlation, which is what the desk
   actually hedges against — but the model can't tell you how that
   sensitivity will evolve through a sell-off.
3. **PDE supports only European KI.** The MC pricer supports
   continuous-KI monitoring (`continuous_ki=True` in the library);
   the 1D PDE doesn't, because that would require an absorbing
   boundary along the strike between observation dates and a
   non-trivial rewrite of the time-stepping. The default product
   spec here uses European KI (checked only at maturity), so the
   gap is fine for this build but flagged for completeness.
4. **Γ noise near the barriers.** As §3.1 above shows, MC Γ is
   structurally noisy near the autocall and KI barriers. Real
   desks use smoothed-payoff variants — replace each hard
   indicator with a steep sigmoid centred at the barrier — which
   gives clean Γ at the cost of a small price bias. Notebook 08
   demonstrates this; the headline numbers here use the hard
   payoff because the "honest MC" story is what nb07 is trying to
   tell.
5. **Flat coupons only on the headline.** This build uses
   `coupon_barrier=None`: a fixed coupon every period until
   autocall. The library *also* supports conditional coupons
   (paid only when the worst-of is above a barrier on the
   observation date) — that's tested but not used in the headline
   numbers here.

**None of these are reasons to distrust the price for this product
under this snapshot.** The cross-validation against the PDE in §1 is
direct evidence that the engines are computing the right thing for
the inputs given. The list above is what the desk would *add* in a
live setting — model risk reserves, smile dynamics, correlation
stress overlays — to cover the gap between the model and reality."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "07_summary.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
