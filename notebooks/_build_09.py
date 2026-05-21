"""Assemble notebooks/09_hedging.ipynb.

Re-run with `python notebooks/_build_09.py` after edits.

This notebook is the hedging addendum: it translates the Phase 5 Greeks into
concrete trader actions (shares to short, listed options to buy) and stress
tests the residuals. It is the closing notebook of the project — anything
about the pricer model itself is in 01-08; this is "what does the desk do
with the numbers."
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
    r"""# 09 — Hedging analysis (the desk view)

Notebooks 01–08 built the pricer and computed Greeks. This notebook does the
*next* thing: it converts those Greeks into the concrete actions the dealing
desk takes on day 1, and quantifies what's left over that can't be neatly
hedged.

I'm coming at this as an IT dev who sees these tickets fly past the desk
every week but has never sat on the other side — the goal is to make the
Greek-to-share-count translation explicit so I can talk about it in an
interview without hand-waving.

**Reading order**

1. Trade context — bank sells $1M of the FCN; desk now owns the short side.
2. Delta hedge — how many of each name to buy.
3. Vega hedge — how many listed options to short.
4. Correlation analysis — the risk with no clean hedge.
5. Unhedgeable risk summary — gap, skew, correlation, in one table.
6. Hedging frequency and rebalancing — desk practice notes.

Everything in this notebook *consumes* the Greeks computed in notebooks 05
and 08; nothing is re-derived. Hedge arithmetic lives in `src/hedging.py`,
tested in `tests/test_hedging.py`.

**Note on settlement method.** The FCN here uses **physical delivery**: if KI
triggers at maturity, the issuer delivers approximately $N / (K \cdot
S_{\text{worst}}(0))$ shares of the worst-performer to the client at the
strike price. The Greek *profile* and the hedge ratios (Δ shares, vega
options) have the same shape as they would under cash settlement — only the
magnitudes and the maturity-payoff slope differ. The operational consequence
on the hedge desk side: the issuer's hedge book needs the **operational
capacity to source and deliver actual shares** of whichever name turns out
to be the worst-performer at maturity. For liquid US large-caps (AMZN, META,
MU) this is routine; on less-liquid names it would require borrow-arrangement
contingencies. Cash settlement avoids the physical-share leg entirely, at
the cost of being a harsher payoff for the holder."""
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
from src.greeks import mc_greeks_bump
from src.hedging import (
    black_scholes_call,
    compute_delta_hedge,
    compute_vega_hedge,
    compute_correlation_risk,
    compute_unhedgeable_risk_summary,
)

pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
np.set_printoptions(suppress=True, precision=6)
"""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.1 — Trade context

Assume the bank has just sold this FCN at notional **$1,000,000** to a
private-banking client. The bank's book now holds the *opposite* side — the
desk is *short* the FCN. As spot rises, the FCN gains in value and the
dealer loses, so the day-1 hedge has to:

* offset the dealer's *short delta* with stock,
* offset the dealer's *long vega* with listed options,
* size up the unhedgeable residual (correlation, gap, skew) so the bid–ask
  spread on the original sale at least pays for it."""
)

code(
    """ISSUE_DATE = date(2025, 10, 17)
TICKERS = ("AMZN", "META", "MU")
TRADE_NOTIONAL = 1_000_000.0   # $1M sold to client
PRICER_NOTIONAL = 50_000.0     # the pricer / Phase 5 Greeks are at this scale

market = load_market_data(tickers=TICKERS, lookback_years=5, target_T=0.5, as_of=ISSUE_DATE)
print(market.summary())

product = FCNProduct(
    notional=PRICER_NOTIONAL,
    coupon_rate=0.01,
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
    issue_date=ISSUE_DATE,
    autocall_barrier=1.00,
    strike=0.70,
    n_autocall_obs=5,
    physical_delivery=True,
)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 8.2 — Delta hedge

We consume the Phase 5 Greeks (recomputed here with the same seed so the
notebook is fully reproducible) and feed them through
`compute_delta_hedge`. The function rescales the deltas from the pricer's
50,000 reference notional to the trade's $1M notional — delta is linear in
notional for this FCN payoff, so this is a simple multiplicative scale."""
)

code(
    """greeks = mc_greeks_bump(
    market=market, product=product,
    n_paths=80_000, antithetic=True, seed=20260511,
    spot_bump_rel=0.01, vol_bump_abs=0.01, corr_bump_abs=0.05,
    smoothing={'k_ac': 100.0, 'k_ki': 100.0, 'k_coupon': 100.0},
)
print(greeks.summary(tickers=list(market.tickers)))
"""
)

code(
    """deltas_dict = {t: float(greeks.delta[i]) for i, t in enumerate(market.tickers)}
spots_dict = {t: float(market.spots[i]) for i, t in enumerate(market.tickers)}

dh = compute_delta_hedge(
    deltas=deltas_dict,
    notional=TRADE_NOTIONAL,
    spots=spots_dict,
    pricer_reference_notional=PRICER_NOTIONAL,
)

rows = []
for t in market.tickers:
    r = dh[t]
    rows.append({
        'ticker': t,
        'Δ (scaled to trade notional)': r.delta,
        'spot': r.spot,
        'shares to BUY': r.shares_to_buy,
        'hedge notional': r.hedge_notional,
        'hedge % of trade': r.hedge_pct_of_trade,
    })
df_dh = pd.DataFrame(rows)
total_hedge = sum(r.hedge_notional for r in dh.values())
df_dh.loc[len(df_dh)] = {
    'ticker': 'TOTAL',
    'Δ (scaled to trade notional)': float('nan'),
    'spot': float('nan'),
    'shares to BUY': sum(r.shares_to_buy for r in dh.values()),
    'hedge notional': total_hedge,
    'hedge % of trade': 100.0 * total_hedge / TRADE_NOTIONAL,
}
df_dh
"""
)

md(
    r"""**What the trader does.** The desk buys roughly the share counts in the
table above — net long stock against the short FCN book. A few practical
observations:

* The *total* hedge notional is materially less than the trade notional
  itself, even though the FCN's economics are tied to all three names. This
  is because Δ is well below 1.0 per name — the FCN is short-vol on the
  basket, not a vanilla equity proxy, so its spot-sensitivity is muted.
* Δ is not stable. It moves as the basket moves (that's Γ) and as time
  passes (that's the autocall feature getting closer). The desk re-runs this
  table at least daily; Γ near the barriers (notebook 08) is exactly why
  intraday re-runs matter when any name approaches `S/S₀ ≈ 1.00` or `≈ 0.70`.
* If the desk doesn't have stock-borrow availability on one of the names
  (rare but possible for MU-class names in stress), the delta hedge would
  partially shift onto futures or onto a short-call position, both of which
  introduce funding/basis residuals."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.3 — Vega hedge

The dealer is on the *opposite* side of the holder's vega. The holder is
short single-name vol (vol up on any name → KI risk up → holder loses), so
the dealer is *long* vol on each name. To neutralise, the dealer **sells**
listed options on the underlyings.

We use ATM 3-month listed calls as the hedge instrument and compute their
vegas + prices via Black–Scholes at the same flat per-name vol used by the
pricer. In practice the desk would pull the real chain, but BS at the
implied vol used to mark the trade is the right *reference* number to size
the hedge against."""
)

code(
    """TENOR_YEARS = 0.25
listed_vegas = {}
listed_prices = {}
for i, t in enumerate(market.tickers):
    price, vega_per_volpt = black_scholes_call(
        spot=float(market.spots[i]),
        strike=float(market.spots[i]),     # ATM
        T=TENOR_YEARS,
        vol=float(market.vols[i]),
        rate=market.rate,
        div=float(market.divs[i]),
    )
    listed_vegas[t] = vega_per_volpt
    listed_prices[t] = price

# Dealer-side vega = -holder vega per +1 vol pt. greeks.vega_per_volpt is
# the holder side at PRICER_NOTIONAL — scale to trade notional + flip sign.
scale = TRADE_NOTIONAL / PRICER_NOTIONAL
dealer_vegas = {
    t: -float(greeks.vega_per_volpt[i]) * scale
    for i, t in enumerate(market.tickers)
}

vh = compute_vega_hedge(
    vegas=dealer_vegas,
    listed_option_vegas=listed_vegas,
    listed_option_prices=listed_prices,
)

rows = []
for t in market.tickers:
    r = vh[t]
    rows.append({
        'ticker': t,
        'dealer vega (USD per +1 vol pt)': r.fcn_vega,
        'listed call price': r.listed_option_price,
        'listed call vega': r.listed_option_vega,
        'n options to TRADE (- = sell)': r.n_options_to_buy,
        'premium (- = received)': r.total_premium,
        'residual vega': r.residual_vega,
    })
pd.DataFrame(rows)
"""
)

md(
    r"""**Reading the table.** The dealer sells ~50–250 listed ATM calls per
name (the exact count depends on each name's vega and the listed option's
vega). The premium is *received*, partially offsetting the cost of warehousing
the position. A few caveats this hedge does not address:

* **Smile / skew.** The desk is buying a *flat-vol* listed option to hedge
  a structured product whose effective vol exposure is concentrated in the
  *downside skew* (the KI is well out-of-the-money put-side). A real desk
  hedges the dominant points on the smile separately — typically the 90% put
  and the 100% call — rather than a single ATM call.
* **Term structure.** The FCN has a 6.5-month tenor; a 3-month listed call
  is *not* perfectly maturity-matched. Theta-decay differential is one
  residual; calendar-spread vega is another.
* **Cross-Greeks.** vanna (∂Δ/∂σ) and volga (∂vega/∂σ) are not in this
  hedge — they're typically a second-order concern but pop up after large
  vol moves.

A production desk would track a *vega ladder* by strike and tenor and hedge
the dominant components, not the parallel level alone."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.4 — Correlation analysis

The 3-asset MC pricer is re-run at the base correlation matrix and at
shifts of ±0.05 and ±0.10 across **every off-diagonal** of the correlation
matrix simultaneously. This is the "parallel correlation shift" stress —
the same number a desk would report as a single-scalar correlation P&L.
We re-use the existing `price_fcn` engine; nothing about the pricer is
re-implemented."""
)

code(
    """def bumped_corr(corr_base, shift):
    c = corr_base.copy()
    d = c.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j:
                c[i, j] = np.clip(c[i, j] + shift, -0.99, 0.99)
    return c

corr_shifts = [-0.10, -0.05, 0.0, +0.05, +0.10]
prices_at_corr = {}
for shift in corr_shifts:
    bumped_market = MarketData(
        tickers=market.tickers,
        spots=market.spots,
        vols=market.vols,
        divs=market.divs,
        corr=bumped_corr(market.corr, shift),
        rate=market.rate,
        history=market.history,
        sources={'corr_shift': shift},
    )
    pr = price_fcn(
        market=bumped_market, product=product,
        n_paths=40_000, antithetic=True, seed=20260511,
    )
    # Rescale price to the trade notional.
    prices_at_corr[shift] = float(pr.price) * (TRADE_NOTIONAL / PRICER_NOTIONAL)

pd.DataFrame({
    'corr_shift': list(prices_at_corr.keys()),
    'FCN price @ trade notional': list(prices_at_corr.values()),
})
"""
)

code(
    """corr_report = compute_correlation_risk(prices_at_corr)
print(f"cega (per +0.01 in ρ): ${corr_report.cega:,.2f}")
print(f"P&L per +5 pp move:    ${corr_report.correlation_pnl_per_5pct_move:,.2f}")
print(f"Hedgeable:             {corr_report.hedgeable}")
print()
print(corr_report.narrative)
"""
)

md(
    r"""**The desk's situation.** A 5pp rise in pairwise correlation across the
basket moves FCN value by the amount above. The issuer is on the opposite
side: a rise in correlation **hurts the holder**, which is a **gain for the
issuer** (or vice versa, depending on the sign — read the narrative). Either
way: there is **no liquid hedge product** for single-stock pairwise
correlation on a 3-name semis basket.

The closest thing desks use is a **dispersion trade**: long single-stock
vol vs short index vol on a related index (here, SOXX or SMH). The
correlation P&L on the FCN doesn't *isolate* cleanly into a dispersion
strategy, though — the dispersion's covered basket isn't your basket, and
the correlation structure inside SOXX moves on its own dynamics. So the
practical answer is: **reserve capital, don't hedge.** The bid–ask on the
original sale to the client has to cover this."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.5 — Unhedgeable risk summary

A single table the trader could put in front of risk management at issuance.
We compose:

1. The correlation P&L from §8.4.
2. A gap-risk scenario: overnight jump of every name to its individual
   knock-in level (the joint worst-case dealer P&L scenario), evaluated by
   re-running the pricer with bumped spots.
3. A vol-skew residual: an approximate "vega left over after the parallel
   hedge", quantified by bumping each name's vol asymmetrically up vs down
   (a crude skew-twist proxy)."""
)

code(
    """# Gap scenario: every name simultaneously drops to its KI level.
# Important: the initial fixing S(0) was locked in at trade date and the
# payoff normalises by it. A "gap" moves only the simulation start, not the
# fixing — so we drive the simulator with bumped spots but call the payoff
# with the original spots. (This is the same spot-bump semantics that
# greeks._price_with_overrides documents.)
from src.gbm_simulation import SimulationConfig, simulate_paths
from src.fcn_payoff import ObservationGrid, payoff_per_path

grid_for_gap = ObservationGrid.from_product(product=product)

def _price_at_simulation_spots(sim_spots, seed=20260511):
    cfg = SimulationConfig(
        spots=sim_spots,
        vols=market.vols,
        divs=market.divs,
        rate=market.rate,
        corr=market.corr,
        T=grid_for_gap.sim_T,
        n_steps=grid_for_gap.sim_n_steps,
        n_paths=40_000,
        antithetic=True,
        seed=seed,
    )
    paths = simulate_paths(cfg)
    pv = payoff_per_path(
        paths=paths,
        spots=np.asarray(market.spots, dtype=float),   # locked initial fixings
        product=product, grid=grid_for_gap, rate=market.rate,
    )
    return float(pv.mean())

base_holder = _price_at_simulation_spots(np.asarray(market.spots, dtype=float))
gap_holder = _price_at_simulation_spots(np.asarray(market.spots * product.strike, dtype=float))
# Dealer (short FCN) P&L on a gap is +base - +gap_value, scaled to trade notional.
gap_pnl_dealer = (base_holder - gap_holder) * (TRADE_NOTIONAL / PRICER_NOTIONAL)
print(f"Holder PV at base:          ${base_holder:,.2f} (per {PRICER_NOTIONAL:,.0f} notional)")
print(f"Holder PV at gap-to-KI:     ${gap_holder:,.2f}")
print(f"Overnight gap-to-KI dealer P&L (at trade notional ${TRADE_NOTIONAL:,.0f}): ${gap_pnl_dealer:,.0f}")

# Vol-skew residual: bump each name's vol asymmetrically (up by 1 vol pt,
# everyone else flat) and read off the cross-vega the listed-call hedge
# wouldn't catch. We use a quick proxy: re-run MC with vol_bump on one name.
skew_residuals = {}
for i, t in enumerate(market.tickers):
    vols_twisted = market.vols.copy()
    vols_twisted[i] += 0.02   # +2 vol pts skew twist on this name
    twisted_market = MarketData(
        tickers=market.tickers, spots=market.spots, vols=vols_twisted,
        divs=market.divs, corr=market.corr, rate=market.rate,
        history=market.history, sources={'scenario': f'skew_{t}'},
    )
    pr_twist = price_fcn(market=twisted_market, product=product,
                          n_paths=40_000, antithetic=True, seed=20260511)
    pnl_dealer_twist = (base_holder - pr_twist.price) * (TRADE_NOTIONAL / PRICER_NOTIONAL)
    # Subtract the parallel-shift portion of the vega hedge (linear in vol).
    parallel_pnl = -2.0 * dealer_vegas[t]    # 2 vol pts × dealer vega
    skew_residuals[t] = float(pnl_dealer_twist - parallel_pnl)

print('Skew residual vega (per +2 vol pt asymmetric bump):')
for t, v in skew_residuals.items():
    print(f"  {t}: ${v:,.0f}")
"""
)

code(
    """summary = compute_unhedgeable_risk_summary(
    greeks={'delta': greeks.delta, 'vega': greeks.vega},
    scenario_pnls={'overnight_gap_to_ki_all_names': gap_pnl_dealer},
    correlation_report=corr_report,
    notional=TRADE_NOTIONAL,
    vol_skew_residual_vega=skew_residuals,
)
df_summary = pd.DataFrame(summary.rows)
df_summary[['risk', 'metric', 'value', 'p_and_l_5pct', 'pct_of_notional', 'hedgeable']]
"""
)

md(
    r"""**Reading the table.** These are the components the bid–ask spread on
the original sale has to compensate for. A few observations the desk would
want me to note:

* The **gap-risk** number is by construction extreme — a joint overnight
  jump of all three names to their KI level is a tail scenario, not a
  daily expectation. It's there to size the reserve, not the daily hedge.
* The **skew residual** is small relative to gap risk, but it's the one
  that bleeds out *daily* if not addressed. A vol surface that twists
  rather than parallel-shifts will cost the desk a few vega per name per
  day until the listed-option hedge is re-struck.
* The **correlation P&L** can be either a gain or a loss depending on
  direction — the desk is short correlation gamma (the issuer's P&L is
  asymmetric across +ρ vs −ρ moves). The reserve is sized to cover the bad
  side."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.6 — Hedging frequency and rebalancing (notes)

This section is discussion-only — no computation. Recording the practitioner
intuition for future-me when I'm trying to explain this in an interview.

**Delta rebalancing.** Most equity-derivatives desks run *intraday* delta
hedging for large structured books — typically a few times per session,
more aggressively when any underlying is approaching a barrier. Smaller
books (or smaller dealers) settle for end-of-day hedging. The trade-off is
hedge-replication accuracy vs transaction-cost drag: tighter rebalancing
captures Γ more cleanly but bleeds bid–ask on every cycle. There's no
universal answer — desks calibrate it to the book's Γ concentration and
the underlyings' average daily range.

**Vega rebalancing.** Materially less frequent than delta — weekly to
monthly is typical for the vega ladder, faster around earnings or vol
regime shifts. The desk's listed-option hedges *do* age (their vegas
decay as their expiries approach), so the hedge book also has to be
re-rolled, typically by selling the front-month and buying the next
quarter, harvesting calendar spread along the way.

**Barrier events.** When any underlying gets within a few percent of an
autocall or KI level the hedge dynamics get *weird*. Specifically:

* Γ spikes (notebook 08 makes this visible). Δ rebalancing demands shrink
  the rebalancing interval.
* **Gap risk** over weekends or overnight is real. The standard mitigation
  is a "barrier shift" convention — for hedging purposes the desk treats
  the effective barrier as *slightly inside* the contractual barrier
  (e.g. 0.71 instead of 0.70 for the KI), which builds in a small safety
  buffer at the cost of a small Δ-hedge bias when the spot is far away.
  This is a well-known desk fudge that's documented in the model approval
  but doesn't show up in the marketing material.

**Theta is the bank's compensation.** Day after day, with nothing else
moving, the FCN's PV drifts: the autocall feature accretes towards a
known terminal payoff, and (more importantly) the *implied* hedge cost
the desk has to bear keeps the bid–ask wide. The integral of theta over
the lifetime of the trade, less the actual cost of running the dynamic
hedge, less the unhedgeable residual reserves, is what the desk earns.
If the desk does its job well, the realised cost of the dynamic hedge
comes in *below* the theta budget, and the difference is the trade's
P&L. If it doesn't (large gaps, vol-of-vol blow-ups, model errors near
barriers), it doesn't — and that's the trade that ends up in the
post-mortem deck."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "09_hedging.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
