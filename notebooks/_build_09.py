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

Notebooks 01–08 built the pricer and computed Greeks. This notebook
takes the *next* step: it translates those Greeks into the concrete
actions a dealing desk would take on day 1 of the trade, and quantifies
the bits of risk that *can't* be cleanly hedged (which the bid-ask
spread on the original sale has to compensate for).

Same dealer-side framing as the dashboard: the bank has sold this FCN
to a client and the desk now holds the short side. Everything below
is from that perspective — Δ shares to *buy* (to offset short basket
exposure), listed options to *sell* (to offset long vol exposure),
and the residual risks that can't be neatly hedged at all.

**Reading order**

1. **Trade context** — bank sells $1M of the FCN; desk owns the short side.
2. **Delta hedge** — how many shares of each name to buy.
3. **Vega hedge** — how many listed options to sell.
4. **Correlation analysis** — the risk with no clean hedge instrument.
5. **Unhedgeable risk summary** — gap, skew, correlation, in one table.
6. **Hedging frequency and rebalancing** — desk-practice notes.

Everything in this notebook *consumes* the Greeks computed in
notebooks 05 and 08; nothing is re-derived. The hedge arithmetic
lives in `src/hedging.py` and is unit-tested in `tests/test_hedging.py`.

**Note on settlement method.** This FCN uses **physical delivery**:
if KI triggers at maturity, the issuer delivers approximately
$N / (K \cdot S_{\text{worst}}(0))$ shares of the worst-performer
to the client at the strike price. The Greek *profile* and the
hedge ratios (Δ shares, vega options) have the same shape as under
cash settlement — only the magnitudes and the maturity-payoff slope
differ. The operational consequence for the hedge desk: the book
needs the capacity to **source and deliver actual shares** of
whichever name turns out to be the worst-performer at maturity.
For liquid US large-caps (AMZN/META/MU) this is routine; on
less-liquid names it would require pre-arranged borrow lines.
Cash settlement avoids the physical-share leg entirely, at the
cost of being a harsher payoff for the holder."""
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
    physical_delivery=True,
)
"""
)

# ---------------------------------------------------------------------------
md(
    """## 8.2 — Delta hedge

Take the Greeks computed in notebook 05 (recomputed here with the same
seed for reproducibility) and feed them into `compute_delta_hedge`.
The function rescales the deltas from the pricer's $50,000 reference
notional to the trade's $1,000,000 notional — Δ is linear in notional
for this FCN payoff, so the rescaling is a single multiplication. The
output is the per-name **share count** the desk needs to buy to
neutralise the spot leg of the book."""
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
    r"""**What the trader does.** The desk buys the share counts in the
table above — net long stock against the short FCN book. Three
practical points worth flagging:

- The **total hedge notional** is much smaller than the FCN's
  notional, even though the structure is tied to all three names.
  That's because per-name Δ is well below 1.0 — the FCN is
  fundamentally a *short-vol* structure on the basket, not a
  vanilla equity proxy, so its spot-sensitivity is muted. Roughly
  speaking, only a fraction of each underlying's notional needs to
  be carried as a hedge.
- **Δ isn't stable.** It changes as the basket moves (that's Γ at
  work) and as time passes (the autocall feature gets closer to
  triggering). The desk re-runs the Δ table at least daily; near
  the barriers — where Γ spikes (see notebook 08) — they re-run
  intraday.
- **Stock-borrow constraints** can complicate the hedge. If the
  desk doesn't have borrow availability on one of the names (rare
  for AMZN/META/MU but possible in stress), the Δ hedge has to
  shift partially onto futures or onto a short-call position, both
  of which introduce funding or basis residuals that the original
  hedge calculation didn't anticipate."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.3 — Vega hedge

The dealer's vega is the **mirror image** of the holder's. The
holder is short single-name vol (higher vol on any name → fatter
KI tail → holder loses), so the dealer's book is **long vol** on
each name. To neutralise the long-vol exposure, the dealer **sells
listed options** on each underlying.

We use ATM 3-month listed calls as the hedge instrument and compute
their vega + price via Black–Scholes at the same flat per-name vol
the pricer uses. In a real production setting the desk would pull
the actual options chain and choose strikes / tenors based on a vega
ladder — but BS at the mark-the-trade vol is the right *reference*
number to size the headline hedge against."""
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
    r"""**Reading the table.** The dealer sells roughly 50–250 listed ATM
calls per name (the exact count depends on the FCN's per-name vega
and on the listed option's vega per contract). The premium from
those short calls is **received** — partial revenue that helps fund
warehousing the structured-product position.

This hedge is the *headline* number. Several things it doesn't
capture:

- **Smile / skew.** We're hedging with a *flat-vol* listed option,
  but the FCN's effective vol exposure is concentrated on the
  **downside skew** — the KI is a put at 70% of spot, well
  out-of-the-money on the put side, which trades at a meaningfully
  higher implied vol than ATM. A real desk would hedge the
  dominant points on the smile separately (typically the 90% put
  alongside the ATM call) rather than just hitting the parallel
  level.
- **Term structure.** The FCN has a 6-month tenor; a 3-month
  listed call isn't perfectly maturity-matched. That mismatch
  shows up as a small theta-decay differential and a residual
  calendar-spread vega.
- **Cross-Greeks.** Vanna (∂Δ/∂σ) and volga (∂vega/∂σ) — the
  cross-derivatives between spot and vol — aren't covered by this
  hedge. They're usually second-order concerns but pop up after
  large vol moves; production desks track them separately.

A production hedge book would maintain a **vega ladder** keyed by
strike and tenor and hedge the dominant cells, not just the
parallel level."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.4 — Correlation analysis

How sensitive is the FCN's value to changes in pairwise correlation?
Run the 3-asset MC pricer at the base correlation matrix and at
shifts of ±0.05 and ±0.10 applied to **every off-diagonal**
simultaneously. That's the standard "parallel correlation shift"
stress test desks report as a single-scalar correlation P&L. The
pricing engine is unchanged from notebook 03; we just re-run it five
times with different correlation matrices."""
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
    r"""**The desk's situation.** A 5-percentage-point rise in pairwise
correlation across the basket moves FCN value by the amount printed
above. The dealer is the mirror of the holder: a rise in correlation
**lifts the structure's value** (good for the long-correlation
holder, bad for the short-correlation dealer's book), and vice versa
on a correlation fall.

The key fact is: **there is no liquid hedge instrument for
single-stock pairwise correlation** on a 3-name semis basket. You
can't go to the exchange and buy "correlation between AMZN and META"
the way you can buy AMZN vol via a listed option.

The closest workaround desks use is a **dispersion trade**: long
single-stock vol versus short index vol on a related index (SOXX or
SMH for a semis basket). The trade captures *average* correlation of
the index, not the specific 3-name pairwise structure inside this
FCN, so it's an *approximate* hedge. The correlation structure
inside SOXX also moves on its own dynamics, adding basis risk.

**Practical answer: reserve capital, don't hedge.** The bid-ask
spread on the original sale to the client has to be wide enough to
absorb correlation P&L within whatever realised range the desk's
capital model says is acceptable. This is exactly the residual that
shows up in §8.5 below."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.5 — Unhedgeable risk summary

This is the **one table** a trader would put in front of risk
management at issuance. It captures the three big risks that
the day-1 hedge book *can't* cleanly neutralise — the ones the
bid-ask spread on the original sale has to compensate the desk
for:

1. **Correlation P&L** — from §8.4 above. Worst-case dollar move
   from a +5pp parallel shift in pairwise correlation. No clean
   hedge instrument; reserve capital.
2. **Gap risk** — an overnight scenario where every name drops to
   its KI level simultaneously. Tail event, not a daily
   expectation, but it sizes how much capital the desk should
   hold against weekend/overnight risk.
3. **Vol-skew residual** — a proxy for the vega left over after
   the parallel-shift hedge in §8.3. Constructed by bumping each
   name's vol asymmetrically (up on one side, down on the other)
   and reading off the residual P&L the ATM hedge couldn't cover.

Together these three numbers say "if any of these tail moves
happens, this is the dollar exposure the desk inherits". That
number sets the **reserve** the trade has to carry."""
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
    r"""**Reading the table.** These three numbers — gap, skew, and
correlation — together are what the bid-ask spread on the original
sale has to compensate the desk for. A few notes:

- **Gap risk** is by construction extreme. A joint overnight jump
  of all three names to their KI level is a tail scenario (worst
  case, not expected), not a daily expectation. It's there to size
  the **reserve**, not the daily hedge book.
- **Skew residual** is small relative to gap risk in dollar terms,
  but it's the one that **bleeds out daily** if not addressed. A
  vol surface that twists (some strikes move up, others move down)
  rather than shifts in parallel will cost the desk a few vega
  worth per name per day until the listed-option hedge is
  re-struck.
- **Correlation P&L** can be either a gain or a loss depending on
  the direction of the correlation move. The dealer is **short
  correlation gamma**: P&L is asymmetric around ρ moves (much like
  spot Γ makes Δ moves asymmetric). The reserve is sized to cover
  the *bad* side of that asymmetry."""
)

# ---------------------------------------------------------------------------
md(
    r"""## 8.6 — Hedging frequency and rebalancing (notes)

No computation in this section — just practitioner notes on how the
hedging shown above actually runs *over time*.

**Delta rebalancing.** Most equity-derivatives desks run **intraday**
Δ hedging on large structured books — typically a few times per
session, more aggressively when any underlying is near a barrier.
Smaller books or smaller dealers settle for end-of-day hedging. The
trade-off: tighter rebalancing captures Γ more cleanly but bleeds
bid-ask on every cycle. No universal answer — desks calibrate the
rebalance frequency to the book's Γ concentration and the
underlyings' average daily range.

**Vega rebalancing.** Materially less frequent than Δ — weekly to
monthly is typical for the vega ladder, with faster cycles around
earnings or vol regime shifts. The listed-option hedges themselves
age (their vegas decay as expiries approach), so the hedge book has
to be **re-rolled**: typically by selling the front-month options
and buying the next quarter, harvesting a small calendar spread
along the way.

**Barrier events.** When any underlying drifts within a few percent
of an autocall or KI level, the hedge dynamics get weird:

- **Γ spikes** (notebook 08 makes this visible). Δ rebalancing
  demands shrink: where you might rebalance Δ a few times a day
  normally, near a barrier you might do it every hour.
- **Gap risk** over weekends or overnight is real. The standard
  mitigation is a **barrier shift** convention — for hedging
  purposes the desk treats the effective barrier as *slightly
  inside* the contractual one (e.g. 0.71 instead of 0.70 for the
  KI). That builds in a small safety buffer at the cost of a small
  Δ-hedge bias when the spot is far away. It's a well-known desk
  fudge that's documented in the model approval but doesn't appear
  in the marketing material.

**Theta is the bank's compensation.** Day after day, with markets
quiet, the FCN's PV drifts in a predictable way — the autocall
feature accretes toward a known terminal payoff. More importantly,
the *expected* hedging cost the desk would have to bear keeps the
bid-ask spread wide. The integral of theta over the trade's life,
**less** the realised cost of running the dynamic hedge, **less**
the reserves consumed by unhedgeable residuals, is what the desk
earns on the trade.

If the desk does its job well — tight Δ hedging, well-sized vega
hedge, no large gaps — the realised hedge cost comes in *below* the
theta budget, and the difference is the trade's P&L. If it doesn't
(big overnight gaps, vol-of-vol blow-ups, model errors near the
barriers), it doesn't, and that trade ends up in the post-mortem
deck."""
)

# ---------------------------------------------------------------------------

out_path = Path(__file__).parent / "09_hedging.ipynb"
nbf.write(NB, out_path)
print(f"Wrote {out_path}")
