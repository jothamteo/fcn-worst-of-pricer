# Hedging the worst-of FCN — practitioner notes

These are the notes I wish I'd had when I first watched a structured-products
desk price up an FCN. I work in IT at a private bank, so I see the
post-trade tickets and the daily risk reports go past — but I've never
priced or hedged one. This document is the bridge from "FCN pricer
notebook" to "what the dealing desk is actually doing all day." Kept short
on purpose: 3–4 pages, no derivations the textbooks already do well.

The product is the textbook worst-of FCN on three large-cap tech names
(AMZN, META, MU) — 6-month tenor, monthly observations, 100% autocall
barrier, 70% European knock-in, flat coupons. The mechanics are in the
main README and METHODOLOGY; the *hedging* is what this doc covers.

---

## 1. Decomposition

The cleanest way to see the FCN is to break it into things the desk can
already price one-by-one. For a worst-of FCN with notional $N$:

$$
V_{\text{FCN}} \;=\; \underbrace{N \cdot \text{ZCB}(T)}_{\text{zero-coupon bond}}
\;+\; \underbrace{\sum_{j=1}^{M} c_j \cdot \text{Digital}_j^{\text{cpn}}}_{\text{strip of coupon digitals}}
\;+\; \underbrace{N \cdot \text{Autocall feature}}_{\text{long: shortens duration when in the money}}
\;-\; \underbrace{N \cdot \text{Down-and-in put}_{\text{worst-of}}}_{\text{short: the only direction the holder loses material money}}
$$

The headline component, economically, is the short worst-of down-and-in
put — that's the source of *every* unhedgeable risk in this product:

- It is *path-dependent* (autocall affects whether the put is even alive
  at maturity).
- It is *worst-of* (correlation matters in a non-linear way; the put kicks
  on whichever name is weakest, not on the basket average).
- It has a *discrete barrier* (Δ and Γ blow up at `S/S₀ = 0.70`).

Everything else is essentially linear and easy: the ZCB is a bond, the
coupon digitals are vanilla(ish), the autocall is "the structure stops
paying if the basket recovers." The dealer's hedging energy goes into the
short worst-of put.

---

## 2. The Greeks the desk monitors

Per name, every day:

- **Δᵢ.** First derivative w.r.t. spot. Hedged with stock. Trivially
  separable across names — short the FCN, long Δᵢ shares per dollar of
  notional. Re-marked intraday for any name within a few percent of either
  barrier.
- **Γᵢ.** Second derivative. Not hedged directly — kept inside the listed
  vanilla option positions used for vega. The desk tracks "Γ-PnL" as the
  cost of running an imperfect Δ hedge: if Γ is large, missing the
  re-hedging schedule by half a day can cost real money.
- **Vegaᵢ.** First derivative w.r.t. each name's vol. Hedged with **listed
  vanilla options** on each underlying (typically ATM 3-month calls or
  90% puts, depending on whether the desk wants to hedge the autocall side
  or the KI side first). For a 3-name basket this is three separate
  vega buckets, not one.
- **Vannaᵢ** (∂Δᵢ/∂σᵢ). Cross-Greek between spot and vol. Matters when a
  name is moving *and* its vol is moving (earnings, macro days). Hedged
  loosely by holding listed options near the autocall / KI strike rather
  than only ATM.
- **Volgaᵢ** (∂vegaᵢ/∂σᵢ). Second derivative in vol. Captures the vol-of-vol
  exposure. Typically reserved-against rather than hedged unless the book
  is very vega-concentrated.
- **ρᵢⱼ** (∂V/∂ρᵢⱼ). Pairwise correlation sensitivity. **No clean hedge.**
  See §3.
- **θ.** Time decay. Not hedged — this is *the desk's revenue.* Every day
  with no spot move, the FCN's value drifts in the direction the desk wants
  (towards the structure paying off as expected); the integral of θ over
  the trade life, less the hedge cost, is the trade's P&L.

For the model in this repo (Phase 5 + Phase 8), Δ, Γ, vega and ρ_pair are
all there. Vanna and volga aren't reported separately but can be derived
from cross-bumps. Smoothed-payoff Γ (Phase 8) is what makes Γ usable near
the barriers — the hard-payoff Γ from the original Phase 5 was too noisy
to drive a daily desk report.

---

## 3. Hedgeable vs not

**Cleanly hedgeable (mature, deep markets).**

- Δ per name — short the underlying stock.
- Vega level per name — listed options on each underlying.
- Funding / interest-rate exposure on the ZCB component — IR swap.
- Dividend exposure — known-dividend curve plus dividend swaps on each
  name if liquid.

**Hedgeable with friction (real, but you bleed bid-ask).**

- Vega skew per name — needs strike-specific listed options, sized off a
  smile model, re-rolled when the FCN's vega ladder shifts.
- Term-structure vega — calendar spreads on listed options.
- Vanna — listed options at the right strike (typically near the autocall
  or KI strike).
- Volga — quadratic in vol; partially captured by holding a strip of
  out-of-the-money options.

**Not cleanly hedgeable (reserved against rather than hedged).**

- **Pairwise correlation** between the three names. There is no single-stock
  correlation product. The closest macro hedges are *dispersion* trades
  (long single-stock vol vs short index vol on a related index — for our
  basket, SOXX or SMH). Dispersion trades the *average* correlation of the
  index basket, not the specific 3-name pairwise structure in the FCN, so
  the hedge is approximate and capital-intensive. Most desks reserve
  capital against correlation risk and don't try to dispersion-hedge a
  small structured-product book.
- **Gap risk.** The Δ hedge is *continuous* in the model; the world is
  not. An overnight (or weekend, worse) move that jumps a name through
  the KI or autocall level isn't hedged. The standard mitigant is a
  **barrier shift**: for hedging purposes the desk treats the effective
  KI as slightly higher than the contractual KI (e.g. 0.71 instead of
  0.70), which builds in a small safety buffer at the cost of a Δ-hedge
  bias when the worst-of is far from the barrier. The convention is
  documented in model approval and doesn't appear in the marketing
  material.
- **Model risk.** The Black-Scholes-style GBM model in the pricer assumes
  log-normal returns, constant vol per name, constant correlation. None
  of these hold under stress. The smile, the term structure, and the
  correlation matrix all move — and they move *together*, in
  hard-to-anticipate ways during a crisis. Desks reserve capital against
  model risk and tighten bid-asks during regime shifts.

---

## 4. Practical issues

**Barrier discontinuities.** §2 covers this with the smoothed-payoff Γ
trick. The desk-level consequence is that hedge intensity *changes* as the
basket approaches a barrier — the Δ-rebalancing schedule tightens, the
vega ladder gets re-marked more often, and the trader keeps a closer eye
on the order flow in the worst-performing name. The "barrier event"
itself, when it happens, is usually a non-event from a hedging perspective
*if* the desk has been re-balancing properly; it becomes an event when the
move is fast enough to skip the rebalancing window.

**Smile-aware vega.** A flat-vol vega number understates the real
exposure. The FCN's effective vega is concentrated in the *downside* part
of each name's smile (the KI is a put-side phenomenon at 70% of spot). A
listed ATM call hedges *level* vega; a listed 90% put hedges the relevant
*skew* vega. Real desks build a per-name vega-by-strike ladder and hedge
the dominant points, accepting that the diagonal entries (single-strike
single-tenor vegas) are what matters and the off-diagonals are residual.
This is the level of detail the pricer in this repo *doesn't* implement
(flat per-name vol, no smile) — and it's the most important thing on the
practitioner-extension list.

**Correlation breakdown in stress.** The pricer assumes a constant
correlation matrix. Empirically, equity correlations are *higher* in
sell-offs ("correlation 1 in a crisis") and lower in calm regimes. The
FCN holder is long correlation (notebook 09 confirms this with positive
cega); the issuer is short correlation. In a sell-off the holder's
correlation P&L moves *in his favour* while everything else (Δ, vega) is
moving against him — a partial natural offset. The desk knows this and
reserves accordingly. Hedging the dynamic correlation exposure is
beyond what a single 3-name book justifies.

---

## 5. What the bid-ask spread compensates for

The desk doesn't make money on the FCN by predicting where the basket
goes. It makes money by collecting *more* in theta and bid-ask spread
than it spends on hedging and reserves. Concretely, the spread at issuance
has to cover:

- **Capital reserves** against unhedgeable risks: correlation, gap, model.
  These are sized off the bank's economic capital model — typically a
  multiple of the tail of the unhedgeable-risk distribution.
- **Slippage** on the daily Δ hedge: bid-ask on each name, market impact
  when rebalancing in size, financing costs on the stock loan for short
  positions (rare for this product but possible).
- **Vega hedge premium decay**: the listed options the desk buys to hedge
  vega cost premium that decays with time; the desk has to earn this
  back from the FCN's theta.
- **Cost of running the model**: salaries, data, infra, model approval,
  legal, sales (this is what gets called "structuring margin" on the
  trade ticket).
- **Issuance buffer**: a margin of safety on top of all of the above, so
  that if the realised hedge cost comes in higher than the modelled hedge
  cost (it usually does, by a little), the trade still makes money.

For a textbook 6M worst-of FCN on liquid US single names, the spread at
issuance is typically 1–3% of notional. At the default 1.0%-per-period
coupon (12% p.a.) the model settles to 49,493.82 / 50,000 = 98.99% of
par — a ~1.0% gap to par that the desk's day-1 margin has to cover
along with the risk components below. If the trade runs cleanly the
desk earns most of it; if it runs into a real gap or a correlation
shock, much of it goes back out the door covering the reserves it was
sized to compensate for.

That asymmetric P&L profile — a fixed upside (the structuring margin) and
a stochastic downside (the realised hedge cost minus the modelled hedge
cost) — is *the* job of a structured-products desk. The job of the pricer
is to estimate the modelled hedge cost honestly enough that the downside
stays inside the upside. This pricer is one model among many a real bank
would run; the honest findings in the README and the unhedgeable-risk
summary in notebook 09 are deliberately included to show I know what this
model *doesn't* tell me.
