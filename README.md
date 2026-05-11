---
title: FCN Worst-of Pricer
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
license: mit
---

# Worst-of FCN Pricer (NVDA / AMD / TSM)

> **Disclaimer.** The structure priced in this repository is a **hypothetical, generic Fixed Coupon Note** built for pedagogical purposes. It is not, and is not intended to resemble, any live or recent commercial issuance. All structural parameters (coupon, barriers, tenor, observation frequency) are round textbook numbers drawn from Bouzoubaa & Osseiran, *Exotic Options and Hybrids* (Wiley, 2010), Ch. 12. All market data is sourced from public APIs (yfinance) only.

## Live demo

Interactive dashboard (Streamlit): **https://fcn-worst-of-pricer.streamlit.app/**

Move the market sliders in the sidebar; the five panels — P&L attribution, per-name Greeks, scenario stress, hedging summary, and price-curve slice — reprice the trade through the same MC + PDE engines used in the notebooks. Cold start loads a pre-built scenario grid from `dashboard/grid_initial.npz` (no MC at startup) and runs a one-shot 10k-path MC Greeks pass for the linearisation point — typically under ~10s on Streamlit Cloud. Subsequent slider moves are sub-100ms via grid interpolation. If you change the defaults in `dashboard/state.py`, rebuild the shipped grid with `python scripts/build_initial_grid.py`.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What this is

A from-scratch Python implementation of a **worst-of Fixed Coupon Note** pricer on a 3-name semiconductor basket (NVDA, AMD, TSM). Pricing is done by Monte Carlo on correlated GBM, cross-validated against a 1D Crank–Nicolson PDE on a reduced single-asset case. Greeks are computed by bump-and-revalue with common random numbers. Everything (GBM engine, Cholesky correlation, antithetic + control variates, FCN payoff, PDE scheme) is written in NumPy — no QuantLib, no FinancePy.

## Why this project

I work in IT for a private bank's structured products desk, so I see FCNs go out the door every week, but I have never priced one myself. This repo is me sitting down with a generic textbook structure and a Quant background and putting the pricing methodology end-to-end: simulate the basket, write the payoff, discount the cashflows, validate against a PDE on the case where a PDE is feasible, and stress the Greeks the way a desk would.

The aim is a defensible, honest portfolio piece — not a glossy backtest. Where MC gets noisy near a barrier I want to *show* it getting noisy and explain why, rather than tune it away.

## The product

Generic worst-of FCN, 1Y tenor, quarterly observations:

- **Notional:** 100
- **Coupon:** 8% p.a., paid quarterly (2.0 per quarter), conditional on coupon barrier
- **Coupon barrier:** 70% of initial spot (worst-performer basis)
- **Autocall barrier:** 100% of initial spot (early redemption at par + coupon if breached on an observation date; first observation excluded by default, configurable)
- **Knock-in barrier:** 65% of initial spot, observed at maturity (European; continuous variant also implemented)
- **Maturity payoff (if not autocalled):**
  - Worst performer ≥ 100%: par + final coupon
  - Knock-in not breached: par + final coupon
  - Knock-in breached and worst < 100%: notional × (worst / initial) + final coupon

See `METHODOLOGY.md` for formal payoff notation. Reference: Bouzoubaa & Osseiran, Ch. 12.

## Pricing approach

| Engine | Use | Why |
|---|---|---|
| Monte Carlo (3-asset) | main pricer | Path-dependent, multi-underlying, discrete observations — the natural fit. |
| Crank–Nicolson PDE (1D) | cross-validation | On a single underlying the FCN is 1D, and a PDE gives smooth Greeks and a benchmark MC must agree with. |

Variance reduction in MC: antithetic variates + a worst-of European put control variate. For Greeks, the payoff's hard autocall / knock-in indicators can be replaced with sigmoids of controllable steepness — a small price bias buys substantially tighter Γ standard errors at the barriers (see notebook 08).

## Key results

AMZN / META / MU snapshot as of 17 Oct 2025 (see notebooks/01).

| | Value |
|---|---|
| **Final MC price** (160k antithetic paths + worst-of put CV) | **47,063.94 USD** (94.13% of notional) |
| MC standard error (with CV) | 11.74 |
| Same engine, no CV (antithetic only) | 47,042.65 ± 21.78 |
| Variance-reduction ratio Var(X)/Var(X_cv) | **3.44×** |
| corr(FCN PV, worst-of put PV) per path | **−0.84** |

**PDE cross-validation** (single-asset reductions, 1600 × 160 grid):

| Reduction | MC ± SE | PDE | \|Δ\| / SE |
|---|---|---|---|
| AMZN | 50,279.53 ± 11.28 | 50,273.42 | 0.54 |
| META | 49,265.50 ± 15.23 | 49,266.82 | 0.09 |
| MU   | 48,908.34 ± 16.46 | 48,911.74 | 0.21 |

All three PDE prices land well inside the MC 1-σ band — the path engine,
payoff arithmetic, observation-grid plumbing and discount accounting all
agree with an independent deterministic solver. (See notebook 04.)

**Greeks** (bump-and-revalue MC, CRN, 80k antithetic paths, 1% spot bump):

| Asset | Δ per 1% | Γ per 1%×1% | vega per vol-pt |
|---|---|---|---|
| AMZN | +47.82  | +0.26  | −54.92  |
| META | +86.59  | −7.38  | −88.93  |
| MU   | +106.18 | −0.69  | −102.60 |

Off-diagonal correlation sensitivities are positive on all three pairs
(the holder is long-correlation — see the discussion in notebook 05).
Probability decomposition: P(autocall) ≈ 44%, P(KI at maturity) ≈ 27%,
P(par at maturity, no AC) ≈ 30%.

## Honest findings

- **MC bump-and-revalue Greeks blow up at the barriers — and Phase 8
  fixes it.** Near `S/S₀ = 1.00` (autocall) and `S/S₀ = 0.70` (KI) the
  hard-payoff bump occasionally flips a path between two qualitatively
  different resolution regimes, creating a discontinuous payoff
  difference that CRN cannot smooth over. The Δ-vs-spot plot
  (notebook 05) shows the MC ribbon widening visibly at exactly those
  levels. The production-desk fix is to **replace each indicator with a
  logistic sigmoid** of controllable steepness, so the autocall becomes
  a soft event and survival probability propagates across observations.
  At `k = 100` on the textbook product, the smoothed price bias is
  ~0.04% of notional (<0.5 hard-MC SE) and the worst-case Γ standard
  error is **~8× tighter** than the hard estimator. The Γ-vs-spot scan
  (notebook 08) lands inside a tight band on the PDE benchmark across
  the full spot range.

- **The worst-of put control variate exploits exactly the loss tail.**
  CV correlation reaches −0.84, giving 3.4× variance reduction
  (≈ 1.85× tighter SE for the same path budget). The mechanism is
  visible in the per-path X-vs-Y scatter (notebook 06): the
  autocalled and par-at-maturity paths sit at Y = 0; the
  knocked-in paths form a tight anti-correlated cloud that the CV
  effectively averages out.

- **Correlation sensitivity is positive on every pair.** The FCN
  holder is long correlation — decorrelated names raise the
  probability of one name dragging the worst-of through the KI
  barrier. Standard worst-of structure, signs match desk intuition.

- **Limitations.** Flat per-name vol (no smile, no term structure),
  constant correlation matrix, European KI only on the PDE side
  (continuous-KI is supported in MC but would need an absorbing
  boundary along the strike in the PDE — not implemented; see
  METHODOLOGY §4). The Phase 5 "MC noise at the barriers" plots are
  deliberately not tuned away — Phase 8 (notebook 08) then closes the
  gap with the smoothed-payoff fix.

## How to run

```bash
git clone https://github.com/jothamteo/fcn-worst-of-pricer
cd fcn-worst-of-pricer

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite
pytest -q

# Run the notebooks in order
jupyter lab notebooks/
```

## Repo layout

```
fcn-worst-of-pricer/
├── README.md            # this file
├── METHODOLOGY.md       # math walkthrough (LaTeX)
├── requirements.txt
├── src/
│   ├── market_data.py   # spot, vol, div, corr loaders
│   ├── gbm_simulation.py
│   ├── fcn_payoff.py
│   ├── mc_pricer.py
│   ├── pde_pricer.py
│   ├── greeks.py
│   └── utils.py
├── tests/
├── notebooks/
└── data/
```

## References

- Bouzoubaa, M. & Osseiran, A. (2010). *Exotic Options and Hybrids*. Wiley. — FCN structures, Ch. 12.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. — variance reduction, pathwise Greeks.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*. Wiley. — finite-difference schemes for exotics.
