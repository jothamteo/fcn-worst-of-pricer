# Worst-of FCN Pricer (NVDA / AMD / TSM)

> **Disclaimer.** The structure priced in this repository is a **hypothetical, generic Fixed Coupon Note** built for pedagogical purposes. It is not, and is not intended to resemble, any live or recent commercial issuance. All structural parameters (coupon, barriers, tenor, observation frequency) are round textbook numbers drawn from Bouzoubaa & Osseiran, *Exotic Options and Hybrids* (Wiley, 2010), Ch. 12. All market data is sourced from public APIs (yfinance) only.

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

Variance reduction in MC: antithetic variates + a worst-of European put control variate.

## Key results

*Filled in at the end of Phase 4 / Phase 7.*

## Honest findings

*Filled in at the end of Phase 6 / Phase 7. The interesting bits are likely:*

- *Bump-and-revalue gamma near the knock-in barrier — MC vs PDE.*
- *Correlation sensitivity of the worst-of FCN (long correlation).*
- *Limitations: flat-vol assumption, constant correlation assumption.*

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
