# Data

All market data in this project comes from public sources, fetched at runtime — nothing proprietary, nothing paid.

## Sources

| Input | Source | Frequency | Notes |
|---|---|---|---|
| Spot closes (5Y daily) | yfinance | daily | Used for realised vols and correlations. |
| Implied vol surfaces | yfinance options chains | live | Black-Scholes inverted per (strike, maturity); a flat-skew surface is fitted on top. Falls back to a hardcoded ATM vol per ticker when the chain is empty / illiquid. |
| Dividend yields | yfinance `Ticker.info["dividendYield"]` | live | Verified manually on first run; documented in `market_data.py` with date stamps. |
| Risk-free rate | yfinance `^IRX` (13W) or `^FVX` (5Y), interpolated to 1Y | live | Falls back to a date-stamped hardcoded value if yfinance is down. |

Cached fetches (if any) are written to `data/cache/` which is gitignored. Re-running the notebook with no cache will hit yfinance directly.

## Notes on data quality

- yfinance options chains can be sparse or stale outside US trading hours. The market-data module logs which strikes were dropped and why.
- Dividend yields on yfinance lag the most recent declaration by a few days. For a hypothetical/textbook pricer this is acceptable; a desk would override with the next ex-div schedule.
