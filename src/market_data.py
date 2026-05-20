"""Market data loaders for the worst-of FCN pricer.

Pulls everything from public sources via yfinance:
    - 5Y daily closes for each ticker (used for realised vols and correlations)
    - Implied vol surfaces from current options chains (with hardcoded fallback)
    - Continuous dividend yields from yfinance fundamentals (with hardcoded fallback)
    - 1Y risk-free rate from US Treasury yield series (with hardcoded fallback)

Everything is cached in a `MarketData` dataclass so the rest of the pricer can
just take `MarketData` as input.

The "fallback" pattern matters here because yfinance is flaky outside US trading
hours and on weekends — the desk would never accept a pricer that crashes when
the data feed has a hiccup, so the loaders log what fell back and keep going.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Hardcoded fallbacks (date-stamped — refresh annually, or when yfinance is down)
# --------------------------------------------------------------------------------------

# Source for fallbacks: yfinance Ticker.info snapshot taken 2026-05.
# These are deliberately round textbook-quality numbers, not last-trade snapshots.
FALLBACK_VOLS: dict[str, float] = {
    "NVDA": 0.45,
    "AMD": 0.50,
    "TSM": 0.32,
    "AMZN": 0.32,
    "META": 0.38,
    "MU": 0.55,
}

# Continuous dividend yields (decimal). yfinance gives a trailing yield; we use it
# as-is with a hardcoded backup. AMD pays no dividend.
FALLBACK_DIVS: dict[str, float] = {
    "NVDA": 0.0003,
    "AMD": 0.0,
    "TSM": 0.015,
    "AMZN": 0.0,
    "META": 0.0035,
    "MU": 0.005,
}

# 1Y US Treasury yield as of the date stamp — used only when yfinance ^IRX/^FVX both fail.
FALLBACK_RATE: float = 0.043
FALLBACK_RATE_DATE: str = "2026-05-01"


# --------------------------------------------------------------------------------------
# Dataclass
# --------------------------------------------------------------------------------------


@dataclass
class MarketData:
    """Container for everything the pricer needs about the market.

    Attributes
    ----------
    tickers : sequence of str
        The basket. Order is preserved everywhere (vols, divs, spots, corr).
    spots : np.ndarray of shape (n_assets,)
        Latest close.
    vols : np.ndarray of shape (n_assets,)
        Annualised volatility per name. Implied where available, else realised, else
        hardcoded fallback.
    divs : np.ndarray of shape (n_assets,)
        Continuous dividend yield per name.
    corr : np.ndarray of shape (n_assets, n_assets)
        Realised log-return correlation matrix.
    rate : float
        1Y risk-free rate (continuous compounding).
    history : pd.DataFrame
        Daily closes used for realised stats.
    sources : dict[str, str]
        Free-form provenance string per field, written into the notebook output.
    as_of : datetime
        UTC timestamp at which the data was pulled.
    """

    tickers: Sequence[str]
    spots: np.ndarray
    vols: np.ndarray
    divs: np.ndarray
    corr: np.ndarray
    rate: float
    history: pd.DataFrame
    sources: dict[str, str] = field(default_factory=dict)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        """Pretty-print all market inputs with sources."""
        lines = [
            f"MarketData snapshot as of {self.as_of.isoformat()}",
            f"Tickers      : {list(self.tickers)}",
            f"Spots        : {dict(zip(self.tickers, np.round(self.spots, 4)))}",
            f"Vols (ann.)  : {dict(zip(self.tickers, np.round(self.vols, 4)))}",
            f"Div yields   : {dict(zip(self.tickers, np.round(self.divs, 5)))}",
            f"Risk-free 1Y : {round(self.rate, 5)}",
            "Correlation matrix:",
        ]
        corr_df = pd.DataFrame(self.corr, index=self.tickers, columns=self.tickers)
        lines.append(corr_df.round(4).to_string())
        if self.sources:
            lines.append("Sources:")
            for k, v in self.sources.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# History + realised stats
# --------------------------------------------------------------------------------------


def _download_history(
    tickers: Sequence[str],
    lookback_years: int = 5,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Pull daily Close history for the basket. Returns a wide DataFrame.

    When `end` is supplied, history is restricted to dates strictly on or before
    `end` so realised stats are computed without look-ahead bias.
    """
    if end is None:
        raw = yf.download(
            list(tickers),
            period=f"{lookback_years}y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    else:
        start = end - timedelta(days=int(round(lookback_years * 365.25)))
        # yfinance `end` is exclusive — add a day so the as-of close itself is included.
        raw = yf.download(
            list(tickers),
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no history for "
            f"{tickers}. Check connectivity or pass an explicit history DataFrame."
        )
    # yfinance returns a multi-indexed frame for >1 ticker; normalise to wide closes.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = pd.concat(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
            axis=1,
        )
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.dropna(how="all").ffill().dropna()
    if end is not None:
        # Belt-and-braces: drop any rows past the as-of date in case yfinance
        # returned an extra bar (timezones, weekend handling).
        end_ts = pd.Timestamp(end)
        if closes.index.tz is not None:
            end_ts = end_ts.tz_localize(closes.index.tz)
        closes = closes.loc[closes.index <= end_ts]
    if closes.empty:
        raise RuntimeError("Downloaded history is empty after cleaning.")
    return closes


def realised_stats(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualised realised vols and correlation matrix from daily log-returns."""
    log_rets = np.log(history / history.shift(1)).dropna()
    daily_vol = log_rets.std(ddof=1)
    ann_vol = (daily_vol * math.sqrt(252)).to_numpy()
    corr = log_rets.corr().to_numpy()
    return ann_vol, corr


# --------------------------------------------------------------------------------------
# Implied vol calibration (per-name ATM, with skew documented as flat)
# --------------------------------------------------------------------------------------


def _bs_implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    div: float,
    T: float,
    option_type: str = "call",
) -> Optional[float]:
    """Invert Black-Scholes for implied vol via Brent. Returns None on failure."""

    def _bs_price(vol: float) -> float:
        if vol <= 0 or T <= 0:
            return float("inf")
        sigma_root_t = vol * math.sqrt(T)
        d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * T) / sigma_root_t
        d2 = d1 - sigma_root_t
        if option_type == "call":
            return (
                spot * math.exp(-div * T) * norm.cdf(d1)
                - strike * math.exp(-rate * T) * norm.cdf(d2)
            )
        return (
            strike * math.exp(-rate * T) * norm.cdf(-d2)
            - spot * math.exp(-div * T) * norm.cdf(-d1)
        )

    try:
        return brentq(lambda v: _bs_price(v) - market_price, 1e-4, 5.0, maxiter=100, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


def _quote_mid(row: pd.Series) -> Optional[float]:
    """Bid/ask midpoint with a sane-quote check; returns None if quotes are bad."""
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    # Reject silly-wide spreads (>50% of mid) — that's a stale quote, not a market.
    mid = 0.5 * (bid + ask)
    if (ask - bid) / mid > 0.5:
        return None
    return mid


def _atm_implied_vol(
    ticker: str,
    spot: float,
    rate: float,
    div: float,
    target_T: float = 1.0,
    sane_range: tuple[float, float] = (0.10, 1.50),
) -> Optional[float]:
    """ATM implied vol at the maturity nearest `target_T` years, calibrated from the chain.

    Strategy:
      1. Pick the expiry whose tenor is closest to `target_T`.
      2. Take the 5 strikes nearest spot, calls + puts.
      3. For each row, compute IV from bid/ask mid (rejecting stale/illiquid rows).
      4. If our own inversion is empty, fall back to yfinance's `impliedVolatility` column.
      5. Median across the surviving IVs.
      6. Apply `sane_range` as a final guard — implied vols outside (0.10, 1.50) on a
         3-name liquid US tech basket are almost certainly bad data, not real signal.
    """
    yf_t = yf.Ticker(ticker)
    expiries = list(yf_t.options or [])
    if not expiries:
        return None
    today = datetime.now(timezone.utc).date()

    def _years_to(exp: str) -> float:
        return max((datetime.strptime(exp, "%Y-%m-%d").date() - today).days / 365.0, 1e-4)

    # Prefer expiries with at least 3 months tenor — short-dated chains have wild IVs.
    candidates = sorted(
        (e for e in expiries if _years_to(e) >= 0.25),
        key=lambda e: abs(_years_to(e) - target_T),
    )
    if not candidates:
        candidates = sorted(expiries, key=lambda e: abs(_years_to(e) - target_T))

    for expiry in candidates[:3]:
        T = _years_to(expiry)
        try:
            chain = yf_t.option_chain(expiry)
        except Exception as exc:
            logger.warning("Options chain fetch failed for %s @ %s: %s", ticker, expiry, exc)
            continue

        ivs_inv: list[float] = []
        ivs_yf: list[float] = []
        for kind, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            # 5 strikes nearest ATM, with non-trivial open interest.
            f = frame.dropna(subset=["strike"]).copy()
            if "openInterest" in f.columns:
                f = f[f["openInterest"].fillna(0) >= 1]
            if f.empty:
                continue
            f["dist"] = (f["strike"] - spot).abs()
            f = f.sort_values("dist").head(5)
            for _, row in f.iterrows():
                mid = _quote_mid(row)
                if mid is not None:
                    iv_inv = _bs_implied_vol(
                        market_price=mid,
                        spot=spot,
                        strike=float(row["strike"]),
                        rate=rate,
                        div=div,
                        T=T,
                        option_type=kind,
                    )
                    if iv_inv is not None and sane_range[0] <= iv_inv <= sane_range[1]:
                        ivs_inv.append(iv_inv)
                yf_iv = row.get("impliedVolatility")
                if yf_iv is not None and not pd.isna(yf_iv):
                    yf_iv = float(yf_iv)
                    if sane_range[0] <= yf_iv <= sane_range[1]:
                        ivs_yf.append(yf_iv)

        # Prefer our own bid/ask-mid inversion when we have ≥4 surviving rows;
        # otherwise fall back to yfinance's reported IV column.
        if len(ivs_inv) >= 4:
            return float(np.median(ivs_inv))
        if ivs_yf:
            return float(np.median(ivs_yf))
        if ivs_inv:
            return float(np.median(ivs_inv))
    return None


# --------------------------------------------------------------------------------------
# Dividend yield + risk-free rate
# --------------------------------------------------------------------------------------


def _dividend_yield(ticker: str) -> Optional[float]:
    """Trailing dividend yield from yfinance, decimal form.

    yfinance's `dividendYield` field is unreliable (it's been flipped between
    fraction and percent multiple times, and for low-yielders it sometimes
    returns silly values). We compute it ourselves from
    `dividendRate / regularMarketPrice`, which is well-defined, and only fall
    back to the `dividendYield` field if the rate or price is missing.
    Sane-range guard rejects anything > 10% as bad data.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Ticker.info failed for %s: %s", ticker, exc)
        return None

    rate = info.get("dividendRate")
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    if rate is not None and price is not None and float(price) > 0:
        y = float(rate) / float(price)
        if 0 <= y <= 0.10:
            return y

    raw = info.get("dividendYield")
    if raw is None:
        return None
    val = float(raw)
    if val > 1.0:
        val /= 100.0
    if 0 <= val <= 0.10:
        return val
    return None


def _trailing_dividend_yield(
    ticker: str,
    as_of: date,
    spot: float,
    window_days: int = 365,
) -> Optional[float]:
    """Trailing 12M dividend yield at `as_of`, decimal form.

    Sum of cash dividends with ex-date in `(as_of - window_days, as_of]`,
    divided by spot. Returns None when the dividend series is empty
    (and the caller should fall through to the hardcoded fallback).
    """
    try:
        divs = yf.Ticker(ticker).dividends
    except Exception as exc:
        logger.warning("Ticker.dividends failed for %s: %s", ticker, exc)
        return None
    if divs is None or divs.empty:
        return None
    if getattr(divs.index, "tz", None) is not None:
        divs = divs.tz_convert(None)
    end_ts = pd.Timestamp(as_of)
    start_ts = end_ts - pd.Timedelta(days=window_days)
    window = divs.loc[(divs.index > start_ts) & (divs.index <= end_ts)]
    if window.empty:
        return 0.0
    if spot <= 0:
        return None
    y = float(window.sum()) / spot
    return y if 0.0 <= y <= 0.20 else None


def _risk_free_rate(
    as_of: Optional[date] = None,
    target_T: float = 1.0,
) -> tuple[float, str]:
    """US Treasury yield interpolated to `target_T` years (decimal). Returns
    (rate, source_string).

    When `as_of` is None, the most recent close of each tenor is used. When
    `as_of` is supplied, the close on or immediately before `as_of` is used,
    so historical pricing has no look-ahead.
    """
    candidates = [("^IRX", 13 / 52), ("^FVX", 5.0)]
    yields: list[tuple[float, float]] = []
    for symbol, tenor_y in candidates:
        try:
            if as_of is None:
                hist = yf.Ticker(symbol).history(period="5d", interval="1d")
            else:
                start = as_of - timedelta(days=10)
                hist = yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(as_of + timedelta(days=1)).isoformat(),
                    interval="1d",
                )
        except Exception as exc:
            logger.warning("Treasury fetch failed for %s: %s", symbol, exc)
            continue
        if hist is None or hist.empty:
            continue
        last = float(hist["Close"].dropna().iloc[-1]) / 100.0
        yields.append((tenor_y, last))
    label_suffix = "" if as_of is None else f" (as_of {as_of.isoformat()})"
    if len(yields) >= 2:
        # Linear interpolate to T=target_T.
        yields.sort()
        t0, y0 = yields[0]
        t1, y1 = yields[1]
        if t0 == t1:
            return y0, f"yfinance {candidates[0][0]} (single point @ {t0:.2f}y){label_suffix}"
        rate = y0 + (y1 - y0) * (target_T - t0) / (t1 - t0)
        return rate, (
            f"yfinance interpolated {candidates[0][0]}/{candidates[1][0]} -> "
            f"{target_T:.2f}Y{label_suffix}"
        )
    if len(yields) == 1:
        return yields[0][1], f"yfinance single tenor @ {yields[0][0]:.2f}y{label_suffix}"
    return FALLBACK_RATE, f"hardcoded fallback (date {FALLBACK_RATE_DATE}){label_suffix}"


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def load_market_data(
    tickers: Sequence[str] = ("AMZN", "META", "MU"),
    lookback_years: int = 5,
    target_T: float = 1.0,
    as_of: Optional[Union[date, str]] = None,
) -> MarketData:
    """Pull a full MarketData snapshot for the basket.

    Parameters
    ----------
    tickers : sequence of str
        Basket order is preserved.
    lookback_years : int
        Years of daily history used for realised stats.
    target_T : float
        Target option tenor for ATM IV calibration *and* the risk-free rate
        interpolation target. Defaults to 1Y.
    as_of : date | str, optional
        When supplied, pricing inputs are taken as of this historical date:
          - history is restricted to dates ≤ `as_of` (no look-ahead),
          - spot is the last close on or before `as_of`,
          - vol is realised (yfinance does not expose historical IV chains),
          - dividend yield is the trailing 12M sum / spot,
          - risk-free rate is the Treasury yield curve on `as_of`.
        When None, the current snapshot is used (live IV calibration is run).

    Returns
    -------
    MarketData
    """
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of).date()

    history = _download_history(tickers, lookback_years=lookback_years, end=as_of)
    realised_vols, corr = realised_stats(history)

    spots = history.iloc[-1].reindex(tickers).to_numpy(dtype=float)
    rate, rate_src = _risk_free_rate(as_of=as_of, target_T=target_T)

    divs = np.empty(len(tickers))
    div_sources: list[str] = []
    for i, t in enumerate(tickers):
        if as_of is None:
            d = _dividend_yield(t)
            src_label = "yfinance Ticker.info"
        else:
            d = _trailing_dividend_yield(t, as_of=as_of, spot=float(spots[i]))
            src_label = f"trailing 12M dividends / spot @ {as_of.isoformat()}"
        if d is None:
            divs[i] = FALLBACK_DIVS.get(t, 0.0)
            div_sources.append(f"{t}: hardcoded fallback ({divs[i]:.4f})")
        else:
            divs[i] = d
            div_sources.append(f"{t}: {src_label} ({d:.4f})")

    vols = np.empty(len(tickers))
    vol_sources: list[str] = []
    for i, t in enumerate(tickers):
        # Historical IV chains are not available via yfinance, so for as_of mode
        # we skip the live IV calibration and use realised vol directly.
        iv: Optional[float] = None
        if as_of is None:
            iv = _atm_implied_vol(
                t, spot=float(spots[i]), rate=rate, div=float(divs[i]), target_T=target_T
            )
        if iv is not None:
            vols[i] = iv
            vol_sources.append(f"{t}: yfinance ATM IV ({iv:.4f})")
        else:
            rv = float(realised_vols[i])
            if rv > 0:
                vols[i] = rv
                if as_of is None:
                    vol_sources.append(f"{t}: realised {lookback_years}Y vol ({rv:.4f}) — IV chain empty")
                else:
                    vol_sources.append(
                        f"{t}: realised {lookback_years}Y vol ({rv:.4f}) @ {as_of.isoformat()}"
                    )
            else:
                vols[i] = FALLBACK_VOLS.get(t, 0.30)
                vol_sources.append(f"{t}: hardcoded fallback ({vols[i]:.4f})")

    as_of_label = "live" if as_of is None else as_of.isoformat()
    sources = {
        "as_of": as_of_label,
        "spots": f"yfinance daily closes (last bar on or before {as_of_label})",
        "history": f"yfinance {lookback_years}Y daily closes ending {as_of_label}",
        "realised_vols": "log-return std × √252",
        "implied_vols": "; ".join(vol_sources),
        "dividends": "; ".join(div_sources),
        "rate": rate_src,
        "correlations": "Pearson on daily log-returns",
        "skew_assumption": "FLAT — ATM vol used at all strikes; documented limitation",
    }

    return MarketData(
        tickers=list(tickers),
        spots=spots,
        vols=vols,
        divs=divs,
        corr=corr,
        rate=rate,
        history=history,
        sources=sources,
    )
