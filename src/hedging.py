r"""Hedging ratios for the worst-of FCN — turning Greeks into trader actions.

This module sits *on top of* the existing pricer: it consumes the Greeks the
desk has already computed (notebook 05) and translates them into the concrete
things the trader has to do — shares to short, listed options to buy, the
correlation P&L the desk has to wear.

**Issuer / dealer perspective.** Throughout this module the bank has just
*sold* the FCN to a private-banking client. The bank's book is therefore the
short side of the FCN; "the Greeks" passed into these functions are the
*holder's* Greeks (matching what `src.greeks.mc_greeks_bump` returns), and the
hedge translates those into the dealer-side action. Sign conventions are
documented per-function and called out in :class:`HedgeReport`.

This is deliberately a thin layer: there is no model recalibration, no smile
construction, no listed-option chain reconciliation. The intent is to make
the path from "Phase 5 Greeks table" to "what the trader actually does on
day 1" explicit and reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaHedgeRow:
    """Per-ticker output of :func:`compute_delta_hedge`.

    `shares_to_buy` carries the sign: positive → BUY shares (dealer is short
    delta on the trade and needs to be long the underlying); negative → SHORT
    shares. The field name `shares_to_short` is also provided as an alias to
    match the addendum spec — it equals ``-shares_to_buy``.
    """

    ticker: str
    delta: float
    spot: float
    shares_to_buy: int
    hedge_notional: float
    hedge_pct_of_trade: float

    @property
    def shares_to_short(self) -> int:
        """Spec-named alias: negative of `shares_to_buy`.

        Positive `shares_to_short` means the dealer literally short-sells the
        stock to hedge (i.e. the FCN delta is negative, which doesn't happen
        for a standard worst-of FCN but is supported for generality).
        """
        return -self.shares_to_buy


@dataclass(frozen=True)
class VegaHedgeRow:
    """Per-ticker output of :func:`compute_vega_hedge`."""

    ticker: str
    fcn_vega: float                  # USD P&L on the *dealer's* book per +1 vol pt
    listed_option_vega: float        # USD vega of one listed option (positive)
    listed_option_price: float
    n_options_to_buy: int            # +ve → BUY listed options; -ve → sell
    total_premium: float             # signed: +ve means dealer pays premium
    residual_vega: float             # post-hedge residual on the dealer's book


@dataclass(frozen=True)
class CorrelationRiskReport:
    """Output of :func:`compute_correlation_risk`."""

    cega: float                       # USD P&L per +0.01 in pairwise correlation
    correlation_pnl_per_5pct_move: float
    base_price: float
    hedgeable: bool
    narrative: str
    samples: dict[float, float] = field(default_factory=dict)


@dataclass(frozen=True)
class UnhedgeableRiskSummary:
    """Composed view of the residual risks the desk wears.

    `rows` is a list of dicts (one per risk), formatted for table display.
    """

    rows: list[dict]
    notional: float

    def as_dict(self) -> dict:
        return {"rows": self.rows, "notional": self.notional}


# ---------------------------------------------------------------------------
# 1) Delta hedge
# ---------------------------------------------------------------------------


def compute_delta_hedge(
    deltas: dict,
    notional: float,
    spots: dict,
    *,
    pricer_reference_notional: Optional[float] = None,
) -> dict:
    r"""Translate FCN deltas into a per-ticker share hedge for a sold FCN.

    Parameters
    ----------
    deltas : dict[str, float]
        Per-ticker holder-side delta from the pricer — i.e.
        :math:`\Delta_i = \partial V_{\text{FCN}} / \partial S_i` in *raw*
        units (USD per USD of spot), as produced by
        :class:`src.greeks.GreekResult.delta`.
    notional : float
        Trade notional in USD (e.g. ``1_000_000`` for $1M sold).
    spots : dict[str, float]
        Per-ticker current spot price.
    pricer_reference_notional : float, optional
        The notional the pricer used when it produced ``deltas`` (e.g. the
        test fixture uses 50,000). If given, deltas are linearly rescaled to
        the trade notional via ``notional / pricer_reference_notional``. If
        omitted, deltas are assumed to already be at ``notional``.

    Sign convention
    ---------------
    The bank has *sold* the FCN. Holder delta is +ve for a standard worst-of
    FCN (price rises with the basket). The dealer is therefore short delta on
    the trade and must hold +Δ shares to neutralise:

    * ``shares_to_buy`` > 0  →  the dealer **buys** that many shares.
    * ``shares_to_buy`` < 0  →  the dealer **short-sells** that many shares.

    ``DeltaHedgeRow.shares_to_short`` is provided as the spec's alias and
    equals ``-shares_to_buy`` (rarely positive for a standard worst-of FCN).

    Returns
    -------
    dict[str, DeltaHedgeRow]
        One row per ticker.
    """
    if notional <= 0:
        raise ValueError(f"notional must be > 0, got {notional}")
    if set(deltas.keys()) != set(spots.keys()):
        raise ValueError(
            f"deltas and spots must cover the same tickers; got "
            f"{sorted(deltas.keys())} vs {sorted(spots.keys())}"
        )

    if pricer_reference_notional is not None:
        if pricer_reference_notional <= 0:
            raise ValueError("pricer_reference_notional must be > 0")
        scale = notional / pricer_reference_notional
    else:
        scale = 1.0

    out: dict[str, DeltaHedgeRow] = {}
    for ticker, delta_raw in deltas.items():
        delta_scaled = float(delta_raw) * scale
        # shares_to_buy is the dealer's required long-stock position to
        # offset the short-FCN book delta. Round to whole shares.
        shares_to_buy = int(round(delta_scaled))
        hedge_notional = shares_to_buy * float(spots[ticker])
        hedge_pct = 100.0 * hedge_notional / notional
        out[ticker] = DeltaHedgeRow(
            ticker=ticker,
            delta=delta_scaled,
            spot=float(spots[ticker]),
            shares_to_buy=shares_to_buy,
            hedge_notional=hedge_notional,
            hedge_pct_of_trade=hedge_pct,
        )
    return out


# ---------------------------------------------------------------------------
# 2) Vega hedge
# ---------------------------------------------------------------------------


def black_scholes_call(
    spot: float, strike: float, T: float, vol: float, rate: float, div: float = 0.0,
) -> tuple[float, float]:
    """Black–Scholes ATM call price + vega.

    Returns ``(price, vega_per_1_vol_pt)`` where vega is the USD P&L impact
    of a +1 vol-point (i.e. +0.01) move in implied vol. Used as a stand-in
    for listed-option vendors when the project is run without a chain feed.
    """
    if T <= 0 or vol <= 0:
        raise ValueError("T and vol must be > 0")
    sqrtT = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    price = spot * math.exp(-div * T) * norm.cdf(d1) - strike * math.exp(-rate * T) * norm.cdf(d2)
    vega_raw = spot * math.exp(-div * T) * sqrtT * norm.pdf(d1)
    # Convert "vega per 1 unit of vol" to "vega per 1 vol pt" (per 0.01).
    return float(price), float(vega_raw / 100.0)


def compute_vega_hedge(
    vegas: dict,
    listed_option_vegas: dict,
    listed_option_prices: Optional[dict] = None,
) -> dict:
    r"""Translate per-name FCN vega into a listed-option hedge.

    Parameters
    ----------
    vegas : dict[str, float]
        Per-ticker vega on the *dealer's* book per +1 vol-pt rise. For a
        standard worst-of FCN, the holder is short vol on each name, so the
        dealer's vega is positive: vol up → dealer gains. Pass in the dealer
        view (i.e. ``-holder_vega_per_volpt``).
    listed_option_vegas : dict[str, float]
        Vega of a single listed vanilla option per ticker (positive, USD per
        +1 vol-pt). The natural hedge instrument is the ATM 3-month listed
        call on the underlying — see :func:`black_scholes_call` if you don't
        have a live chain.
    listed_option_prices : dict[str, float], optional
        Listed option price per ticker. If supplied, the hedge cost
        (``total_premium``) is reported; otherwise it's NaN.

    Sign convention
    ---------------
    The dealer carries +ve vega from selling the FCN (he gains on vol up). To
    neutralise, he must **sell** listed options (negative vega):

    * ``n_options_to_buy`` < 0  →  the dealer **short-sells** options.
    * ``n_options_to_buy`` > 0  →  the dealer **buys** options (would happen
      only if the FCN's vega sign flips — atypical).
    * ``total_premium`` is signed: positive means the dealer pays premium.

    Returns
    -------
    dict[str, VegaHedgeRow]
    """
    if set(vegas.keys()) != set(listed_option_vegas.keys()):
        raise ValueError("vegas and listed_option_vegas must share tickers")
    prices = listed_option_prices or {}
    out: dict[str, VegaHedgeRow] = {}
    for ticker, dealer_vega in vegas.items():
        opt_vega = float(listed_option_vegas[ticker])
        if opt_vega == 0:
            raise ValueError(f"listed_option_vegas[{ticker!r}] must be non-zero")
        n_raw = -float(dealer_vega) / opt_vega
        n_int = int(round(n_raw))
        residual = float(dealer_vega) + n_int * opt_vega
        opt_price = float(prices.get(ticker, float("nan")))
        total_premium = n_int * opt_price if not math.isnan(opt_price) else float("nan")
        out[ticker] = VegaHedgeRow(
            ticker=ticker,
            fcn_vega=float(dealer_vega),
            listed_option_vega=opt_vega,
            listed_option_price=opt_price,
            n_options_to_buy=n_int,
            total_premium=total_premium,
            residual_vega=residual,
        )
    return out


# ---------------------------------------------------------------------------
# 3) Correlation risk
# ---------------------------------------------------------------------------


def compute_correlation_risk(
    price_at_corr_levels: dict,
    *,
    base_corr_shift: float = 0.0,
) -> CorrelationRiskReport:
    r"""Compose a desk-level read on correlation risk from a price scan.

    Parameters
    ----------
    price_at_corr_levels : dict[float, float]
        Mapping of (signed) shift in pairwise correlation from base → FCN
        price at that bumped correlation matrix. Standard usage is to pass
        in :math:`\{-0.10, -0.05, 0.0, +0.05, +0.10\}` (or any subset
        containing both a negative and a positive shift).
    base_corr_shift : float
        Where the "base" sits in the input dict. Defaults to 0.0.

    Computation
    -----------
    * ``cega`` is a central difference around ``base_corr_shift`` using the
      closest available symmetric pair of shifts.
    * ``correlation_pnl_per_5pct_move`` is reported as the linearised P&L
      from cega, i.e. ``cega × 5`` (units: USD per +5 pp of pairwise corr).

    `hedgeable` is hard-coded ``False`` — there is no liquid single-stock
    correlation product for a generic 3-name basket. See HEDGING_NOTES.md
    for the (limited) macro-hedge palette desks actually use.

    Returns
    -------
    CorrelationRiskReport
    """
    shifts = sorted(price_at_corr_levels.keys())
    if len(shifts) < 2:
        raise ValueError("need at least 2 correlation levels for a finite difference")
    base = float(price_at_corr_levels.get(base_corr_shift,
                                          price_at_corr_levels[min(shifts, key=abs)]))

    # Pick the smallest symmetric (positive, negative) pair around base_corr_shift.
    pos_shifts = [s for s in shifts if s - base_corr_shift > 1e-9]
    neg_shifts = [s for s in shifts if base_corr_shift - s > 1e-9]
    if not pos_shifts or not neg_shifts:
        raise ValueError(
            "need at least one positive and one negative shift around base"
        )
    h_pos = min(pos_shifts, key=lambda s: abs(s - base_corr_shift))
    h_neg = max(neg_shifts, key=lambda s: abs(s - base_corr_shift))
    dx = h_pos - h_neg
    dP = price_at_corr_levels[h_pos] - price_at_corr_levels[h_neg]
    # cega is per +0.01 in pairwise correlation.
    cega = (dP / dx) * 0.01
    pnl_5pct = cega * 5.0

    direction = "long" if cega > 0 else "short"
    narrative = (
        f"The holder is {direction} correlation. A +5 percentage-point rise in "
        f"pairwise correlation across the basket changes FCN value by "
        f"${pnl_5pct:,.0f} (≈ cega × 5). For the issuer this is a "
        f"{'loss' if cega > 0 else 'gain'} that has no liquid hedge — the desk "
        f"reserves capital against it."
    )
    return CorrelationRiskReport(
        cega=cega,
        correlation_pnl_per_5pct_move=pnl_5pct,
        base_price=base,
        hedgeable=False,
        narrative=narrative,
        samples={float(s): float(p) for s, p in price_at_corr_levels.items()},
    )


# ---------------------------------------------------------------------------
# 4) Unhedgeable-risk summary
# ---------------------------------------------------------------------------


def compute_unhedgeable_risk_summary(
    greeks: dict,
    scenario_pnls: Optional[dict] = None,
    correlation_report: Optional[CorrelationRiskReport] = None,
    *,
    notional: float = 0.0,
    vol_skew_residual_vega: Optional[dict] = None,
) -> UnhedgeableRiskSummary:
    """Compose a table of residual risks the desk cannot cleanly hedge.

    The function is deliberately a *composer* — it does not re-run the pricer.
    All inputs come from prior computations (Phase 5 Greeks, the correlation
    scan, an overnight gap scenario revaluation, etc.).

    Parameters
    ----------
    greeks : dict
        Dict-like of named Greeks already computed (e.g. ``{"delta": [...]}``).
        Used for headline numbers in the summary table; not all keys are
        required.
    scenario_pnls : dict[str, float], optional
        Pre-computed P&L for stress scenarios — e.g.
        ``{"overnight_jump_to_ki_worst": -42_000.0}``. Each entry becomes a
        row labelled "Gap risk: <scenario>".
    correlation_report : CorrelationRiskReport, optional
        Output of :func:`compute_correlation_risk`. Becomes one row.
    notional : float
        Trade notional; used only to express each P&L as ``% of notional``.
    vol_skew_residual_vega : dict[str, float], optional
        Estimated residual vega per name when the vol surface twists (skew
        bumped) rather than parallel-shifts (level bumped). Each entry
        becomes a row.

    Returns
    -------
    UnhedgeableRiskSummary
    """
    rows: list[dict] = []

    def _pct(x: float) -> float:
        return 100.0 * x / notional if notional > 0 else float("nan")

    if correlation_report is not None:
        rows.append({
            "risk": "Correlation",
            "metric": "cega (per +0.01 ρ)",
            "value": correlation_report.cega,
            "p_and_l_5pct": correlation_report.correlation_pnl_per_5pct_move,
            "pct_of_notional": _pct(correlation_report.correlation_pnl_per_5pct_move),
            "hedgeable": correlation_report.hedgeable,
            "notes": correlation_report.narrative,
        })

    if scenario_pnls:
        for name, pnl in scenario_pnls.items():
            rows.append({
                "risk": f"Gap risk: {name}",
                "metric": "scenario P&L",
                "value": float(pnl),
                "p_and_l_5pct": float("nan"),
                "pct_of_notional": _pct(float(pnl)),
                "hedgeable": False,
                "notes": (
                    "Overnight jump to the knock-in barrier is not hedged by "
                    "the day-1 delta book; rebalancing the hedge intraday is "
                    "the only mitigant, and gaps over weekends are impossible "
                    "to neutralise."
                ),
            })

    if vol_skew_residual_vega:
        for ticker, residual in vol_skew_residual_vega.items():
            rows.append({
                "risk": f"Vol-of-vol / skew residual: {ticker}",
                "metric": "residual vega per +1 vol pt of skew twist",
                "value": float(residual),
                "p_and_l_5pct": float("nan"),
                "pct_of_notional": _pct(float(residual)),
                "hedgeable": False,
                "notes": (
                    "Listed-vanilla vega hedge neutralises parallel vol-level "
                    "shifts; a skew twist (steepening / flattening of the "
                    f"{ticker} smile) leaves a residual the desk must wear."
                ),
            })

    return UnhedgeableRiskSummary(rows=rows, notional=notional)
