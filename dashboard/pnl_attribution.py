"""First-order P&L attribution from ``initial`` → ``current`` snapshot.

The decomposition reported by the dashboard's P&L panel uses Greeks at the
*initial* snapshot as the linearisation point:

    ΔV  ≈  Σ_i  Δ_i  · ΔS_i                  (spot)
         + Σ_i  Vega_i · Δσ_i                 (vol)
         + Σ_{i<j} cega_ij · Δρ_ij · 100      (corr; cega is per 0.01)
         + Theta · Δt                          (time decay — proxy below)
    residual = V_current − V_initial − (sum above)

Theta. We use the drift-only proxy ``θ ≈ −r · V``, computed at the initial
snapshot. It is the bond-component of theta for a long-FCN holder and is the
right order of magnitude for an attribution rollup, but it is not a full
revaluation theta — the residual will pick up the difference for trades far
from issue. The UI labels it as a proxy.

The point of this module is to keep the attribution arithmetic in one place
so the P&L panel and the (later) hedging "what-if hedged" view share the
same decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dashboard.state import MarketSnapshot


@dataclass(frozen=True)
class PnLAttribution:
    """First-order P&L decomposition. All numbers in trade-notional currency."""

    spot_pnl: float
    vol_pnl: float
    corr_pnl: float
    theta_pnl: float
    residual_pnl: float
    total_pnl: float            # = current_price − initial_price
    initial_price: float
    current_price: float

    # Per-name breakdowns for the hover tooltip / table view.
    spot_pnl_by_name: np.ndarray
    vol_pnl_by_name: np.ndarray

    def as_dict(self) -> dict:
        return {
            "spot": self.spot_pnl,
            "vol": self.vol_pnl,
            "corr": self.corr_pnl,
            "theta": self.theta_pnl,
            "residual": self.residual_pnl,
            "total": self.total_pnl,
        }


def attribute(
    *,
    initial: MarketSnapshot,
    current: MarketSnapshot,
    initial_price: float,
    current_price: float,
    initial_delta: np.ndarray,
    initial_vega: np.ndarray,
    initial_cega_pair: np.ndarray,
    rate_for_theta: Optional[float] = None,
) -> PnLAttribution:
    """Run the first-order decomposition.

    Parameters
    ----------
    initial, current : MarketSnapshot
        The two market states being compared.
    initial_price, current_price : float
        Full-reval price at each snapshot.
    initial_delta : np.ndarray, shape (d,)
        Per-name raw delta at the initial snapshot (USD per USD of spot).
    initial_vega : np.ndarray, shape (d,)
        Per-name raw vega at the initial snapshot (USD per +1.0 in vol).
    initial_cega_pair : np.ndarray, shape (d, d)
        Pairwise correlation sensitivity (per +0.01 in ρ) at initial.
        Off-diagonal, symmetric; diagonal is ignored.
    rate_for_theta : float, optional
        Rate used for the ``θ ≈ −r·V`` proxy. Defaults to ``initial.rate``.
    """
    d = len(initial.spots)
    if rate_for_theta is None:
        rate_for_theta = float(initial.rate)

    # --- Δ·ΔS, per-name --------------------------------------------------
    dS = np.asarray(current.spots, dtype=float) - np.asarray(initial.spots, dtype=float)
    spot_by_name = np.asarray(initial_delta, dtype=float) * dS
    spot_pnl = float(spot_by_name.sum())

    # --- Vega·Δσ, per-name -----------------------------------------------
    dv = np.asarray(current.vols, dtype=float) - np.asarray(initial.vols, dtype=float)
    vol_by_name = np.asarray(initial_vega, dtype=float) * dv
    vol_pnl = float(vol_by_name.sum())

    # --- cega·Δρ, pairwise -----------------------------------------------
    drho = np.asarray(current.corr, dtype=float) - np.asarray(initial.corr, dtype=float)
    cp = np.asarray(initial_cega_pair, dtype=float)
    corr_pnl = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            # cega_pair is reported per +0.01, so multiply by 100 × Δρ.
            corr_pnl += float(cp[i, j]) * float(drho[i, j]) * 100.0

    # --- θ·Δt proxy ------------------------------------------------------
    dt_years = (current.as_of - initial.as_of).days / 365.0
    theta_proxy = -rate_for_theta * initial_price
    theta_pnl = float(theta_proxy * dt_years)

    total_pnl = float(current_price - initial_price)
    residual_pnl = total_pnl - (spot_pnl + vol_pnl + corr_pnl + theta_pnl)

    return PnLAttribution(
        spot_pnl=spot_pnl,
        vol_pnl=vol_pnl,
        corr_pnl=corr_pnl,
        theta_pnl=theta_pnl,
        residual_pnl=residual_pnl,
        total_pnl=total_pnl,
        initial_price=float(initial_price),
        current_price=float(current_price),
        spot_pnl_by_name=spot_by_name,
        vol_pnl_by_name=vol_by_name,
    )
