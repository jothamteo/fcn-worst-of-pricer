"""Small helpers used across the pricer modules."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import norm


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if `matrix` is symmetric positive semi-definite.

    Uses the smallest eigenvalue of the symmetric part, with `tol` slack
    for floating-point noise.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(sym)
    return bool(np.all(eigvals >= -tol))


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix (Higham 1988, simplified).

    For correlation matrices that have been bumped slightly out of PSD by a
    pairwise correlation perturbation. Not industrial-grade — adequate for the
    sensitivities we compute here.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    # Re-normalise diagonals to 1 so it remains a correlation matrix.
    diag = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    return psd / np.outer(diag, diag)


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    div: float,
    vol: float,
    T: float,
    option_type: str = "call",
) -> float:
    """Closed-form Black-Scholes price for a European option on a single underlying.

    Used as a benchmark for the MC and PDE engines.
    """
    if T <= 0:
        intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
        return intrinsic
    sigma_root_t = vol * math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * T) / sigma_root_t
    d2 = d1 - sigma_root_t
    if option_type == "call":
        return spot * math.exp(-div * T) * norm.cdf(d1) - strike * math.exp(-rate * T) * norm.cdf(d2)
    if option_type == "put":
        return strike * math.exp(-rate * T) * norm.cdf(-d2) - spot * math.exp(-div * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def discount_factors(rate: float, times: Iterable[float]) -> np.ndarray:
    """Continuous-compounding discount factors at each time."""
    return np.exp(-rate * np.asarray(list(times), dtype=float))
