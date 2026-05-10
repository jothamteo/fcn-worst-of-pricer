"""Correlated geometric Brownian motion path simulator.

Used by the FCN Monte Carlo pricer and as the workhorse for the variance
reduction / Greeks experiments. Path generation is fully vectorised in NumPy.

Notation matches `METHODOLOGY.md` §1. For each underlying $S_i$,

    dS_i/S_i = (r - q_i) dt + sigma_i dW_i,    d<W_i, W_j> = rho_{ij} dt,

with exact log-Euler update between observation grid points:

    S_i(t + dt) = S_i(t) * exp((r - q_i - 0.5 * sigma_i^2) * dt
                               + sigma_i * sqrt(dt) * Z_i),

where (Z_1, ..., Z_d) ~ N(0, Sigma). This is exact for constant-coefficient
GBM, so the only source of discretisation error is the observation grid
(which is chosen to coincide with the FCN observation dates anyway).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import is_psd


@dataclass(frozen=True)
class SimulationConfig:
    """Inputs to a single GBM simulation run.

    All arrays are 1D of length `d` (number of underlyings), except `corr`
    which is `(d, d)`. Times are in years.
    """

    spots: np.ndarray
    vols: np.ndarray
    divs: np.ndarray
    rate: float
    corr: np.ndarray
    T: float
    n_steps: int
    n_paths: int
    antithetic: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        spots = np.asarray(self.spots, dtype=float)
        vols = np.asarray(self.vols, dtype=float)
        divs = np.asarray(self.divs, dtype=float)
        corr = np.asarray(self.corr, dtype=float)

        d = spots.shape[0]
        if vols.shape != (d,):
            raise ValueError(f"vols shape {vols.shape} != ({d},)")
        if divs.shape != (d,):
            raise ValueError(f"divs shape {divs.shape} != ({d},)")
        if corr.shape != (d, d):
            raise ValueError(f"corr shape {corr.shape} != ({d}, {d})")
        if self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be > 0, got {self.n_paths}")
        if not is_psd(corr):
            raise ValueError("correlation matrix is not PSD")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("correlation matrix must have unit diagonal")

        # dataclass(frozen=True) — bypass __setattr__ to store coerced arrays.
        object.__setattr__(self, "spots", spots)
        object.__setattr__(self, "vols", vols)
        object.__setattr__(self, "divs", divs)
        object.__setattr__(self, "corr", corr)

    @property
    def d(self) -> int:
        return self.spots.shape[0]

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


def cholesky_lower(corr: np.ndarray) -> np.ndarray:
    """Cholesky factor L such that L L^T = corr.

    Uses `np.linalg.cholesky` directly. If the matrix has been bumped just
    out of PSD by a correlation perturbation, the caller should project it
    onto the PSD cone via `utils.nearest_psd` first.
    """
    return np.linalg.cholesky(corr)


def draw_normals(
    n_paths: int, n_steps: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw independent standard normals of shape (n_paths, n_steps, d).

    Exposed as a separate function so callers can reuse the same draws across
    parameter bumps (CRN). The simulator accepts pre-drawn normals via the
    `normals` kwarg of `simulate_paths`.
    """
    return rng.standard_normal(size=(n_paths, n_steps, d))


def simulate_paths(
    cfg: SimulationConfig,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate correlated GBM paths.

    Returns an array of shape (n_paths_total, n_steps + 1, d) where
    `n_paths_total = n_paths * (2 if antithetic else 1)`. The first slice
    along axis 1 is the initial spot vector, repeated across paths.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation inputs (see dataclass docstring).
    normals : np.ndarray, optional
        Pre-drawn standard normals of shape (n_paths, n_steps, d). When
        provided, `cfg.seed` is ignored. This is the mechanism for CRN
        across parameter bumps.

    Notes
    -----
    With `antithetic=True`, the second half of the returned array is driven
    by `-eta` where `eta` drives the first half. The Cholesky transform
    commutes with the sign flip, so this is identical to flipping the
    correlated draws.
    """
    d = cfg.d
    n_paths = cfg.n_paths
    n_steps = cfg.n_steps
    dt = cfg.dt
    sqrt_dt = np.sqrt(dt)

    if normals is None:
        rng = np.random.default_rng(cfg.seed)
        eta = draw_normals(n_paths, n_steps, d, rng)
    else:
        eta = np.asarray(normals, dtype=float)
        if eta.shape != (n_paths, n_steps, d):
            raise ValueError(
                f"normals shape {eta.shape} != ({n_paths}, {n_steps}, {d})"
            )

    L = cholesky_lower(cfg.corr)
    correlated = eta @ L.T  # (n_paths, n_steps, d), each step row has cov = Sigma

    if cfg.antithetic:
        correlated = np.concatenate([correlated, -correlated], axis=0)

    drift = (cfg.rate - cfg.divs - 0.5 * cfg.vols ** 2) * dt  # (d,)
    diffusion_scale = cfg.vols * sqrt_dt  # (d,)
    log_increments = drift + diffusion_scale * correlated  # broadcast over (paths, steps, d)
    log_paths = np.cumsum(log_increments, axis=1)  # (n_paths_total, n_steps, d)

    n_paths_total = log_paths.shape[0]
    paths = np.empty((n_paths_total, n_steps + 1, d), dtype=float)
    paths[:, 0, :] = cfg.spots
    paths[:, 1:, :] = cfg.spots * np.exp(log_paths)
    return paths


def observation_indices(n_steps: int, n_obs: int) -> np.ndarray:
    """Indices into the time grid corresponding to `n_obs` equally-spaced
    observation dates including maturity, excluding $t_0$.

    For `n_steps = 252, n_obs = 4` (quarterly observations on a daily grid)
    this returns [63, 126, 189, 252].
    """
    if n_obs <= 0 or n_steps <= 0:
        raise ValueError("n_steps and n_obs must be positive")
    if n_steps % n_obs != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of n_obs ({n_obs}) "
            "for equally-spaced observations"
        )
    step = n_steps // n_obs
    return np.arange(step, n_steps + 1, step, dtype=int)
