"""Build the shipped scenario grid for the dashboard's cold-start path.

Run this once locally (and again only when ``dashboard/state.py`` defaults
or the product definition change):

    python scripts/build_initial_grid.py

It writes ``dashboard/grid_initial.npz``. The dashboard loads this file at
startup whenever the fingerprint matches the live initial snapshot,
skipping the MC build entirely (~30 s → ~3 s on Streamlit Cloud).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow ``python scripts/build_initial_grid.py`` from the repo root without
# requiring ``PYTHONPATH=.`` — the project layout uses flat top-level
# packages (``dashboard/``, ``src/``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.precompute import (
    GridAxes,
    build_scenario_grid,
    save_grid_npz,
    snapshot_fingerprint,
)
from dashboard.state import (
    DEFAULT_ISSUE_DATE,
    DEFAULT_NOTIONAL,
    DEFAULT_TICKERS,
    MarketSnapshot,
    _DEFAULT_ISSUE_CORR,
    _DEFAULT_ISSUE_DIVS,
    _DEFAULT_ISSUE_RATE,
    _DEFAULT_ISSUE_SPOTS,
    _DEFAULT_ISSUE_VOLS,
    _default_product,
)


OUTPUT = Path(__file__).resolve().parents[1] / "dashboard" / "grid_initial.npz"


def main() -> None:
    initial = MarketSnapshot(
        tickers=DEFAULT_TICKERS,
        spots=_DEFAULT_ISSUE_SPOTS.copy(),
        vols=_DEFAULT_ISSUE_VOLS.copy(),
        divs=_DEFAULT_ISSUE_DIVS.copy(),
        corr=_DEFAULT_ISSUE_CORR.copy(),
        rate=_DEFAULT_ISSUE_RATE,
        as_of=DEFAULT_ISSUE_DATE,
        notional=DEFAULT_NOTIONAL,
    )
    product = _default_product()
    axes = GridAxes.default()
    total = axes.total_cells()

    print(f"Building grid: {axes.shape} = {total} cells @ 8000 paths/cell")
    print(f"Fingerprint: {snapshot_fingerprint(initial, product)}")

    last = time.time()

    def cb(done: int, total_: int) -> None:
        nonlocal last
        now = time.time()
        if done == total_ or now - last > 2.0:
            print(f"  {done}/{total_} ({100 * done / total_:.0f}%)")
            last = now

    t0 = time.time()
    grid = build_scenario_grid(
        initial=initial, product=product, axes=axes, n_paths=8_000,
        progress_cb=cb,
    )
    dt = time.time() - t0
    print(f"Built in {dt:.1f}s. Price range: "
          f"[{grid.price.min():.2f}, {grid.price.max():.2f}]")

    save_grid_npz(grid, OUTPUT)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
