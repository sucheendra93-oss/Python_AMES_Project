"""
Load Yahoo-style multi-index CSV and build close prices, returns, and aligned panel.

DataFrame roles (do not confuse these):
  close   — levels; NaN on a row when a market has no close that day.
  returns — daily % change; probabilities, z-test, chi-square on direction, etc.
  aligned — close.dropna(): same calendar date has all four closes. Joint-level
            correlation, covariance, PCA (not returns.dropna(), which would be returns).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from consts import CSV_PATH, LONG, TICKERS


def load_close_returns_aligned(csv_path: str | Path | None = None):
    """
    Read two-row header CSV, keep Close columns for TICKERS, compute daily % returns
    and the all-ticker-complete subset (aligned).

    Returns
    -------
    close : DataFrame
        Levels; NaN where a market has no close on that calendar row.
    returns : DataFrame
        pct_change * 100, first row dropped.
    aligned : DataFrame
        close.dropna() — joint dates for correlation / PCA on levels.
    """
    path = Path(CSV_PATH) if csv_path is None else Path(csv_path)
    raw = pd.read_csv(path, header=[0, 1], index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[~raw.index.isna()].sort_index()

    close = raw["Close"].copy()
    close.columns = [c.strip() for c in close.columns]
    close = close[TICKERS].dropna(how="all").astype(float)

    returns = close.pct_change().dropna() * 100
    aligned = close.dropna()
    return close, returns, aligned


def print_step1_summary(close, aligned) -> None:
    """Console summary for Step 1 (shape, range, missing counts, preview)."""
    print(f"\n  Data type     : Multivariate Time-Series — Numerical Continuous")
    print(f"  Shape         : {close.shape[0]:,} rows × {close.shape[1]} columns")
    print(f"  Date range    : {close.index.min().date()}  →  {close.index.max().date()}")
    print(f"  Aligned rows  : {len(aligned):,}  (dates where all 4 indices have data)")

    print(f"\n  Missing values per index:")
    for col in TICKERS:
        n_miss = close[col].isna().sum()
        print(f"    {LONG[col]:25s}: {n_miss:>3d} missing")

    print(f"\n  Tabular preview (first 5 rows):")
    print(close.head().rename(columns=LONG).to_string())
