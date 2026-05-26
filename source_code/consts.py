"""Paths, tickers, labels, colours, and matplotlib rc defaults for the indices project."""

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_NAME = "global_indices_full.csv"

# Resolved paths; override at runtime via configure_paths() before loading data.
CSV_PATH: Path = _PROJECT_ROOT / DEFAULT_CSV_NAME
PLOT_DIR: Path = _PROJECT_ROOT / "plots"
LOG_DIR: Path = _PROJECT_ROOT / "logs"


def configure_paths(
    csv_file: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> None:
    """Set CSV and plot output locations. Call from main() before ensure_plot_dir / load."""
    global CSV_PATH, PLOT_DIR
    if csv_file is not None:
        CSV_PATH = Path(csv_file).expanduser().resolve()
    else:
        CSV_PATH = _PROJECT_ROOT / DEFAULT_CSV_NAME
    if plots_dir is not None:
        PLOT_DIR = Path(plots_dir).expanduser().resolve()
    else:
        PLOT_DIR = _PROJECT_ROOT / "plots"


# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME  (all plot colours + colormap names — single place to tweak look)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChartTheme:
    """Palette for indices, semantic accents, axes chrome, and seaborn colormaps."""

    # One colour per index (same order as TICKERS)
    series: tuple[str, str, str, str] = (
        "#2563EB",
        "#F59E0B",
        "#10B981",
        "#EF4444",
    )
    navy: str = "#1B2A4A"
    bg: str = "#F8FAFC"
    white: str = "#FFFFFF"
    spine: str = "#E2E8F0"
    red_accent: str = "#DC2626"
    slate: str = "#64748B"
    amber: str = "#D97706"
    grid_y_alpha: float = 0.3

    cmap_correlation: str = "RdYlGn"
    cmap_covariance: str = "Blues"
    cmap_pca_loadings: str = "RdBu_r"
    cmap_pca_scatter: str = "plasma"

    @property
    def sig_pass(self) -> str:
        return self.series[2]

    @property
    def sig_fail(self) -> str:
        return self.series[3]

    @property
    def bayes_colors(self) -> tuple[str, str, str, str]:
        s = self.series
        return (s[0], s[2], s[1], s[3])

    @property
    def scatter_gs_pc_others(self) -> tuple[str, str, str]:
        s = self.series
        return (s[0], s[2], s[3])


THEME = ChartTheme()

# Short aliases used across many zips / bars (same values as THEME)
COLORS = list(THEME.series)
NAVY = THEME.navy
BG = THEME.bg

PLOT_RC_PARAMS = {
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": THEME.bg,
    "axes.facecolor": THEME.white,
}

TICKERS = ["^FTSE", "^GSPC", "^N225", "^NSEI"]
LONG = {
    "^FTSE": "FTSE 100 (UK)",
    "^GSPC": "S&P 500 (USA)",
    "^N225": "Nikkei 225 (Japan)",
    "^NSEI": "Nifty 50 (India)",
}
SHORT = {
    "^FTSE": "FTSE 100",
    "^GSPC": "S&P 500",
    "^N225": "Nikkei 225",
    "^NSEI": "Nifty 50",
}
