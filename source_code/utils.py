"""Console banners, figure save, axes styling, session log tee, and plot output setup."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from consts import LOG_DIR, PLOT_DIR, PLOT_RC_PARAMS, THEME


class _TeeStream:
    """Write to multiple text streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


@contextmanager
def capture_console_to(log_path: Path):
    """
    Duplicate stdout and stderr to log_path while still printing to the terminal.
    UTF-8 text file; one file per run when used from main().
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, file_obj)
    sys.stderr = _TeeStream(old_err, file_obj)
    try:
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        file_obj.close()


def default_session_log_path() -> Path:
    """Timestamped path under LOG_DIR."""
    return LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.txt"


def ensure_plot_dir() -> None:
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


def apply_plot_rc_params() -> None:
    plt.rcParams.update(PLOT_RC_PARAMS)


def print_run_notice() -> None:
    """Tell the user to wait while the pipeline runs (longer copy on first / empty plots)."""
    plot_dir = Path(PLOT_DIR)
    has_pngs = plot_dir.is_dir() and any(plot_dir.glob("*.png"))
    line = "─" * 68
    print(f"\n  {line}")
    if not has_pngs:
        print("  Welcome — first run (no plot images in the output folder yet).")
        print("  Please wait: generating statistics, tests, insights, and 14 chart PNGs.")
        print("  This usually takes a few minutes; you'll see sections appear below.")
    else:
        print("  Please wait — running the full pipeline and refreshing outputs.")
        print("  Regenerating plots and summaries may take a few minutes.")
    print(f"  {line}\n")


def banner(title: str) -> None:
    """Print a visible section divider to the console."""
    line = "═" * 72
    print(f"\n{line}\n  {title}\n{line}")


def save(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    path = Path(PLOT_DIR) / filename
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=THEME.bg)
    plt.close()
    print(f"  [plot saved] {filename}")


def style(ax) -> None:
    """Apply clean consistent styling to an axes object."""
    ax.set_facecolor(THEME.white)
    for sp in ax.spines.values():
        sp.set_color(THEME.spine)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=THEME.grid_y_alpha, linestyle="--")
    ax.xaxis.grid(False)
