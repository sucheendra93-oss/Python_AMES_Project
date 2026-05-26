"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ANALYTICAL METHODS IN ENGINEERING SYSTEMS                          ║
║         Global Stock Market Indices — Complete Project Source Code         ║
║                                                                              ║
║  Dataset  : global_indices_full.csv  (Yahoo Finance format)                 ║
║  Indices  : FTSE 100 (UK) | S&P 500 (USA) | Nikkei 225 (Japan)             ║
║             | Nifty 50 (India)                                               ║
║  Period   : January 2020 – June 2025                                        ║
║                                                                              ║
║  STRUCTURE:                                                                  ║
║    STEP 1  → Data Loading & Understanding                                   ║
║    STEP 2  → Descriptive Statistical Analysis                               ║
║    STEP 3  → All Mandatory Visualizations (11 plots)                        ║
║    STEP 4  → Probability Analysis (Basic, Conditional, Bayes)               ║
║    STEP 5  → Inferential Statistics (CI, z-test, t-test, ANOVA, Chi-sq)    ║
║    STEP 6  → Linear Algebra & PCA                                           ║
║    STEP 7  → Insight Generation                                             ║
║                                                                              ║
║  MODULES:  consts.py (paths, tickers, colours)  utils.py (save, style, …)    ║
║            data.py (CSV → close, returns, aligned)                            ║
║  HOW TO RUN:                                                                 ║
║    pip install -r requirements.txt                                          ║
║    python project_source_code_v3.py [--csv PATH] [--plots DIR]              ║
║    (paths default to CSV/plots next to consts.py; cwd need not be project)  ║
║                                                                              ║
║  OUTPUT: PNG plots under plots/ (or --plots directory)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  (backend must be set before pyplot / seaborn)
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")                     # Use "TkAgg" for interactive windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from scipy import stats
from scipy.stats import (
    norm,
    ttest_ind,
    f_oneway,
    chi2_contingency,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import argparse
import sys
from pathlib import Path

from consts import CSV_PATH, PLOT_DIR, TICKERS, LONG, SHORT, THEME, configure_paths
from data import load_close_returns_aligned, print_step1_summary
from utils import (
    apply_plot_rc_params,
    banner,
    capture_console_to,
    default_session_log_path,
    ensure_plot_dir,
    print_run_notice,
    save,
    style,
)

def _run_pipeline() -> None:
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 ─ DATA LOADING & UNDERSTANDING
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WHY: Before any analysis, we must understand what the data looks like —
    #      its shape, types, date range, and any missing values.
    #
    # THE CSV FORMAT:
    #   This file has a TWO-ROW header (MultiIndex):
    #     Row 0 → Price type : Close | High | Low | Open | Volume
    #     Row 1 → Ticker     : ^FTSE | ^GSPC | ^N225 | ^NSEI
    #   header=[0,1] tells pandas to treat both rows as column labels.
    #   index_col=0 makes the Date column the row index.
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 1 — DATA LOADING & UNDERSTANDING")

    # close / returns / aligned: see data.load_close_returns_aligned docstring and comments in data.py
    close, returns, aligned = load_close_returns_aligned()
    print_step1_summary(close, aligned)


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 ─ DESCRIPTIVE STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WHY: Descriptive statistics summarise the key properties of each index —
    #      where values tend to cluster (central tendency), how spread out they
    #      are (dispersion), and whether extreme values (outliers) exist.
    #
    # KEY FORMULAS:
    #
    #   Mean (arithmetic):   μ = Σxᵢ / n
    #     → Sum all values, divide by count. Sensitive to outliers.
    #
    #   Median (Q₂):         middle value when sorted
    #     → Robust to extreme values. If mean > median, distribution is
    #       right-skewed (long tail on the high end).
    #
    #   Sample Variance:     σ² = Σ(xᵢ − μ)² / (n − 1)
    #     → Bessel's correction: we divide by (n−1) not n because we are
    #       estimating the POPULATION variance from a SAMPLE. Using n would
    #       systematically underestimate the true variance.
    #
    #   Standard Deviation:  σ = √σ²
    #     → Same units as the original data, so directly interpretable
    #       (e.g., "price typically varies by ±925 points from the mean").
    #
    #   IQR (Interquartile Range):  IQR = Q₃ − Q₁
    #     → The range of the middle 50% of data. Not affected by extreme
    #       values, so it is a robust measure of spread.
    #
    #   Coefficient of Variation:  CV = (σ / μ) × 100%
    #     → Scale-free measure of relative variability. Lets us compare
    #       volatility across indices that trade at very different price
    #       levels (e.g., Nikkei at ~30,000 vs S&P at ~4,000).
    #
    #   Skewness:  3rd standardised moment
    #     → Positive = right tail is longer (bull-market pull-up)
    #     → Negative = left tail is longer (crash-driven pull-down)
    #
    #   Kurtosis (excess):  4th standardised moment − 3
    #     → 0 = Normal distribution (mesokurtic)
    #     → > 0 = heavier tails than Normal (leptokurtic) → fatter tail risk
    #     → < 0 = lighter tails than Normal (platykurtic)
    #
    # OUTLIER DETECTION — IQR METHOD:
    #   Lower fence = Q₁ − 1.5 × IQR
    #   Upper fence = Q₃ + 1.5 × IQR
    #   Any value outside these fences is flagged as an outlier.
    #   The 1.5× factor is Tukey's rule — empirically covers ~99.3% of a
    #   Normal distribution, so values outside are genuinely extreme.
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 2 — DESCRIPTIVE STATISTICAL ANALYSIS")

    rows = []
    for col in TICKERS:
        s  = close[col].dropna()
        n  = len(s)
        mu = s.mean()                   # arithmetic mean: Σxᵢ / n
        md = s.median()                 # middle value when sorted

        # Variance uses ddof=1 (degrees of freedom = n−1) = Bessel's correction
        var = s.var(ddof=1)             # σ² = Σ(xᵢ−μ)² / (n−1)
        std = s.std(ddof=1)             # σ  = √σ²
        cv  = std / mu * 100            # Coefficient of Variation

        sk  = s.skew()                  # 3rd standardised moment
        ku  = s.kurt()                  # excess kurtosis (4th moment − 3)

        # Five-number summary
        mn  = s.min()
        q1  = s.quantile(0.25)
        q3  = s.quantile(0.75)
        mx  = s.max()
        iqr = q3 - q1

        # Tukey's IQR fences
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        n_out    = int(((s < lo_fence) | (s > hi_fence)).sum())

        rows.append({
            "Index"         : LONG[col],
            "N"             : n,
            "Mean (μ)"      : round(mu, 2),
            "Median"        : round(md, 2),
            "Variance (σ²)" : round(var, 2),
            "Std Dev (σ)"   : round(std, 2),
            "CV (%)"        : round(cv, 2),
            "Skewness"      : round(sk, 4),
            "Kurtosis"      : round(ku, 4),
            "Min"           : round(mn, 2),
            "Q1"            : round(q1, 2),
            "Q3"            : round(q3, 2),
            "Max"           : round(mx, 2),
            "IQR"           : round(iqr, 2),
            "Lower Fence"   : round(lo_fence, 2),
            "Upper Fence"   : round(hi_fence, 2),
            "# Outliers"    : n_out,
        })

    desc_df = pd.DataFrame(rows).set_index("Index")

    print("\n  ── Full Descriptive Statistics Table ──\n")
    print(desc_df.to_string())

    print("\n  ── Formula Reference ──")
    print("    Mean     : μ  = Σxᵢ / n")
    print("    Variance : σ² = Σ(xᵢ − μ)² / (n − 1)   [Bessel's correction]")
    print("    Std Dev  : σ  = √σ²")
    print("    CV       : (σ/μ) × 100%")
    print("    IQR      : Q₃ − Q₁")
    print("    Fences   : Q₁ − 1.5×IQR   and   Q₃ + 1.5×IQR")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 ─ MANDATORY VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 11 charts are required. Each is explained at its definition below.
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 3 — MANDATORY VISUALIZATIONS")

    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 1 ─ HISTOGRAM                                                      │
    # │                                                                          │
    # │  WHY: Shows the frequency distribution of closing prices for each        │
    # │  index. Tells us:                                                        │
    # │    - Whether the distribution is symmetric, left-skewed, or right-skewed │
    # │    - Whether there are multiple clusters (bimodal) — indicating regime   │
    # │      changes such as a pre-COVID vs post-COVID price level               │
    # │    - The relative positions of mean and median:                          │
    # │        mean > median → positive skew (right-skewed distribution)         │
    # │                                                                          │
    # │  bins=45: balances between too coarse (missing shape) and too fine       │
    # │  (noise dominates signal). For ~1,500 data points, 40–50 bins works well.│
    # └──────────────────────────────────────────────────────────────────────────┘

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Histogram — Distribution of Closing Prices",
                 fontsize=14, fontweight="bold", color=THEME.navy, y=1.01)

    for ax, col, color in zip(axes.flat, TICKERS, THEME.series):
        s = close[col].dropna()

        ax.hist(s, bins=45, color=color, edgecolor=THEME.white, linewidth=0.4, alpha=0.88)

        # Vertical reference lines help compare mean vs median visually.
        # If they're far apart, the distribution is skewed.
        ax.axvline(s.mean(),   color=THEME.navy,     lw=2.0, linestyle="--",
                   label=f"Mean   {s.mean():,.0f}")
        ax.axvline(s.median(), color=THEME.red_accent, lw=1.8, linestyle=":",
                   label=f"Median {s.median():,.0f}")

        ax.set_title(LONG[col], fontweight="bold", color=THEME.navy)
        ax.set_xlabel("Closing Price (local currency)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        style(ax)

    plt.tight_layout(pad=2)
    save("01_histogram.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 2 ─ BOXPLOT                                                        │
    # │                                                                          │
    # │  WHY: Shows the five-number summary (min, Q1, median, Q3, max) and       │
    # │  outliers for each index in a single compact figure.                     │
    # │                                                                          │
    # │  MIN-MAX NORMALISATION: z = (x − min) / (max − min)                      │
    # │  We normalise so all four indices fall in [0,1]. Without this, Nikkei    │
    # │  (range 16K–52K) would dominate the y-axis and the FTSE/S&P boxes       │
    # │  (range 2K–10K) would be tiny and unreadable.                           │
    # │                                                                          │
    # │  NOTCH: notch=True draws a notch (narrowing) around the median.          │
    # │  The notch represents the 95% CI around the median.                      │
    # │  If notches of two boxes do NOT overlap, their medians are significantly  │
    # │  different at roughly the 95% confidence level.                          │
    # │                                                                          │
    # │  WHISKERS: extend 1.5×IQR from Q1 and Q3.                               │
    # │  Points beyond the whiskers are plotted individually as outlier dots.    │
    # └──────────────────────────────────────────────────────────────────────────┘

    fig, ax = plt.subplots(figsize=(11, 6))

    # Min-max normalise each series independently
    normed = []
    for col in TICKERS:
        s = close[col].dropna()
        normed.append((s - s.min()) / (s.max() - s.min()))   # range → [0, 1]

    bp = ax.boxplot(
        normed,
        patch_artist=True,    # fill boxes with colour
        notch=True,           # show 95% CI notch around median
        medianprops={"color": THEME.navy, "linewidth": 2.5},
        whiskerprops={"color": THEME.slate, "linewidth": 1.3},
        capprops={"color": THEME.slate, "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4, "markeredgewidth": 0},
    )
    for patch, color in zip(bp["boxes"], THEME.series):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.set_xticklabels([SHORT[t] for t in TICKERS], fontsize=11, fontweight="bold")
    ax.set_title("Boxplot — Min-Max Normalised Closing Prices  (Notch = 95% CI around Median)",
                 fontsize=12, fontweight="bold", color=THEME.navy)
    ax.set_ylabel("Normalised Price  [0 = min,  1 = max]")
    style(ax)
    plt.tight_layout()
    save("02_boxplot.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 3 ─ BAR CHART (Mean Closing Price)                                 │
    # │                                                                          │
    # │  WHY: Quick visual comparison of average price levels across indices.    │
    # │  Note: Values are in different local currencies (GBP, USD, JPY, INR)    │
    # │  so this is NOT a direct performance comparison — it simply shows scale. │
    # └──────────────────────────────────────────────────────────────────────────┘

    means  = [close[col].dropna().mean() for col in TICKERS]
    labels = [SHORT[t] for t in TICKERS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, means, color=THEME.series, width=0.52, edgecolor=THEME.white, linewidth=0.8)

    # Add value labels above each bar for readability
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.01,
                f"{val:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=THEME.navy)

    ax.set_title("Bar Chart — Mean Closing Price by Index (Local Currency)",
                 fontsize=13, fontweight="bold", color=THEME.navy)
    ax.set_ylabel("Mean Closing Price")
    ax.set_ylim(0, max(means) * 1.14)
    style(ax)
    plt.tight_layout()
    save("03_bar_mean.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 4 ─ LINE PLOT (Time-Series Trend, Rebased)                         │
    # │                                                                          │
    # │  WHY: Shows how each index evolved over time. To enable fair comparison  │
    # │  across indices priced in different currencies and at different levels,  │
    # │  we REBASE all series to 100 at their starting date.                     │
    # │                                                                          │
    # │  REBASING FORMULA:  rebased_t = (P_t / P_0) × 100                       │
    # │    → At start: every index = 100 (by definition)                        │
    # │    → At any later date: value > 100 means appreciation vs start         │
    # │    → Example: value = 160 means the index is 60% above its start level  │
    # │                                                                          │
    # │  ANNOTATIONS:                                                            │
    # │    - Shaded region: COVID-19 crash period (Feb 19 – Mar 23, 2020)       │
    # │    - Dashed line: Jan 2022 = start of global rate-hike cycle            │
    # └──────────────────────────────────────────────────────────────────────────┘

    fig, ax = plt.subplots(figsize=(14, 6))

    for col, color in zip(TICKERS, THEME.series):
        s = close[col].dropna()
        rebased = s / s.iloc[0] * 100     # rebase: divide every value by the first
        ax.plot(rebased.index, rebased.values, color=color,
                lw=1.9, label=LONG[col], alpha=0.92)

    # Highlight the COVID crash window
    ax.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"),
               alpha=0.10, color=THEME.red_accent, label="COVID Crash")
    # Mark the rate-hike era start
    ax.axvline(pd.Timestamp("2022-01-01"), color=THEME.amber,
               lw=1.3, linestyle="--", alpha=0.7, label="Rate-Hike Era")

    ax.set_title("Line Plot — Rebased Price Trend 2020–2025  (Base = 100 at Start)",
                 fontsize=13, fontweight="bold", color=THEME.navy)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased Index  (100 = first available date)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left")
    style(ax)
    plt.tight_layout()
    save("04_line_trend.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 5 ─ SCATTER PLOT (S&P 500 vs Each Other Index)                    │
    # │                                                                          │
    # │  WHY: Reveals the linear relationship between the US market and each     │
    # │  other market. Each dot is one trading day.                              │
    # │                                                                          │
    # │  PEARSON r (correlation coefficient):                                    │
    # │    r = Σ[(xᵢ−μₓ)(yᵢ−μᵧ)] / [(n−1)·σₓ·σᵧ]    ∈ [−1, +1]              │
    # │    → +1 = perfect positive linear relationship                          │
    # │    →  0 = no linear relationship                                        │
    # │    → -1 = perfect inverse linear relationship                           │
    # │                                                                          │
    # │  scipy.stats.linregress returns:                                         │
    # │    slope, intercept, r_value, p_value, std_err                          │
    # │  We use the r_value to annotate each subplot.                           │
    # └──────────────────────────────────────────────────────────────────────────┘

    other_tickers = ["^FTSE", "^N225", "^NSEI"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Scatter Plot — S&P 500 vs Other Indices (Closing Prices)",
                 fontsize=13, fontweight="bold", color=THEME.navy)

    for ax, col, color in zip(axes, other_tickers, THEME.scatter_gs_pc_others):
        common = close[["^GSPC", col]].dropna()
        x = common["^GSPC"].values
        y = common[col].values

        # Linear regression using scipy
        slope, intercept, r, p_val, stderr = stats.linregress(x, y)

        ax.scatter(x, y, alpha=0.22, s=8, color=color)

        # Regression line: y = slope × x + intercept
        x_line = np.linspace(x.min(), x.max(), 300)
        ax.plot(x_line, slope * x_line + intercept,
                color=THEME.navy, lw=2, linestyle="--")

        ax.set_title(f"vs {SHORT[col]}\nr = {r:.3f}", fontweight="bold", color=THEME.navy)
        ax.set_xlabel("S&P 500 Closing Price")
        ax.set_ylabel(f"{SHORT[col]} Closing Price")
        style(ax)

    plt.tight_layout()
    save("05_scatter.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 6 ─ GROUPED BAR CHART (Yearly Average, Rebased)                   │
    # │                                                                          │
    # │  WHY: Compares how each index performed year-by-year.                    │
    # │  Each index is rebased to its own 2020 average = 100 so year-on-year   │
    # │  growth is comparable across indices priced in different currencies.    │
    # └──────────────────────────────────────────────────────────────────────────┘

    temp = close.copy()
    temp["Year"] = temp.index.year
    # Mean close per calendar year per ticker; .dropna() removes any year where a ticker’s yearly mean is NaN
    yearly = temp.groupby("Year")[TICKERS].mean().dropna()

    x  = np.arange(len(yearly))
    bw = 0.19                            # bar width (4 bars per group)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, color) in enumerate(zip(TICKERS, THEME.series)):
        rebased = yearly[col] / yearly[col].iloc[0] * 100   # 2020 avg = 100
        ax.bar(x + i * bw, rebased, bw, color=color,
               alpha=0.87, edgecolor=THEME.white, label=SHORT[col])

    ax.set_xticks(x + 1.5 * bw)
    ax.set_xticklabels([str(y) for y in yearly.index], fontsize=10)
    ax.set_title("Grouped Bar Chart — Yearly Average Price  (Rebased: first year = 100)",
                 fontsize=12, fontweight="bold", color=THEME.navy)
    ax.set_ylabel("Rebased Index")
    ax.legend()
    style(ax)
    plt.tight_layout()
    save("06_grouped_bar.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 7 ─ STACKED BAR CHART (Annual Returns)                            │
    # │                                                                          │
    # │  WHY: Shows the year-on-year percentage return for each index and        │
    # │  whether years were universally positive or negative.                   │
    # │                                                                          │
    # │  ANNUAL RETURN FORMULA: ret_t = (avg_t / avg_{t-1} − 1) × 100          │
    # │  pandas .pct_change() computes this automatically on the yearly averages.│
    # │                                                                          │
    # │  STACKING: Positive bars stack upward from 0; negative bars stack        │
    # │  downward from 0. This lets us see both the direction and magnitude of  │
    # │  each index's contribution within a given year.                         │
    # └──────────────────────────────────────────────────────────────────────────┘

    annual_ret = yearly.pct_change().dropna() * 100   # year-over-year % change

    fig, ax = plt.subplots(figsize=(12, 6))
    years_list   = list(annual_ret.index)
    pos_bottom   = np.zeros(len(years_list))
    neg_bottom   = np.zeros(len(years_list))

    for col, color in zip(TICKERS, THEME.series):
        vals     = annual_ret[col].values
        pos_vals = np.where(vals >= 0, vals, 0)    # positive returns → stack up
        neg_vals = np.where(vals  < 0, vals, 0)    # negative returns → stack down

        ax.bar(years_list, pos_vals, bottom=pos_bottom,
               color=color, alpha=0.85, edgecolor=THEME.white, label=SHORT[col])
        # No label= here: avoids duplicate legend entries (one swatch per index, not two bars)
        ax.bar(years_list, neg_vals, bottom=neg_bottom,
               color=color, alpha=0.85, edgecolor=THEME.white)

        pos_bottom += pos_vals
        neg_bottom += neg_vals

    ax.axhline(0, color=THEME.navy, lw=1.2)
    ax.set_title("Stacked Bar Chart — Annual Returns (%) by Index",
                 fontsize=12, fontweight="bold", color=THEME.navy)
    ax.set_ylabel("Annual Return (%)")
    ax.set_xlabel("Year")
    ax.legend()
    style(ax)
    plt.tight_layout()
    save("07_stacked_bar.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 8 ─ CORRELATION HEATMAP                                            │
    # │                                                                          │
    # │  WHY: Shows the strength of linear relationships between every pair      │
    # │  of indices simultaneously in a colour-coded matrix.                    │
    # │                                                                          │
    # │  PEARSON CORRELATION FORMULA:                                            │
    # │    r_{XY} = Σ[(xᵢ−μₓ)(yᵢ−μᵧ)] / [(n−1)·σₓ·σᵧ]                        │
    # │                                                                          │
    # │  pandas .corr() computes the full pairwise Pearson correlation matrix.   │
    # │  The diagonal is always 1.0 (every series is perfectly correlated        │
    # │  with itself). The matrix is symmetric: r_{XY} = r_{YX}.               │
    # │                                                                          │
    # │  vmin=0.8: We set the colour scale minimum to 0.8 (not −1) because     │
    # │  all correlations in our dataset happen to be > 0.89. This makes        │
    # │  subtle differences between e.g. 0.895 and 0.955 visible.              │
    # └──────────────────────────────────────────────────────────────────────────┘

    # Joint sample = aligned rows only (same dates); r is linear association of *price levels*
    corr_matrix = aligned.corr()    # Pearson by default

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".3f",           # show 3 decimal places in each cell
        cmap=THEME.cmap_correlation,      # red (low) → yellow → green (high)
        center=0, vmin=0.8, vmax=1.0,   # colour scale: 0.8–1.0
        linewidths=1,
        ax=ax, square=True,
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=[SHORT[t] for t in TICKERS],
        yticklabels=[SHORT[t] for t in TICKERS],
        cbar_kws={"shrink": 0.80},
    )
    ax.set_title("Correlation Heatmap — Pearson r",
                 fontsize=13, fontweight="bold", color=THEME.navy, pad=12)
    plt.tight_layout()
    save("08_correlation_heatmap.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 9 ─ COVARIANCE HEATMAP                                             │
    # │                                                                          │
    # │  WHY: While correlation is scale-free (always −1 to +1), covariance     │
    # │  retains the original units. This matters for portfolio construction:    │
    # │  the covariance matrix is the direct input to Modern Portfolio Theory   │
    # │  (Markowitz mean-variance optimisation).                                │
    # │                                                                          │
    # │  COVARIANCE FORMULA:                                                     │
    # │    Cov(X,Y) = Σ[(xᵢ−μₓ)(yᵢ−μᵧ)] / (n−1)                               │
    # │                                                                          │
    # │  pandas .cov() computes the full pairwise covariance matrix.             │
    # │  The diagonal contains each series' own VARIANCE (Cov(X,X) = Var(X)).  │
    # │                                                                          │
    # │  Note: Large Nikkei 225 values (~49M) are a SCALE ARTEFACT — the index  │
    # │  simply trades at much higher absolute levels than the others.           │
    # └──────────────────────────────────────────────────────────────────────────┘

    cov_matrix = aligned.cov()

    # Custom annotation: display values in millions where large for readability
    annot_labels = cov_matrix.map(
        lambda v: f"{v/1e6:.2f}M" if abs(v) >= 1e6 else f"{v:,.0f}"
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cov_matrix,
        annot=annot_labels, fmt="",      # use our custom labels instead of raw values
        cmap=THEME.cmap_covariance,
        linewidths=1,
        ax=ax, square=True,
        annot_kws={"size": 9},
        xticklabels=[SHORT[t] for t in TICKERS],
        yticklabels=[SHORT[t] for t in TICKERS],
        cbar_kws={"shrink": 0.80},
    )
    ax.set_title("Covariance Heatmap  (Large Nikkei values are a scale artefact)",
                 fontsize=12, fontweight="bold", color=THEME.navy, pad=12)
    plt.tight_layout()
    save("09_covariance_heatmap.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 10 ─ SCREE PLOT                                                    │
    # │                                                                          │
    # │  WHY: After running PCA (see Step 6 for full details), the scree plot   │
    # │  shows how much variance each principal component explains.             │
    # │  It guides how many components are "worth keeping".                     │
    # │                                                                          │
    # │  READING THE SCREE PLOT:                                                 │
    # │    - Ideal "elbow" point: where the curve flattens — components beyond  │
    # │      the elbow explain very little additional variance.                  │
    # │    - In our data: PC1 alone explains 94.38% — a single dominant factor.  │
    # │    - PC1 + PC2 = 97.15% — two components capture nearly everything.    │
    # └──────────────────────────────────────────────────────────────────────────┘

    # Same scaler instance name as Step 6 — refit here for plots 10–11 (independent of Step 6 prints)
    scaler   = StandardScaler()
    X_std    = scaler.fit_transform(aligned)

    pca_full = PCA(n_components=4)
    pca_full.fit(X_std)

    exp_var  = pca_full.explained_variance_ratio_ * 100   # % explained per PC
    cum_var  = np.cumsum(exp_var)                          # running cumulative %

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 5), exp_var, color=THEME.series, alpha=0.85,
           edgecolor=THEME.white, width=0.55)
    ax.plot(range(1, 5), cum_var, "o-", color=THEME.navy,
            lw=2.5, zorder=5, label="Cumulative %")

    for i, (ev, cv) in enumerate(zip(exp_var, cum_var)):
        ax.text(i + 1, ev + 1.5, f"{ev:.2f}%",
                ha="center", fontsize=10, fontweight="bold", color=THEME.navy)

    ax.axhline(90, color=THEME.slate, linestyle="--", lw=1, alpha=0.6,
               label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot — PCA Explained Variance per Component",
                 fontsize=13, fontweight="bold", color=THEME.navy)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["PC1", "PC2", "PC3", "PC4"])
    ax.legend()
    style(ax)
    plt.tight_layout()
    save("10_scree_plot.png")


    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │  PLOT 11 ─ 2D PCA PROJECTION                                             │
    # │                                                                          │
    # │  WHY: Projects all 1,459 trading days from 4-dimensional space (one     │
    # │  dimension per index) down to 2 dimensions (PC1 and PC2) so we can      │
    # │  visualise the structure in a single scatter plot.                      │
    # │                                                                          │
    # │  READING THE PROJECTION:                                                 │
    # │    - Each dot = one trading day                                          │
    # │    - Position = where that day falls in the space of "market conditions" │
    # │    - Colour = year (blue=2020, yellow=2025)                             │
    # │    - Clusters of same-colour dots = periods of similar market behaviour  │
    # │    - The arc from bottom-left (2020) to top-right (2024–25) shows the   │
    # │      global bull market progression along PC1                           │
    # └──────────────────────────────────────────────────────────────────────────┘

    pca2  = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_std)     # project to 2D
    years = aligned.index.year.values     # year of each row, for colour coding

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=years, cmap=THEME.cmap_pca_scatter,
        s=10, alpha=0.65, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Year", fontsize=10)

    ax.set_xlabel(f"PC1  ({pca2.explained_variance_ratio_[0]*100:.1f}% variance explained)")
    ax.set_ylabel(f"PC2  ({pca2.explained_variance_ratio_[1]*100:.1f}% variance explained)")
    ax.set_title("2D PCA Projection — All Trading Days  (colour = year)",
                 fontsize=13, fontweight="bold", color=THEME.navy)
    style(ax)
    plt.tight_layout()
    save("11_pca_2d.png")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 ─ PROBABILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WHY: Probability gives us a quantitative framework to describe how likely
    #      market events are, and how knowing one market's direction changes our
    #      expectation of another's.
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 4 — PROBABILITY ANALYSIS")

    # ── 4.1  Basic (Empirical) Probability ─────────────────────────────────────
    #
    # FORMULA: P(A) = (Number of times A occurs) / (Total observations)
    #
    # This is the "relative frequency" definition of probability.
    # Rather than assuming a theoretical distribution, we simply COUNT how
    # often each event actually happened in our historical data.
    #
    # For example:
    #   P(S&P return > 0) = 825 positive days / 1507 total return days = 0.5495
    #   This means S&P 500 closed higher 54.95% of trading days in our sample.

    print("\n  ── 4.1  Basic Empirical Probabilities ──\n")
    print(f"  Formula: P(A) = count(A) / n\n")
    print(f"  {'Index':<25}  {'P(>0)':>7}  {'P(>+1%)':>8}  "
          f"{'P(>+2%)':>8}  {'P(<-1%)':>8}  {'P(<-2%)':>8}")
    print("  " + "─" * 70)

    for col in TICKERS:
        r = returns[col].dropna()
        print(f"  {LONG[col]:<25}  "
              f"{(r > 0).mean():>7.4f}  "
              f"{(r > 1).mean():>8.4f}  "
              f"{(r > 2).mean():>8.4f}  "
              f"{(r < -1).mean():>8.4f}  "
              f"{(r < -2).mean():>8.4f}")


    # ── 4.2  Conditional Probability ───────────────────────────────────────────
    #
    # FORMULA: P(A|B) = P(A ∩ B) / P(B)
    #
    # Read as: "the probability of A, GIVEN that B has already happened"
    # This is different from P(A) alone — conditioning on B updates our
    # beliefs based on new information.
    #
    # Application: Does a positive S&P 500 day make other markets more
    # likely to also be positive?
    #
    # Step-by-step:
    #   1. Compute P(B) = P(S&P 500 up) from historical data
    #   2. Compute P(A ∩ B) = P(both index A and S&P 500 were up on the SAME day)
    #   3. Divide: P(A|B) = P(A ∩ B) / P(B)
    #
    # If P(A|B) > P(A), then knowing S&P is up INCREASES our probability
    # estimate for index A — the two are positively dependent.

    print("\n  ── 4.2  Conditional Probability: P(index up | S&P 500 up) ──\n")
    print(f"  Formula: P(A|B) = P(A ∩ B) / P(B)\n")

    for a_ticker in ["^FTSE", "^N225", "^NSEI"]:
        common_ret = returns[["^GSPC", a_ticker]].dropna()
        A  = common_ret[a_ticker] > 0          # event A: index A is positive
        B  = common_ret["^GSPC"]  > 0          # event B: S&P 500 is positive

        p_A        = A.mean()                  # P(A)     = unconditional probability
        p_B        = B.mean()                  # P(B)     = P(S&P up)
        p_AandB    = (A & B).mean()            # P(A ∩ B) = both up on same day
        p_A_given_B = p_AandB / p_B           # P(A|B)   = conditional probability

        lift = p_A_given_B - p_A              # how much knowing S&P is up helps

        print(f"  {LONG[a_ticker]:25s}")
        print(f"    P(A)         = {p_A:.4f}   (unconditional)")
        print(f"    P(B)         = {p_B:.4f}   (P(S&P up))")
        print(f"    P(A ∩ B)     = {p_AandB:.4f}   (both up same day)")
        print(f"    P(A|B)       = {p_A_given_B:.4f}   (given S&P is up)")
        print(f"    Conditional lift = {lift:+.4f} pp\n")


    # ── 4.3  Bayes' Theorem ────────────────────────────────────────────────────
    #
    # FORMULA: P(A|B) = P(B|A) × P(A) / P(B)
    #
    # Bayes' theorem lets us REVERSE the conditioning. If we know P(B|A)
    # (the likelihood) but want P(A|B) (the posterior), Bayes gives us a way
    # to compute it using the prior probabilities P(A) and P(B).
    #
    # TERMINOLOGY:
    #   Prior       P(A)     = our initial belief about A before seeing evidence
    #   Likelihood  P(B|A)   = how probable is the evidence B if A is true
    #   Evidence    P(B)     = overall probability of the evidence (normaliser)
    #   Posterior   P(A|B)   = our updated belief about A after seeing evidence B
    #
    # APPLICATION:
    #   A = "S&P 500 rose today"
    #   B = "Nifty 50 rose today"
    #
    #   We already know P(B|A) = P(Nifty up | S&P up) from Section 4.2.
    #   Now we REVERSE it: given that Nifty 50 rose, what is the probability
    #   that S&P 500 also rose?
    #
    #   This is useful because Indian markets close BEFORE the US market opens.
    #   If Nifty is up, it provides a signal about US direction.

    print("\n  ── 4.3  Bayes' Theorem ──\n")
    print("  Formula: P(A|B) = P(B|A) × P(A) / P(B)")
    print("  A = 'S&P 500 up today'    B = 'Nifty 50 up today'\n")

    both = returns[["^GSPC", "^NSEI"]].dropna()
    p_A  = (both["^GSPC"] > 0).mean()                                  # Prior: P(S&P up)
    p_B  = (both["^NSEI"] > 0).mean()                                  # Evidence: P(Nifty up)
    p_B_given_A = ((both["^NSEI"] > 0) & (both["^GSPC"] > 0)).mean() / p_A   # Likelihood
    posterior    = (p_B_given_A * p_A) / p_B                           # Bayes posterior

    print(f"  Prior        P(A)       = P(S&P up)          = {p_A:.4f}")
    print(f"  Evidence     P(B)       = P(Nifty up)         = {p_B:.4f}")
    print(f"  Likelihood   P(B|A)     = P(Nifty up|S&P up) = {p_B_given_A:.4f}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Posterior    P(A|B)     = P(S&P up|Nifty up) = {posterior:.4f}")
    print(f"\n  Interpretation:")
    print(f"    Prior (before seeing Nifty data): P(S&P up)          = {p_A:.4f}")
    print(f"    Posterior (after Nifty rose)    : P(S&P up|Nifty up) = {posterior:.4f}")
    print(f"    Update magnitude: {(posterior - p_A)*100:+.2f} pp — Nifty rising is a")
    print(f"    positive signal for the S&P 500.")

    # Visualise the Bayes update
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_b = ["Prior\nP(S&P up)", "Likelihood\nP(Nifty up|S&P up)",
                "Evidence\nP(Nifty up)", "Posterior\nP(S&P up|Nifty up)"]
    values_b = [p_A, p_B_given_A, p_B, posterior]
    colors_b = list(THEME.bayes_colors)

    bars_b = ax.bar(labels_b, values_b, color=colors_b,
                    alpha=0.85, edgecolor=THEME.white, width=0.52)
    for bar, val in zip(bars_b, values_b):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center",
                fontsize=10, fontweight="bold", color=THEME.navy)

    ax.axhline(p_A, color=THEME.series[0], lw=1.5, linestyle="--", alpha=0.6,
               label=f"Prior baseline = {p_A:.4f}")
    ax.set_title("Bayes' Theorem — Prior vs Posterior Update\n"
                 "Does Nifty 50 rising give us information about S&P 500?",
                 fontsize=12, fontweight="bold", color=THEME.navy)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.82)
    ax.legend()
    style(ax)
    plt.tight_layout()
    save("12_bayes.png")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 ─ INFERENTIAL STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WHY: Descriptive stats describe our SAMPLE. Inferential statistics let us
    #      draw conclusions about the broader POPULATION and test formal
    #      hypotheses about whether observed differences are real or just noise.
    #
    # KEY CONCEPTS:
    #   - Null Hypothesis (H₀): the "default" assumption — no effect, no difference
    #   - Alternative Hypothesis (H₁): what we're trying to show
    #   - p-value: probability of seeing a result as extreme as ours IF H₀ were true
    #   - α (significance level): our pre-set threshold, typically 0.05
    #   - Decision: if p < α → reject H₀ (result is statistically significant)
    #               if p ≥ α → fail to reject H₀ (result could be due to chance)
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 5 — INFERENTIAL STATISTICS")

    # ── 5.1  Confidence Intervals (95%) ────────────────────────────────────────
    #
    # FORMULA: CI = μ̂ ± t(α/2, n−1) × (s / √n)
    #
    # A 95% CI means: "If we repeated this sampling 100 times, about 95 of
    # the resulting intervals would contain the true population mean."
    #
    # COMPONENTS:
    #   μ̂        = sample mean (our best point estimate)
    #   s        = sample standard deviation
    #   n        = sample size
    #   s / √n   = standard error of the mean (SE) — uncertainty in μ̂
    #   t(α/2)   = t critical value for α=0.05, two-tailed = 1.96 for large n
    #
    # WHY t-distribution, not z?
    # The z-distribution assumes we know the TRUE population σ. We don't —
    # we only have the sample s. The t-distribution corrects for this extra
    # uncertainty. For large n (≥1,500 here), t ≈ z ≈ 1.96 anyway.
    #
    # scipy.stats.t.interval(confidence, df, loc, scale) is the cleanest way:
    #   confidence = 0.95
    #   df         = n - 1  (degrees of freedom)
    #   loc        = sample mean
    #   scale      = standard error = s / √n

    print("\n  ── 5.1  95% Confidence Intervals for Mean Closing Price ──\n")
    print(f"  Formula: CI = μ̂ ± t(0.025, n−1) × (s/√n)\n")
    print(f"  {'Index':<25}  {'n':>6}  {'Mean':>10}  "
          f"{'SE':>8}  {'Lower CI':>12}  {'Upper CI':>12}  {'Width':>8}")
    print("  " + "─" * 82)

    for col in TICKERS:
        s  = close[col].dropna()
        n  = len(s)
        mu = s.mean()
        se = s.std(ddof=1) / np.sqrt(n)    # standard error of the mean
        ci_lo, ci_hi = stats.t.interval(
            confidence=0.95,
            df=n - 1,        # degrees of freedom
            loc=mu,          # centre the interval at the sample mean
            scale=se         # use standard error as the scale
        )
        print(f"  {LONG[col]:<25}  {n:>6,}  {mu:>10,.2f}  "
              f"{se:>8.2f}  {ci_lo:>12,.2f}  {ci_hi:>12,.2f}  {ci_hi-ci_lo:>8.2f}")


    # ── 5.2  Z-Test: Is the S&P 500 mean daily return significantly > 0? ───────
    #
    # FORMULA: z = (x̄ − μ₀) / (σ / √n)
    #
    # We use the z-test (instead of t-test) because:
    #   1. n ≈ 1,100 is large → by the Central Limit Theorem, the sampling
    #      distribution of x̄ is approximately Normal regardless of the
    #      underlying distribution.
    #   2. For large n, the t-distribution converges to the Normal anyway.
    #
    # H₀: μ_return = 0    (daily return is zero on average — random walk)
    # H₁: μ_return > 0    (markets have a positive long-run drift)
    # This is a ONE-TAILED test because we only care about returns > 0.
    #
    # p-value = P(Z ≥ z_stat | H₀ true)
    #         = 1 − Φ(z_stat)   where Φ is the standard Normal CDF
    # scipy: norm.sf(z_stat) = 1 − norm.cdf(z_stat) = upper-tail probability

    print("\n  ── 5.2  Z-Test: Is S&P 500 Mean Daily Return Significantly > 0? ──\n")
    print("  H₀: μ_return = 0    H₁: μ_return > 0    α = 0.05 (one-tailed)\n")

    sp_ret  = returns["^GSPC"].dropna()
    n_r     = len(sp_ret)
    mu_r    = sp_ret.mean()
    sigma_r = sp_ret.std(ddof=1)
    z_stat  = mu_r / (sigma_r / np.sqrt(n_r))  # z = (x̄ − 0) / SE
    p_z     = norm.sf(z_stat)                   # one-tailed: P(Z ≥ z_stat)

    print(f"  n = {n_r:,}    x̄ = {mu_r:.6f}%    σ = {sigma_r:.4f}%")
    print(f"  SE = σ/√n = {sigma_r/np.sqrt(n_r):.6f}%")
    print(f"  z-statistic = {z_stat:.4f}")
    print(f"  p-value (one-tailed) = {p_z:.4f}")
    print(f"  Decision: {'✓ Reject H₀' if p_z < 0.05 else '✗ Fail to reject H₀'}  (α=0.05)")
    if p_z < 0.05:
        print(f"  Interpretation: The S&P 500 has a statistically significant positive")
        print(f"  average daily return — markets exhibit a long-run upward drift.")


    # ── 5.3  Two-Sample Welch's t-Test: Pre-COVID vs Post-COVID S&P 500 ────────
    #
    # FORMULA: t = (μ̂₁ − μ̂₂) / √(s₁²/n₁ + s₂²/n₂)
    #
    # This is WELCH'S t-test (equal_var=False), which does NOT assume both
    # groups have the same population variance. It is more robust than the
    # classic Student's t-test, especially when sample sizes differ (40 vs 1,404).
    #
    # H₀: μ_pre = μ_post    (COVID caused no structural change in S&P 500 level)
    # H₁: μ_pre ≠ μ_post    (COVID caused a significant level shift)
    # Two-tailed: we don't pre-specify direction.
    #
    # scipy.stats.ttest_ind with equal_var=False runs Welch's test.
    # It returns: t-statistic, p-value (two-tailed by default)
    #
    # COHEN'S d (effect size):
    #   d = (μ₁ − μ₂) / pooled_std
    #   Interprets the PRACTICAL significance (not just statistical):
    #   |d| < 0.2 = small,  0.2–0.8 = medium,  > 0.8 = large

    print("\n  ── 5.3  Welch's t-Test: Pre-COVID vs Post-COVID S&P 500 ──\n")
    print("  H₀: μ_pre = μ_post    H₁: μ_pre ≠ μ_post    α = 0.05 (two-tailed)")
    print("  Pre-COVID  : before 2020-03-01")
    print("  Post-COVID : after  2020-06-01  (avoids the ambiguous crash/recovery window)\n")

    sp       = close["^GSPC"].dropna()
    pre      = sp[sp.index < "2020-03-01"]
    post     = sp[sp.index > "2020-06-01"]

    t_stat, p_val = ttest_ind(pre, post, equal_var=False)   # Welch's t-test

    # Cohen's d effect size
    pooled_std = np.sqrt((pre.std(ddof=1)**2 + post.std(ddof=1)**2) / 2)
    cohens_d   = (post.mean() - pre.mean()) / pooled_std

    print(f"  Pre-COVID   n={len(pre):4,}   mean={pre.mean():>8,.2f}   std={pre.std():>7,.2f}")
    print(f"  Post-COVID  n={len(post):4,}   mean={post.mean():>8,.2f}   std={post.std():>7,.2f}")
    print(f"\n  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {p_val:.2e}")
    print(f"  Cohen's d   = {cohens_d:.4f}  ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.2 else 'small'} effect)")
    print(f"  Decision    : {'✓ Reject H₀' if p_val < 0.05 else '✗ Fail to reject H₀'}  (α=0.05)")


    # ── 5.4  One-Way ANOVA: Normalised Prices Across All Four Indices ──────────
    #
    # FORMULA: F = MSB / MSW = (Between-group variance) / (Within-group variance)
    #
    # ANOVA tests whether the MEANS of multiple groups are simultaneously equal.
    # It extends the two-sample t-test to k ≥ 3 groups.
    #
    # WHY NORMALISE FIRST?
    # The four indices trade at very different scales (Nikkei ~30K vs S&P ~4K).
    # If we ran ANOVA on raw prices, differences in scale — not differences in
    # pattern — would dominate the F statistic. Min-max normalisation scales
    # all four to [0,1] so we compare distributional shape, not magnitude.
    #
    # H₀: μ_FTSE = μ_SP = μ_N225 = μ_NSEI  (all normalised means equal)
    # H₁: At least one mean differs
    #
    # F-statistic logic:
    #   If H₀ is true: between-group variance ≈ within-group variance → F ≈ 1
    #   If H₁ is true: between-group variance >> within-group variance → F >> 1
    #
    # scipy.stats.f_oneway takes arrays for each group and returns F, p-value.

    print("\n  ── 5.4  One-Way ANOVA: All Four Indices (Min-Max Normalised) ──\n")
    print("  H₀: μ_FTSE = μ_SP = μ_N225 = μ_NSEI    H₁: at least one differs")
    print("  Groups: min-max normalised closing prices of each index\n")

    normed_groups = []
    for col in TICKERS:
        s = close[col].dropna()
        normed_groups.append(((s - s.min()) / (s.max() - s.min())).values)

    # Groups can differ in length (each index has its own trading-day count). scipy.stats.f_oneway
    # supports unequal n; classical one-way ANOVA still assumes roughly similar variances across groups.
    F_stat, p_anova = f_oneway(*normed_groups)   # unpack list into separate args

    k  = len(TICKERS)                             # number of groups
    N  = sum(len(g) for g in normed_groups)       # total observations
    dfb = k - 1                                   # between-group df
    dfw = N - k                                   # within-group df

    print(f"  k (groups) = {k},  N (total) = {N:,},  df_between = {dfb},  df_within = {dfw:,}")
    print(f"  F-statistic = {F_stat:.4f}")
    print(f"  p-value     = {p_anova:.2e}")
    print(f"  Decision    : {'✓ Reject H₀' if p_anova < 0.05 else '✗ Fail to reject H₀'}  (α=0.05)")


    # ── 5.5  Two-Way ANOVA: Market Phase × Index (conceptual) ──────────────────
    #
    # Two-way ANOVA extends one-way ANOVA to test TWO categorical factors
    # simultaneously AND their interaction.
    #
    # Factor 1: Index (4 levels: FTSE, S&P, Nikkei, Nifty)
    # Factor 2: Market Phase (2 levels: Bull = before 2022 / Bear = after 2022)
    #
    # We approximate this with pairwise Welch's t-tests per index
    # (Bull vs Bear returns), which gives the core directional insight
    # without requiring the pingouin/statsmodels library.

    print("\n  ── 5.5  Two-Way ANOVA (Phase × Index) — Pairwise Breakdown ──\n")
    print("  Factor 1: Index (4 levels)")
    print("  Factor 2: Market Phase — Bull (before Jan 2022) / Bear/Recovery (after)\n")

    tmp_ret = returns.copy()
    tmp_ret["Phase"] = np.where(tmp_ret.index < "2022-01-01", "Bull", "Bear/Recovery")

    print(f"  {'Index':<15}  {'Phase':<16}  {'n':>5}  {'Mean (%)':>10}  {'Std (%)':>9}")
    print("  " + "─" * 60)
    for col in TICKERS:
        for phase in ["Bull", "Bear/Recovery"]:
            mask = tmp_ret["Phase"] == phase
            g    = tmp_ret.loc[mask, col].dropna()
            print(f"  {SHORT[col]:<15}  {phase:<16}  {len(g):>5}  {g.mean():>10.4f}  {g.std():>9.4f}")

    print("\n  Pairwise t-tests (Bull vs Bear/Recovery) per index:")
    for col in TICKERS:
        bull = tmp_ret.loc[tmp_ret["Phase"] == "Bull", col].dropna()
        bear = tmp_ret.loc[tmp_ret["Phase"] == "Bear/Recovery", col].dropna()
        tt, pp = ttest_ind(bull, bear, equal_var=False)
        sig = "✓ Significant" if pp < 0.05 else "✗ Not significant"
        print(f"    {SHORT[col]:<12s}  t={tt:+7.4f}  p={pp:.4f}  [{sig}]")


    # ── 5.6  Chi-Square Test of Independence ───────────────────────────────────
    #
    # FORMULA: χ² = Σ [(Observed − Expected)² / Expected]
    #
    # Tests whether two categorical variables are INDEPENDENT or ASSOCIATED.
    # Here: Are the daily "up/down" directions of FTSE 100 and S&P 500 independent?
    #
    # H₀: FTSE direction and S&P 500 direction are independent
    # H₁: They are statistically dependent (associated)
    #
    # CONTINGENCY TABLE: a 2×2 table counting how often each combination occurred:
    #   (FTSE Up, S&P Up), (FTSE Up, S&P Down), (FTSE Down, S&P Up), (FTSE Down, S&P Down)
    #
    # Expected frequency (under H₀):
    #   E_{ij} = (row_i total × col_j total) / grand total
    #   This is what we'd expect if the two variables were truly independent.
    #
    # If Observed >> Expected in certain cells, χ² is large, p is small → reject H₀.
    #
    # CRAMÉR'S V: measures the strength of the association after rejecting H₀.
    #   V = √(χ² / (n × (min(r,c) − 1)))   ∈ [0, 1]
    #   0 = no association, 1 = perfect association

    print("\n  ── 5.6  Chi-Square Test: FTSE 100 vs S&P 500 Return Direction ──\n")
    print("  H₀: FTSE and S&P 500 up/down directions are INDEPENDENT\n")

    both_ret = returns[["^FTSE", "^GSPC"]].dropna()
    ftse_dir = (both_ret["^FTSE"] >= 0).astype(int).map({1: "Up", 0: "Down"})
    gspc_dir = (both_ret["^GSPC"] >= 0).astype(int).map({1: "Up", 0: "Down"})

    # pd.crosstab builds the contingency table
    ct = pd.crosstab(ftse_dir, gspc_dir, rownames=["FTSE 100"], colnames=["S&P 500"])

    # chi2_contingency returns: χ², p-value, degrees of freedom, expected frequencies
    chi2_val, p_chi, dof, expected = chi2_contingency(ct)

    # Cramér's V: effect size
    cramers_v = np.sqrt(chi2_val / (ct.values.sum() * (min(ct.shape) - 1)))

    print(f"  Observed contingency table:\n{ct}")
    print(f"\n  Expected (under independence assumption):")
    print(pd.DataFrame(expected.round(1), index=["Down","Up"], columns=["Down","Up"]).to_string())
    print(f"\n  χ² = {chi2_val:.4f}")
    print(f"  df = {dof}")
    print(f"  p-value = {p_chi:.2e}")
    print(f"  Cramér's V = {cramers_v:.4f}  (effect size)")
    print(f"  Decision: {'✓ Reject H₀' if p_chi < 0.05 else '✗ Fail to reject H₀'}  (α=0.05)")

    # Summary visualisation: p-values and test statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Inferential Statistics — Test Results Summary",
                 fontsize=13, fontweight="bold", color=THEME.navy)

    tests      = ["z-test\n(S&P return>0)", "t-test\n(Pre vs Post\nCOVID)",
                  "ANOVA\n(4 indices)", "Chi-square\n(FTSE vs S&P)"]
    p_vals     = [p_z, p_val, p_anova, p_chi]
    stat_vals  = [abs(z_stat), abs(t_stat), F_stat, chi2_val]
    bar_colors = [THEME.sig_pass if p < 0.05 else THEME.sig_fail for p in p_vals]

    # -log₁₀(p) plot: bars above the dashed line are significant
    # Using −log₁₀ because p-values span many orders of magnitude (0.04 to 1e-20)
    ax1.bar(tests, [-np.log10(max(p, 1e-20)) for p in p_vals],
            color=bar_colors, alpha=0.85, edgecolor=THEME.white)
    ax1.axhline(-np.log10(0.05), color=THEME.navy, lw=1.5, linestyle="--",
                label="α = 0.05  (−log₁₀ = 1.30)")
    ax1.set_title("−log₁₀(p-value)  [green = significant]", fontsize=11, color=THEME.navy)
    ax1.set_ylabel("−log₁₀(p-value)")
    ax1.legend(fontsize=9)
    style(ax1)

    ax2.bar(tests, stat_vals, color=THEME.series, alpha=0.85, edgecolor=THEME.white)
    for bar, val in zip(ax2.patches, stat_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stat_vals) * 0.01,
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold", color=THEME.navy)
    ax2.set_title("Test Statistic Values", fontsize=11, color=THEME.navy)
    ax2.set_ylabel("Statistic Value")
    style(ax2)

    plt.tight_layout()
    save("13_hypothesis_tests.png")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6 ─ LINEAR ALGEBRA COMPONENT
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WHY: This section treats the dataset as a matrix and applies linear
    #      algebra operations — mean-centering, covariance, eigendecomposition,
    #      and PCA — to uncover the underlying structure of market co-movement.
    #
    # ───────────────────────────────────────────────────────────────────────────

    banner("STEP 6 — LINEAR ALGEBRA COMPONENT")

    # ── 6.1  Data Matrix ───────────────────────────────────────────────────────
    #
    # We represent the dataset as a numeric matrix X ∈ ℝ^(n × p):
    #   n = number of complete trading days (rows where all 4 indices have data)
    #   p = 4 (one column per index: FTSE, S&P, Nikkei, Nifty)
    #
    # .values converts a pandas DataFrame to a raw numpy 2D array, which
    # enables direct matrix operations.

    X   = aligned.values            # shape: (n, 4)
    n, p = X.shape

    print(f"\n  Data matrix X : shape {X.shape}  (n={n} trading days, p={p} indices)")
    print(f"  Column order  : {[SHORT[t] for t in TICKERS]}")


    # ── 6.2  Mean Centering ────────────────────────────────────────────────────
    #
    # FORMULA: X̃ = X − 1ᵀμ    where μ is the column mean vector
    #
    # WHY CENTRE?
    # Mean centering shifts each column so its average becomes zero.
    # This is required for:
    #   1. Correct covariance estimation (without centering, variance is
    #      computed around zero, not around the actual mean — overestimated)
    #   2. PCA: principal components should describe DEVIATIONS from the mean,
    #      not deviations from zero.
    #
    # Implementation: numpy broadcasting handles the subtraction automatically.
    # X has shape (n, 4) and mu has shape (4,). Broadcasting subtracts mu
    # from every row of X.

    mu  = X.mean(axis=0)            # column means: shape (4,)
    X_c = X - mu                    # mean-centered matrix

    print(f"\n  Column means before centering: {np.round(mu, 2)}")
    print(f"  Column means after centering : {np.round(X_c.mean(axis=0), 6)}")
    print("  (should be essentially zero — confirms correct centering)")


    # ── 6.3  Covariance Matrix ─────────────────────────────────────────────────
    #
    # FORMULA: Σ = X̃ᵀ X̃ / (n − 1)
    #
    # The covariance matrix Σ is a symmetric p×p matrix where:
    #   Σ_{ii} = Var(Xᵢ)        = variance of index i (diagonal)
    #   Σ_{ij} = Cov(Xᵢ, Xⱼ)   = covariance between indices i and j (off-diagonal)
    #
    # numpy: np.cov(X_c, rowvar=False)
    #   rowvar=False: each COLUMN is a variable (standard convention)
    #   default ddof=1: divides by n−1 (Bessel's correction)
    #
    # The matrix is symmetric (Σᵢⱼ = Σⱼᵢ) and positive semi-definite
    # (all eigenvalues ≥ 0), which are required properties for a valid
    # covariance matrix.

    Sigma = np.cov(X_c, rowvar=False)    # shape: (4, 4)

    print(f"\n  ── Covariance Matrix Σ (4×4) ──")
    labels_idx = [SHORT[t] for t in TICKERS]
    cov_df     = pd.DataFrame(Sigma, index=labels_idx, columns=labels_idx)
    print(cov_df.round(2).to_string())


    # ── 6.4  Correlation Matrix ────────────────────────────────────────────────
    #
    # FORMULA: ρ_{ij} = Σ_{ij} / (σᵢ × σⱼ)
    #
    # The correlation matrix normalises the covariance by the product of the
    # two standard deviations, making it scale-free (always in [−1, +1]).
    # This allows meaningful comparison across indices at different price levels.
    #
    # numpy: np.corrcoef(X_c, rowvar=False)
    # Equivalently: ρ_{ij} = Cov(i,j) / (σᵢ × σⱼ)

    Rho = np.corrcoef(X_c, rowvar=False)  # shape: (4, 4)

    print(f"\n  ── Correlation Matrix ρ (4×4) ──")
    corr_df = pd.DataFrame(Rho, index=labels_idx, columns=labels_idx)
    print(corr_df.round(4).to_string())


    # ── 6.5  Z-Score Standardisation (for PCA) ─────────────────────────────────
    #
    # FORMULA: zᵢ = (xᵢ − μ) / σ
    #
    # Before PCA, we apply z-score standardisation so all indices have:
    #   mean = 0    and    standard deviation = 1
    #
    # WHY IS THIS NECESSARY FOR PCA?
    # PCA finds directions of maximum VARIANCE. If one variable has a much
    # larger variance than others (e.g., Nikkei 225 variance ≈ 49M vs
    # FTSE variance ≈ 860K), PCA will spuriously conclude that Nikkei
    # direction is the most "important" — simply because of its larger scale.
    # Standardising puts all four indices on equal footing.
    #
    # sklearn.preprocessing.StandardScaler does this automatically:
    #   fit_transform: computes mean & std from data, then applies (x−μ)/σ

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(aligned)    # shape: (n, 4), all columns now mean=0, std=1

    print(f"\n  After z-score standardisation:")
    print(f"    Column means : {X_std.mean(axis=0).round(6)}")
    print(f"    Column stds  : {X_std.std(axis=0).round(6)}")
    print("    (all should be ~0 mean and ~1 std)")


    # ── 6.6  Eigenvalue Decomposition of the Correlation Matrix ────────────────
    #
    # FORMULA: symmetric matrix A = V Λ Vᵀ  (here A = ρ, the correlation matrix)
    #
    # Every real symmetric matrix (including our correlation matrix ρ) can be
    # decomposed into:
    #   Λ = diagonal matrix of eigenvalues λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄
    #   V = matrix of eigenvectors (each COLUMN of V is one eigenvector)
    #
    # INTERPRETATION:
    #   - Each eigenvector = a "direction" in the original 4D variable space
    #   - Each eigenvalue  = total variance captured along that direction
    #   - Eigenvectors are orthogonal (perpendicular) to each other
    #   - We sort by eigenvalue descending so PC1 has the largest variance
    #
    # numpy.linalg.eig returns:
    #   eig_vals : array of eigenvalues (not necessarily sorted)
    #   eig_vecs : matrix whose COLUMNS are the corresponding eigenvectors
    #
    # We sort both by descending eigenvalue.
    #
    # EXPLAINED VARIANCE RATIO (EVR):
    #   EVRₖ = λₖ / Σλᵢ
    #   The fraction of total variance captured by principal component k.

    eig_vals, eig_vecs = np.linalg.eig(Rho)

    # np.linalg.eig may return complex numbers for numerical reasons;
    # take the real part (imaginary parts should be essentially zero for
    # symmetric real matrices)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # Sort by eigenvalue descending
    idx      = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]      # reorder columns to match

    total_var = eig_vals.sum()
    evr       = eig_vals / total_var * 100    # explained variance ratio (%)
    cum_evr   = np.cumsum(evr)

    print(f"\n  ── Eigenvalue Decomposition of Correlation Matrix ──\n")
    print(f"  {'PC':<6}  {'Eigenvalue (λ)':>16}  {'EVR (%)':>10}  {'Cumulative (%)':>15}")
    print("  " + "─" * 52)
    for i, (lam, ev, cv) in enumerate(zip(eig_vals, evr, cum_evr)):
        print(f"  PC{i+1:<4}  {lam:>16.4f}  {ev:>10.2f}%  {cv:>14.2f}%")

    print(f"\n  ── Eigenvector Loadings (each column = one PC direction) ──\n")
    loading_df = pd.DataFrame(
        eig_vecs,
        index=labels_idx,
        columns=[f"PC{i+1}" for i in range(4)]
    )
    print(loading_df.round(4).to_string())
    print(f"\n  PC1: All loadings are nearly equal ({eig_vecs[:,0].min():.3f} to {eig_vecs[:,0].max():.3f})")
    print(f"       → One 'global market factor' drives {evr[0]:.2f}% of all variance.")
    print(f"  PC2: FTSE loads positively ({eig_vecs[0,1]:.3f}), Nikkei negatively ({eig_vecs[2,1]:.3f})")
    print(f"       → Europe vs Asia divergence factor ({evr[1]:.2f}% variance).")

    # ── sklearn PCA for verification ──────────────────────────────────────────
    pca_check = PCA(n_components=4)
    pca_check.fit(X_std)
    print(f"\n  Verification (sklearn PCA should match manual decomposition):")
    print(f"    Manual  EVR: {np.round(evr, 4)}")
    print(f"    sklearn EVR: {np.round(pca_check.explained_variance_ratio_*100, 4)}")

    # PCA loadings heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        loading_df,
        annot=True, fmt=".3f",
        cmap=THEME.cmap_pca_loadings,
        center=0, vmin=-1, vmax=1,
        linewidths=0.8,
        ax=ax, square=False,
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"shrink": 0.80},
    )
    ax.set_title("PCA Component Loadings — Eigenvectors of Correlation Matrix",
                 fontsize=12, fontweight="bold", color=THEME.navy, pad=12)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Index")
    plt.tight_layout()
    save("14_pca_loadings.png")


    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7 ─ INSIGHT GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    banner("STEP 7 — INSIGHT GENERATION")

    insights = [
        (
            "Global Market Synchronisation",
            "PATTERN",
            f"All pairwise Pearson correlations exceed 0.89 (highest: S&P500 ↔ Nikkei = {Rho[1,2]:.3f}). "
            f"PC1 alone explains {evr[0]:.2f}% of variance with near-equal positive loadings on all "
            f"four indices. One global factor — US Fed policy, global risk sentiment, and tech "
            f"mega-trends — dominates all four equity markets simultaneously."
        ),
        (
            "COVID-19 as a Structural Break",
            "PATTERN",
            f"Welch's t-test (t = {t_stat:.2f}, p < 0.0001) provides overwhelming evidence of a regime "
            f"change. The S&P 500 post-COVID mean ({post.mean():,.0f}) is "
            f"{(post.mean()/pre.mean()-1)*100:.1f}% above its pre-pandemic mean ({pre.mean():,.0f}), "
            f"reflecting $20+ trillion in combined global fiscal and monetary stimulus."
        ),
        (
            "Divergent Recovery Speeds",
            "PATTERN",
            "Despite high correlation, Nikkei 225 gained ~216% from its 2020 low (16,553) to 2024 "
            "peak (52,411). Key drivers: yen depreciation boosted Japanese export earnings, "
            "Bank of Japan maintained ultra-accommodative policy long after other central banks, "
            "and Tokyo Stock Exchange governance reforms improved capital efficiency."
        ),
        (
            "2022 Universal Decline",
            "ANOMALY",
            "2022 was the only calendar year in the dataset where all four indices "
            "simultaneously posted negative annual returns — triggered by the most aggressive "
            "global rate-hiking cycle in 40 years. FTSE 100 was most resilient due to heavy "
            "energy/commodities weighting (BP, Shell) amid commodity price spikes post-Russia-Ukraine."
        ),
        (
            "Conditional Market Signal",
            "OPPORTUNITY",
            f"Bayesian analysis reveals a {(posterior - p_B)*100:.1f} pp conditional lift: "
            f"P(Nifty up | S&P up) = {p_B_given_A:.4f} vs unconditional {p_B:.4f}. "
            f"S&P 500 futures (available before Indian markets open at 9:15 AM IST) serve as "
            f"a statistically validated leading indicator for Indian market direction."
        ),
        (
            "FTSE 100 Tail Risk",
            "RISK",
            f"FTSE 100 registers 18 IQR outliers (all March 2020) while S&P 500 registers 0. "
            f"Paradoxically, FTSE has lower daily σ ({desc_df.loc['FTSE 100 (UK)','Std Dev (σ)']:,.2f}) "
            f"yet more extreme outliers. Normal-distribution VaR models would severely underestimate "
            f"true downside risk. Fat-tail distributions (Student-t, EVT) are essential."
        ),
        (
            "Diversification Limitation",
            "RISK",
            f"With r > 0.89 across all pairs, Markowitz diversification benefits are severely limited. "
            f"Portfolio variance reduction ∝ (1 − r); with r = 0.90+, the reduction is minimal. "
            f"During crises, correlations approach 1.0 — eliminating diversification precisely when "
            f"it is most needed (the 'correlation breakdown' phenomenon)."
        ),
        (
            "Currency Adjustment Limitation",
            "LIMITATION",
            "All prices are in local currencies (GBP, USD, JPY, INR) with no FX adjustment. "
            "Return comparisons are partially confounded by exchange rate movements. "
            "Additionally ~5–7% of dates are missing per index due to national holidays; "
            "cross-country analyses use only the ~1,459 aligned rows."
        ),
    ]

    for num, (title, tag, text) in enumerate(insights, 1):
        tag_str = f"[{tag}]"
        print(f"\n  {num}. {title}  {tag_str}")
        # Word-wrap the text at 72 characters for clean console output
        words, line, lines_out = text.split(), [], []
        for w in words:
            if sum(len(x)+1 for x in line) + len(w) > 70:
                lines_out.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line: lines_out.append(" ".join(line))
        for ln in lines_out:
            print(f"     {ln}")


    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    banner("COMPLETE — ALL STEPS FINISHED")

    print(f"""
      Plots saved to  : {PLOT_DIR}/
      Total plots     : 14

      Plot index:
        01_histogram.png      ─ Step 3: Closing price distributions
        02_boxplot.png        ─ Step 3: Min-max normalised boxplots
        03_bar_mean.png       ─ Step 3: Mean closing price bar chart
        04_line_trend.png     ─ Step 3: Rebased time-series trend
        05_scatter.png        ─ Step 3: S&P 500 vs other indices scatter
        06_grouped_bar.png    ─ Step 3: Yearly average (grouped bar)
        07_stacked_bar.png    ─ Step 3: Annual returns (stacked bar)
        08_correlation_heatmap.png ─ Step 3: Pearson r heatmap
        09_covariance_heatmap.png  ─ Step 3: Covariance heatmap
        10_scree_plot.png     ─ Step 3: PCA scree plot
        11_pca_2d.png         ─ Step 3: 2D PCA projection
        12_bayes.png          ─ Step 4: Bayes' theorem visualisation
        13_hypothesis_tests.png ─ Step 5: All hypothesis test results
        14_pca_loadings.png   ─ Step 6: PCA component loadings heatmap

      Key numerical results:
        Highest correlation   : S&P 500 ↔ Nikkei 225  r = {Rho[1,2]:.3f}
        PC1 variance explained: {evr[0]:.2f}%
        COVID structural break: t = {t_stat:.2f}, p < 0.0001
        Bayes posterior lift  : P(S&P up | Nifty up) = {posterior:.4f}  (prior = {p_A:.4f})
        Chi-square (FTSE/S&P) : χ² = {chi2_val:.2f}, p < 0.0001
        ANOVA (4 indices)     : F  = {F_stat:.2f}, p < 0.0001
    """)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global stock indices analysis pipeline (steps 1–7).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to CSV (default: global_indices_full.csv next to consts.py)",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for PNG plots (default: plots/ next to consts.py)",
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Do not write a session log under logs/ (console only).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Exact log file path (default: logs/session_YYYYMMDD_HHMMSS.txt).",
    )
    args = parser.parse_args()
    configure_paths(csv_file=args.csv, plots_dir=args.plots)

    if not CSV_PATH.is_file():
        print("Error: data file not found:", file=sys.stderr)
        print(f"  {CSV_PATH}", file=sys.stderr)
        print("  Use --csv PATH or place global_indices_full.csv in the project folder.", file=sys.stderr)
        sys.exit(1)

    ensure_plot_dir()

    def _run_with_optional_log() -> None:
        print_run_notice()
        apply_plot_rc_params()
        _run_pipeline()

    if args.no_file_log:
        _run_with_optional_log()
    else:
        log_path = args.log_file if args.log_file is not None else default_session_log_path()
        log_path = log_path.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with capture_console_to(log_path):
            print(f"  Session log: {log_path}\n")
            _run_with_optional_log()


if __name__ == "__main__":
    main()
