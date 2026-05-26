# Global Stock Market Indices — Project

Analytical pipeline for four indices (FTSE 100, S&P 500, Nikkei 225, Nifty 50) using `global_indices_full.csv` (Yahoo Finance–style, two-row header).

## Quick start

```bash
cd "/path/to/files-v3"
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install --upgrade pip
pip install -r requirements.txt
python3 project_source_code_v3.py
```

Plots appear under `plots/` next to `consts.py` (14 PNG files, `01_` … `14_`; re-running overwrites files with the same names).

## What you need

- **Python 3.10+** (3.11 or 3.12 recommended; 3.9 may work but is not tested here)
- **Terminal** (macOS Terminal, Windows PowerShell, or VS Code integrated terminal)

## Step 1 — Install Python (if needed)

- **macOS:** Install from [python.org](https://www.python.org/downloads/) or use Homebrew: `brew install python`
- **Windows:** Install from [python.org](https://www.python.org/downloads/) and tick **“Add Python to PATH”**

Check in a terminal:

```bash
python3 --version
```

You should see something like `Python 3.10.x` or newer.

## Step 2 — Go to the project folder

```bash
cd "/path/to/files-v3"
```

Replace `/path/to/files-v3` with the folder that contains:

- `project_source_code_v3.py`
- `consts.py`, `utils.py`, `data.py`
- `requirements.txt`
- `global_indices_full.csv` (your data file), **or** pass `--csv` when you run (see Step 6).

**Imports:** Python adds the **folder that contains the script** to the import path. You can run `python3 /full/path/to/project_source_code_v3.py` from any working directory as long as `consts.py`, `utils.py`, and `data.py` sit beside that script.

**Virtual environment:** If you use a `.venv` in this folder, add it to `.gitignore` if you use Git (the venv is large and machine-specific).

## Step 3 — (Optional) Use a virtual environment

Keeps packages separate from other projects.

```bash
cd "/path/to/files-v3"
python3 -m venv .venv
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 4 — Install dependencies

With the virtual environment activated (or without, if you skipped Step 3):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs NumPy, pandas, Matplotlib, Seaborn, SciPy, and scikit-learn using **compatible version ranges** (not single fixed versions).

If a package fails to build, try upgrading pip/wheel first, then retry. On Windows, prefer official Python from python.org so wheels install cleanly.

## Step 5 — Data file and CSV format

**Default location:** the loader expects a file named **`global_indices_full.csv`** in the **same folder as `consts.py`** (the project root for paths).

**Override:** change `DEFAULT_CSV_NAME` in `consts.py`, or pass **`--csv PATH`** at runtime (Step 6).

**Expected CSV shape:** two header rows (MultiIndex: price type row + ticker row), dates in the first column — the same layout the script’s `data.load_close_returns_aligned()` was written for (Yahoo Finance export style). If your file differs, you must adjust `data.py` / column names to match.

**Default plot output directory:** **`<folder containing consts.py>/plots/`** unless you pass **`--plots DIR`**.

**Path rules for flags:** relative paths for `--csv` and `--plots` are resolved from your **current working directory** (where you ran the command), then normalized to absolute paths.

## Step 6 — Run the project

```bash
cd "/path/to/files-v3"
python3 project_source_code_v3.py
```

**CLI help:**

```bash
python3 project_source_code_v3.py --help
```

**Optional arguments:**

| Argument | Meaning |
|----------|--------|
| `--csv PATH` | Path to your CSV (absolute, or relative to the **current working directory**). |
| `--plots DIR` | Directory for PNG output (created if missing; absolute or relative to **cwd**). |
| `--no-file-log` | Skip writing a text session log (console only). |
| `--log-file PATH` | Write the session log to this exact file instead of `logs/session_YYYYMMDD_HHMMSS.txt`. |

**Examples:**

```bash
python3 project_source_code_v3.py --csv ~/Downloads/global_indices_full.csv
python3 project_source_code_v3.py --plots ./output_plots
python3 project_source_code_v3.py --csv ./data/indices.csv --plots ./results/plots
```

**Windows (PowerShell) example:**

```powershell
python project_source_code_v3.py --csv "$env:USERPROFILE\Downloads\global_indices_full.csv"
```

On some systems the command is `python` instead of `python3`:

```bash
python project_source_code_v3.py
```

If the CSV path does not exist, the program prints a short error to **stderr** and exits with code **1** (no long traceback).

**Note:** The full analysis runs only when you execute this file as the main program (`python … project_source_code_v3.py`). Importing the module loads dependencies but does **not** run the pipeline.

## Step 7 — Check the output

- **Console:** Printed summaries for each step (descriptive stats, tests, insights).
- **Session log (default on):** The same text is **also** saved under **`logs/session_YYYYMMDD_HHMMSS.txt`** (next to `consts.py`). Stdout and stderr are both copied. Use `--no-file-log` to disable, or `--log-file PATH` to pick the file.
- **Plots:** 14 PNG files in your plots directory (default **`plots/`** next to `consts.py`): `01_histogram.png` through `14_pca_loadings.png` (see the final console summary for the full list).

### Other ways to capture output (no code changes)

- **Shell `tee`:** `python3 project_source_code_v3.py 2>&1 | tee my_run.txt` — duplicates terminal + file; you can still use `--no-file-log` to avoid two log files.
- **Redirect only:** `python3 project_source_code_v3.py > out.txt 2>&1` — file only unless you also use `tee`.
- **Python `logging`:** A future refactor could route `print` through `logging` with `FileHandler` + `StreamHandler`; the built-in session log avoids rewriting the whole script.

## Project layout (short)

| File | Purpose |
|------|--------|
| `consts.py` | Paths, `configure_paths()`, tickers, labels, **`THEME`** (`ChartTheme`: series colours, accents, colormaps), `PLOT_RC_PARAMS` |
| `utils.py` | Save figures, styling, banners, **`capture_console_to()`** session log tee, plot folder setup |
| `data.py` | Load CSV → `close`, `returns`, `aligned` DataFrames |
| `requirements.txt` | Dependency version ranges for `pip install -r` |
| `project_source_code_v3.py` | CLI (`argparse`), `main()`, `_run_pipeline()` for steps 1–7 |

## Troubleshooting

| Issue | What to try |
|--------|----------------|
| `ModuleNotFoundError` (e.g. `pandas`, `matplotlib`) | Run Step 4 again in the **same** terminal; ensure `.venv` is activated if you use one. |
| `Error: data file not found` | Use `--csv` with the correct path, or place `global_indices_full.csv` next to `consts.py`. |
| `ImportError` for `consts` / `utils` / `data` | Keep those three modules in the **same directory** as `project_source_code_v3.py`. |
| `pip install` errors / build failures | `pip install --upgrade pip setuptools wheel`, then `pip install -r requirements.txt` again. |
| Plots do not pop up | The code uses the **non-interactive** backend (`Agg`). Images are written to disk only. |

## Optional: interactive plot windows

If you want charts in a window instead of only PNG files, in `project_source_code_v3.py` change:

```python
matplotlib.use("Agg")
```

to:

```python
matplotlib.use("TkAgg")
```

You may need a working GUI and, on Linux, `python3-tk` (or equivalent).
