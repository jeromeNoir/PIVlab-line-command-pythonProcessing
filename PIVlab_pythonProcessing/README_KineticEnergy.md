# Kinetic-energy post-processing of PIVlab runs

These scripts turn a directory of PIVlab results into a kinetic-energy resonance
curve:

| Script | Role |
| --- | --- |
| [`batch_KineticEnergy.py`](batch_KineticEnergy.py) | Loop over every run folder, compute the kinetic-energy time series, write a per-run `.npz`, a combined summary (`.csv` + `.xlsx`), and the resonance figure. |
| [`process_one_KineticEnergy.py`](process_one_KineticEnergy.py) | Same computation as the batch but for a **single** run; inserts/updates that run's row in the shared summary `.xlsx`/`.csv` and refreshes the figure. |
| [`plot_KineticEnergy_summary.py`](plot_KineticEnergy_summary.py) | Standalone: (re)draw the resonance figure from an existing summary `.xlsx`/`.csv` without re-reading the (large) `.mat` files. |

---

## 1. What the batch does

For each sub-folder of `BASE_DIR` it:

1. Loads `PIVlab_results_uncalibrated.mat` (the `wienerwurst` / `scipy.io.loadmat`
   branch), flipping axes and masking invalid vectors exactly as in
   `processPIVLab_results.ipynb`.
2. Reads calibration from the **last row** of `acquisition_log.txt`:
   - `dt_vel = pulse_sep` (µs → s) — image separation *within a pair*, used to
     convert displacement to velocity.
   - `fps = cam_fps` (Hz) — PIV sampling frequency, used for the per-field
     timestamp `t[i] = i / fps`.
   - Pixel scale is fixed at `XSCALE = YSCALE = 1.2323e-4 m/px`.
   - If the log cannot be read, the run stays **uncalibrated**
     (`dt = fps = scales = 1` → velocities in px/frame) and a warning is printed.
3. Crops to the fixed ROI `PTS_ROI` (calibrated metres).
4. Computes `Ek(t) = 0.5·⟨U² + V²⟩_ROI` per frame, plus its mean and std.

### Outputs

- Per run: `<run>/PostProcessing/KineticEnergy_timeSeries.npz`
  (`t`, `Ek_frame`, `mean_Ekin`, `std_Ekin`, `dt_vel`, `fps`, `xscale`,
  `yscale`, `pts_ROI`, `nframes`, `calibrated`). **The presence of this file is
  what marks a run as "processed"** (see [§3.1](#31-incremental-processing)).
- At `BASE_DIR`: `KineticEnergy_summary.csv` and `KineticEnergy_summary.xlsx`
  with columns `run, run idx, processed, frot_Hz, flib_Hz, dphi_deg, fstar,
  calibrated, dt_vel_s, fps_Hz, nframes, mean_Ekin, std_Ekin, npz`.
- At `BASE_DIR`: `KineticEnergy_vs_fstar.png` — two panels, `⟨Ek⟩` and
  `std(Ek)` versus `f* = flib/frot` (only processed runs are plotted).

`frot`, `flib`, `dphi` are parsed from the folder name
(e.g. `frot050_flib0400_dphi2.5deg_SS1` → 0.5 Hz, 0.4 Hz, 2.5°), and `run idx`
is the integer after `SS` (e.g. `_SS2` → 2).

Two summary columns describe processing state:

- **`processed`** — `True` if the run has a result `.npz` (freshly computed or
  reused from cache), `False` for a folder with no PIV `.mat` (its metric
  columns are then `NaN`).
- **`run idx`** — the `SSn` session index, useful for telling repeat
  acquisitions apart.

---

## 2. Requirements

Use a Python environment with **numpy, scipy, pandas, matplotlib** (and
`openpyxl` for the `.xlsx`). The `dpivsoft` conda env works; the base env has a
NumPy 2.x / pandas conflict, so call the env's interpreter explicitly:

```bash
PY=/Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python
```

(or `conda activate dpivsoft` first, then just use `python`).

---

## 3. Running the batch

Edit the configuration block at the top of `batch_KineticEnergy.py` if needed
(`BASE_DIR`, `XSCALE`/`YSCALE`, `PTS_ROI`, `FROT_DIVISOR`, `FLIB_DIVISOR`),
then run:

```bash
cd /Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVlab_pythonProcessing

# process only runs not yet processed (default)
/Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python batch_KineticEnergy.py

# force reprocessing of every run
/Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python batch_KineticEnergy.py --reprocess-all
```

Console output — `[ok]` for a freshly processed run, `[skip]` for one reused
from cache:

```
  [ok]   frot050_flib1250_dphi2deg_SS1   nframes=1500 dt_vel=0.04s fps=25Hz  <Ek>=5.1443e-08  std=9.4427e-09
  [skip] frot050_flib0400_dphi2deg_SS1   already processed (cached)
  ...
Summary (20 runs: 19 processed, 1 unprocessed) written to:
  .../k6_TopBottom/KineticEnergy_summary.csv
  .../k6_TopBottom/KineticEnergy_summary.xlsx
Figure written to:
  .../k6_TopBottom/KineticEnergy_vs_fstar.png
```

### 3.1 Incremental processing

By default the batch **only processes runs that have no result yet** — a run
counts as processed when its `PostProcessing/KineticEnergy_timeSeries.npz`
exists. Already-processed runs are skipped and their summary row is rebuilt
cheaply from that `.npz`, so the large `.mat` is **not** re-read. The summary is
regenerated in full every run, so it always reflects the current state.

To reprocess everything (e.g. after changing `XSCALE`, `PTS_ROI`, or the
calibration logic), either:

- pass `--reprocess-all` on the command line, or
- set `REPROCESS_ALL = True` in the configuration block.

> **Note** — reprocessing reads every `.mat` (hundreds of MB each), so it is
> slow and I/O-heavy; the default incremental run is fast because it skips those
> reads. To only redraw the figure without touching the `.mat` files, use the
> plot script below.
>
> If a cached `.npz` is unreadable/corrupt, that run is automatically
> reprocessed (with a warning) regardless of the flag.

### Point it at a different dataset

Change `BASE_DIR` near the top of the script, e.g.:

```python
BASE_DIR = (".../CylinderExperimentsGMA/k20_bottomOnly")
```

### 3.2 Processing a single run

`process_one_KineticEnergy.py` does exactly what the batch does — it reuses the
batch's functions, so calibration / ROI / `Ek` are identical — but for one run
folder. The resulting row is **inserted (or updated) in the shared summary**
that lives in the run's parent directory (`KineticEnergy_summary.xlsx` and
`.csv`), and the resonance figure is refreshed. Re-running the same run never
duplicates its row (the existing one is replaced).

```bash
PY=/Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python

# process one run, add/update it in the shared summary, refresh the figure
$PY process_one_KineticEnergy.py .../k6_TopBottom/frot050_flib0430_dphi2deg_SS1

# force re-reading the .mat even if a cached result exists, and skip the figure
$PY process_one_KineticEnergy.py .../frot050_flib0430_dphi2deg_SS1 --reprocess --no-plot
```

| Argument | Meaning |
| --- | --- |
| `RUN_DIR` (positional) | The single run folder to process. |
| `--reprocess` | Re-read the `.mat` even if a cached result `.npz` exists (default: reuse the cache). |
| `--no-plot` | Do not redraw `KineticEnergy_vs_fstar.png`. |

Use this to add a newly acquired run to an existing summary without re-scanning
the whole dataset. (The batch's default incremental mode does the same for many
runs at once; this script is the one-run convenience form.)

---

## 4. Re-plotting from the summary

`plot_KineticEnergy_summary.py` rebuilds the figure from an existing summary
file — fast, since it never touches the `.mat` files.

```bash
cd /Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVlab_pythonProcessing
PY=/Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python

# default: reads k6_TopBottom/KineticEnergy_summary.xlsx, saves the PNG beside it
$PY plot_KineticEnergy_summary.py

# a specific summary file (.xlsx or .csv accepted)
$PY plot_KineticEnergy_summary.py /path/to/KineticEnergy_summary.xlsx

# choose the output PNG path
$PY plot_KineticEnergy_summary.py /path/to/KineticEnergy_summary.xlsx -o /tmp/resonance.png

# also pop up an interactive window (needs a GUI backend)
$PY plot_KineticEnergy_summary.py --show
```

Options:

| Flag | Meaning |
| --- | --- |
| `SUMMARY_FILE` (positional) | Summary `.xlsx`/`.csv`. Defaults to the `k6_TopBottom` xlsx. |
| `-o`, `--output` | Output PNG path. Default: `KineticEnergy_vs_fstar.png` next to the summary. |
| `--show` | Display the figure window in addition to saving. |

---

## 5. Reading the figure

- **Blue connected curve** — the main resonance sweep (most common `δφ`, i.e.
  2° here) as `f*` is varied.
- **Coloured squares** — runs at other `δφ` (the `flib1500` amplitude sweep,
  all at `f*=3`), shown separately so they are not read as part of the sweep.
- **Open diamonds** — repeat acquisitions of an identical `(frot, flib, dphi)`
  point (e.g. `_SS1`/`_SS2`), detected automatically; not folded into the curve.

Both panels use a logarithmic `Ek` axis.
