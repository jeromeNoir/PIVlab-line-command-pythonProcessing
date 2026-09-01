# How to process PIV data: PIVlab (MATLAB) → Python post-processing

This guide covers the whole chain, in two parts:

- **Part I — PIV processing (MATLAB)**, sections 1–8. Turns raw image pairs into
  a `PIVlab_results_uncalibrated.mat` per run.
- **Part II — Post-processing (Python notebooks)**, sections 9–15. Turns those
  `.mat` files into kinetic-energy and velocity spectra, summary tables and
  figures.

Run Part I first; every Part II notebook reads the `.mat` it produces.

---

# Part I — PIV processing (MATLAB)

This part explains how to run the two PIV processing scripts:

- **`PIVlab_process_commandline.m`** — processes a **single** run folder.
- **`PIV_batchprocessing_commandline.m`** — processes **every** sub-folder of a
  project root in one go (batch version of the script above).

Both scripts run the same four-step PIVlab workflow on each image pair:

1. **Load** an image pair (images `1+2`, `3+4`, …).
2. **Preprocess** both images (CLAHE contrast enhancement, high-pass filter, …).
3. **Run the PIV analysis** (`piv.piv_FFTmulti`, multi-pass FFT cross-correlation).
4. **Postprocess** the vector field (velocity limits, outlier rejection,
   interpolation of removed vectors).

Results are written as a `PIVlab_results_uncalibrated.mat` file plus a
first-frame preview figure, one per run folder.

---

## 1. Prerequisites

- **MATLAB** (a recent release — the scripts use the `name=value` function-call
  syntax, which requires R2021a or newer).
- The **PIVlab command-line code** must be on the MATLAB path. It lives in:

  ```
  /Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVLAB_commandLine/
  ```

  The scripts call functions from the PIVlab packages:
  `preproc.PIVlab_preproc`, `piv.piv_FFTmulti`, `postproc.PIVlab_postproc`,
  and `misc.inpaint_nans`. These are package folders (`+preproc`, `+piv`,
  `+postproc`, `+misc`) inside the PIVlab installation.

- The **image source** must be reachable. In this project the images live on a
  mounted volume, e.g. `/Volumes/Archives/TOPOLIB_TopBottom/k6_TopBottom`, so
  that drive must be mounted before running.

### Set up the MATLAB path

Open MATLAB, then add the PIVlab code to the path. Either run once in the
Command Window:

```matlab
addpath(genpath('/Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVLAB_commandLine'))
```

or `cd` into the `PIVLAB_commandLine` folder before running the scripts so the
package folders are visible.

---

## 2. Image layout assumptions

- Images sit **directly** inside each run folder (one level deep). Nested
  layouts are not traversed.
- Images are processed **as pairs** (`1+2`, `3+4`, …), so each folder must
  contain an **even** number of images. An odd count is an error
  (single-folder script) or a skip-with-warning (batch script).
- Images are matched by a file pattern (`file_pattern`), e.g. `*.tif`, `*.bmp`,
  `*.png`, `*.jpg`. File names are sorted alphabetically before pairing, so the
  naming must put consecutive frames in order.

---

## 3. `PIVlab_process_commandline.m` — single folder

Use this when you want to process **one** run folder.

### What to edit

Open the script and edit the path block near the top:

| Variable        | Meaning                                                              | Example                                                        |
| --------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| `project_root`  | Root directory that contains the run folder (image source).         | `/Volumes/Archives/TOPOLIB_TopBottom/k6_TopBottom`             |
| `run_folder`    | Name of the specific run sub-folder to process.                     | `frot0.50Hz_flib0.405Hz_dphi2deg_SS1`                          |
| `file_pattern`  | Glob pattern that matches the images.                               | `*.tif`                                                        |
| `local_folder`  | Where results are written (mirrors `project_root` locally).         | `.../CylinderExperimentsGMA/k6_TopBottom`                      |

The images are read from `fullfile(project_root, run_folder)` and results are
written to `fullfile(local_folder, run_folder)`.

### How to run

1. Set up the MATLAB path (section 1).
2. Edit `project_root`, `run_folder`, `file_pattern`, `local_folder`.
3. Optionally tune the processing settings (section 5).
4. Run the script:

   ```matlab
   PIVlab_process_commandline
   ```

### What it produces

In `local_folder/<run_folder>/`:

- **`PIVlab_results_uncalibrated.mat`** — saves the **entire workspace**
  (`save(file_results)` with no variable list), including all result arrays and
  the settings used.
- **`PIVlab_figure_uncalibrated_firstFrame.jpg`** — a quiver plot of the
  filtered velocity field of the **first** image pair. This figure is left
  **visible** on screen.
- Copies of any auxiliary files found next to the images
  (`acquisition_log.txt`, `background.mat`, `PIVlab_Capture_Session.mat`).

> **Do not modify this file.** It is the reference/original script. To process
> many folders or change behaviour, use the batch script below.

---

## 4. `PIV_batchprocessing_commandline.m` — all folders

Use this to process **every** sub-folder of a project root with one command.
This is the workhorse for the project.

### What to edit

Only three variables at the top:

| Variable        | Meaning                                                         | Example                                              |
| --------------- | --------------------------------------------------------------- | ---------------------------------------------------- |
| `project_root`  | Root directory whose sub-folders are each a run to process.     | `/Volumes/Archives/TOPOLIB_TopBottom/k6_TopBottom`   |
| `local_folder`  | Where results are written (mirrors `project_root` structure).   | `.../CylinderExperimentsGMA/k6_TopBottom`            |
| `file_pattern`  | Glob pattern that matches the images.                           | `*.tif`                                              |

There is **no** `run_folder` here — the script auto-discovers all sub-folders.

### How it differs from the single-folder script

- **Auto-discovers** every sub-folder of `project_root`, skipping `.`, `..`,
  and hidden folders, processing them in sorted order.
- Processing settings are defined **once** and applied to all folders.
- Per-folder result variables are **reset each iteration** so runs don't mix.
- Folders with **no images** or an **odd** image count are **skipped with a
  warning** instead of aborting the whole batch.
- Figures are created **invisibly** (`'Visible','off'`) and closed after saving,
  so no windows pile up.
- `save` uses an **explicit variable list** (result arrays + all settings used)
  rather than dumping the whole workspace.

### How to run

1. Set up the MATLAB path (section 1).
2. Edit `project_root`, `local_folder`, `file_pattern`.
3. Optionally tune the processing settings (section 5).
4. Run the script:

   ```matlab
   PIV_batchprocessing_commandline
   ```

### What it produces

For each valid sub-folder `<run_folder>`, in `local_folder/<run_folder>/`:

- **`PIVlab_results_uncalibrated.mat`** — result arrays plus the settings used.
- **`PIVlab_figure_uncalibrated_firstFrame.jpg`** — first-pair velocity field
  (saved invisibly).
- Copies of any auxiliary files found next to the images.

Progress is printed to the Command Window per folder and per image pair.

---

## 5. Processing settings (shared by both scripts)

The settings blocks are identical in both scripts. Tune them to your imaging
conditions.

### Preprocessing (`preproc.PIVlab_preproc`)

| Setting          | Default | Meaning                                                       |
| ---------------- | ------- | ------------------------------------------------------------- |
| `roi_inpt`       | `[]`    | Region of interest `[x y width height]`; `[]` = full image.   |
| `clahe`          | `1`     | Contrast-limited adaptive histogram equalization on/off.      |
| `clahesize`      | `64`    | Size of the local contrast tiles.                             |
| `highp`          | `1`     | High-pass filter on/off.                                      |
| `highpsize`      | `15`    | High-pass filter size.                                        |
| `intenscap`      | `0`     | Intensity capping on/off.                                     |
| `wienerwurst`    | `0`     | Wiener denoise filter on/off.                                 |
| `wienerwurstsize`| `3`     | Wiener filter size.                                           |
| `minintens`      | `0.0`   | Lower intensity limit.                                        |
| `maxintens`      | `1.0`   | Upper intensity limit.                                        |

### PIV analysis (`piv.piv_FFTmulti`)

| Setting                  | Default     | Meaning                                                       |
| ------------------------ | ----------- | ------------------------------------------------------------- |
| `interrogationarea`      | `64`        | First-pass interrogation window size (px).                    |
| `step`                   | `32`        | Spacing between neighbouring vectors (px).                    |
| `subpixfinder`           | `1`         | 1 = 3-point Gauss, 2 = 2D Gauss sub-pixel estimator.          |
| `mask_inpt`              | `[]`        | Logical mask (same size as image, `true` = masked); `[]`=none.|
| `passes`                 | `2`         | Number of refinement passes.                                  |
| `int2`/`int3`/`int4`     | `32/16/16`  | Interrogation window sizes for passes 2/3/4.                  |
| `imdeform`               | `'*linear'` | Image deformation interpolation (`'*linear'` or `'*spline'`). |
| `repeat`                 | `0`         | Repeated correlation.                                         |
| `mask_auto`              | `0`         | Auto-correlation masking in first pass.                       |
| `do_linear_correlation`  | `0`         | 0 = circular, 1 = linear correlation.                         |
| `repeat_last_pass`       | `0`         | Repeat the last pass.                                         |
| `delta_diff_min`         | `0.025`     | Stop repeating last pass below this improvement.              |
| `limit_peak_search_area` | `1`         | Limit peak search to the central region (recommended).        |

> **Masking:** to mask a region, build a logical matrix the same size as the
> images (`true` = masked out) and assign it to `mask_inpt`. The single-folder
> script has a commented example near the top.

### Postprocessing (`postproc.PIVlab_postproc`)

| Setting          | Default                | Meaning                                                       |
| ---------------- | ---------------------- | ------------------------------------------------------------- |
| `calu` / `calv`  | `1` / `1`              | Calibration factors for `u` / `v` (1 = uncalibrated px/frame).|
| `valid_vel`      | `[-50;50;-50;50]`      | Velocity limits `[u_min;u_max;v_min;v_max]`; outside = removed.|
| `do_stdev_check` | `1`                    | Global standard-deviation outlier check on/off.               |
| `stdthresh`      | `7`                    | Threshold for the std-dev check.                              |
| `do_local_median`| `1`                    | Local median outlier check on/off.                            |
| `neigh_thresh`   | `3`                    | Threshold for the local median check.                         |
| `paint_nan`      | `1`                    | Interpolate (fill) removed vectors via `misc.inpaint_nans`.   |

> Results are **uncalibrated** by default (`calu = calv = 1`), hence the
> `_uncalibrated` in the output file names. Velocities are in pixels/frame.

---

## 6. Output `.mat` contents

The key result arrays are 3-D matrices with dimensions
`[vertical position, horizontal position, image-pair index]`:

| Variable          | Meaning                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `x`, `y`          | Grid coordinates of the vectors.                                 |
| `u`, `v`          | Raw velocity components (px/frame).                              |
| `typevector`      | Vector type from PIV (0 = masked, 1 = valid).                    |
| `correlation_map` | Correlation peak value per vector.                               |
| `u_filt`, `v_filt`| Postprocessed velocity components (outliers removed/filled).     |
| `typevector_filt` | Vector type after filtering (2 = removed by postprocessing).     |

Masked regions are forced back to `NaN` in `u_filt`/`v_filt` after inpainting.

The batch script additionally saves the run metadata (`image_folder`,
`results_folder`, `image_names`, `num_pairs`, `run_folder`, `file_pattern`) and
**all** the settings listed in section 5, so each `.mat` is self-documenting.

> **Two PIVlab layouts exist in this project.** The batch script above writes
> `u_filt`/`v_filt` as 3-D arrays `[y, x, pair]`. Files exported from the PIVlab
> **GUI session** instead hold `u_filtered`/`v_filtered` (and `u_original`/
> `v_original`) as *cell arrays*, one 2-D frame per cell. The Python loader
> (section 14) reads both, so either export can be post-processed.

---

## 7. Quick recipes

**Process one specific run:**

```matlab
% edit run_folder inside PIVlab_process_commandline.m, then:
PIVlab_process_commandline
```

**Process a whole dataset (all sub-folders):**

```matlab
% edit project_root / local_folder / file_pattern inside
% PIV_batchprocessing_commandline.m, then:
PIV_batchprocessing_commandline
```

**Point at a different dataset:** change `project_root`, `local_folder`, and
`file_pattern` at the top of the relevant script. The output folder structure
under `local_folder` mirrors the run-folder names found under `project_root`.

---

## 8. Troubleshooting

| Symptom                                             | Likely cause / fix                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------------  |
| `Undefined ... preproc.PIVlab_preproc` (etc.)       | PIVlab code not on the path — `addpath(genpath(...))` the `PIVLAB_commandLine` dir. |
| `No images found`                                   | Wrong `file_pattern` or wrong `image_folder`/`project_root`; volume not mounted.    |
| Odd-image-count error / skip                        | A frame is missing or extra; PIV needs an even count (pairs). Check the folder.     |
| `No sub-folders found in project_root` (batch)      | `project_root` is wrong, empty, or only contains files/hidden folders.              |
| Results not appearing                               | Check `local_folder` is writable; the scripts create the per-run folder if missing. |
| Velocities look wrong/clipped                       | Adjust `valid_vel` limits and the outlier thresholds (`stdthresh`, `neigh_thresh`). |

---

# Part II — Post-processing (Python notebooks)

Everything below lives in:

```
/Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVlab_pythonProcessing/
```

## 9. Setup

- **Interpreter.** Use the Anaconda Python (`~/anaconda3/bin/python`, 3.11) — it
  has numpy / scipy / pandas / matplotlib. The macOS system Python
  (`/usr/bin/python3`) has **none** of them, and picking it is the usual cause of
  `ModuleNotFoundError: No module named 'pandas'`. In VS Code choose it with
  *Python: Select Interpreter*; a `.vscode/settings.json` in the folder already
  pins it.
- **Shared library.** `piv_postprocessing_lib.py` holds every helper the
  notebooks share (section 14). **All notebooks import from it**, so a fix lands
  once.
- **Per-dataset parameters.** Calibration, ROI and the other shared constants
  live in a `param_postProcessing.json` file **inside each dataset folder** (the
  one holding the run sub-folders), so two datasets can carry different
  calibrations. Each notebook loads it right after its dataset path is set:
  ```python
  _P = read_paramPostprocessing(BASE_DIR)   # BASE_DIR, or the parent of a single RUN_DIR
  XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
  PTS_ROI = _P.PTS_ROI
  ```
  The call returns the values as a namespace **and** rebinds them inside the
  library, so its own helpers (`read_acquisition_params`, `parse_run_name`,
  `libration_ke_scale`, …) use the dataset's values too.
  `piv_postprocessing_lib.py` holds **no** parameter values itself: the defaults
  live in **`param_postProcessing_default.json`** next to the notebooks (loaded
  once at import, so the helpers work even in tools that don't call
  `read_paramPostprocessing`). **If a dataset folder has no
  `param_postProcessing.json`, `read_paramPostprocessing` copies the default one
  there, prints a message and stops** — edit the copy and relaunch. Keep the
  default file in sync when you change a dataset's file.
- **Run notebooks from their own folder.** `import piv_postprocessing_lib`
  resolves because Jupyter uses the notebook's directory; starting elsewhere
  breaks the import.
- **Restart the kernel after editing the library.** Python caches imported
  modules, so a running kernel keeps the old version — the usual cause of
  `ImportError: cannot import name ...` for something you just added.
- `obsolete/` holds the retired `.py` twins of these tools; the notebooks are the
  live versions.

## 10. Run-folder naming

Post-processing reads the physical parameters straight from the folder name:

```
frot0.50Hz_flib0.400Hz_dphi2deg_SS1
   │         │           │      └── acquisition index; SS2, SS3… mark repeats
   │         │           └───────── libration amplitude δφ  [deg]
   │         └───────────────────── libration frequency f_lib [Hz]
   └─────────────────────────────── rotation frequency f_rot [Hz]
```

`parse_run_name()` returns `(frot_Hz, flib_Hz, dphi_deg)`; the summaries add
`fstar = flib/frot`. The **legacy** form (`frot050_flib0400_…`, zero-padded
integers) is still parsed, so old summary tables keep working.

## 11. Regions: ROI vs FULL

Every quantity can be computed over the fixed ROI rectangle (`PTS_ROI` in the
library) or over the whole field. Each notebook sets this **locally**:

| value | meaning |
| --- | --- |
| `REGION = 'ROI'`  | crop to `PTS_ROI` |
| `REGION = 'FULL'` | whole field |
| `REGION = 'BOTH'` | **batch notebooks only** — process both in one run |

The region is written into **every output name** (`…_ROI.npz`, `…_FULL.png`,
`…_summary_ROI.csv`), so the two never overwrite each other. With `'BOTH'` a
batch produces two of everything — including two figures, which is expected, not
a duplicate.

## 12. The notebooks

Run the batches first: they create the `.npz` files the others read.

| Notebook | What it does |
| --- | --- |
| `batch_KineticEnergy.ipynb` | **Start here.** For every run: `Ek(t) = ½⟨u²+v²⟩` over the region → `KineticEnergy_timeSeries_<region>.npz`, a summary table, and the resonance figure. |
| `batch_KineticEnergyFFT.ipynb` | FFT of each run's `Ek(t)`, both `fft(⟨Ek⟩)` and `⟨fft(Ek)⟩`, → `KineticEnergy_FFT_<region>.png` + summary. Needs the batch above. |
| `batch_VelocityFFT.ipynb` | Per-point velocity spectra averaged over the region + polarization → `VelocityFFT_<region>.npz`, figures, summary, and the `f*`/`δφ`/`f_lib` colormaps. |
| `single_KineticEnergy.ipynb` | The two steps above for **one** run, with an option to refresh that run's row in the summaries. |
| `single_KineticEnergyFFT.ipynb` / `single_VelocityFFT.ipynb` | Inspect one run's stored `.npz` interactively (no recomputation). The velocity one draws a single panel with `|FFT(U)|` and `|FFT(V)|`. |
| `PIV_subset_analysis_singleExperiment.ipynb` | One run over a **time window**: movie of the velocity field, `Ek(t)`, and its spectrum. |
| `filter_velocity_bandpass.ipynb` | Band-pass the velocity in time at every grid point → a filtered `.mat` beside the original. |
| `plot_resonance_summary.ipynb` | Resonance figure from a summary `.csv`; x-axis `fstar` or `flib`, optional normalization, optional filtering by frot/flib/dphi. |
| `make_thumbmail_summary.ipynb` | Tiles every run's figures into one contact sheet per figure type, written beside the summaries. `SOURCE_NORMALIZED` (raw vs `_normalized`) and `SOURCE_EXT` (`'png'`/`'pdf'`) pick which per-run figures to tile; missing files are reported per group. PDF sources need a raster backend (pymupdf/pypdfium2/pdf2image), else the build reports and stops. |

Common options in the config cell: `BASE_DIR` (dataset root), `REGION`,
`REPROCESS_ALL` (recompute vs reuse cached `.npz`), and `NORMALIZE_EK`.

**Normalization.** `NORMALIZE_EK = True` divides `Ek` by the libration kinetic
energy scale `(δφ[rad]·2π·f_lib·R)²` (`R` = cylinder radius, in the library),
making it dimensionless and adding `_normalized` to the figure name. Only the
figures are normalized — the `.npz` and summary tables stay in physical units.

## 13. Where the outputs go

Per run, in `<run>/PostProcessing/`:

| File | From |
| --- | --- |
| `KineticEnergy_timeSeries_<region>.npz` | `batch_KineticEnergy` |
| `KineticEnergy_FFT_<region>.png` | `batch_KineticEnergyFFT` |
| `VelocityFFT_<region>.npz` / `.png`, `Velocity_polarization_<region>.png` | `batch_VelocityFFT` |

At the **dataset root** (next to the run folders):

| File | From |
| --- | --- |
| `KineticEnergy_summary_<region>.csv` | `batch_KineticEnergy` |
| `KineticEnergyFFT_summary_<region>.csv` | `batch_KineticEnergyFFT` |
| `VelocityFFT_summary_<region>.csv` | `batch_VelocityFFT` |
| `KineticEnergy_vs_<xaxis>_<region>[_normalized].png` | resonance figure |
| `Thumbnails_<figure>.png` | `make_thumbmail_summary` |

> Summaries are written as **`.csv` only**. Excel output was removed — some
> pandas/openpyxl versions silently wrote boolean columns (`processed`,
> `calibrated`) as blank cells.

## 14. Library reference — `piv_postprocessing_lib.py`

**Constants** — `PIV_FILENAME`, `LOG_FILENAME`, `XSCALE`/`YSCALE` (m/px),
`PTS_ROI` (ROI corners in metres), `FRAMES_PER_FIELD`,
`FROT_DIVISOR`/`FLIB_DIVISOR` (legacy names), `REGIONS`, `R` (cylinder radius, m),
`H` (container height, m), `k0` (dimensionless azimuthal wavenumber = 6),
`nu` (kinematic viscosity, m²/s), `UNCAL_*` fallbacks. The library defines **none**
of these as literals — they are loaded at import from
`param_postProcessing_default.json` (next to the notebooks) and overridden per
dataset by `read_paramPostprocessing(dataset_dir)`, which rebinds these module
globals. Edit the JSON files, not the `.py`. The physical wavenumber `k = k0·π/R`
and wavelength `l = 2π/k` are **derived** — only `k0` is stored, and `k`/`l` are
recomputed from `k0` and `R` on every load, so editing `R` (or `k0`) alone keeps
them consistent. **The default file must exist** (import fails without it) and
should be kept in sync when a dataset's file changes.

**Parameter file:** `read_paramPostprocessing(base_dir, apply=True)` →
namespace of the constants (`P.XSCALE`, `P.PTS_ROI`, …), also rebinding the
module globals so the library's own helpers use the dataset's values;
`write_paramPostprocessing(base_dir, overwrite=False)` writes a fresh JSON from
the defaults.

**Normalisation (`_star` columns/arrays).** Every summary row and per-run `.npz`
carries both raw and normalised quantities. Frequencies are divided by `f_rot`
(`flib_star`, `f_peak_star`, `f_low_star`, and the `f_star` array), velocities by
`U0` (`amp_u_star` = `amp_u`/`U0`), and kinetic energy by `U0²`
(`mean_Ekin_star`, `Ek_frame_star`), with `U0 = libration_velocity_scale`. Each
summary also gets the eight `dimensionless_numbers` columns.

**Figures: format + two versions.** Each figure-producing notebook has a
`FIG_FORMAT = 'png'` config option (`'png'` or `'pdf'`) and writes **every figure
twice** — a raw version and a `_normalized` one (built via
`figure_filename(stem, fmt, normalized=...)`). The normalised spectra use
`f / f_rot` on the frequency axis and divide amplitude by `U0` (velocity) or `U0²`
(kinetic energy); the velocity colormaps' normalised version is the `f/f_rot`
frequency axis. So e.g. `VelocityFFT_ROI.png` now comes with
`VelocityFFT_ROI_normalized.png`.

**Key functions:**

| Function | Purpose |
| --- | --- |
| `load_piv(path)` | `X, Y, U, V, nframes` from a PIVlab `.mat`. Prefers `u_filt`/`v_filt`, then `u_filtered`/`v_filtered`; falls back to the unfiltered `u`/`v` (or `u_original`/`v_original`) with a printed message, and prints *"No velocity field found"* if there is none. Handles both the 3-D-array and cell-array layouts. |
| `read_acquisition_params(log)` | `(dt_pulse, fps, ok)` from `acquisition_log.txt`. **`fps = cam_fps / 2`** — PIVlab pairs images, so one velocity field is produced every two camera frames. |
| `region_fields(...)`, `region_tag`, `regions_to_run` | ROI/FULL cropping, filename tag, and `'BOTH'` expansion. |
| `parse_run_name`, `parse_frot`, `parse_flib` | Physical parameters from a folder name (both naming conventions). |
| `compute_fft`, `amp_from_rfft`, `peak_freq` | One-sided amplitude spectrum (`|FFT|·2/N`), peak ignoring DC. |
| `fft_axis_limits(freq, amp, frot, flib)` | Axis limits for spectra: x to `max(4·f_rot, 4·f_lib)`, y to decade bounds. |
| `libration_ke_scale(dphi_deg, flib)` | `(δφ[rad]·2π·f_lib·R)²`, the Ek normalization scale. |
| `libration_velocity_scale(flib, dphi_deg)` | `U0 = 2π·f_lib·R·δφ[rad]`, the velocity scale for `U_star = U/U0` (its square is `libration_ke_scale`). |
| `dimensionless_numbers(frot, flib, dphi)` | Dict of the run's dimensionless numbers — `E`, `E_l` (Ekman, height/wavelength), `delta_nu_m` (viscous BL), `U0_mps`, `Ro`, `Re`, `Re_l`, `Re_bl` — added to every summary row. |
| `figure_filename(stem, fmt, normalized)` | `'<stem>[_normalized].<fmt>'` — the shared figure-naming/format helper (`fmt` = `'png'`/`'pdf'`). |
| `read_KineticEnergy(path)`, `read_VelocityFFT(path)` | Load an output `.npz`, print its variables, return them as a dict. |
| `calibrate`, `create_mask`, `extract_roi`, `resolve_npz`, `compute_polarization` | Supporting helpers. |

## 15. Troubleshooting (Python)

| Symptom | Likely cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` | Wrong interpreter — you are on the system Python. Select `~/anaconda3/bin/python` (section 9). |
| `ImportError: cannot import name 'X' from piv_postprocessing_lib` | Stale kernel holding the old module. **Restart the kernel.** |
| `ModuleNotFoundError: No module named 'piv_postprocessing_lib'` | Notebook started outside its own folder. |
| `no filtered velocity found using the unfiltered velocities` | Normal message: the `.mat` has no `u_filt`/`u_filtered`, so raw `u`/`v` were used. |
| `No velocity field found` | The `.mat` holds no velocity at all — wrong file, or PIV never ran. |
| Batch reports every run as `[skip] no PIVlab_results_uncalibrated.mat` | The `.mat` is named differently in those folders; rename it or change `PIV_FILENAME`. |
| Two near-identical figures from a batch | `REGION = 'BOTH'` — one is ROI, the other FULL (see the bold heading on each). |
| A summary row named `obsolete` with NaNs | The batch treats every sub-folder as a run; non-run folders show up as unprocessed rows and are dropped from the plots. |
| `f*` looks wrong / NaN | The folder name does not match section 10, so `frot`/`flib` could not be parsed. |
