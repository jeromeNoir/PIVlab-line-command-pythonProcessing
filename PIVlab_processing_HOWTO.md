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
- **Notebooks + auto-generated `.py` twins.** Each tool is a notebook plus a
  `.py` twin regenerated with `python ipynb_to_py.py` (figures saved, never
  shown; magics stripped). Edit the NOTEBOOK and regenerate — never the twin.
- `obsolete/` holds every retired tool (the old `batch_*`/`plot_*`/
  `colormaps_*` generation and their docs). Ignore it unless you need one
  specific file.

## 10. Run-folder naming

Post-processing reads the physical parameters straight from the folder name:

```
frot0.50Hz_flib0.400Hz_dphi2deg_SS1
   │         │           │      └── acquisition index; SS2, SS3… mark repeats
   │         │           └───────── libration amplitude δφ  [deg]
   │         └───────────────────── libration frequency f_lib [Hz]
   └─────────────────────────────── rotation frequency f_rot [Hz]
```

`parse_run_name()` returns `(frot_Hz, flib_Hz, dphi_deg)`; the summary adds
`flib_star = flib/frot`. Only this decimal-Hz form is parsed — the legacy
zero-padded form (`frot050_flib0400_…`) was retired together with the divisor
parameters.

## 11. Regions: ROI vs FULL — one mask, one file

`PTS_ROI` (in the dataset's `param_postProcessing.json`, in metres, **in the
rotated frame** — see section 12) defines the analysis rectangle; pick it with
`select_ROI_quiver.py`. Processing builds `MASK_ROI` (1 inside, NaN outside)
and stores BOTH regions' statistics in the SAME per-run `.npz` — there are no
per-region files or `REGION` batch options any more. The ROI-averaged spectra
(`*_ROIaveraged`) are what the viewers and the resonance curves read; a
rectangle that selects nothing falls back to the full field with a warning.

## 12. The notebooks

Run `PIV_processing` first: it creates the `.npz` files every other tool
reads.

| Tool | What it does |
| --- | --- |
| `PIV_processing.ipynb` | **Start here.** `BATCH = True` processes every run of every dataset in the `BASE_DIRS` list; `BATCH = False` only `RUN_DIR`. Each run's `.mat` is loaded once, the fields are **rotated by the dataset's `ROTATE`** (0/±90/180°, so everything downstream lives in the final frame), and the `'KE'` and `'VELOCITY'` analyses write `KineticEnergy.npz` + `Velocity.npz` per run plus one `Runs_summary.csv` per dataset. `REPROCESS_ALL = False` reuses cached `.npz`. Every stored variable is documented in `FILE_STRUCTURE_VELOCITY.md` / `FILE_STRUCTURE_ENERGY.md`. |
| `MAPS_VELOCITY.ipynb` | Mean/std velocity maps (`MAP_COMPONENT = 'U'`, `'V'` or `'both'`), ROI overlaid, ±θ inertial-wave characteristics on the **std** panels (at `f_lib`, or at `FREQ` when set). |
| `MAPS_FFT.ipynb` | Spatial FFT maps: amplitude of `|FFT(U)|+|FFT(V)|` at `f_lib` (with the characteristics) and the band integral over `[FMIN, FMAX]` (`PEAK_SELECT = False` → `[DELTA_F, f_lib − DELTA_F]`), each panel with its marginal profile. |
| `PLOT_FFT.ipynb` | ROI-averaged spectra: `FFT_U + FFT_V` (velocity) and `FFT_EK` (energy) vs frequency, with the forcing guides and the stored detected peaks. |
| `POLARIZATION.ipynb` | Polarization `P_V/P_U` from the stored per-point spectra, thresholded (`MIN_FFT_AMP` [m/s]: weaker `FFT_U`/`FFT_V` samples become NaN and drop out), against the inertial-wave relation. |
| `PHASE_AVERAGE.ipynb` | Phase-averages `U`, `V` at the libration period (all complete periods, one bin per frame interval) -> `Velocity_phaseAveraged.npz` (`UPA`, `VPA`, `PHASE`, `N_SAMPLES` + shared header) and the phase-0 figure: velocity magnitude + quiver + ROI. |
| `RES_CURVES.ipynb` | Resonance curves over a `BASE_DIRS` list: velocity amplitude at `f_lib` and energy amplitude at `2 f_lib` vs `f_lib` — per-dataset figures (symbols distinguish repeated `SSn` runs) plus overlay figures with mean ± std; save-only. |
| `select_ROI_quiver.py` | Interactive ROI picker: quiver over the background image, **in the rotated frame**; writes `PTS_ROI` into the dataset's parameter file. Run it from a terminal. |
| `READ_velocityFile.py` / `READ_energyFile.py` | Load a `.npz` and return every stored variable under its in-file name. |

The viewers (`MAPS_*`, `PLOT_FFT`, `POLARIZATION`, `PHASE_AVERAGE`) share the `BATCH` switch: `False` draws the
single `RUN_DIR` and SHOWS the figures, `True` sweeps every run of `BASE_DIR`
and only saves them. Other common options: `SAVE`, `FIG_FORMAT`
(`'png'`/`'pdf'`).

**DIM / NODIM.** Every viewer figure is written twice: `_DIM` in physical
units and `_NODIM` non-dimensional, where every quantity is simply starred
(`x*`, `z*`, `f*`, amplitude`*`; lengths / `R`, frequencies / `f_rot`,
velocities / `U₀`, energies / `EK_SCALE`) with no unit suffixes. Nothing in
the `.npz` files is ever stored normalised.

## 13. Where the outputs go

Per run, in `<run>/PostProcessing/`:

| File | From |
| --- | --- |
| `KineticEnergy.npz`, `Velocity.npz` | `PIV_processing` |
| `MAPS_VELOCITY_DIM/_NODIM.png` | `MAPS_VELOCITY` |
| `MAPS_FFT_DIM/_NODIM.png` | `MAPS_FFT` |
| `PLOT_FFT_velocity_DIM/_NODIM.png`, `PLOT_FFT_energy_DIM/_NODIM.png` | `PLOT_FFT` |
| `POLARIZATION_DIM/_NODIM.png` | `POLARIZATION` |
| `Velocity_phaseAveraged.npz`, `PHASE_AVERAGE_DIM/_NODIM.png` | `PHASE_AVERAGE` |

At the **dataset root** (next to the run folders):

| File | From |
| --- | --- |
| `Runs_summary.csv` | `PIV_processing` |
| `RES_CURVE_ENERGY.png`, `RES_CURVE_VELOCITY.png` | `RES_CURVES` |
| `ROI_selection.png` | `select_ROI_quiver` |

At the datasets' **common parent**, when `RES_CURVES` is given several
datasets: `RES_CURVE_ENERGY_ALL.png`, `RES_CURVE_VELOCITY_ALL.png` (one curve
per dataset, repeated runs collapsed to mean ± std).

> Summaries are written as **`.csv` only**. Excel output was removed — some
> pandas/openpyxl versions silently wrote boolean columns (`processed`,
> `calibrated`) as blank cells.

### The dataset summary — `Runs_summary.csv`

`PIV_processing` maintains **one** summary CSV per
dataset (`Runs_summary.csv` at the dataset root — it replaces the earlier
`KineticEnergy_summary.csv` / `Velocity_summary.csv` pair). One row per run,
holding only run parameters and dimensionless control parameters — the
measured quantities stay in the per-run `.npz` files. A batch replaces the
rows of the runs it processed and keeps every other row; the single-run mode
(`BATCH = False`) does the same for its one run, so both modes create/update
the same file.

Columns: `run`, `run indx`, `processed`, `k0`, `lambda`, `topo_top`,
`topo_bottom`, `frot(Hz)`, `flib(Hz)`, `dphi(deg)`, `flib_star`
(= `flib/frot`), `Pulse_sep(s)` (PIV pulse separation), `PIV_fps(Hz)`,
`nframe`, `U_SCALE (m/s)`, `V_SCALE (m/s)`, `LENGTH_SCALE`, `TIME_SCALE`,
and the dimensionless control parameters below.

**Dimensionless control parameters** (computed by `control_parameters` in
`piv_postprocessing_lib` from the run's forcing — `frot`, `flib`, `dphi` —
and the dataset constants `R`, `H`, `nu`, `k0`), with
`U_SCALE = 2π·flib·R·dphi[rad]` the peak libration wall velocity:

| column | definition |
| --- | --- |
| `lambda` | topography wavelength `2·R / k0` [m] |
| `Ekman` | `nu / (2π·frot·H²)` |
| `BL thickness (m)` | `H·√Ekman` |
| `Rossby` | `U_SCALE / (2π·frot·R)` |
| `TOPO Rossby` | `Rossby · R / lambda` |
| `TOPO Reynolds` | `U_SCALE · lambda / nu` |
| `BL Reynolds` | `U_SCALE · BL thickness / nu` |

Anything requiring `frot` is NaN when `frot` is unknown or zero; the TOPO
numbers are NaN for a full cylinder (`k0 = 0`, `lambda` undefined).

## 14. Library reference — `piv_postprocessing_lib.py`

**Constants** — `PIV_FILENAME`, `LOG_FILENAME`, `XSCALE`/`YSCALE` (m/px),
`ROTATE` (rotation applied to the PIV fields by `PIV_processing` and
`select_ROI_quiver`, deg: 0, ±90 or 180),
`PTS_ROI` (ROI corners in metres), `FRAMES_PER_FIELD`, `REGIONS`, `R` (cylinder radius, m),
`H` (container height, m), `k0` (dimensionless azimuthal wavenumber),
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

**Nothing is stored calibrated or normalised.** Every `.npz` field is in
native PIV units (px, px/frame, frames, 1/frame) together with the
multiplicative calibration factors (`XCAL`, `YCAL`, `TCAL`, `UCAL`, `VCAL`,
`ECAL`, `FCAL`; native → physical) and the non-dimensional divisors
(`U_SCALE = U₀ = 2π·f_lib·R·δφ[rad]`, `V_SCALE`, `LENGTH_SCALE = R`,
`TIME_SCALE = 1/f_rot`, `F_SCALE = f_rot`, `EK_SCALE`). The viewers calibrate
on the fly; the summary carries the `control_parameters` columns (section
13).

**Figures: format + two versions.** Each figure-producing notebook has a
`FIG_FORMAT` option (`'png'`/`'pdf'`) and writes every figure twice via
`figure_filename(stem, fmt, normalized=...)`: `<stem>_DIM.<fmt>` (physical
units) and `<stem>_NODIM.<fmt>` (non-dimensional, starred labels — section
12).

**Key functions:**

| Function | Purpose |
| --- | --- |
| `load_piv(path)` | `X, Y, U, V, nframes` from a PIVlab `.mat`. Prefers `u_filt`/`v_filt`, then `u_filtered`/`v_filtered`; falls back to the unfiltered `u`/`v` (or `u_original`/`v_original`) with a printed message. Handles both the 3-D-array and cell-array layouts; `VALIDATE_VELOCITY` picks filtered vs original-with-NaN. |
| `rotate_fields(X, Y, U, V, rotate)` | The dataset's `ROTATE` applied to freshly loaded fields — 180: `max(x)−x`, `max(z)−z`, `−u`, `−v`; −90: `max(z)−z`, `x`, `−v`, `u`; +90: `z`, `max(x)−x`, `v`, `−u` (±90 also transpose the arrays). Shared by `PIV_processing` and `select_ROI_quiver`. |
| `read_acquisition_params(log)` | `(dt_pulse, fps, ok)` from `acquisition_log.txt`. **`fps = cam_fps / FRAMES_PER_FIELD`** — PIVlab pairs images, so one velocity field per two camera frames. |
| `parse_run_name`, `parse_frot`, `parse_flib` | Physical parameters from a folder name (decimal-Hz form only). |
| `peak_freq`, `amp_at_freq`, `peak_freq_in_band` | Spectrum maximum (ignoring DC), amplitude at a target frequency, peak inside a band. |
| `fft_guide_lines(frot, flib, f_low)` | The dashed guide lines (`f_rot`, `f_lib`, `2 f_lib`, `f_low` and its sidebands) shared by the spectra viewers. |
| `fft_axis_limits(freq, amps, frot, flib)` | Axis limits for spectra: x to `max(4·f_rot, 4·f_lib)`, y to decade bounds. |
| `libration_velocity_scale(flib, dphi)` / `libration_ke_scale(dphi, flib)` | `U₀ = 2π·f_lib·R·δφ[rad]` and its square. |
| `control_parameters(frot, flib, dphi)` | The summary's dimensionless control parameters (section 13). |
| `dimensionless_numbers(frot, flib, dphi)` | The older `E`, `E_l`, `delta_nu_m`, `U0_mps`, `Ro`, `Re`, `Re_l`, `Re_bl` set, printed by the viewers. |
| `figure_filename(stem, fmt, normalized)` | `'<stem>_DIM.<fmt>'` / `'<stem>_NODIM.<fmt>'` — the shared figure-naming helper (`fmt` = `'png'`/`'pdf'`). |
| `topography_arrangement(path)` | `(top_topo, bottom_topo)` from the dataset folder name. |
| `resolve_npz`, `create_mask`, `compute_polarization` | Supporting helpers. |

## 15. Troubleshooting (Python)

| Symptom | Likely cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` | Wrong interpreter — you are on the system Python. Select `~/anaconda3/bin/python` (section 9). |
| `ImportError: cannot import name 'X' from piv_postprocessing_lib` | Stale kernel holding the old module. **Restart the kernel.** |
| `ModuleNotFoundError: No module named 'piv_postprocessing_lib'` | Notebook started outside its own folder. |
| `no filtered velocity found using the unfiltered velocities` | Normal message: the `.mat` has no `u_filt`/`u_filtered`, so raw `u`/`v` were used. |
| `No velocity field found` | The `.mat` holds no velocity at all — wrong file, or PIV never ran. |
| Batch reports every run as `[skip] no PIVlab_results_uncalibrated.mat` | The `.mat` is named differently in those folders; rename it or change `PIV_FILENAME`. |
| `[warn] PTS_ROI selects nothing -> ROI mask covers the FULL field` | The ROI rectangle lies outside the (rotated) field — re-pick it with `select_ROI_quiver`, which works in the rotated frame. |
| A summary row with `processed = False` and NaNs | The batch treats every sub-folder as a run; non-run folders show up as unprocessed rows. |
| `flib_star` looks wrong / NaN | The folder name does not match section 10, so `frot`/`flib` could not be parsed (the legacy zero-padded names are no longer accepted). |
