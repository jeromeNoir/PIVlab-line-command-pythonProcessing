# How to use the PIVlab command-line processing scripts

This guide explains how to run the two PIV processing scripts used in this
project:

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
| `run_folder`    | Name of the specific run sub-folder to process.                     | `frot050_flib0405_dphi2deg_SS1`                                |
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

| Variable        | Meaning                                                          | Example                                              |
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

| Symptom                                             | Likely cause / fix                                                                 |
| --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `Undefined ... preproc.PIVlab_preproc` (etc.)       | PIVlab code not on the path — `addpath(genpath(...))` the `PIVLAB_commandLine` dir. |
| `No images found`                                   | Wrong `file_pattern` or wrong `image_folder`/`project_root`; volume not mounted.    |
| Odd-image-count error / skip                        | A frame is missing or extra; PIV needs an even count (pairs). Check the folder.     |
| `No sub-folders found in project_root` (batch)      | `project_root` is wrong, empty, or only contains files/hidden folders.              |
| Results not appearing                               | Check `local_folder` is writable; the scripts create the per-run folder if missing. |
| Velocities look wrong/clipped                       | Adjust `valid_vel` limits and the outlier thresholds (`stdthresh`, `neigh_thresh`). |
