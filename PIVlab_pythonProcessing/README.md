# PIVlab Python post-processing

Python toolchain that turns the `PIVlab_results_uncalibrated.mat` files
produced by the MATLAB PIVlab scripts into per-run `.npz` archives, summary
tables and figures, for the cylinder-libration experiments.

For the full workflow (including the MATLAB PIV step) see
[`../PIVlab_processing_HOWTO.md`](../PIVlab_processing_HOWTO.md). This README
is the reference: what each routine does, every variable of the parameter
file, and every variable stored in the `.npz` files.

---

## 1. How to process data

### Data layout

Each **dataset** is a folder (a `BASE_DIR`, e.g. `k6_TopBottom/`) holding one
`param_postProcessing.json` and the **run** sub-folders named

```
frot0.50Hz_flib0.400Hz_dphi2deg_SS1
   │         │           │      └── acquisition index (SS2, SS3… = repeats)
   │         │           └───────── libration amplitude δφ [deg]
   │         └───────────────────── libration frequency f_lib [Hz]
   └─────────────────────────────── rotation frequency f_rot [Hz]
```

Each run folder contains the PIVlab `.mat` file and `acquisition_log.txt`.
All datasets sit under a common `ROOT_DIR`; the batch tools carry the list of
datasets as `BASE_DIRS`.

### The pipeline, in order

1. **`select_ROI_quiver.py`** (once per dataset, from a terminal) — pick the
   analysis rectangle on the rotated velocity field; writes `PTS_ROI` into
   the dataset's `param_postProcessing.json`.
2. **`PIV_processing.ipynb`** — reads each run's `.mat`, rotates the fields
   by the dataset's `ROTATE`, and writes `Velocity.npz` +
   `KineticEnergy.npz` in `<run>/PostProcessing/` plus one
   `Runs_summary.csv` per dataset. Everything downstream reads these `.npz`.
3. **Viewers** (any order): `MAPS_VELOCITY`, `MAPS_FFT`, `PLOT_FFT`,
   `POLARIZATION`, `PHASE_AVERAGE` — per-run figures (and, for
   `PHASE_AVERAGE`, the `Velocity_phaseAveraged.npz`).
4. **Dataset-level products**: `RES_CURVES` (resonance curves),
   `MAKE_thumbmails` (A4 contact sheets of the per-run figures).

### The shared `BATCH` pattern

`PIV_processing` and every viewer share the same config cell:

| option | meaning |
| --- | --- |
| `BATCH = True` | sweep every run of every dataset in `BASE_DIRS`; figures are **saved only** |
| `BATCH = False` | process the single `RUN_DIR`; figures are saved **and shown** |
| `ROOT_DIR` / `BASE_DIRS` | common parent / list of dataset folders |
| `SAVE`, `OVERWRITE_FIG` | write figures / overwrite existing ones |
| `FIG_FORMAT` | `'png'` or `'pdf'` |

`RES_CURVES` and `MAKE_thumbmails` are dataset-level: no `BATCH`/`RUN_DIR`,
they always sweep `BASE_DIRS`. Each run is processed inside `try/except`, so
one bad run never aborts a batch.

**DIM / NODIM.** Every viewer figure is written twice:
`<stem>_DIM.<fmt>` in physical units and `<stem>_NODIM.<fmt>` non-dimensional,
where every quantity is simply starred (`x*`, `z*`, `f*`, amplitude`*`;
lengths /`R`, frequencies /`f_rot`, velocities /`U₀`, energies /`EK_SCALE`).

### Practical rules

- Use the Anaconda interpreter (`~/anaconda3/bin/python`); the system Python
  lacks numpy/pandas/matplotlib.
- Run notebooks **from this folder** so `import piv_postprocessing_lib`
  resolves; restart the kernel after editing the library.
- Each tool is a **notebook + auto-generated `.py` twin**. Edit the notebook,
  then regenerate the twin with `python ipynb_to_py.py <notebook.ipynb>`
  (Agg backend, figures saved never shown) — never edit the `.py` by hand.
- `obsolete/` holds retired tools — ignore it.

---

## 2. The routines

### `PIV_processing.ipynb` — start here

Turns each run's PIVlab `.mat` into the two `.npz` archives.
Per run: load the `.mat` (`load_piv`), **rotate** `X, Y, U, V` by the
dataset's `ROTATE` (all stored fields live in the rotated frame), read
`acquisition_log.txt` for the time base, build the ROI mask from `PTS_ROI`,
then run the two analyses:

- `'VELOCITY'` → `Velocity.npz` (fields, time statistics, per-point spectra,
  ROI/FULL-averaged spectra, detected peaks — section 4);
- `'KE'` → `KineticEnergy.npz` (kinetic energy `0.5(U²+V²)` series, maps and
  spectra — section 5).

It also creates/updates the per-dataset `Runs_summary.csv`: one row per run
with the run parameters and the dimensionless control parameters (Ekman,
boundary-layer thickness, Rossby, TOPO Rossby, TOPO/BL Reynolds). A batch
replaces only the rows of the runs it processed. Specific options:
`ANALYSES = ('KE', 'VELOCITY')`, `REPROCESS_ALL` (`False` = keep existing
`.npz`), peak detection (`DELTA_F`, `F_LOW_MIN`, `THRESHOLD_PEAK`).

### `MAPS_VELOCITY.ipynb`

Maps of the per-point **time** mean and std of the velocity
(`MAP_COMPONENT = 'U'`, `'V'` or `'both'`), with the ROI rectangle overlaid.
The ±θ inertial-wave characteristics (`sin θ = f / 2f_rot`, θ from the
z-axis) are drawn on the **std** panels only — at `f_lib`, or at `FREQ` when
set. Outputs `MAPS_VELOCITY_DIM/_NODIM`.

### `MAPS_FFT.ipynb`

Spatial maps of the velocity-spectrum amplitude `|FFT(U)| + |FFT(V)|`:
panel 1 at `f_lib` (with the characteristics at `FREQ` or `f_lib`), panel 2
the band integral over `[FMIN, FMAX]` divided by the bandwidth
(`PEAK_SELECT = False` → the band `[DELTA_F, f_lib − DELTA_F]`). Each panel
carries its horizontal-marginal profile; ROI overlaid. Outputs
`MAPS_FFT_DIM/_NODIM`.

### `PLOT_FFT.ipynb`

ROI-averaged spectra vs frequency: velocity (`FFT_U + FFT_V`) and kinetic
energy (`FFT_EK`), with the forcing guide lines (`f_rot`, `f_lib`, `2f_lib`,
`f_low` + sidebands) and the stored detected peaks. `FMAX` limits the axis.
Outputs `PLOT_FFT_velocity_DIM/_NODIM` and `PLOT_FFT_energy_DIM/_NODIM`.

### `POLARIZATION.ipynb`

Polarization `(FFT_V / FFT_U)²` from the stored per-point spectra, compared
with the inertial-wave prediction. Samples whose `FFT_U` or `FFT_V` amplitude
is below `MIN_FFT_AMP` (physical units, m/s) are set to NaN and drop out.
Outputs `POLARIZATION_DIM/_NODIM`.

### `PHASE_AVERAGE.ipynb`

Phase-averages `U`, `V` at the libration period `T = 1/f_lib` over **all
complete periods** in the record: `T_frames = PIV_fps / f_lib` (generally
non-integer), `n_phase = floor(T_frames)` bins, frame `k` lands in the bin of
its phase `(k mod T_frames)/T_frames`. Writes `Velocity_phaseAveraged.npz`
(section 6) and the phase-0 figure: velocity-magnitude background + quiver
(`QUIVER_SKIP`, `QUIVER_SCALE`, `QUIVER_COLOR`) + ROI + the ±θ
characteristics at `f_lib`. Outputs `PHASE_AVERAGE_DIM/_NODIM`.

### `RES_CURVES.ipynb`

Resonance curves per dataset and overlays across datasets: velocity amplitude
of `FFT_TOTAL_ROIaveraged` at `f_lib` and energy amplitude of
`FFT_EK_ROIaveraged` at `2f_lib`, plotted vs `f_lib`. Per-dataset figures use
one symbol per `SSn` repeat and one series per `δφ`
(`RES_CURVE_VELOCITY.png`, `RES_CURVE_ENERGY.png` at the dataset root); the
overlays collapse repeats to mean ± std (`RES_CURVE_*_ALL.png` at the common
parent). Save-only, never shown.

### `MAKE_thumbmails.ipynb`

A4 contact sheets: for each entry of `GROUPS` (a figure stem + a sheet
title), tiles that figure from every run into a multi-page
`Thumbnails_<stem>.pdf` at the dataset root, runs sorted by
`f_rot`/`f_lib`/`δφ` and annotated. Current groups: `PLOT_FFT_velocity_DIM`,
`PLOT_FFT_energy_DIM`, `MAPS_FFT_DIM`, `MAPS_VELOCITY_DIM`,
`PHASE_AVERAGE_DIM`. Options: page orientation, `DPI`, `SOURCE_EXT`,
`EXCLUDE` (folder-name substrings to skip), `OVERWRITE`.

### `select_ROI_quiver.py` (terminal script)

Interactive ROI picker: shows the background image and quiver **in the
rotated frame** (same `rotate_fields` rules as `PIV_processing`), lets you
drag the rectangle, and writes the two corners as `PTS_ROI` [m] into the
dataset's `param_postProcessing.json`. Also saves `ROI_selection.png` at the
dataset root. Run it whenever the camera frame or `ROTATE` changes.

### `READ_velocityFile.py` / `READ_energyFile.py`

`read_velocity_file(path)` / `read_energy_file(path)` load a `.npz` and
return a plain `{name: value}` dict (0-d arrays unwrapped to scalars). Run
from the command line they print the full variable inventory of a file.

### `piv_postprocessing_lib.py` (shared library)

Every helper the notebooks share: `read_paramPostprocessing` (loads a
dataset's parameter file and rebinds the module globals),
`load_piv`, `rotate_fields`, `read_acquisition_params`
(**`fps = cam_fps / FRAMES_PER_FIELD`** — image pairing halves the camera
rate), `parse_run_name`, `create_mask`, the spectra helpers (`peak_freq`,
`amp_at_freq`, `fft_guide_lines`, `fft_axis_limits`), the scales
(`libration_velocity_scale`, `libration_ke_scale`), `control_parameters`,
`dimensionless_numbers`, `figure_filename`, `topography_arrangement`,
`resolve_npz`. It defines **no parameter values** — they come from the JSON
files (section 3).

### `ipynb_to_py.py`

Regenerates a notebook's `.py` twin: `python ipynb_to_py.py <notebook.ipynb>`.
The twin uses the Agg backend, strips magics, saves figures instead of
showing them, and carries a "do not edit by hand" banner.

---

## 3. The parameter file — `param_postProcessing.json`

One per dataset folder, next to the run sub-folders. Loaded by
`read_paramPostprocessing(BASE_DIR)`, which returns the values **and**
rebinds them inside the library so every helper uses the dataset's values.
If a dataset has no file, the default
(`param_postProcessing_default.json`, in this folder) is copied there and the
tool stops so you can edit it. **Keep the default file in sync** when a
dataset's file gains or loses a key.

| key | type / unit | meaning |
| --- | --- | --- |
| `PIV_FILENAME` | str | name of the PIVlab result file in each run folder (`PIVlab_results_uncalibrated.mat`) |
| `LOG_FILENAME` | str | name of the acquisition log (`acquisition_log.txt`), source of pulse separation and camera fps |
| `XSCALE`, `YSCALE` | m/px | geometric calibration of the images |
| `ROTATE` | deg (0, ±90, 180) | rotation applied to the raw PIV fields by `PIV_processing` and `select_ROI_quiver`, so all stored fields live in the final (lab) frame. 180: `x→max(x)−x, z→max(z)−z, u→−u, v→−v`; −90: `x→max(z)−z, z→x, u→−v, v→u`; +90: `x→z, z→max(x)−x, u→v, v→−u` |
| `PTS_ROI` | 2×2, m | two opposite corners `[[x1, z1], [x2, z2]]` of the analysis rectangle, **in the rotated frame**; pick with `select_ROI_quiver.py` |
| `FRAMES_PER_FIELD` | int | camera frames per velocity field (2 — PIVlab pairs images, so the PIV field rate is `cam_fps / 2`) |
| `UNCAL_DT`, `UNCAL_FPS`, `UNCAL_SCALE` | s, Hz, — | fallback values used when the acquisition log is missing/unreadable (run flagged `calibrated = False`, all `*CAL` = 1) |
| `REGIONS` | list | statistics regions stored in the `.npz` (`["ROI", "FULL"]` — both always in the same file) |
| `R` | m | cylinder radius (length scale; also sets the topography wavelength) |
| `H` | m | fluid column height (Ekman number, boundary-layer thickness) |
| `k0` | — | topography azimuthal wavenumber parameter; wavelength `λ = 2R/k0`; `k0 = 0` = full (smooth) cylinder |
| `nu` | m²/s | kinematic viscosity |
| `VALIDATE_VELOCITY` | bool | `true` = use PIVlab's filtered velocities (`u_filt`/`u_filtered`); `false` = the unfiltered originals with NaN where filtering removed vectors |

---

## 4. `Velocity.npz`

One per run in `<run>/PostProcessing/`, written by `PIV_processing`
(analysis `'VELOCITY'`). **Nothing is stored calibrated or normalised**:
everything is in native PIV units (px, px/frame, frames, 1/frame) with the
multiplicative calibration factors (`*CAL`, native → physical) and the
non-dimensionalisation divisors (`*_SCALE`, physical → dimensionless) stored
alongside. Shapes: `ny × nx` grid, `nt` fields, `nf = nt//2 + 1` FFT bins.
Full details: [`FILE_STRUCTURE_VELOCITY.md`](FILE_STRUCTURE_VELOCITY.md).

### Shared header (identical keys in `KineticEnergy.npz`)

| key | description |
| --- | --- |
| `run`, `run_idx` | run-folder name; `SSn` index (−1 if absent) |
| `calibrated` | acquisition log read OK; if `False` every `*CAL` = 1 and no peaks are stored |
| `ROTATE` | rotation applied to the stored fields [deg] — everything is in the rotated frame |
| `frot_Hz`, `flib_Hz`, `dphi_deg` | forcing parameters, from the folder name |
| `R`, `H`, `nu`, `k0`, `lambda` | dataset constants; `lambda = 2R/k0` [m], NaN for `k0 = 0` |
| `top_topo`, `bottom_topo` | topography on the top / bottom lid (from the dataset name) |
| `PIV_FPS` | PIV field rate [Hz] = `cam_fps / FRAMES_PER_FIELD` |
| `DT_VEL` | pulse separation inside an image pair [s] — the velocity time base |
| `TSCALE` | `1 / PIV_FPS` [s per field] — the series/FFT time base |
| `XSCALE`, `YSCALE` | geometric calibration [m/px] |
| `pts_ROI`, `MASK_ROI` | ROI corners [m]; mask map (1 inside, NaN outside) |
| `X`, `Y` | (ny, nx) grid positions [px]; `× XCAL`/`YCAL` → m |
| `nframes`, `ny`, `nx`, `npoints_ROI`, `npoints_FULL` | dimensions and point counts |
| `XCAL`, `YCAL` | px → m (`= XSCALE`, `YSCALE`) |
| `TCAL` | frames → s (`= 1/PIV_FPS`) |
| `UCAL`, `VCAL` | px/frame → m/s (`= XCAL/DT_VEL`, `YCAL/DT_VEL`) |
| `ECAL` | px²/frame² → m²/s² (`= 0.5(UCAL² + VCAL²)`) |
| `FCAL` | 1/frame → Hz (`= 1/TCAL`) |
| `U_SCALE`, `V_SCALE` | `U₀ = 2π·flib·R·δφ[rad]`, the peak libration wall velocity — divides velocities |
| `LENGTH_SCALE`, `TIME_SCALE`, `F_SCALE` | `R`; `1/frot`; `frot` |
| `EK_SCALE` | `0.5(U_SCALE² + V_SCALE²)` — divides kinetic energy |

### Velocity-specific variables

`mean`/`std` = statistics over **time**; `averaged` = mean over **space**
(NaN-ignoring; ROI = with `MASK_ROI`, FULL = whole grid). No `TIME` axis is
stored — rebuild as `np.arange(nframes) * TCAL` [s].

| key | shape | description |
| --- | --- | --- |
| `U`, `V` | (ny, nx, nt) | full native velocity fields [px/frame]; `× UCAL`/`VCAL` → m/s |
| `Umean`, `Vmean` | (ny, nx) | per-point time mean |
| `Ustd`, `Vstd` | (ny, nx) | per-point time std |
| `FREQ` | (nf,) | frequency axis [1/frame]; `× FCAL` → Hz |
| `FFT_U`, `FFT_V` | (ny, nx, nf) | per-point one-sided FFT amplitude of U(t), V(t) |
| `FFT_U_ROIaveraged`, `FFT_V_ROIaveraged` | (nf,) | ROI space average of the per-point spectra |
| `FFT_U_FULLaveraged`, `FFT_V_FULLaveraged` | (nf,) | full-field space average |
| `FFT_TOTAL_ROIaveraged`, `FFT_TOTAL_FULLaveraged` | (nf,) | `FFT_U_* + FFT_V_*` — peaks are detected on the ROI one |
| `window`, `detrend` | scalars | FFT taper name; mean removed before FFT |
| `F_PEAKS`, `AMP_PEAKS`, `N_PEAKS` | (n_peaks,) | detected sub-harmonic peaks in `[F_LOW_MIN, flib − DELTA_F]`, sorted by decreasing amplitude (strongest = `f_low`); native units |

---

## 5. `KineticEnergy.npz`

One per run, written by `PIV_processing` (analysis `'KE'`). Kinetic energy
per grid point `Ek = 0.5(U² + V²)` in native px²/frame²; `× ECAL` → m²/s²,
then `/ EK_SCALE` → dimensionless. Carries the same shared header as
`Velocity.npz` (section 4). Full details:
[`FILE_STRUCTURE_ENERGY.md`](FILE_STRUCTURE_ENERGY.md).

Naming: `averaged`/`rms` = mean/std over **space**; `mean`/`std` = over
**time**; names read in operation order —
`EK_ROIaveraged_FFT` = FFT(⟨Ek⟩) (average first),
`FFT_EK_ROIaveraged` = ⟨FFT(Ek)⟩ (FFT first).

| key | shape | description |
| --- | --- | --- |
| `TIME` | (nt,) | time axis [frames]; `× TCAL` → s |
| `EK_ROIaveraged`, `EK_FULLaveraged` | (nt,) | space-averaged Ek vs time |
| `EK_ROIrms`, `EK_FULLrms` | (nt,) | space std of Ek vs time |
| `EK_FULLmean`, `EK_FULLstd` | (ny, nx) | per-point time mean / std maps |
| `FREQ` | (nf,) | frequency axis [1/frame]; `× FCAL` → Hz |
| `EK_ROIaveraged_FFT`, `EK_FULLaveraged_FFT` | (nf,) | FFT of the space-averaged series |
| `EK_ROIrms_FFT`, `EK_FULLrms_FFT` | (nf,) | FFT of the space-rms series |
| `FFT_EK` | (ny, nx, nf) | per-point FFT amplitude of Ek(t) |
| `FFT_EK_ROIaveraged`, `FFT_EK_FULLaveraged` | (nf,) | space average of `FFT_EK` (the ROI one feeds the resonance curves) |
| `FFT_EK_ROIrms`, `FFT_EK_FULLrms` | (nf,) | space std of `FFT_EK` |
| `detrend` | scalar | mean removed before the FFTs |

---

## 6. `Velocity_phaseAveraged.npz`

One per run, written by `PHASE_AVERAGE` next to `Velocity.npz`. Carries the
**shared header of `Velocity.npz`** (section 4: run identification, geometry,
`*CAL` factors, `*_SCALE` divisors — everything except the big fields, the
time statistics and the spectra) plus:

| key | shape | description |
| --- | --- | --- |
| `UPA`, `VPA` | (ny, nx, n_phase) | phase-averaged velocity fields [px/frame]; `× UCAL`/`VCAL` → m/s |
| `PHASE` | (n_phase,) | phase of each bin, as a fraction of the libration period `[0, 1)` |
| `TIME_PHASE` | (n_phase,) | bin centres [frames]; `× TCAL` → s |
| `N_SAMPLES` | (n_phase,) | number of velocity fields averaged into each bin |
| `n_phase` | scalar int | number of phase bins = `floor(PIV_fps / f_lib)` |
| `n_periods` | scalar int | complete libration periods used |
| `nframes_used` | scalar int | velocity fields actually used (`floor(n_periods · T_frames)`) |
| `frames_per_period` | scalar | `T_frames = PIV_fps / f_lib` [frames], generally non-integer |

---

## 7. Where the outputs go

Per run, in `<run>/PostProcessing/`: `Velocity.npz`, `KineticEnergy.npz`,
`Velocity_phaseAveraged.npz`, and the figures
(`MAPS_VELOCITY_*`, `MAPS_FFT_*`, `PLOT_FFT_velocity_*`, `PLOT_FFT_energy_*`,
`POLARIZATION_*`, `PHASE_AVERAGE_*`; each `_DIM` + `_NODIM`).

At the dataset root: `Runs_summary.csv`, `RES_CURVE_VELOCITY.png`,
`RES_CURVE_ENERGY.png`, the `Thumbnails_*.pdf` sheets, `ROI_selection.png`.

At the datasets' common parent: `RES_CURVE_VELOCITY_ALL.png`,
`RES_CURVE_ENERGY_ALL.png`.
