# PIV post-processing — HOW-TO

Python tools that turn PIVlab velocity exports into kinetic-energy and
velocity-spectrum analyses (time series, FFTs, resonance sweeps, polarization,
dimensionless numbers) with per-run figures and dataset-level summary tables.

Everything lives in this folder (`PIVlab_pythonProcessing/`): the notebooks, the
shared library `piv_postprocessing_lib.py`, and the default parameter file
`param_postProcessing_default.json`.

---

## 1. Requirements

**Python:** 3.11 (any 3.9+ works). Run the notebooks with an interpreter that has
the scientific stack — on this machine `~/anaconda3/bin/python`. The system
`/usr/bin/python3` has **none** of the packages and is the usual cause of
`ModuleNotFoundError: No module named 'pandas'`. `.vscode/settings.json` in this
folder already pins the anaconda interpreter; in VS Code use *Python: Select
Interpreter* if needed.

**Required modules**

| module | used for |
|---|---|
| `numpy` | arrays, FFTs |
| `scipy` | `scipy.io.loadmat/savemat` (read/write `.mat`), `scipy.signal` (band-pass filter) |
| `pandas` | summary tables (`.csv`) |
| `matplotlib` | all figures (`%matplotlib widget`/`qt`/`inline` per notebook) |

**Optional**

| module | used for |
|---|---|
| `pymupdf` (imported as `fitz`), or `pypdfium2`, or `pdf2image` | tiling **PDF** source figures in `make_thumbmail_summary` only. Not needed if the per-run figures are PNG (the default). |

Install the optional one with e.g. `~/anaconda3/bin/python -m pip install pymupdf`.

**Standard-library** modules also used: `os`, `re`, `json`, `glob`, `shutil`.

> **Run notebooks from this folder.** `import piv_postprocessing_lib` resolves
> because Jupyter uses the notebook's directory. Starting elsewhere breaks the
> import.

**Script twins.** Each notebook has a `.py` twin (`python batch_KineticEnergy.py`,
etc.) that runs the same pipeline headlessly — figures are **saved, not shown**
(matplotlib `Agg` backend). They are generated from the notebooks by
`python ipynb_to_py.py` (all) or `python ipynb_to_py.py <one>.ipynb`; regenerate
after editing a notebook, since the `.py` is a snapshot.

---

## 2. Setup — parameter files

The library holds **no** hardcoded calibration/ROI values. They live in JSON:

- **`param_postProcessing_default.json`** — next to the notebooks. The single
  source of default values, loaded once when the library is imported. **It must
  exist** (the library fails to import without it). Do not delete it.
- **`param_postProcessing.json`** — one **inside each dataset folder** (the
  directory that holds the run sub-folders, i.e. `BASE_DIR`). Overrides the
  defaults so two datasets can carry different calibrations.

Each tool loads its dataset's file right after the dataset path is set:

```python
_P = read_paramPostprocessing(BASE_DIR)     # or the parent of a single RUN_DIR
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
```

`read_paramPostprocessing` returns the values as a namespace (`_P.XSCALE`, …)
**and** rebinds the library's module globals, so the library's own helpers
(`read_acquisition_params`, `parse_run_name`, `libration_ke_scale`,
`dimensionless_numbers`, …) use the dataset's values.

> **If a dataset has no `param_postProcessing.json`,
> `read_paramPostprocessing` copies the default one there, prints
> *"No parameter file found, a default one has been created, please edit and
> relaunch the notebook"*, and stops.** Edit the copy and re-run.

### Parameters (keys in the JSON)

| key | meaning |
|---|---|
| `PIV_FILENAME` | PIVlab export to read (`PIVlab_results_uncalibrated.mat`) |
| `LOG_FILENAME` | acquisition log (`acquisition_log.txt`) |
| `XSCALE`, `YSCALE` | pixel calibration [m/px] |
| `PTS_ROI` | ROI rectangle, two opposite corners `[(x0,y0),(x1,y1)]` in metres |
| `FRAMES_PER_FIELD` | camera frames per PIV field (**2** — PIVlab pairs images) |
| `UNCAL_DT`, `UNCAL_FPS`, `UNCAL_SCALE` | fallbacks used when the log can't be read (all `1`) |
| `FROT_DIVISOR`, `FLIB_DIVISOR` | legacy folder-name divisors |
| `REGIONS` | the two spatial extents, `["ROI","FULL"]` |
| `R` | cylinder radius [m] |
| `H` | container height [m] |
| `k0` | dimensionless azimuthal wavenumber (number of wavelengths, 6) |
| `nu` | kinematic viscosity [m²/s] |
| `VALIDATE_VELOCITY` | `True` (default): `load_piv` returns PIVlab's **filtered/validated** velocity (`u_filt`/`v_filt`). `False`: returns the **original** velocity (`u`/`v`) with NaN wherever the vector failed validation (`typevector_filt != 1`). |

Derived (not stored, recomputed on load): the physical wavenumber `k = k0·π/R`
and wavelength `l = 2π/k`. Edit `R`/`k0`, never `k`/`l`.

> When you change a dataset's `param_postProcessing.json`, mirror the change into
> `param_postProcessing_default.json` so freshly-seeded datasets stay current.

---

## 3. Run-folder naming

Physical parameters are parsed straight from the folder name:

```
frot0.50Hz_flib0.400Hz_dphi2deg_SS1
   │         │           │      └── acquisition index; SS2, SS3… mark repeats
   │         │           └───────── libration amplitude δφ  [deg]
   │         └───────────────────── libration frequency f_lib [Hz]
   └─────────────────────────────── rotation frequency f_rot [Hz]
```

`parse_run_name()` returns `(frot_Hz, flib_Hz, dphi_deg)`. The legacy form
(`frot050_flib0400_…`, zero-padded integers) is still parsed.

> **Sampling rate.** PIVlab pairs images, so one velocity field is produced every
> two camera frames: the PIV field rate is `fps = cam_fps / 2`. All time and
> frequency axes use this field rate, not the camera rate.

---

## 4. How to process the data

### Regions

Every quantity is computed over a spatial region:

- `'ROI'` — cropped to `PTS_ROI` (tagged `_ROI` in filenames)
- `'FULL'` — the whole field (tagged `_FULL`)

The **batch** tools take a local `REGION` of `'BOTH'` (both, default), `'ROI'`,
or `'FULL'`. The single-run / plotting tools pick one region locally.

### The tools

| notebook | what it does |
|---|---|
| `batch_KineticEnergy.ipynb` | For every run, in **one pass** (the `.mat` is read once): ROI/FULL-averaged kinetic-energy time series `⟨Ek⟩(t)` **and** both FFT spectra of `Ek` — `FFT(⟨Ek⟩)` (average-then-FFT) and `⟨FFT(Ek)⟩` (FFT-then-average). Writes one combined per-run `.npz`, one dataset summary `.csv` (time-series stats + the `⟨FFT(Ek)⟩` quantities), a per-run 2-panel figure, and a resonance figure `⟨Ek⟩,std vs f*`. See Annex B. |
| `batch_Velocity.ipynb` | Per-point FFT of U(t), V(t), ROI-averaged amplitude spectra, polarization, dimensionless numbers. Per-run `.npz`, spectrum + polarization figures, summary, and dataset colormaps. |
| `process_single_KineticEnergy.ipynb` | Runs `batch_KineticEnergy`'s per-run step for **one** run from a PIV `.mat` — **identical** combined `.npz`, two-panel figure and summary row to the batch. `PIV_FILENAME` (local, always used) selects the `.mat`; a band-passed one (`..._bp<lo>-<hi>Hz.mat`) has its `_bp<lo>-<hi>Hz` tag appended to every output. |
| `single_Velocity.ipynb` | Viewer: reads one `VelocityFFT` `.npz`, redraws the velocity spectrum + polarization, prints peaks and dimensionless numbers. |
| `process_single_Velocity.ipynb` | Runs `batch_Velocity`'s per-run step for **one** run from a PIV `.mat` — identical `.npz`, figures and summary row to the batch. `PIV_FILENAME` (local, always used) selects the `.mat`; a band-passed one has its `_bp<lo>-<hi>Hz` tag appended to every output. `OVERWRITE_NPZ` / `OVERWRITE_FIG` / `UPDATE_SUMMARY` switches. |
| `plot_batch_Velocity.ipynb` | Redraws every run's `batch_Velocity` figures (spectrum + polarization) **from the cached `.npz`** — no `.mat`, no recompute; byte-identical to the batch figures. |
| `plot_batch_KineticEnergy.ipynb` | Redraws every run's `batch_KineticEnergy` two-panel figure **from the cached `.npz`** — no `.mat`, no recompute; byte-identical to the batch. Physical/normalized `XLIM`/`YLIM` set the spectrum-panel limits. |
| `plot_resonance_summary.ipynb` | Overlays one or more **dataset folders** and plots a **single** quantity vs `f*` set by `PLOT_QUANTITY` (`velocity_fft` / `mean_Ek` / `std_Ek` / `ke_fft`), saving just that figure (named after the quantity). Colour = dataset sweep, marker = run idx (filled circle + line for idx 1; open diamond/square/… scattered for repeats idx ≥ 2). `SELECT_*` / `SELECT_TOPO` filter the rows. |
| `filter_velocity_bandpass.ipynb` | Zero-phase Butterworth band-pass of U/V between two cut-offs; writes a new **uncalibrated** `.mat`. |
| `PIV_subset_analysis_singleExperiment.ipynb` | One run over a user time window: quiver movie (`.mp4`), `⟨Ek⟩(t)` + FFT figure, `.npz`. |
| `make_velocity_movie.ipynb` | Reads one run's velocity from a **local** `PIV_FILENAME` `.mat` and writes an `.mp4` of the field (velocity norm `|u|` colour map) over a time window in seconds (`T_START`/`T_END`, `None` = first/last), at `OUTPUT_FPS`. A `_bp<lo>-<hi>Hz` filename is tagged onto the movie. |
| `colormaps_single_FFTVelocity.ipynb` | For one run (local `PIV_FILENAME`): per-point `|FFT(U)|+|FFT(V)|`, then two spatial colormaps over the whole field — the peak amplitude at `f_lib`, and the band integral over `[FMIN, FMAX]` — each with a marginal of the quantity summed over x vs z. |
| `select_ROI_quiver.py` | **Interactive, script only** (no notebook twin). Draws one run's velocity quiver in m/s on top of its background camera image, in metres, and asks you to drag the ROI rectangle. Prints the top-left / bottom-right corners in physical units, writes them into the dataset's `param_postProcessing.json` as `PTS_ROI`, and saves `ROI_selection.png` next to it with the corner coordinates drawn on the picture. |
| `make_thumbmail_summary.ipynb` | Tiles all runs' figures into A4-wide contact-sheet PDFs, one per figure type. |

### Typical workflow

1. Put the runs under a dataset folder (`BASE_DIR`), each named as in §3, each
   holding `PIVlab_results_uncalibrated.mat` and `acquisition_log.txt`.
2. Open a batch notebook, set `BASE_DIR` (and `REGION`, `FIG_FORMAT`, …) in the
   config cell, *Run All*. First run in a new dataset seeds the parameter file
   and stops — edit it and re-run.
   The calibration (`XSCALE`/`YSCALE`) has to be typed in by hand; the ROI does
   not — run `python select_ROI_quiver.py <run_folder>` and drag the rectangle
   (Annex B).
3. `batch_KineticEnergy` computes the kinetic-energy time series **and** both of
   its FFT spectra in a single pass. `batch_Velocity` is independent.
4. Inspect single runs with the `single_*` viewers; redraw the resonance curve
   with `plot_resonance_summary`.
5. `make_thumbmail_summary` for a whole-sweep contact sheet.

---

## 5. Outputs

All per-run outputs go in `<run>/PostProcessing/`; summaries and dataset-level
figures go in `BASE_DIR`.

### Files

| file | written by | contents |
|---|---|---|
| `KineticEnergy_<region>.npz` | batch/single KineticEnergy | Ek time series **and** both FFT spectra (see §6) |
| `KineticEnergy_summary_<region>.csv` | batch/single KineticEnergy | one combined row per run (time series + `⟨FFT(Ek)⟩`) |
| `KineticEnergy_vs_<fstar\|flib>_<region>[_normalized].<fmt>` | KineticEnergy / plot_summary | resonance figure |
| `KineticEnergy_FFT_<region>[_normalized].<fmt>` | batch/single KineticEnergy | 2-panel Ek time series + spectra |
| `VelocityFFT_<region>.npz` | batch Velocity | velocity spectra + polarization (see §6) |
| `VelocityFFT_<region>[_normalized].<fmt>` | batch/single Velocity | amplitude spectrum figure |
| `Velocity_polarization_<region>[_normalized].<fmt>` | batch/single Velocity | polarization figure |
| `VelocityFFT_summary_<region>.csv` | batch Velocity | one row per run |
| `VelocityFFT_colormap_*_<region>[_normalized].<fmt>` | batch Velocity | sweep colormaps (vs f*, δφ, f_lib) |
| `PIVlab_results_uncalibrated_bp<lo>-<hi>Hz.mat` | filter_velocity_bandpass | band-passed velocity (uncalibrated) |
| `Thumbnails_<figure>[_normalized].pdf` | make_thumbmail_summary | A4 contact sheets |

### Naming conventions

- **Region tag** `_ROI` / `_FULL` on every output.
- **Figures come in two versions**, always: a raw one and a `_normalized` one.
  Normalized spectra use `f / f_rot` on the frequency axis and divide the
  quantity by its scale — velocity by `U0`, kinetic energy by `U0²`
  (`U0 = 2π·f_lib·R·δφ[rad]`, the peak libration wall velocity). Built by
  `figure_filename(stem, fmt, normalized=…)`.
- **`FIG_FORMAT`** (config, each figure notebook) selects `'png'` or `'pdf'`.
- Summaries are **CSV only**.

---

## 6. Variables

### `KineticEnergy_<region>.npz`

| variable | meaning |
|---|---|
| `run`, `PIV_file`, `region`, `calibrated` | run name, source `.mat`, `ROI`/`FULL`, calibration flag |
| `dt_vel`, `fps`, `xscale`, `yscale`, `pts_ROI` | calibration + sampling used |
| `nframes`, `npoints` | number of PIV fields, ROI grid points |
| `t`, `Ek_frame` | time [s] and `⟨Ek⟩_region(t)` [m²/s²] |
| `mean_Ekin`, `std_Ekin` | time-mean and std of `Ek_frame` |
| `U0`, `Ek_frame_star`, `mean_Ekin_star`, `std_Ekin_star` | libration velocity scale and the `/U0²`-normalized versions |
| `detrend`, `freq` | whether the mean was removed before the FFT; FFT frequency axis [Hz] |
| `amp_fft_of_mean` | `\|FFT(⟨Ek⟩)\|` — average, then FFT (figure only) |
| `amp_mean_of_fft` | `⟨\|FFT(Ek)\|⟩` — FFT at each point, then average (**this is what the summary keeps**) |
| `f_peak`, `amp_peak` | peak frequency of `⟨FFT(Ek)⟩` and its amplitude |
| `amp_flib`, `amp_2flib` | `⟨FFT(Ek)⟩` amplitude at `f_lib` and `2·f_lib` |

### `VelocityFFT_<region>.npz`

| variable | meaning |
|---|---|
| `run`, `PIV_file`, `region`, `calibrated`, `dt_vel`, `fps`, `xscale`, `yscale`, `pts_ROI`, `nframes`, `npoints`, `window`, `detrend` | metadata / FFT settings |
| `f`, `amp_u`, `amp_v`, `amp_total` | frequency [Hz] and ROI-averaged FFT amplitudes of U, V and their sum [m/s] |
| `f_peak`, `f_low` | dominant peak, and strongest peak in `[0.01, f_lib/2]` Hz |
| `U0`, `f_star`, `amp_u_star`, `amp_v_star`, `amp_total_star` | `f/f_rot` and amplitudes `/U0` |
| `powerU`, `powerV`, `powerU_mean`, `powerV_mean` | ROI power spectra of U, V |
| `f_pol`, `pol_IW`, `polarization` | inertial-wave prediction + measured curve |
| `powerRatio`, `powerRatio_std/_med/_p25/_p75` | ROI `P_V/P_U` statistics |

### Summary tables (`*_summary_<region>.csv`)

Common to all: `run`, `run idx`, `region`, `processed`, `frot_Hz`, `flib_Hz`,
`dphi_deg`, `fstar` (= `flib_star` = `f_lib/f_rot`), `calibrated`, `dt_vel_s`,
`fps_Hz`, `nframes`, `npoints`, `top_topo`/`bottom_topo` (topography arrangement
from the dataset folder name — `TopBottom` → both `True`, `bottomOnly` →
`top_topo=False`), `npz`/`figure`.

Dimensionless numbers (added to every summary, from `dimensionless_numbers`):
`U0_mps`, `E`, `E_l`, `delta_nu_m`, `Ro`, `Re`, `Re_l`, `Re_bl` — see the annex.

Per-tool result columns:

- **KineticEnergy (combined):** time series — `mean_Ekin`, `std_Ekin`,
  `mean_Ekin_star`, `std_Ekin_star`; and the `⟨FFT(Ek)⟩` spectrum — `f_peak_Hz`,
  `f_peak_star`, `amp_peak`, `amp_peak_star`, `amp_flib`, `amp_flib_star`,
  `amp_2flib`, `amp_2flib_star`. Only `⟨FFT(Ek)⟩` is stored (the old `_perPoint`
  suffix is gone; the `FFT(⟨Ek⟩)` spectrum is drawn in the figure but not tabled).
- **VelocityFFT:** `f_peak_Hz`, `f_peak_star`, `f_low_Hz`, `f_low_star`, and the
  summed-spectrum amplitude at each marked frequency — `amp_flib`, `amp_flow`,
  `amp_flib_minus_flow`, `amp_flib_plus_flow` (m/s) with `_star` twins (÷`U0`).

---

## Annex A — `piv_postprocessing_lib.py` function reference

### Parameter files

| function | purpose |
|---|---|
| `read_paramPostprocessing(base_dir, apply=True)` | Load `base_dir/param_postProcessing.json` into a `_Param` namespace, filling missing keys from the default file and deriving `k`, `l`. With `apply=True`, rebinds the module globals. **If no file exists, copies the default there and raises `SystemExit`.** |
| `write_paramPostprocessing(base_dir, params=None, overwrite=False)` | Write a `param_postProcessing.json` from the default template (optionally overriding some values); refuses to clobber an existing file unless `overwrite=True`. |

### Data loading

| function | purpose |
|---|---|
| `load_piv(file_path)` | `X, Y, U, V, nframes` from a PIVlab `.mat`. Prefers `u_filt`/`v_filt`, then `u_filtered`/`v_filtered`, then falls back to unfiltered `u`/`v` (or `u_original`/`v_original`) with a message; prints *"No velocity field found"* if none. Handles both 3-D-array and cell-array layouts. |
| `create_mask(x, y, pts_roi)` | Boolean mask of grid points inside the ROI rectangle. |
| `calibrate(x_px, y_px, u_px, v_px, xscale, yscale, dt_pulse)` | Pixels/pulse → metres and m/s, z pointing up. |
| `extract_roi(x, y, u, v, mask)` | Crop `x, y` (2-D) and `u, v` (3-D, frames last) to the ROI bounding box. |
| `region_tag(region)` | Validate a region and return the filename tag `'ROI'`/`'FULL'`. |
| `region_fields(x, y, u, v, pts_roi, region)` | `(x, y, u, v)` restricted to the chosen region. |
| `regions_to_run(region)` | Expand a batch `REGION` (`'BOTH'`/`'ROI'`/`'FULL'`) into the tuple of regions to process. |

### Metadata / naming

| function | purpose |
|---|---|
| `read_acquisition_log(log_path)` | The last data row of the tab-separated PIVlab acquisition log, as a dict. |
| `read_acquisition_params(log_path)` | `(dt_vel_s, fps_Hz, ok)` from the log. **`fps = cam_fps / FRAMES_PER_FIELD`.** `ok=False` (with fallbacks) if the log can't be read. |
| `parse_run_name(name)` | `(frot_Hz, flib_Hz, dphi_deg)` from a folder name (both naming conventions). |
| `parse_frot(name)`, `parse_flib(name)` | Single frequency [Hz] from a run name, or `None`. |
| `parse_freq_token(name, tag, divisor)` | Frequency [Hz] carried by one folder-name token. |
| `resolve_npz(path, npz_name)` | Accept an `.npz` path directly, or a run folder containing it. |
| `read_KineticEnergy(filepath)`, `read_VelocityFFT(filepath)` | Load an output `.npz`, print its variables, return them as a dict. |

### Spectra

| function | purpose |
|---|---|
| `amp_from_rfft(fft, n)` | One-sided amplitude `abs(FFT)·2/N` along the last axis (DC/Nyquist keep `1/N`). |
| `compute_fft(ek, fps, detrend=True)` | `(freq, amp, fft)` for the positive-frequency half of a series. |
| `peak_freq(f, amp)` | `(frequency, amplitude)` of the spectrum maximum, ignoring the DC bin. |
| `peak_freq_in_band(f, amp, fmin, fmax)` | Same, restricted to `[fmin, fmax]` (used for `f_low`). |
| `fft_guide_lines(frot, flib, f_low)` | Reference frequencies to mark on a spectrum: `[(freq, label, colour)]` — `f_rot` (blue), `f_lib`/`2f_lib` (red), `f_low`/`f_lib±f_low` (green). |
| `fft_axis_limits(freq, amp, frot, flib)` | Axis limits for a spectrum: x to `max(4·f_rot, 4·f_lib)`, y to decade bounds keeping the peak in view. |

### Physics / scales

| function | purpose |
|---|---|
| `libration_velocity_scale(flib_hz, dphi_deg)` | `U0 = 2π·f_lib·R·δφ[rad]` [m/s] — the velocity scale for `U_star = U/U0`. |
| `libration_ke_scale(dphi_deg, flib_hz)` | `U0²` — the Ek normalization scale. |
| `dimensionless_numbers(frot_hz, flib_hz, dphi_deg)` | Dict keyed by summary-column name: `U0_mps`; `E = ν/(2π·f_rot·H²)` and `E_l = ν/(2π·f_rot·l²)` (Ekman, height/wavelength); `delta_nu_m = √E·H` (viscous BL); `Ro = U0/(2π·f_rot·R)`; `Re = U0·R/ν`, `Re_l = U0·l/ν`, `Re_bl = U0·δ_ν/ν`. `f_rot`-dependent ones are NaN when `f_rot` is unknown. |
| `compute_polarization(f, powerU, powerV, frot)` | `(f_pol, pol_IW, pol_data)`: inertial-wave prediction `2[(2f_rot/f)²−1]` and the measured `(P_V/P_U)` curve. |

### Figures

| function | purpose |
|---|---|
| `figure_filename(stem, fmt, normalized=False)` | `'<stem>[_normalized].<fmt>'` with `fmt ∈ {png, pdf}` — the shared naming/format helper. |

*(Names beginning with `_` — `_read_default_params`, `_params_from_raw`,
`_apply_params`, `_first_present`, `_as_frames`, `_as_grid`, `_load_npz` — are
internal helpers.)*

---

## Annex B — Notebooks in detail

Every notebook has a runnable `.py` twin (same base name) generated by
`ipynb_to_py.py`; the twin forces the matplotlib `Agg` backend, so it **saves
every figure but shows none** — ideal for headless batch runs
(`python batch_KineticEnergy.py`). Regenerate the twins after editing a notebook.
All notebooks read their per-dataset constants from `param_postProcessing.json`
(via `read_paramPostprocessing`) and share the helpers in
`piv_postprocessing_lib.py`.

### `batch_KineticEnergy.ipynb` / `.py`

The kinetic-energy workhorse. For every run under `BASE_DIR` and each requested
region it reads the PIV `.mat` **once** and, from the same ROI-cropped velocity
field, produces both the time-domain and the frequency-domain products:

- **Time series** `Ek(t) = 0.5·⟨U²+V²⟩` (region-averaged per frame), with
  `⟨Ek⟩` and `std(Ek)` and their `/U0²` normalized twins.
- **Two FFT spectra of the kinetic energy** — the two differ *only* in when the
  spatial average is taken:
  - **`FFT(⟨Ek⟩)` — average, then FFT.** Transform the single region-averaged
    series `⟨Ek⟩(t)`.
  - **`⟨FFT(Ek)⟩` — FFT, then average.** Transform `Ek(t)` at *every* grid point,
    then average the amplitudes over the region.

  Averaging first cancels fluctuations that are incoherent across the region, so
  `FFT(⟨Ek⟩)` is normally the smaller of the two; the gap between the curves
  measures how spatially coherent the signal is at each frequency. **Both curves
  are drawn** in the per-run figure (panel 2), but **only `⟨FFT(Ek)⟩` is written
  to the summary** — its peak (`f_peak_Hz`, `amp_peak`) and its amplitude at
  `f_lib` and `2·f_lib` (`amp_flib`, `amp_2flib`, with `_star` twins). Because
  `Ek ~ velocity²`, the libration forcing shows up at `2·f_lib`.

  Merging the old `batch_KineticEnergy` + `batch_KineticEnergyFFT` removed the
  need to re-read the `.mat` for the per-point spectrum — the field is already in
  hand — so both spectra are essentially free.

Outputs per run: one combined `KineticEnergy_<region>.npz` (time series + both
spectra + the kept `⟨FFT(Ek)⟩` scalars) and a two-panel figure
`KineticEnergy_FFT_<region>[_normalized].<fmt>` (panel 1: `⟨Ek⟩(t)`; panel 2:
both spectra with dashed guides at `f_lib`, `2·f_lib`). Across the dataset: one
`KineticEnergy_summary_<region>.csv` and the resonance figure
`KineticEnergy_vs_fstar_<region>[_normalized].<fmt>` (mean and std of `Ek` vs
`f* = f_lib/f_rot`). Config: `DETREND`, `LOGY`/`LOGX` (spectrum panel),
`REPROCESS_ALL` (else cached `.npz` rows are reused), and `SELECT_FROT/FLIB/DPHI`
(restrict only the resonance figure — the CSV always keeps every run).

### `batch_Velocity.ipynb` / `.py`

Independent of the kinetic-energy chain. Per run, per region: the per-point FFT
amplitude spectra of `U(t)` and `V(t)`, ROI-averaged (`amp_u`, `amp_v`,
`amp_total = |U|+|V|`), plus the velocity polarization `P_V/P_U` against the
inertial-wave prediction. It locates `f_peak`, and `f_low` (the strongest peak in
`[F_LOW_FMIN, f_lib/2]`, kept only if it exceeds `THRESHOLD_PEAK ×` the band
mean — otherwise `f_low` and its sidebands are `NaN`), and records the
total-spectrum amplitude at `f_lib`, `f_low`, and `f_lib∓f_low`. Writes a
`VelocityFFT_<region>.npz`, a spectrum figure (total curve only, with on-curve
diamond markers), a polarization figure, the `VelocityFFT_summary_<region>.csv`,
and dataset-level sweep colormaps (vs `f*`, `δφ`, `f_lib`). `SAVE_COLORMAP`
toggles whether the colormaps are written.

### `process_single_KineticEnergy.ipynb` / `.py`

Runs `batch_KineticEnergy`'s per-run step for **one** run (`PATH`). It reuses the
batch's `compute_fft_per_point`, `make_fft_figure` and `build_row` verbatim, so
the outputs are **identical** to what the batch writes for that run: the same
combined `KineticEnergy_<region>.npz`, a pixel-identical two-panel figure, and —
when `UPDATE_SUMMARY` is `True` — the same 38-column row in
`KineticEnergy_summary_<region>.csv`. `REPROCESS=False` reuses a cached `.npz`
(exactly like the batch's skip path). It subsumes the former
`single_KineticEnergyFFT` — both `FFT(⟨Ek⟩)` and `⟨FFT(Ek)⟩` are computed here.

`PIV_FILENAME` (local, **always** used — the `param_postProcessing.json` value is
ignored) selects the `.mat`; if it is a band-passed file
(`..._bp<lo>-<hi>Hz.mat` from `filter_velocity_bandpass`) its `_bp<lo>-<hi>Hz` tag
is appended to every output (`KineticEnergy_<region>_bp...Hz.npz`, the figure and
`KineticEnergy_summary_<region>_bp...Hz.csv`), so filtered results never overwrite
the raw ones.
Section 6 adds a **wavelet** (time-frequency) analysis on top, writing its own
`KineticEnergyWavelet_<region>.npz` and a raw + normalized scalogram; it is a separate product and
never reaches the summary. `PERFORM_WAVELET` switches it off — the CWT is by far
the slowest step, so `False` is the quick way to re-run only the FFT products,
and it leaves any wavelet files already on disk untouched. PyWavelets is only
imported opportunistically, so with the switch off the notebook runs without it.


### `single_Velocity.ipynb` / `.py`

A viewer for one `VelocityFFT_<region>.npz`: redraws the velocity amplitude
spectrum (total only, diamonds on the curve) and the polarization figure, prints
the peak frequencies, `f_low` significance test, and dimensionless numbers.
`OVERWRITE_FIG` guards the spectrum figures; `UPDATE_SUMMARY` optionally pushes
this run's row into `VelocityFFT_summary_<region>.csv`.

### `process_single_Velocity.ipynb` / `.py`

Runs `batch_Velocity`'s per-run step for **one** run from a PIV `.mat`, reusing
the batch's helper, figure and `build_row` functions verbatim, so the outputs are
**identical** to what the batch writes for that run: the same
`VelocityFFT_<region>.npz`, pixel-identical spectrum and polarization figures, and
the same summary row (with the `kept` de-duplication). `OVERWRITE_NPZ` /
`OVERWRITE_FIG` / `UPDATE_SUMMARY` decide what is written; use `single_Velocity`
to only redraw from an existing `.npz`.

`PIV_FILENAME` (local, **always** used — the `param_postProcessing.json` value is
ignored) selects the `.mat`; if it is a band-passed file
(`..._bp<lo>-<hi>Hz.mat` from `filter_velocity_bandpass`) its `_bp<lo>-<hi>Hz` tag
is appended to **every output** (`VelocityFFT_<region>_bp...Hz.npz`, the spectrum
+ polarization figures, and `VelocityFFT_summary_<region>_bp...Hz.csv`), so
filtered results never overwrite the raw ones.
Section 6 adds a **wavelet** (time-frequency) analysis on top, writing its own
`VelocityWavelet_<region>.npz` and a raw + normalized scalogram; it is a separate product and
never reaches the summary. `PERFORM_WAVELET` switches it off — the CWT is by far
the slowest step, so `False` is the quick way to re-run only the FFT products,
and it leaves any wavelet files already on disk untouched. PyWavelets is only
imported opportunistically, so with the switch off the notebook runs without it.


### `plot_batch_Velocity.ipynb` / `.py`

Reproduces `batch_Velocity`'s per-run figures **without reprocessing**: it loops
over the runs and redraws the spectrum (`VelocityFFT_<region>`) and polarization
(`Velocity_polarization_<region>`) figures straight from each run's cached
`VelocityFFT_<region>.npz`, reusing the batch's `_save_fft_figure` /
`_save_polarization_v2_figure` verbatim -- so the output is byte-identical to the
batch. Only plotting options apply (`FIG_FORMAT`, `LOGX`, `LOGY`); nothing is
recomputed and no `.npz`/summary is written. `OVERWRITE=False` skips runs whose
figures already exist.

### `plot_resonance_summary.ipynb` / `.py`

Cross-dataset overlays of **one** quantity vs `f* = f_lib/f_rot`. `SUMMARY_FILES`
is a list of **dataset folders**; each must hold `KineticEnergy_summary_<region>.csv`
and `VelocityFFT_summary_<region>.csv` (both are loaded, so any quantity works).
`PLOT_QUANTITY` chooses what is plotted:

- `'velocity_fft'` — normalised velocity FFT amplitude at `f_lib` (`amp_flib_star`);
- `'mean_Ek'` — mean kinetic energy (`mean_Ekin_star`);
- `'std_Ek'` — std of kinetic energy (`std_Ekin_star`);
- `'ke_fft'` — per-point `⟨FFT(Ek)⟩` amplitude at `2·f_lib` (`amp_2flib_star`).

Colour encodes the (dataset, `f_rot`, `dphi`) sweep; the **run index** sets the
marker — run idx 1 is a connected curve with filled circles, each higher idx
(repeats of an identical `f_rot/f_lib/dphi` run) is drawn as **scattered open
markers with no line** (diamond for 2, square for 3, …), in the sweep's colour.
The legend keeps both the run information and the run idx. `SELECT_FROT/FLIB/FSTAR/DPHI`
(value / list / `(lo, hi)` range) and `SELECT_TOPO` (`'all'` / `'bottom only'` /
`'top and bottom'`) filter the rows.

**Only the selected figure is saved**, `<quantity>_vs_fstar_<tag>[_filtered].<fmt>`
(the quantity is in the name). A single dataset → written in that dataset folder
with the region tag; several → the project root with the `overlay` tag; an active
`SELECT_*` filter appends `_filtered`.
### `filter_velocity_bandpass.ipynb` / `.py`

Zero-phase Butterworth band-pass of `U/V` between two cut-off frequencies; writes
a new **uncalibrated** `.mat` (`..._bp<lo>-<hi>Hz.mat`) that the other tools can
then post-process.

### `PIV_subset_analysis_singleExperiment.ipynb` / `.py`

One run over a user-chosen time window: a quiver movie (`.mp4`), an `⟨Ek⟩(t)` +
FFT figure, and a `.npz` — for zooming into a transient.


### `make_velocity_movie.ipynb` / `.py`

Writes an `.mp4` of one run's velocity field (the velocity norm `|u|` as a colour map) straight from a PIV `.mat`. `PIV_FILENAME` is a
**local** constant (not from `param_postProcessing.json`) so it can point at a
raw or a band-passed `.mat`; calibration, the ROI and the log name still come
from the dataset parameters, and the velocity honours `VALIDATE_VELOCITY` via
`load_piv`. `T_START`/`T_END` are in seconds (`None` -> first/last),
`OUTPUT_FPS` is the playback rate, `MAX_FRAMES`/`MAG_CLIM` control the drawing, and `REGION`/`SHOW_ROI_BOX` zoom to / outline the ROI. The movie is
`PostProcessing/velocity_movie_<region>_t<start>-<end>s[_bp<lo>-<hi>Hz].mp4`; the
`_bp` tag (when the `.mat` is band-passed) keeps it from overwriting the raw movie.
### `colormaps_single_FFTVelocity.ipynb` / `.py`

Spatial FFT maps for **one** run. It computes the single-sided FFT amplitude of
`U(t)` and `V(t)` at **every grid point** (NaN-in-time interpolated per point;
all-NaN points blank) and forms `|FFT(U)| + |FFT(V)|` per point, then draws two
colormaps over the whole PIV field: the **peak amplitude at `f_lib`** (max within
±`HALFWIDTH_BINS` bins) and the **band integral** `∫(|FFT(U)|+|FFT(V)|) df` over
`[FMIN, FMAX]`. Each colormap carries a right-hand marginal of the same quantity summed over all x as a function of z (shared z axis). `PIV_FILENAME` is a local constant (raw or band-passed); the
velocity honours `VALIDATE_VELOCITY`. `LOGC` toggles a log colour scale,
`SHOW_ROI_BOX` overlays the ROI. Figures →
`VelocityFFT_spatialmap_peak_flib[_bp…Hz]` and
`VelocityFFT_spatialmap_integral_<FMIN>-<FMAX>Hz[_bp…Hz]`.

### `select_ROI_quiver.py` (interactive, no notebook)

Picks `PTS_ROI` by eye instead of by trial and error. Run it from a terminal —
it opens a window and blocks, so it is deliberately **not** part of
`ipynb_to_py.py`'s Agg twins:

```
python select_ROI_quiver.py                    # the RUN_DIR set at the top
python select_ROI_quiver.py <run_folder>       # or any run folder
```

What it draws, all on axes in metres:

- the run's **background image** (`<run>/background.mat`, `bg_img_A`), contrast-
  stretched with `BG_CLIP` percentiles — raw PIV frames are almost black. Runs
  recorded without one get a warning and a blank field; the picking still works.
- the **velocity quiver** in m/s, calibrated exactly as the batch tools do
  (`XSCALE`/`YSCALE` from the parameter file, `pulse_sep` from the acquisition
  log, and the dataset's `VALIDATE_VELOCITY` — so the ROI is picked on the same
  field the batch tools will analyse). `FRAME` chooses what is shown: `'rms'`
  (default — RMS magnitude with the sign of the mean, so an oscillating flow does
  not average itself away), `'mean'`, or an integer frame index.
- the ROI **already in the parameter file**, dashed, for reference.

The axes are clipped to the background image in both directions, and never go
negative. The physical origin sits on the **bottom row of the PIV grid**, which
is a few pixels above the bottom of the frame, so the raw image extent dips
slightly below zero — a sliver `create_mask` could never select, so it is cut
away. A drag that runs off the axes is clamped to them, so a picked ROI can
never be negative nor reach past the image.

| control | |
|---|---|
| drag left mouse | draw / redraw the rectangle |
| `enter` | accept — print the corners, save them |
| `r` | clear and start again |
| `escape` / close window | quit, write nothing |

On accept it prints

```
PTS_ROI = [[x_left, y_top], [x_right, y_bottom]]
```

— top-left corner first, bottom-right second, the order `create_mask` expects —
and (with `SAVE_TO_PARAM`, the default) writes just that key into the dataset's
`param_postProcessing.json`, leaving every other key and the key order alone.

`SAVE_FIGURE` (also the default) saves `ROI_selection.png` **next to that
parameter file**, in the dataset folder rather than under any one run — the ROI
belongs to the dataset, not to the run it happened to be picked on. The corner
coordinates are written **on the picture**: labelled at the two corners and
spelled out as the full `PTS_ROI = [[…], […]]` line in the title, so the figure
documents the parameter file on its own. One ROI per dataset, so one picture:
re-picking overwrites it.

`SAVE_TO_DEFAULT` is **False** on purpose: unlike the other keys, the ROI is
genuinely per-dataset, so mirroring one dataset's rectangle into
`param_postProcessing_default.json` would mislead the next dataset seeded from
it. An uncalibrated run (unreadable acquisition log ⇒ positions in px) refuses to
write, since the tools read `PTS_ROI` as metres.

**Alignment.** The image extent is derived from the PIV grid's own pixel
coordinates, not from the image size: `x = x_px·XSCALE` and
`y = (max(y_px) − y_px)·YSCALE`, so `y = 0` sits on the bottom row of the PIV
grid — the same origin `load_piv` gives the vectors. That is what puts the
picture, the arrows and the saved rectangle on one set of axes.

### `make_thumbmail_summary.ipynb` / `.py`

Tiles every run's figures of a given type into A4-wide contact-sheet PDFs
(`Thumbnails_<figure>[_normalized].pdf`), one per figure type, for a whole-sweep
overview.

---

### `plot_batch_KineticEnergy.ipynb` / `.py`

The kinetic-energy counterpart of `plot_batch_Velocity`: redraws every run's
two-panel `KineticEnergy_FFT_<region>` figure (`<Ek>(t)` + both spectra) straight
from the cached `KineticEnergy_<region>.npz`, reusing `batch_KineticEnergy`'s
`save_fft_figure` verbatim -- byte-identical to the batch when the limits are
left at `None`. `XLIM_PHYS`/`YLIM_PHYS` and `XLIM_NORM`/`YLIM_NORM` set the
**spectrum panel** limits (physical: Hz / m^2/s^2; normalized: `f/f_rot` /
`/E_lib`); the time-series panel stays auto. No `.mat`, no recompute, no
`.npz`/summary written.

## Annex C — troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: pandas` | wrong interpreter — use `~/anaconda3/bin/python`. |
| `ImportError: cannot import name X from piv_postprocessing_lib` | stale kernel holding the old module — **restart the kernel**. |
| `No module named 'piv_postprocessing_lib'` | notebook started outside this folder. |
| *"No parameter file found, a default one has been created…"* then the cell stops | expected on a new dataset — edit the seeded `param_postProcessing.json` and re-run. |
| `FileNotFoundError: default parameter file missing` on import | `param_postProcessing_default.json` was deleted — restore it. |
| Thumbnail build reports it needs a PDF backend | `SOURCE_EXT='pdf'` but no `pymupdf`/`pypdfium2`/`pdf2image` — set `SOURCE_EXT='png'` or install one. |
