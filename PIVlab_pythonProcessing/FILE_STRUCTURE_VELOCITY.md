# Velocity.npz — file structure

One `Velocity.npz` per run, written by `PIV_processing`
(analysis `'VELOCITY'`) in `<run>/PostProcessing/`. Read it with
`READ_velocityFile.py` or directly with `np.load(path)`.

Everything is stored in **native PIV units** (px, px/frame, frames, 1/frame)
together with the multiplicative calibration factors (`*CAL`, native →
physical) and the non-dimensionalisation divisors (`*_SCALE`, physical →
dimensionless). Nothing is stored calibrated or dimensionless.

Shapes: `ny × nx` grid points, `nt` PIV fields, `nf = nt//2 + 1` one-sided FFT
bins. 3-D arrays and 2-D maps are float32; series, axes and scalars float64.

## Shared header block (identical keys in `KineticEnergy.npz`)

### Run identification and physical parameters

| key | shape | description |
|-----|-------|-------------|
| `run` | scalar str | run-folder name, e.g. `frot0.50Hz_flib0.400Hz_dphi2deg_SS1` |
| `run_idx` | scalar int | session index `SSn` from the folder name (−1 if absent) |
| `calibrated` | scalar bool | acquisition log read OK; if False **every `*CAL` factor = 1** and no peaks are stored |
| `ROTATE` | scalar int | rotation applied to the stored fields [deg] (0, 90, 180, −90); all fields are stored in the rotated frame |
| `frot_Hz` | scalar | rotation frequency [Hz] (from the folder name) |
| `flib_Hz` | scalar | libration frequency [Hz] |
| `dphi_deg` | scalar | libration angular amplitude [deg] |
| `R`, `H` | scalar | cylinder radius / height [m] |
| `nu` | scalar | kinematic viscosity [m²/s] |
| `k0` | scalar | topography wavenumber parameter |
| `lambda` | scalar | topography wavelength `2πR/(k0·π)` [m]; NaN if `k0 = 0` |
| `top_topo`, `bottom_topo` | scalar bool | topography on the top / bottom lid |

### Acquisition and geometry

| key | shape | description |
|-----|-------|-------------|
| `PIV_FPS` | scalar | PIV field rate [Hz] = `cam_fps / FRAMES_PER_FIELD` (image pairing **halves** the camera rate) |
| `DT_VEL` | scalar | pulse separation inside an image pair [s] — the velocity time base |
| `TSCALE` | scalar | `1 / PIV_FPS` [s per field] — the series/FFT time base |
| `XSCALE`, `YSCALE` | scalar | geometric calibration [m/px] (from `param_postProcessing.json`) |
| `pts_ROI` | (2, 2) | ROI rectangle corners [m] |
| `X`, `Y` | (ny, nx) | grid positions [px]; `× XCAL`/`YCAL` → m |
| `MASK_ROI` | (ny, nx) | 1.0 inside the ROI, NaN outside |
| `nframes`, `ny`, `nx` | scalar int | array dimensions (`nframes` = nt) |
| `npoints_ROI`, `npoints_FULL` | scalar int | grid points inside the ROI / in total |

### Calibration factors (native → physical, multiply)

| key | converts | into | definition |
|-----|----------|------|------------|
| `XCAL` | px | m | `XSCALE` |
| `YCAL` | px | m | `YSCALE` |
| `TCAL` | frames | s | `1 / PIV_FPS` |
| `UCAL` | px/frame (U, its FFT) | m/s | `XCAL / DT_VEL` |
| `VCAL` | px/frame (V, its FFT) | m/s | `YCAL / DT_VEL` |
| `ECAL` | px²/frame² | m²/s² | `0.5·(UCAL² + VCAL²)` (unused by this analysis, present via the shared header) |
| `FCAL` | 1/frame (`FREQ` axis) | Hz | `1 / TCAL` |

### Non-dimensionalisation scales (physical → dimensionless, divide)

| key | definition | divides |
|-----|------------|---------|
| `U_SCALE` | `2π · flib · R · dphi[rad]` = `U0`, the peak libration wall velocity | velocities, their FFT amplitudes |
| `V_SCALE` | `U_SCALE` | idem for V |
| `LENGTH_SCALE` | `R` | positions |
| `TIME_SCALE` | `1 / frot` | times |
| `F_SCALE` | `frot` | frequencies |
| `EK_SCALE` | `0.5·(U_SCALE² + V_SCALE²)` | kinetic energy (unused here) |

## Velocity data

Velocities in px/frame; `× UCAL`/`VCAL` → m/s. `mean`/`std` = statistics over
**TIME**; `averaged` = mean over **SPACE** (ROI = with `MASK_ROI`, FULL =
whole grid, NaN-ignoring). There is no `TIME` axis stored — rebuild it as
`np.arange(nframes) * TCAL` [s].

| key | shape | description |
|-----|-------|-------------|
| `U`, `V` | (ny, nx, nt) | full native velocity fields U(x, y, t), V(x, y, t) — reprocessing can start here instead of the PIV `.mat` |
| `Umean`, `Vmean` | (ny, nx) | per-point **time** mean of U / V |
| `Ustd`, `Vstd` | (ny, nx) | per-point **time** std of U / V |

## Spectra

One-sided amplitude spectra along time, `window` taper (coherent gain divided
out), mean removed when `detrend`. Amplitudes in px/frame (`× UCAL`/`VCAL` →
m/s).

| key | shape | description |
|-----|-------|-------------|
| `FREQ` | (nf,) | frequency axis [1/frame]; `× FCAL` → Hz |
| `FFT_U`, `FFT_V` | (ny, nx, nf) | per-point FFT amplitude of U(t) / V(t) |
| `FFT_U_ROIaveraged`, `FFT_V_ROIaveraged` | (nf,) | ROI space average of `FFT_U` / `FFT_V` |
| `FFT_U_FULLaveraged`, `FFT_V_FULLaveraged` | (nf,) | full-field space average |
| `FFT_TOTAL_ROIaveraged` | (nf,) | `FFT_U_ROIaveraged + FFT_V_ROIaveraged` — the spectrum peaks are detected on |
| `FFT_TOTAL_FULLaveraged` | (nf,) | `FFT_U_FULLaveraged + FFT_V_FULLaveraged` |
| `window` | scalar str | FFT taper (e.g. `hann`; `boxcar` = none) |
| `detrend` | scalar bool | mean removed before the FFTs |

## Detected peaks

Local maxima of the **calibrated** ROI-averaged total spectrum in the band
`F_LOW_MIN ≤ f ≤ flib − DELTA_F` Hz, kept if above `THRESHOLD_PEAK ×` the
band-mean amplitude; sorted by **decreasing amplitude** (the strongest is the
summary's `f_low`). Empty for uncalibrated runs.

| key | shape | description |
|-----|-------|-------------|
| `F_PEAKS` | (n_peaks,) | peak frequencies [1/frame]; `× FCAL` → Hz |
| `AMP_PEAKS` | (n_peaks,) | peak amplitudes [px/frame]; `× UCAL` → m/s |
| `N_PEAKS` | scalar int | number of detected peaks |

## Example

```python
from READ_velocityFile import read_velocity_file
d = read_velocity_file("<run>/PostProcessing/Velocity.npz")

u_mps   = d["U"] * d["UCAL"]                       # physical [m/s]
u_star  = d["U"] * d["UCAL"] / d["U_SCALE"]        # dimensionless
f_hz    = d["FREQ"] * d["FCAL"]                    # [Hz]
t_s     = np.arange(d["nframes"]) * d["TCAL"]      # [s]
```
