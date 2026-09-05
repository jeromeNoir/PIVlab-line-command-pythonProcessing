# KineticEnergy.npz — file structure

One `KineticEnergy.npz` per run, written by `PIV_processing`
(analysis `'KE'`) in `<run>/PostProcessing/`. Read it with
`READ_energyFile.py` or directly with `np.load(path)`.

Kinetic energy per grid point `Ek = 0.5·(U² + V²)` in **native px²/frame²**;
`× ECAL` → m²/s², `/ EK_SCALE` (after calibration) → dimensionless. Nothing is
stored calibrated or dimensionless — the calibration factors (`*CAL`,
multiply) and non-dimensionalisation divisors (`*_SCALE`, divide) are stored
alongside the data.

Shapes: `ny × nx` grid points, `nt` PIV fields, `nf = nt//2 + 1` one-sided FFT
bins. The 3-D array and 2-D maps are float32; series, axes and scalars
float64.

## Shared header block (identical keys in `Velocity.npz`)

### Run identification and physical parameters

| key | shape | description |
|-----|-------|-------------|
| `run` | scalar str | run-folder name, e.g. `frot0.50Hz_flib0.400Hz_dphi2deg_SS1` |
| `run_idx` | scalar int | session index `SSn` from the folder name (−1 if absent) |
| `calibrated` | scalar bool | acquisition log read OK; if False **every `*CAL` factor = 1** |
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
| `TCAL` | frames (`TIME` axis) | s | `1 / PIV_FPS` |
| `UCAL` | px/frame | m/s | `XCAL / DT_VEL` |
| `VCAL` | px/frame | m/s | `YCAL / DT_VEL` |
| `ECAL` | px²/frame² (Ek, its FFT) | m²/s² | `0.5·(UCAL² + VCAL²)` (exact when `XSCALE == YSCALE`) |
| `FCAL` | 1/frame (`FREQ` axis) | Hz | `1 / TCAL` |

### Non-dimensionalisation scales (physical → dimensionless, divide)

| key | definition | divides |
|-----|------------|---------|
| `U_SCALE` | `2π · flib · R · dphi[rad]` = `U0`, the peak libration wall velocity | velocities |
| `V_SCALE` | `U_SCALE` | idem |
| `LENGTH_SCALE` | `R` | positions |
| `TIME_SCALE` | `1 / frot` | times |
| `F_SCALE` | `frot` | frequencies |
| `EK_SCALE` | `0.5·(U_SCALE² + V_SCALE²)` | kinetic energy, its FFT amplitudes |

## Naming conventions

- **`averaged` / `rms`** = mean / std over **SPACE** (ROI = with `MASK_ROI`,
  FULL = whole grid; NaN-ignoring);
- **`mean` / `std`** = mean / std over **TIME**;
- names read in **operation order**, left to right:
  `EK_ROIaveraged_FFT` = space-average first, FFT second → FFT(⟨Ek⟩);
  `FFT_EK_ROIaveraged` = per-point FFT first, space-average second → ⟨FFT(Ek)⟩.

## Time series (space statistics vs. time)

| key | shape | description |
|-----|-------|-------------|
| `TIME` | (nt,) | time axis [frames]; `× TCAL` → s |
| `EK_ROIaveraged` | (nt,) | Ek space-averaged over the ROI vs. time |
| `EK_FULLaveraged` | (nt,) | Ek space-averaged over the full field vs. time |
| `EK_ROIrms` | (nt,) | space std of Ek over the ROI vs. time |
| `EK_FULLrms` | (nt,) | space std of Ek over the full field vs. time |

## Time-statistics maps

| key | shape | description |
|-----|-------|-------------|
| `EK_FULLmean` | (ny, nx) | per-point **time** mean of Ek (full field) |
| `EK_FULLstd` | (ny, nx) | per-point **time** std of Ek (full field) |

## Spectra

One-sided amplitude spectra (no window, mean removed when `detrend`).
Amplitudes in px²/frame²; `× ECAL` → m²/s².

| key | shape | description |
|-----|-------|-------------|
| `FREQ` | (nf,) | frequency axis [1/frame]; `× FCAL` → Hz |
| `EK_ROIaveraged_FFT` | (nf,) | FFT of `EK_ROIaveraged` — FFT(⟨Ek⟩), average first |
| `EK_FULLaveraged_FFT` | (nf,) | FFT of `EK_FULLaveraged` |
| `EK_ROIrms_FFT` | (nf,) | FFT of `EK_ROIrms` |
| `EK_FULLrms_FFT` | (nf,) | FFT of `EK_FULLrms` |
| `FFT_EK` | (ny, nx, nf) | per-point FFT amplitude of Ek(t) — (i, j) space, k frequency |
| `FFT_EK_ROIaveraged` | (nf,) | ROI space average of `FFT_EK` — ⟨FFT(Ek)⟩, FFT first (the summary's peak spectrum) |
| `FFT_EK_FULLaveraged` | (nf,) | full-field space average of `FFT_EK` |
| `FFT_EK_ROIrms` | (nf,) | ROI space std of `FFT_EK` |
| `FFT_EK_FULLrms` | (nf,) | full-field space std of `FFT_EK` |
| `detrend` | scalar bool | mean removed before the FFTs |

## Example

```python
from READ_energyFile import read_energy_file
d = read_energy_file("<run>/PostProcessing/KineticEnergy.npz")

ek_phys = d["EK_ROIaveraged"] * d["ECAL"]                 # [m²/s²]
ek_star = d["EK_ROIaveraged"] * d["ECAL"] / d["EK_SCALE"] # dimensionless
t_s     = d["TIME"] * d["TCAL"]                           # [s]
f_hz    = d["FREQ"] * d["FCAL"]                           # [Hz]
```
