"""Auto-generated .py twin of PIV_processing.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # PIV post-processing of PIVlab runs -- kinetic energy **and** velocity spectra
# 
# Processes either **every run sub-folder of every dataset in `BASE_DIRS`**
# (`BATCH = True`) or the **single run `RUN_DIR`** (`BATCH = False`). Each run **loads the PIV
# `.mat` once**, **rotates the fields by the dataset's `ROTATE`** (from
# `param_postProcessing.json`: 0, 90, 180 or -90 deg -- see `rotate_fields`),
# so every `.npz` stores the fields in the final frame, and computes, in a
# single pass, the analyses selected in `ANALYSES`:
# 
# | analysis | per-run output |
# |---|---|
# | `'KE'` | `PostProcessing/KineticEnergy.npz` |
# | `'VELOCITY'` | `PostProcessing/Velocity.npz` |
# 
# plus ONE summary CSV per dataset (`Runs_summary.csv` in each `BASE_DIR`) --
# a run's row replaces its existing entry, every other row is kept, so batch
# and single-run mode maintain the same file. **No figures are produced here** -- plotting lives in
# the separate `BATCH_plot_energy` / `BATCH_plotting_velocity` routines.
# 
# ## Conventions
# 
# **Units.** Every field in the `.npz` files is stored in NATIVE PIV units
# (px, px/frame, frames, 1/frame) together with the multiplicative calibration
# factors that turn them into physical units when needed (plotting, summaries):
# 
# | factor | converts | to |
# |---|---|---|
# | `XCAL`, `YCAL` | px | m |
# | `TCAL` = 1/`PIV_FPS` | frames (the `TIME` axis) | s |
# | `UCAL` = `XCAL`/`DT_VEL`, `VCAL` = `YCAL`/`DT_VEL` | px/frame | m/s |
# | `ECAL` = 0.5 (`UCAL`² + `VCAL`²) | px²/frame² | m²/s² |
# | `FCAL` = 1/`TCAL` | 1/frame (the `FREQ` axis) | Hz |
# 
# (`DT_VEL` is the pulse separation inside an image pair -- the time base of the
# velocity -- while `TCAL` is the PIV field interval, the time base of the series;
# `ECAL` is exact when `XSCALE == YSCALE`.) A run without a readable acquisition
# log gets every factor = 1 (`calibrated = False`).
# 
# Physical units become **dimensionless** by dividing by the hardcoded scales
# `U_SCALE`, `V_SCALE`, `LENGTH_SCALE`, `TIME_SCALE`, `F_SCALE`, `EK_SCALE`
# (stored per run, used only where needed -- fields are never stored
# dimensionless).
# 
# **Space vs. time statistics.** `averaged` / `rms` = mean / std over SPACE;
# `mean` / `std` = mean / std over TIME.
# 
# **ROI vs. FULL.** One mask (`MASK_ROI`: 1 inside the ROI, NaN outside) and
# NaN-ignoring statistics: the ROI version of any quantity is the same routine
# with the mask applied, the FULL version is the routine without it. Both live in
# the SAME `.npz` -- there are no per-region files any more.
# 
# **Array names read in operation order**, left to right:
# `EK_ROIaveraged_FFT` = space-average Ek first, FFT second (FFT⟨Ek⟩), while
# `FFT_EK_ROIaveraged` = per-point FFT first (`FFT_EK`), space-average second
# (⟨FFT(Ek)⟩).


import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.signal import get_window, find_peaks

from piv_postprocessing_lib import (control_parameters, create_mask,
                        load_piv, parse_run_name,
                        read_acquisition_params, read_paramPostprocessing,
                        rotate_fields, topography_arrangement)

# nan-statistics over an all-NaN slice (fully masked frames / points) are
# expected -- the NaN result is the right answer, so silence the noise.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle.
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print

MUTE_PRINT = False  # toggle this to True to silence ALL print() output

builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print


# --- Configuration --------------------------------------------------------- #
# BATCH = True  -> process every run sub-folder of EVERY dataset in BASE_DIRS;
# BATCH = False -> process only the single run RUN_DIR (its row replaces the
#                  run's entry in its dataset's summary, all other rows kept).
BATCH = True

RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_topBottom/"
           "frot0.50Hz_flib0.400Hz_dphi2deg_SS1")  # the ONE run when BATCH = False

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
        "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets processed when BATCH = True (each holds the run sub-folders).
# BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
#     "FullCylinder", "k20_bottomOnly", "k20_topBottom",
#     "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "k20_topBottom_centerTight_spacer64mm",)]

# Which analyses to run on each loaded field. The .mat is read ONCE per run
# whatever this contains, so asking for both is far cheaper than two passes.
#   ('KE', 'VELOCITY')  -> both        ('KE',) -> kinetic energy only
#   ('VELOCITY',)       -> velocity spectra only
ANALYSES = ('KE', 'VELOCITY')

# Only runs without an existing result are processed unless this is True. The
# two analyses are cached independently: with REPROCESS_ALL = False an existing
# KineticEnergy.npz / Velocity.npz is reused (its summary row is rebuilt from
# it) and the .mat is read only if a requested analysis is actually missing.
REPROCESS_ALL = True

# Restrict processing to these run-folder names (for testing). Empty -> all runs.
ONLY_RUNS = []


# --- Analysis settings and dataset parameters ------------------------------ #
# --- Kinetic energy -------------------------------------------------------- #
KE_DETREND = True   # remove the mean before every Ek FFT (kills the DC spike)

# --- Velocity spectra ------------------------------------------------------ #
FFT_WINDOW = "hann"   # taper before the FFT to suppress spectral leakage;
                      # "boxcar" or None -> no window. Coherent gain is divided
                      # out so amplitudes stay calibrated.
FFT_DETREND = True    # remove the per-point mean before the FFT (kills DC)

# --- Peak detection (ROI-averaged total velocity spectrum) ----------------- #
# All local maxima of |FFT(U)|+|FFT(V)| with F_LOW_MIN <= f <= flib - DELTA_F
# are detected -- the response BELOW the libration forcing. A peak is kept only
# if its amplitude exceeds THRESHOLD_PEAK x the mean amplitude over the search
# band.
DELTA_F = 0.05          # Hz, margin kept below flib (upper bound of the band)
F_LOW_MIN = 0.06        # Hz, lower bound of the search band (keeps the
                        # near-DC bins -- drift, finite record -- out)
THRESHOLD_PEAK = 1.5

###########################################################################################

def load_dataset_params(base_dir):
    """Bind the module globals to `base_dir`'s param_postProcessing.json
    (also rebinding the library's) -- called once per dataset, before its
    runs are processed. BATCH mode calls it for each entry of BASE_DIRS;
    single-run mode for RUN_DIR's parent."""
    _P = read_paramPostprocessing(base_dir)
    rot = int(_P.ROTATE)
    if rot not in (0, 90, 180, -90):
        raise ValueError("ROTATE must be 0, 90, 180 or -90 deg, got %r" % rot)
    top, bottom = topography_arrangement(base_dir)
    globals().update(
        BASE_DIR=base_dir,
        k0=_P.k0, R=_P.R, H=_P.H, nu=_P.nu, LAMBDA=_P.l, ROTATE=rot,
        TOP_TOPO=top, BOTTOM_TOPO=bottom,
        PIV_FILENAME=_P.PIV_FILENAME, LOG_FILENAME=_P.LOG_FILENAME,
        XSCALE=_P.XSCALE, YSCALE=_P.YSCALE, PTS_ROI=_P.PTS_ROI,
        UNCAL_DT=_P.UNCAL_DT, UNCAL_FPS=_P.UNCAL_FPS)

# --- Non-dimensionalisation scales (hardcoded defaults) -------------------- #
def nondim_scales(frot_hz, flib_hz, dphi_deg):
    """Physical -> dimensionless divisors for one run.

    Defaults: U_SCALE = 2*pi*flib*R*dphi[rad] -- the peak libration wall
    velocity U0, the SAME definition as libration_velocity_scale / the summary
    column U0_mps -- V_SCALE = U_SCALE, LENGTH_SCALE = R, TIME_SCALE = 1/frot,
    F_SCALE = frot, EK_SCALE = 0.5*(U_SCALE**2 + V_SCALE**2). NaN where the run
    parameters are unknown. Dividing a physical quantity by its scale makes it
    dimensionless; the fields themselves are never stored that way.
    """
    ok_lib = np.isfinite(flib_hz) and flib_hz and np.isfinite(dphi_deg)
    ok_rot = np.isfinite(frot_hz) and frot_hz
    U_SCALE = (2.0 * np.pi * flib_hz * R * dphi_deg * np.pi / 180.0
               if ok_lib else np.nan)
    V_SCALE = U_SCALE
    return {"U_SCALE": U_SCALE, "V_SCALE": V_SCALE, "LENGTH_SCALE": R,
            "TIME_SCALE": 1.0 / frot_hz if ok_rot else np.nan,
            "F_SCALE": frot_hz if ok_rot else np.nan,
            "EK_SCALE": 0.5 * (U_SCALE ** 2 + V_SCALE ** 2)}


# Output names. One file per run (ROI and FULL share it) + ONE dataset summary
# CSV at BASE_DIR, maintained by both the BATCH and the single-run mode.
RESULT_NAME = {"KE": "KineticEnergy.npz", "VELOCITY": "Velocity.npz"}
SUMMARY_NAME = "Runs_summary.csv"

# The summary columns: one row per run, only run parameters and dimensionless
# control parameters (see control_parameters in piv_postprocessing_lib and the
# HOWTO) -- the measured quantities stay in the per-run .npz files.
SUMMARY_COLUMNS = ["run", "run indx", "processed", "k0", "lambda",
                   "topo_top", "topo_bottom", "frot(Hz)", "flib(Hz)",
                   "dphi(deg)", "flib_star", "Pulse_sep(s)", "PIV_fps(Hz)",
                   "nframe", "U_SCALE (m/s)", "V_SCALE (m/s)",
                   "LENGTH_SCALE", "TIME_SCALE", "Ekman",
                   "BL thickness (m)", "Rossby", "TOPO Rossby",
                   "TOPO Reynolds", "BL Reynolds"]


# ## Calibration factors
# 
# Native (px, frame) -> physical (m, s) multiplicative factors, stored in every
# `.npz` so any tool can calibrate on the fly.


def calibration_factors(dt_vel, fps, ok):
    """The native -> physical multiplicative factors for one run.

    XCAL/YCAL: px -> m. TCAL = 1/fps: frames -> s (fps is the PIV FIELD rate,
    cam_fps/2, the time base of the series). UCAL = XCAL/dt_vel, VCAL =
    YCAL/dt_vel: px/frame -> m/s -- the velocity time base is dt_vel, the pulse
    separation inside an image pair, NOT the field interval TCAL. ECAL converts
    the kinetic energy 0.5*(u**2+v**2) (exact when XSCALE == YSCALE). FCAL =
    1/TCAL: 1/frame -> Hz. ok=False (unreadable log) -> every factor 1, so the
    'calibrated' data stay native.
    """
    if not ok:
        return {"XCAL": 1.0, "YCAL": 1.0, "TCAL": 1.0, "UCAL": 1.0,
                "VCAL": 1.0, "ECAL": 1.0, "FCAL": 1.0}
    XCAL, YCAL = XSCALE, YSCALE
    TCAL = 1.0 / fps
    UCAL, VCAL = XCAL / dt_vel, YCAL / dt_vel
    return {"XCAL": XCAL, "YCAL": YCAL, "TCAL": TCAL, "UCAL": UCAL,
            "VCAL": VCAL, "ECAL": 0.5 * (UCAL ** 2 + VCAL ** 2),
            "FCAL": 1.0 / TCAL}


# ## ROI mask and space statistics
# 
# `MASK_ROI` is 1 inside the `PTS_ROI` rectangle and NaN outside. With
# NaN-ignoring statistics the ROI version of any space reduction is the same
# routine with the mask applied and the FULL version is the routine without it.


def roi_mask(X_px, Y_px):
    """(ny, nx) ROI mask: 1.0 inside the PTS_ROI rectangle, NaN outside.

    PTS_ROI is stored in metres, so the native px grids are scaled with
    XSCALE/YSCALE here (the geometric calibration comes from the param file and
    is known even when the acquisition log is unreadable). An empty ROI falls
    back to the full field, matching the batch tools' long-standing behaviour.
    """
    inside = create_mask(XSCALE * X_px, YSCALE * Y_px, PTS_ROI)
    if not inside.any():
        print("  [warn] PTS_ROI selects nothing -> ROI mask covers the FULL field")
        return np.ones(np.shape(X_px))
    return np.where(inside, 1.0, np.nan)


def space_averaged(F, mask=None):
    """Space average of F (ny, nx, n) over the grid, NaNs ignored -> (n,).

    mask=None -> FULL; mask=MASK_ROI -> ROI (points outside become NaN and drop
    out of the nan-statistics).
    """
    A = F if mask is None else F * mask[:, :, None]
    return np.nanmean(A, axis=(0, 1))


def space_rms(F, mask=None):
    """Space std of F (ny, nx, n) over the grid, NaNs ignored -> (n,)."""
    A = F if mask is None else F * mask[:, :, None]
    return np.nanstd(A, axis=(0, 1))


# ## FFT
# 
# One routine for every spectrum in this notebook: series (1-D) and per-point
# maps (3-D) alike. Frequencies come out in 1/frame (`* FCAL` -> Hz) and
# amplitudes in the units of the input (`* UCAL`/`ECAL`/... -> physical).


def _fill_nan_time(A):
    """Linearly interpolate NaNs along the last axis of A (2-D), in place.

    Edge NaNs take the nearest valid value; all-NaN rows are left untouched
    (the caller zeroes them and restores NaN in the spectrum).
    """
    n = A.shape[-1]
    idx = np.arange(n)
    for row in A:
        m = np.isnan(row)
        if m.any() and not m.all():
            row[m] = np.interp(idx[m], idx[~m], row[~m])
    return A


def amp_spectrum(A, detrend=True, window=None):
    """One-sided FFT amplitude spectrum along the last (time) axis of A.

    A may be a series (n,) or a per-point map (ny, nx, n); the spectrum keeps
    the leading shape. NaNs are linearly interpolated along time first
    (all-NaN series give an all-NaN spectrum). detrend removes the mean;
    `window` tapers before the FFT, with the coherent gain divided out so a
    pure tone reads its true amplitude; every bin except DC (and Nyquist, for
    an even record) is doubled. Returns (freq, amp) with freq in 1/frame
    (multiply by FCAL for Hz) and amp in the units of A.
    """
    A = np.array(A, dtype=float)               # copy: filled/detrended in place
    shp = A.shape
    n = shp[-1]
    A2 = _fill_nan_time(A.reshape(-1, n))
    allnan = np.all(np.isnan(A2), axis=1)
    A2[allnan] = 0.0
    if detrend:
        A2 = A2 - A2.mean(axis=1, keepdims=True)
    if window and window != "boxcar":
        w = get_window(window, n)
    else:
        w = np.ones(n)
    amp = np.abs(np.fft.rfft(A2 * w, axis=-1)) / w.sum()
    if amp.shape[-1] > 1:
        amp[:, 1:] *= 2.0                      # single-sided doubling
        if n % 2 == 0:
            amp[:, -1] /= 2.0                  # ... except the Nyquist bin
    amp[allnan] = np.nan
    freq = np.fft.rfftfreq(n, d=1.0)
    return freq, amp.reshape(shp[:-1] + (freq.size,))


# ## Peak detection
# 
# All significant peaks of the ROI-averaged total velocity spectrum between
# `F_LOW_MIN` and `flib - DELTA_F` -- the response below the libration
# forcing.


def detect_peaks(freq_hz, amp, flib_hz):
    """All significant spectral peaks in [F_LOW_MIN, flib - DELTA_F] Hz.

    Local maxima of `amp` (scipy find_peaks) inside the band below the
    libration forcing, kept only if their amplitude exceeds THRESHOLD_PEAK x
    the mean amplitude over the search band. Returns (f_peaks, amp_peaks)
    sorted by DECREASING amplitude, both empty when flib is unknown or nothing
    qualifies. Amplitude units follow `amp`; the frequency axis must be in Hz
    (the band limits are).
    """
    empty = (np.array([]), np.array([]))
    freq_hz = np.asarray(freq_hz, dtype=float)
    amp = np.asarray(amp, dtype=float)
    if not (np.isfinite(flib_hz) and flib_hz > 0) or amp.size < 3:
        return empty
    band = (freq_hz >= F_LOW_MIN) & (freq_hz <= flib_hz - DELTA_F)
    if not band.any() or not np.any(np.isfinite(amp[band])):
        return empty
    base = np.nanmean(amp[band])
    a = np.where(np.isfinite(amp), amp, 0.0)
    idx, _ = find_peaks(a)
    idx = idx[band[idx] & (a[idx] >= THRESHOLD_PEAK * base)]
    if idx.size == 0:
        return empty
    order = np.argsort(a[idx])[::-1]
    return freq_hz[idx][order], a[idx][order]


# ## Shared metadata and the summary row
# 
# Both `.npz` files open with the same header block (run parameters, acquisition
# parameters, calibration factors, non-dimensionalisation scales), so the run's
# single summary row can be built from either file. `_get` reads a plain payload
# dict and a loaded `.npz` alike, so cached results rebuild their row without
# touching the `.mat`.


def _get(d, key, default=np.nan):
    """d[key] from a dict or a loaded npz; 0-d arrays unwrap to scalars."""
    keys = d.files if hasattr(d, "files") else d.keys()
    if key not in keys:
        return default
    v = np.asarray(d[key])
    return v.item() if v.ndim == 0 else v


def run_metadata(name, dt_vel, fps, ok):
    """Header block shared by both npz files: run parameters (k0, lambda, frot,
    flib, dphi, SS index, topography), acquisition parameters (PIV_FPS, DT_VEL,
    TSCALE, XSCALE, YSCALE), calibration factors and non-dim scales."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    ss = re.search(r"SS(\d+)", name)
    meta = {"run": name, "calibrated": ok, "ROTATE": ROTATE,
            "k0": k0, "lambda": LAMBDA, "R": R, "H": H, "nu": nu,
            "frot_Hz": frot_hz, "flib_Hz": flib_hz, "dphi_deg": dphi_deg,
            "run_idx": int(ss.group(1)) if ss else -1,
            "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO,
            "PIV_FPS": fps, "DT_VEL": dt_vel, "TSCALE": 1.0 / fps,
            "XSCALE": XSCALE, "YSCALE": YSCALE,
            "pts_ROI": np.array(PTS_ROI)}
    meta.update(calibration_factors(dt_vel, fps, ok))
    meta.update(nondim_scales(frot_hz, flib_hz, dphi_deg))
    return meta


def build_summary_row(d, processed=True):
    """The run's single summary row, from a payload dict / loaded npz.

    Only run parameters and dimensionless control parameters -- the measured
    quantities stay in the per-run `.npz` files.
    """
    frot, flib = _get(d, "frot_Hz"), _get(d, "flib_Hz")
    dphi = _get(d, "dphi_deg")
    row = {"run": str(_get(d, "run", "")), "run indx": _get(d, "run_idx"),
           "processed": processed, "k0": _get(d, "k0"),
           "lambda": _get(d, "lambda"),
           "topo_top": _get(d, "top_topo"),
           "topo_bottom": _get(d, "bottom_topo"),
           "frot(Hz)": frot, "flib(Hz)": flib, "dphi(deg)": dphi,
           "flib_star": (flib / frot if (np.isfinite(frot) and frot)
                         else np.nan),
           "Pulse_sep(s)": _get(d, "DT_VEL"),
           "PIV_fps(Hz)": _get(d, "PIV_FPS"),
           "nframe": _get(d, "nframes"),
           "U_SCALE (m/s)": _get(d, "U_SCALE"),
           "V_SCALE (m/s)": _get(d, "V_SCALE"),
           "LENGTH_SCALE": _get(d, "LENGTH_SCALE"),
           "TIME_SCALE": _get(d, "TIME_SCALE")}
    row.update(control_parameters(frot, flib, dphi))
    return row


# ## Kinetic energy
# 
# Per-point `Ek = 0.5*(U**2 + V**2)` in native px²/frame², then every output the
# `KineticEnergy.npz` must hold: the four space-statistics time series and their
# FFTs, the per-point time statistics maps, the per-point FFT `FFT_EK` and its
# four space statistics. The run's summary row is built by the shared driver.


def analyse_ke(name, out_file, meta, X, Y, U, V, mask, nframes):
    """Kinetic-energy analysis of one loaded run (native units).

    Writes <run>/PostProcessing/KineticEnergy.npz (ROI and FULL together) and
    returns the payload.
    """
    Ek = 0.5 * (U ** 2 + V ** 2)                 # (ny, nx, nt), px^2/frame^2
    ny, nx = Ek.shape[:2]
    TIME = np.arange(nframes, dtype=float)       # frames; * TCAL -> s

    # Space statistics vs. time: averaged/rms over the ROI (mask) and FULL.
    EK_ROIaveraged = space_averaged(Ek, mask)
    EK_FULLaveraged = space_averaged(Ek)
    EK_ROIrms = space_rms(Ek, mask)
    EK_FULLrms = space_rms(Ek)

    # FFT of each series ('<series>_FFT': space statistic first, FFT second).
    FREQ, EK_ROIaveraged_FFT = amp_spectrum(EK_ROIaveraged, KE_DETREND)
    _, EK_FULLaveraged_FFT = amp_spectrum(EK_FULLaveraged, KE_DETREND)
    _, EK_ROIrms_FFT = amp_spectrum(EK_ROIrms, KE_DETREND)
    _, EK_FULLrms_FFT = amp_spectrum(EK_FULLrms, KE_DETREND)

    # Per-point time statistics (2-D maps; no ROI needed, the full field).
    EK_FULLmean = np.nanmean(Ek, axis=2)
    EK_FULLstd = np.nanstd(Ek, axis=2)

    # Per-point FFT (i, j, k) and its space statistics ('FFT_EK_<...>': FFT
    # first, space statistic second). float32 halves the file.
    _, FFT_EK = amp_spectrum(Ek, KE_DETREND)
    FFT_EK_ROIaveraged = space_averaged(FFT_EK, mask)
    FFT_EK_FULLaveraged = space_averaged(FFT_EK)
    FFT_EK_ROIrms = space_rms(FFT_EK, mask)
    FFT_EK_FULLrms = space_rms(FFT_EK)

    payload = dict(meta, nframes=nframes, ny=ny, nx=nx,
                   npoints_ROI=int(np.isfinite(mask).sum()),
                   npoints_FULL=int(mask.size), detrend=KE_DETREND,
                   X=X.astype(np.float32), Y=Y.astype(np.float32),
                   MASK_ROI=mask.astype(np.float32),
                   TIME=TIME, FREQ=FREQ,
                   EK_ROIaveraged=EK_ROIaveraged,
                   EK_FULLaveraged=EK_FULLaveraged,
                   EK_ROIrms=EK_ROIrms, EK_FULLrms=EK_FULLrms,
                   EK_ROIaveraged_FFT=EK_ROIaveraged_FFT,
                   EK_FULLaveraged_FFT=EK_FULLaveraged_FFT,
                   EK_ROIrms_FFT=EK_ROIrms_FFT, EK_FULLrms_FFT=EK_FULLrms_FFT,
                   EK_FULLmean=EK_FULLmean.astype(np.float32),
                   EK_FULLstd=EK_FULLstd.astype(np.float32),
                   FFT_EK=FFT_EK.astype(np.float32),
                   FFT_EK_ROIaveraged=FFT_EK_ROIaveraged,
                   FFT_EK_FULLaveraged=FFT_EK_FULLaveraged,
                   FFT_EK_ROIrms=FFT_EK_ROIrms, FFT_EK_FULLrms=FFT_EK_FULLrms)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, **payload)

    print("  [ok] %-40s KE       nframes=%d npts_ROI=%d/%d"
          % (name, nframes, payload["npoints_ROI"], payload["npoints_FULL"]))
    return payload


# ## Velocity
# 
# The full native velocity matrices `U(x, y, t)` / `V(x, y, t)` (so later
# reprocessing never has to reopen the huge PIV `.mat`), the per-point time
# statistics (`Umean`, `Vmean`, `Ustd`, `Vstd`), the per-point amplitude spectra
# `FFT_U` / `FFT_V`, their ROI/FULL space averages and the peak detection.
# Plotting and polarization are handled by `BATCH_plotting_velocity`.


def analyse_velocity(name, out_file, meta, X, Y, U, V, mask, nframes):
    """Velocity analysis of one loaded run (native units).

    Writes <run>/PostProcessing/Velocity.npz (ROI and FULL together) and
    returns the payload.
    """
    ny, nx = U.shape[:2]

    # Per-point time statistics (2-D maps).
    Umean = np.nanmean(U, axis=2)
    Vmean = np.nanmean(V, axis=2)
    Ustd = np.nanstd(U, axis=2)
    Vstd = np.nanstd(V, axis=2)

    # Per-point amplitude spectra and their space averages (FFT first,
    # space-average second).
    FREQ, FFT_U = amp_spectrum(U, FFT_DETREND, FFT_WINDOW)
    _, FFT_V = amp_spectrum(V, FFT_DETREND, FFT_WINDOW)
    FFT_U_ROIaveraged = space_averaged(FFT_U, mask)
    FFT_V_ROIaveraged = space_averaged(FFT_V, mask)
    FFT_U_FULLaveraged = space_averaged(FFT_U)
    FFT_V_FULLaveraged = space_averaged(FFT_V)
    FFT_TOTAL_ROIaveraged = FFT_U_ROIaveraged + FFT_V_ROIaveraged
    FFT_TOTAL_FULLaveraged = FFT_U_FULLaveraged + FFT_V_FULLaveraged

    # Peak detection on the CALIBRATED ROI-averaged total spectrum (the band
    # limits F_LOW_MIN / DELTA_F are in Hz); stored back in native units.
    FCAL = meta["FCAL"]
    f_peaks_hz, amp_peaks = np.array([]), np.array([])
    if meta["calibrated"]:
        f_peaks_hz, amp_peaks = detect_peaks(FREQ * FCAL,
                                             FFT_TOTAL_ROIaveraged,
                                             meta["flib_Hz"])

    payload = dict(meta, nframes=nframes, ny=ny, nx=nx,
                   npoints_ROI=int(np.isfinite(mask).sum()),
                   npoints_FULL=int(mask.size),
                   window=FFT_WINDOW or "boxcar", detrend=FFT_DETREND,
                   X=X.astype(np.float32), Y=Y.astype(np.float32),
                   MASK_ROI=mask.astype(np.float32),
                   # full native velocity matrices (px/frame): reprocessing can
                   # start from here instead of the far bigger PIV .mat
                   U=U.astype(np.float32), V=V.astype(np.float32),
                   Umean=Umean.astype(np.float32),
                   Vmean=Vmean.astype(np.float32),
                   Ustd=Ustd.astype(np.float32), Vstd=Vstd.astype(np.float32),
                   FREQ=FREQ,
                   FFT_U=FFT_U.astype(np.float32),
                   FFT_V=FFT_V.astype(np.float32),
                   FFT_U_ROIaveraged=FFT_U_ROIaveraged,
                   FFT_V_ROIaveraged=FFT_V_ROIaveraged,
                   FFT_U_FULLaveraged=FFT_U_FULLaveraged,
                   FFT_V_FULLaveraged=FFT_V_FULLaveraged,
                   FFT_TOTAL_ROIaveraged=FFT_TOTAL_ROIaveraged,
                   FFT_TOTAL_FULLaveraged=FFT_TOTAL_FULLaveraged,
                   F_PEAKS=f_peaks_hz / FCAL, AMP_PEAKS=amp_peaks,
                   N_PEAKS=int(f_peaks_hz.size))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, **payload)

    print("  [ok] %-40s VELOCITY nframes=%d npts_ROI=%d/%d n_peaks=%d"
          % (name, nframes, payload["npoints_ROI"], payload["npoints_FULL"],
             payload["N_PEAKS"]))
    return payload


# ## The shared driver
# 
# `process_run` resolves what each requested analysis still needs, reads the
# `.mat` **once** if anything does (in native units -- no calibration applied to
# the stored fields), builds the ROI mask and hands the same `(U, V)` to both
# analyses. Each analysis caches independently through its own `.npz`.


def analyses_to_run(analyses=None):
    """Normalise the ANALYSES config to a validated tuple in a fixed order."""
    a = ANALYSES if analyses is None else analyses
    if isinstance(a, str):
        a = (a,)
    sel = tuple(x for x in ("KE", "VELOCITY") if x.upper() in
                {str(v).upper() for v in a})
    if not sel:
        raise ValueError("ANALYSES must contain 'KE' and/or 'VELOCITY', got %r"
                         % (a,))
    return sel


def process_run(run_dir, analyses=None, reprocess=False):
    """Process one run folder (ROI and FULL together, via the mask).

    Returns the run's single summary row. The PIV .mat is read at most ONCE
    however many analyses are requested. With reprocess=False an existing
    per-run .npz is reused (the row rebuilt from it) and the .mat is opened
    only if a requested analysis is actually missing. A folder with no PIV
    file yields a processed=False row.
    """
    analyses = analyses_to_run(analyses)
    name = os.path.basename(run_dir.rstrip("/"))
    piv_file = os.path.join(run_dir, PIV_FILENAME)
    out_dir = os.path.join(run_dir, "PostProcessing")
    out_files = {a: os.path.join(out_dir, RESULT_NAME[a]) for a in analyses}

    row = None
    todo = []
    for a in analyses:
        if os.path.isfile(out_files[a]) and not reprocess:
            try:
                d = np.load(out_files[a], allow_pickle=True)
                if row is None:
                    row = build_summary_row(d)
                print("  [skip] %-40s %-8s already processed (cached)"
                      % (name, a))
                continue
            except Exception as exc:
                print("  [warn] %s %s: cached result unreadable (%s) -> "
                      "reprocessing" % (name, a, exc))
        todo.append(a)
    if not todo:
        return row

    if not os.path.isfile(piv_file):
        print("  [skip] no %s in %s" % (PIV_FILENAME, name))
        if row is None:
            meta = run_metadata(name, UNCAL_DT, UNCAL_FPS, False)
            row = build_summary_row(meta, processed=False)
        return row

    dt_vel, fps, ok = read_acquisition_params(os.path.join(run_dir,
                                                           LOG_FILENAME))
    if not ok:
        print("  [warn] %s: no readable acquisition log -- data stay in "
              "native px/frame units" % name)
    meta = run_metadata(name, dt_vel, fps, ok)

    # --- the expensive step, done once for every requested analysis --------- #
    # NATIVE units throughout: X, Y in px, U, V in px/frame. No calibration is
    # applied to the stored fields -- only the factors in `meta` travel along.
    X, Y, U, V, nframes = load_piv(piv_file)
    X, Y, U, V = rotate_fields(X, Y, U, V, ROTATE)   # into the final frame
    mask = roi_mask(X, Y)

    analyse = {"KE": analyse_ke, "VELOCITY": analyse_velocity}
    for a in todo:
        payload = analyse[a](name, out_files[a], meta, X, Y, U, V, mask,
                             nframes)
        if row is None:
            row = build_summary_row(payload)
    return row


def update_summary(rows):
    """Create/update the single dataset summary CSV at BASE_DIR.

    Rows for the runs in `rows` replace their existing entries; every other
    existing row is kept, so a single run or a partial batch (ONLY_RUNS)
    updates the table rather than rewriting it.
    """
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    csv_path = os.path.join(BASE_DIR, SUMMARY_NAME)
    if os.path.isfile(csv_path):
        try:
            old = pd.read_csv(csv_path)
            df = pd.concat([old[~old["run"].isin(df["run"])], df],
                           ignore_index=True)
        except Exception as exc:
            print("  [warn] existing summary unreadable (%s) -> rewriting"
                  % exc)
    df = (df.reindex(columns=SUMMARY_COLUMNS)
            .sort_values(["frot(Hz)", "flib(Hz)", "dphi(deg)"])
            .reset_index(drop=True))
    df.to_csv(csv_path, index=False)
    n_proc = int((df["processed"] == True).sum())
    print("\n  summary (%d runs: %d processed, %d unprocessed):\n    %s"
          % (len(df), n_proc, len(df) - n_proc, csv_path))
    return df


# ## The batch
# 
# ONE dataset summary CSV (`Runs_summary.csv`) at `BASE_DIR`, nothing else -- no
# figures (see `BATCH_plot_energy` / `BATCH_plotting_velocity`). Rows of runs not
# in this batch are kept, so a partial batch (`ONLY_RUNS`) updates rather than
# rewrites the table.


def run_batch(reprocess_all=REPROCESS_ALL, only_runs=None, analyses=None):
    """Process every run under BASE_DIR for each requested analysis.

    only_runs (list of folder names) restricts processing for testing; None
    falls back to the ONLY_RUNS config (empty -> all runs).
    Returns the dataset summary DataFrame.
    """
    analyses = analyses_to_run(analyses)
    if only_runs is None:
        only_runs = ONLY_RUNS
    subdirs = sorted(dd for dd in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(dd)
                     and not os.path.basename(dd).startswith("."))
    if only_runs:
        wanted = set(only_runs)
        subdirs = [dd for dd in subdirs if os.path.basename(dd) in wanted]

    print("Found %d subfolders to process in %s%s\n  analyses: %s"
          % (len(subdirs), BASE_DIR,
             "  (reprocessing ALL)" if reprocess_all else
             "  (skipping already-processed)", ", ".join(analyses)))

    rows = []
    for run_dir in subdirs:
        try:
            row = process_run(run_dir, analyses, reprocess=reprocess_all)
        except Exception as exc:  # keep the batch going
            print("  [error] %s: %s" % (os.path.basename(run_dir), exc))
            row = None
        if row is not None:
            rows.append(row)

    if not rows:
        print("  No runs found.")
        return None
    return update_summary(rows)


# ## Run
# 
# `BATCH = True` processes every run of every dataset in `BASE_DIRS`
# (`ONLY_RUNS` in the config cell restricts each batch to a few folder names
# for testing; clear it to process **all** runs). `BATCH = False` processes
# only `RUN_DIR` and updates its row in its dataset's summary. Set
# `REPROCESS_ALL = True` to recompute from the `.mat` files.


if BATCH:
    # Every run of every dataset in BASE_DIRS -> {dataset: summary DataFrame}.
    results = {}
    for _bd in BASE_DIRS:
        _bd = _bd.rstrip("/")
        print("\n=== dataset %s ===" % os.path.basename(_bd))
        load_dataset_params(_bd)
        results[os.path.basename(_bd)] = run_batch(reprocess_all=REPROCESS_ALL)
else:
    # The single RUN_DIR -> its row replaces the run's entry in the summary
    # of its dataset (RUN_DIR's parent).
    load_dataset_params(os.path.dirname(os.path.abspath(RUN_DIR.rstrip("/"))))
    row = process_run(RUN_DIR, ANALYSES, reprocess=REPROCESS_ALL)
    results = update_summary([row])

# Preview: the single summary, or {dataset: rows} in batch mode.
(results.head(30) if hasattr(results, "head") else
 {k: (0 if v is None else len(v)) for k, v in results.items()})
