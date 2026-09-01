"""Auto-generated .py twin of process_single_Velocity.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Reprocess a single run -- Velocity FFT (from the `.mat`)
# 
# Runs `batch_Velocity`'s per-run step for ONE run, straight from a PIV `.mat`,
# reusing the batch's helper, figure and `build_row` functions verbatim -- so the
# outputs are identical to what the batch would write for that run:
# `VelocityFFT_<region>.npz`, the spectrum and polarization figures, and the
# `VelocityFFT_summary_<region>.csv` row (with the `kept` de-duplication).
# 
# Set `PIV_FILENAME` to the PIV `.mat` to reprocess -- this notebook **always** uses
# that local name (the value in `param_postProcessing.json` is ignored). If it is a
# band-passed file from `filter_velocity_bandpass` (`..._bp<lo>-<hi>Hz.mat`), its
# `_bp<lo>-<hi>Hz` tag is appended to **every output** so the filtered results never
# overwrite the raw ones.
# 
# Everything is recomputed fresh from the `.mat`; the switches only decide what is
# written to disk:
# 
# | switch | effect |
# |---|---|
# | `OVERWRITE_NPZ` | `False` keeps an existing `.npz`; `True` re-writes it |
# | `OVERWRITE_FIG` | `False` keeps existing figure files; `True` re-writes them |
# | `UPDATE_SUMMARY` | `False` leaves the summary CSV untouched; `True` inserts/replaces this run's row |
# 
# Use `single_Velocity` if you only want to redraw from an existing `.npz`.


# ## 1. Imports


import os
import re
import numpy as np
import pandas as pd
from scipy.signal import get_window
try:
    import pywt
except ImportError:              # only needed when PERFORM_WAVELET
    pywt = None
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, amp_at_freq,
                        compute_polarization, dimensionless_numbers,
                        fft_axis_limits, fft_guide_lines, figure_filename,
                        libration_velocity_scale, load_piv, parse_run_name,
                        peak_freq, peak_freq_in_band, read_acquisition_params,
                        read_paramPostprocessing, region_fields, region_tag)


# ## 2. Configuration


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library).
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = False
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# The single run FOLDER to reprocess (must hold the .mat and the acquisition log).
PATH = ('/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
        'CylinderExperimentsGMA/k20_topBottom/frot0.50Hz_flib0.400Hz_dphi2deg_SS1')

# Band-passed PIV .mat to reprocess -- this notebook ALWAYS uses this local
# file; the PIV_FILENAME in param_postProcessing.json is ignored here. Point it
# at a filter_velocity_bandpass output (..._bp<lo>-<hi>Hz.mat).
PIV_FILENAME = 'PIVlab_results_uncalibrated.mat'

# Spatial extent: 'ROI' (crop to PTS_ROI) or 'FULL' (whole field).
REGION = 'ROI'

# --- Overwrite switches --------------------------------------------------
OVERWRITE_NPZ  = True   # (over)write VelocityFFT_<region>.npz (False -> keep an
                        #  existing one; figures + row still come from the fresh
                        #  computation from the .mat)
OVERWRITE_FIG  = True   # (over)write the spectrum + polarization figures
UPDATE_SUMMARY = True   # add/refresh this run's row in VelocityFFT_summary_<region>.csv

# Per-run output + summary stems (region tag appended) -- identical to the batch.
RESULT_STEM  = "VelocityFFT"            # per-run npz -> <stem>_<region>.npz
SUMMARY_STEM = "VelocityFFT_summary"    # summary at the dataset root
POLFIG_STEM  = "Velocity_polarization"  # per-run polarization figure

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'

# --- FFT settings (identical defaults to batch_Velocity) ------------------
FFT_WINDOW = "hann"        # taper before the FFT ("boxcar"/None -> none)
FFT_DETREND = "constant"   # "constant" removes the per-point mean; None keeps it
F_LOW_FMIN = 0.06          # Hz, lower edge of the f_low search band
THRESHOLD_PEAK = 1.5       # f_low significant only if peak >= THRESHOLD_PEAK * band mean
LOGX = False               # log frequency axis on the spectrum figure
LOGY = True                # log amplitude axis

# ----------------------------------------------------------------------
# Parameters live in the dataset folder (PATH's parent).


_P = read_paramPostprocessing(os.path.dirname(PATH.rstrip('/')))
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(PATH)
LOG_FILENAME = _P.LOG_FILENAME
# PIV_FILENAME stays the local band-passed file set above (param value ignored).
# If the PIV file is band-passed (name has a '_bp<lo>-<hi>Hz' tag from
# filter_velocity_bandpass), append that tag to every output name so the
# filtered results never overwrite the raw ones.
_bpm = re.search(r"_bp[-0-9.]+Hz", PIV_FILENAME)
BP_TAG = _bpm.group(0) if _bpm else ""
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
UNCAL_SCALE = _P.UNCAL_SCALE
tag = region_tag(REGION)


# ### Wavelet analysis parameters
# 
# All settings for the time-frequency (wavelet) analysis in section 6, kept in their own cell.


# --- Wavelet (time-frequency) analysis settings --------------------------
# The wavelet analysis uses the SAME REGION ('ROI'/'FULL') and BP_TAG as the
# FFT above. The CWT is computed at EVERY grid point of the region and the
# amplitude |W U| + |W V| is then averaged over the domain -- the time-resolved
# analog of averaged_fft. It is a separate product: figures + a dedicated .npz
# are written, but nothing is added to the summary.
#
# PERFORM_WAVELET is the master switch for section 6: False skips the wavelet
# analysis entirely -- no CWT, no .npz, no figures -- and leaves whatever is
# already on disk untouched. The CWT is by far the slowest step here, so turning
# it off is the quick way to re-run only the FFT products. Everything else in
# this cell only matters when it is True.
PERFORM_WAVELET = True

WAVELET         = "cmor1.5-1.0"  # PyWavelets continuous wavelet (complex Morlet:
                                 #  "cmor<B>-<C>", B=bandwidth, C=centre freq).
                                 #  Others: "morl", "mexh", "gaus4", "shan1.0-1.0".
WAVELET_FMIN    = 0.05           # Hz, lowest frequency shown on the map
WAVELET_FMAX    = None           # Hz, highest frequency; None -> Nyquist (fps/2)
WAVELET_NFREQ   = 128            # number of frequency rows (scales)
WAVELET_SPACING = "log"          # 'log' or 'linear' spacing of the frequencies
WAVELET_METHOD  = "fft"          # CWT implementation: 'fft' (fast, recommended)
                                 #  or 'conv' (direct convolution, much slower)
WAVELET_MAX_POINTS = None        # cap on the number of grid points averaged
                                 #  (evenly subsampled) to speed up FULL; None ->
                                 #  use EVERY valid point in the region
WAVELET_LOGF    = True           # log frequency (y) axis
WAVELET_LOGC    = True          # log colour scale for the amplitude
WAVELET_CMAP    = "viridis"
WAVELET_STEM    = "VelocityWavelet"   # per-run npz + figure stem


# ## 3. Helper functions (verbatim from batch_Velocity)


def _interp_nan_rows(A):
    """Linearly interpolate NaNs along time for each row (point) of A.

    A has shape (npoints, nframes). Rows that are entirely NaN are flagged in the
    returned `keep` mask (they are excluded from the ROI average). Interior /
    edge NaNs are filled by linear interpolation (edges use the nearest value).
    """
    A = A.astype(float, copy=True)
    n = A.shape[1]
    idx = np.arange(n)
    keep = np.ones(A.shape[0], dtype=bool)
    for i in range(A.shape[0]):
        row = A[i]
        m = np.isnan(row)
        if m.all():
            keep[i] = False
            continue
        if m.any():
            row[m] = np.interp(idx[m], idx[~m], row[~m])
    return A, keep


def _amp_spectrum(A, window=FFT_WINDOW, detrend=FFT_DETREND):
    """Single-sided FFT amplitude spectrum along the last axis.

    A has shape (npoints, nframes). Returns an array (npoints, nfreq) of
    amplitudes: |rfft| divided by the window's coherent gain (so a pure tone
    reads its true amplitude), then doubled for every bin except DC and, for an
    even-length record, Nyquist. The mean is removed first when detrend is
    "constant".
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[-1]
    if detrend == "constant":
        A = A - np.nanmean(A, axis=-1, keepdims=True)
    if window and window != "boxcar":
        w = get_window(window, n)
    else:
        w = np.ones(n)
    cg = w.sum()                       # coherent gain -> amplitude-correct scaling
    F = np.fft.rfft(A * w, axis=-1)
    amp = np.abs(F) / cg
    if amp.shape[-1] > 1:
        amp[..., 1:] *= 2.0            # single-sided: fold negative frequencies
        if n % 2 == 0:
            amp[..., -1] /= 2.0        # Nyquist bin is not doubled
    return amp



def _power_spectrum(A, window=FFT_WINDOW, detrend=FFT_DETREND):
    """Raw FFT power spectrum |rfft|**2 along the last axis.

    A has shape (npoints, nframes). The mean is removed (detrend="constant")
    and the same window as the amplitude spectrum is applied, then power =
    np.abs(rfft)**2 (no single-sided doubling / coherent-gain scaling; this is
    the power definition used for the polarization ratio powerV/powerU).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[-1]
    if detrend == "constant":
        A = A - np.nanmean(A, axis=-1, keepdims=True)
    if window and window != "boxcar":
        w = get_window(window, n)
    else:
        w = np.ones(n)
    F = np.fft.rfft(A * w, axis=-1)
    return np.abs(F) ** 2


def averaged_fft(u_roi, v_roi, fps, window=FFT_WINDOW, detrend=FFT_DETREND):
    """ROI-averaged single-sided amplitude spectrum of U(t) and V(t).

    For every grid point in the ROI the amplitude spectrum of its time series is
    computed with np.fft.rfft, then averaged over all valid points.
    Returns (f, amp_u, amp_v, amp_total, powerU, powerV, npoints) where
    amp_total = amp_u+amp_v (amplitudes, m/s) and powerU/powerV are the
    ROI-averaged power spectra |FFT|**2 of each component (m^2/s^2). Returns
    NaNs if no valid point exists.
    """
    nf = u_roi.shape[-1]
    U = u_roi.reshape(-1, nf)
    V = v_roi.reshape(-1, nf)
    U, keepU = _interp_nan_rows(U)
    V, keepV = _interp_nan_rows(V)

    f = np.fft.rfftfreq(nf, d=1.0 / fps)
    if not keepU.any() or not keepV.any():
        nanv = np.full_like(f, np.nan, dtype=float)
        return (f, nanv, nanv.copy(), nanv.copy(),
                nanv.copy(), nanv.copy(), 0)

    Au = _amp_spectrum(U[keepU], window, detrend)
    Av = _amp_spectrum(V[keepV], window, detrend)
    amp_u = np.nanmean(Au, axis=0)
    amp_v = np.nanmean(Av, axis=0)
    amp_total = amp_u + amp_v

    # ROI-averaged power spectra |FFT|**2 of each component.
    powerU = np.nanmean(_power_spectrum(U[keepU], window, detrend), axis=0)
    powerV = np.nanmean(_power_spectrum(V[keepV], window, detrend), axis=0)

    npoints = int(keepU.sum())
    return f, amp_u, amp_v, amp_total, powerU, powerV, npoints



def roi_polarization_stats(u_roi, v_roi, fps, window=FFT_WINDOW,
                           detrend=FFT_DETREND):
    """ROI statistics of the velocity-polarization ratio powerV/powerU.

    Definitions (per grid point, |.| = np.abs, all vs the one-sided FFT grid f):
      powerU = |FFT(U)|**2 , powerV = |FFT(V)|**2 .
    All spectra use the same detrend + window as the amplitude spectrum.
    Returns a dict with:
      powerU_mean, powerV_mean : ROI mean of powerU / powerV
      powerRatio, powerRatio_std: ROI mean/std of the per-point powerV/powerU
      powerRatio_med           : ROI median of the per-point powerV/powerU
      powerRatio_p25, powerRatio_p75 : ROI 25/75 percentiles of that ratio
      f, npoints
    """
    nf = u_roi.shape[-1]
    U = u_roi.reshape(-1, nf)
    V = v_roi.reshape(-1, nf)
    U, keepU = _interp_nan_rows(U)
    V, keepV = _interp_nan_rows(V)
    keep = keepU & keepV
    f = np.fft.rfftfreq(nf, d=1.0 / fps)
    nanv = np.full_like(f, np.nan, dtype=float)
    out = {"f": f, "npoints": int(keep.sum())}
    keys = ("powerU_mean", "powerV_mean", "powerRatio", "powerRatio_std",
            "powerRatio_med", "powerRatio_p25", "powerRatio_p75")
    if not keep.any():
        for k in keys:
            out[k] = nanv.copy()
        return out

    Uk = U[keep]
    Vk = V[keep]

    # (1) power spectra |FFT|**2 per point and their ROI mean.
    pU = _power_spectrum(Uk, window, detrend)
    pV = _power_spectrum(Vk, window, detrend)
    powerU_mean = np.nanmean(pU, axis=0)
    powerV_mean = np.nanmean(pV, axis=0)

    # (2) per-point power ratio powerV/powerU, then ROI statistics.
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = pV / pU
    rp[~np.isfinite(rp)] = np.nan
    powerRatio = np.nanmean(rp, axis=0)
    powerRatio_std = np.nanstd(rp, axis=0)
    # Median + 25-75 percentiles: a log-axis-friendly, skew-robust ROI spread.
    powerRatio_med = np.nanmedian(rp, axis=0)
    powerRatio_p25 = np.nanpercentile(rp, 25, axis=0)
    powerRatio_p75 = np.nanpercentile(rp, 75, axis=0)

    out.update(powerU_mean=powerU_mean, powerV_mean=powerV_mean,
               powerRatio=powerRatio, powerRatio_std=powerRatio_std,
               powerRatio_med=powerRatio_med, powerRatio_p25=powerRatio_p25,
               powerRatio_p75=powerRatio_p75)
    return out


def build_row(name, out_file, processed, region=np.nan, fps=np.nan,
              nframes=np.nan, npoints=np.nan, f_peak=np.nan, f_low=np.nan,
              dt_vel=np.nan, ok=np.nan, peak_amp=None, peak_amp_star=None):
    """Assemble one summary-table row (folder-name metadata + results/flags)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    # Session index: the integer following 'SS' in the folder name (SS2 -> 2).
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    # Dimensionless numbers + normalised (_star) frequencies (/ frot).
    dl = dimensionless_numbers(frot_hz, flib_hz, dphi_deg)
    fn = frot_hz if np.isfinite(frot_hz) and frot_hz else np.nan
    row = {"run": name, "run idx": run_idx, "region": region,
           "processed": processed, "frot_Hz": frot_hz, "flib_Hz": flib_hz,
           "dphi_deg": dphi_deg, "fstar": fstar, "flib_star": fstar,
           "calibrated": ok, "dt_vel_s": dt_vel, "fps_Hz": fps,
           "nframes": nframes, "npoints": npoints,
           "f_peak_Hz": f_peak, "f_peak_star": f_peak / fn,
           "f_low_Hz": f_low, "f_low_star": f_low / fn,
           "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO, "npz": out_file}
    # Amplitude of the summed spectrum at f_lib, f_low and the f_lib -/+ f_low
    # sidebands -- raw (m/s) and normalised by U0.
    _pa = peak_amp or {}
    _pas = peak_amp_star or {}
    for _k in ("flib", "flow", "flib_minus_flow", "flib_plus_flow"):
        row["amp_%s" % _k] = _pa.get(_k, np.nan)
        row["amp_%s_star" % _k] = _pas.get(_k, np.nan)
    row.update(dl)
    return row


def _save_fft_figure(out_png, f, amp_u, amp_v, amp_total, name, calibrated,
                     f_peak, frot=None, flib=None, f_low=None, dphi_deg=None,
                     normalize=False):
    """Save a per-run figure of the ROI-averaged FFT amplitude spectra.

    Written next to the run's .npz. Log-log axes; the DC bin (f = 0) is dropped.
    Dashed guides mark f_rot (blue), f_lib and 2*f_lib (red), and the
    low-frequency peak f_low with its sidebands f_lib -/+ f_low (green); each
    carries its value in the legend. f_peak is stated in the title.
    normalize=True divides the amplitudes by U0 and the frequency axis by f_rot.
    Closed immediately so a batch does not leave many windows open.
    """
    funit = "Hz" if calibrated else "1/frame"
    aunit = r"m\,s^{-1}" if calibrated else r"px/frame"
    U0 = (libration_velocity_scale(dphi_deg, flib)
          if flib and dphi_deg is not None else np.nan)
    ascale = U0 if (normalize and calibrated and np.isfinite(U0) and U0) else 1.0
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    at = amp_total / ascale
    ff = f / fscale
    _frot = frot / fscale if frot else frot
    _flib = flib / fscale if flib else flib
    _flow = f_low / fscale if (f_low is not None and np.isfinite(f_low)) else f_low
    _fpk = f_peak / fscale if np.isfinite(f_peak) else f_peak
    _fu = "" if fscale != 1.0 else " %s" % funit
    m = f > 0
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot = ax.semilogy if LOGY else ax.plot
    if np.any(m):
        _plot(ff[m], at[m], color="C0", lw=1.2,
              label=r"$|\widehat{U}|+|\widehat{V}|$")
    ax.set_xlabel(r"$f / f_{\mathrm{rot}}$" if fscale != 1.0
                  else "frequency (%s)" % funit, fontsize=12)
    ax.set_ylabel(r"ROI-averaged amplitude $/\,U_0$" if ascale != 1.0
                  else r"ROI-averaged amplitude  ($\mathrm{%s}$)" % aunit, fontsize=12)
    title = name + ("   (normalized)" if normalize else "") \
        + "   ($k_0 = %g$, top=%s, bottom=%s)" % (k0, TOP_TOPO, BOTTOM_TOPO)
    if np.isfinite(_fpk) and _fpk > 0:
        title += "      $f_{\mathrm{peak}}$ = %.4g%s" % (_fpk, _fu)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    # Frequency axis to [0, max(4*frot,4*flib)]; y to decade bounds.
    _xmax, _ymin, _ymax = fft_axis_limits(ff, [at], _frot, _flib)
    # y lower bound: the closest power of 10 below the minimum of the visible
    # total amplitude (keeps the whole curve in view on the log axis).
    _vis = at[(f > 0) & (ff <= _xmax) & (at > 0)]
    if _vis.size:
        _ymin = 10.0 ** np.floor(np.log10(_vis.min()))
    if LOGX:
        ax.set_xscale("log")
        _pos = ff[(ff > 0) & (ff <= _xmax)]
        ax.set_xlim(_pos.min() if _pos.size else _xmax / 100.0, _xmax)
    else:
        ax.set_xlim(0, _xmax)
    if _ymin and _ymax:
        ax.set_ylim(_ymin, _ymax)
    # Reference frequencies: colour-coded by family, value in the legend. Drawn
    # after the limits are set so the off-scale ones are dropped.
    for fq, lbl, colr in fft_guide_lines(_frot, _flib, _flow):
        if fq > _xmax:
            continue
        ax.axvline(fq, color=colr, ls="--", lw=1.0, alpha=0.85,
                   label="%s = %.4g%s" % (lbl, fq, _fu))
    # Black diamonds at the detected peaks (f_lib, f_low, f_lib -/+ f_low), at
    # the total amplitude *on the plotted curve* (interpolated at each exact
    # frequency) so every marker sits exactly on the total-amplitude line.
    _pk = []
    if _flib is not None and np.isfinite(_flib):
        _pk.append(_flib)
    if _flow is not None and np.isfinite(_flow):
        _pk.append(_flow)
        if _flib is not None and np.isfinite(_flib):
            _pk += [_flib - _flow, _flib + _flow]
    _dlbl = False
    for _pf in _pk:
        if not (0.0 <= _pf <= _xmax):
            continue
        _pa = float(np.interp(_pf, ff, at))
        if np.isfinite(_pa):
            ax.plot(_pf, _pa, "D", color="k", ms=6, zorder=5,
                    label=None if _dlbl else "detected peaks")
            _dlbl = True
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_polarization_v2_figure(out_png, f, pol, frot, name, normalize=False):
    """Polarization figure next to the .npz (log y-axis).

    Overlays the inertial-wave relation 2*((2*frot/f)**2 - 1) with the measured
    powerV/powerU ratio over the ROI, shown both as the ROI mean and as the
    median with a shaded 25-75 percentile band. The ratio is dimensionless, so
    normalize=True only rescales the frequency axis to f/f_rot (0.01..1 Hz).
    """
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    ff = f / fscale
    m = f > 0
    fig, ax = plt.subplots(figsize=(9, 6))
    if frot:
        theory = 2.0 * ((2.0 * frot / f[m]) ** 2 - 1.0)
        ax.plot(ff[m], theory, "k-", lw=2.2,
                label=r"$2[(2f_{\mathrm{rot}}/f)^2-1]$  (IW)")
    # powerV/powerU over the ROI: both the plain ROI mean and the median with
    # a shaded 25-75 percentile band (skew-robust, always positive -> sensible
    # on a log axis).
    ax.plot(ff[m], pol["powerRatio"][m], lw=1.2, label=r"mean $P_V/P_U$")
    line, = ax.plot(ff[m], pol["powerRatio_med"][m], lw=1.2,
                    label=r"median $P_V/P_U$")
    ax.fill_between(ff[m], pol["powerRatio_p25"][m], pol["powerRatio_p75"][m],
                    color=line.get_color(), alpha=0.20, lw=0,
                    label="ROI 25-75%")
    ax.set_yscale("log")
    ax.set_xlim(0.01 / fscale, 1.0 / fscale)
    ax.set_xlabel(r"$f / f_{\mathrm{rot}}$" if fscale != 1.0 else "frequency (Hz)",
                  fontsize=12)
    ax.set_ylabel(r"polarization  $P_V/P_U$", fontsize=12)
    ax.set_title("Polarization - %s%s   ($k_0 = %g$, top=%s, bottom=%s)"
                 % (name, "   (normalized)" if normalize else "", k0,
                    TOP_TOPO, BOTTOM_TOPO),
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def update_summary(csv_path, row):
    """Insert/replace this run's row in the VelocityFFT summary CSV.

    Drops any existing (run, region) row, appends the new one, and re-derives the
    'kept' flag (one run per (frot, flib, dphi) group -- the highest SS index),
    exactly as batch_Velocity does."""
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        for _c in ("processed", "kept", "calibrated"):
            if _c in df.columns and df[_c].dtype == object:
                df[_c] = df[_c].astype(str).str.strip().str.lower().isin(
                    ("true", "1", "1.0", "yes"))
    else:
        df = pd.DataFrame()
        print("  (no existing summary -> creating it)")
    if not df.empty and {"run", "region"}.issubset(df.columns):
        df = df[~((df["run"] == row["run"]) & (df["region"] == row["region"]))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df["kept"] = False
    _proc = df[df["processed"] == True].copy()
    if not _proc.empty:
        _proc["_ss"] = _proc["run idx"].fillna(1)
        for _, _g in _proc.groupby(["frot_Hz", "flib_Hz", "dphi_deg"],
                                   dropna=False):
            df.loc[_g["_ss"].idxmax(), "kept"] = True
    df.to_csv(csv_path, index=False)
    return df


# ## 4. Reprocess the run (from the .mat)


# batch_Velocity's per-run computation, for this one run -- always from the .mat.
name = os.path.basename(PATH.rstrip("/"))
piv_file = os.path.join(PATH, PIV_FILENAME)
out_dir = os.path.join(PATH, "PostProcessing")
out_file = os.path.join(out_dir, "%s_%s%s.npz" % (RESULT_STEM, tag, BP_TAG))
if not os.path.isfile(piv_file):
    raise FileNotFoundError("no %s in %s" % (PIV_FILENAME, PATH))

dt_vel, fps, ok = read_acquisition_params(os.path.join(PATH, LOG_FILENAME))
if ok:
    xscale, yscale = XSCALE, YSCALE
else:
    print("[warn] Calibration not possible - all velocities will be in px/frame")
    xscale = yscale = UNCAL_SCALE

X, Y, U, V, nframes = load_piv(piv_file)
X = xscale * X
Y = yscale * Y
U = xscale * U / dt_vel
V = yscale * V / dt_vel
Xr, Yr, Ur, Vr = region_fields(X, Y, U, V, PTS_ROI, REGION)

# ROI-averaged amplitude spectra of U, V (fps is the PIV field rate = cam_fps/2).
f, amp_u, amp_v, amp_total, powerU, powerV, npoints = averaged_fft(Ur, Vr, fps)
f_peak, _ = peak_freq(f, amp_total)

_frotN, _flibN, _dphiN = parse_run_name(name)
U0 = libration_velocity_scale(_flibN, _dphiN)
f_star = f / _frotN if np.isfinite(_frotN) and _frotN else np.full_like(f, np.nan)
_uscale = U0 if np.isfinite(U0) and U0 else np.nan
amp_u_star = amp_u / _uscale
amp_v_star = amp_v / _uscale
amp_total_star = amp_total / _uscale

# f_low: strongest peak in [F_LOW_FMIN, f_lib/2], kept only if significant.
_frot, _flib, _ = parse_run_name(name)
_frot = None if not np.isfinite(_frot) else _frot
_flib = None if not np.isfinite(_flib) else _flib
f_low, _a_low = peak_freq_in_band(f, amp_total, F_LOW_FMIN,
                                  _flib / 2.0 if _flib else np.nan)
if _flib and np.isfinite(f_low):
    _fb = (f >= F_LOW_FMIN) & (f <= _flib / 2.0)
    _fbmean = np.nanmean(amp_total[_fb]) if np.any(_fb) else np.nan
    if not (np.isfinite(_fbmean) and _a_low >= THRESHOLD_PEAK * _fbmean):
        f_low = np.nan

_peak_amp = {"flib": amp_at_freq(f, amp_total, _flibN),
             "flow": amp_at_freq(f, amp_total, f_low),
             "flib_minus_flow": amp_at_freq(f, amp_total, _flibN - f_low),
             "flib_plus_flow": amp_at_freq(f, amp_total, _flibN + f_low)}
_peak_amp_star = {_k: _a / _uscale for _k, _a in _peak_amp.items()}

# Polarization: IW prediction + measured power ratio, and ROI statistics.
f_pol, pol_IW, pol_data = compute_polarization(f, powerU, powerV,
                                               parse_run_name(name)[0])
pol = roi_polarization_stats(Ur, Vr, fps)

# The .npz is always (re)written -- reprocessing from the .mat is the point.
os.makedirs(out_dir, exist_ok=True)
if OVERWRITE_NPZ or not os.path.isfile(out_file):
    np.savez(out_file,
             run=name, PIV_file=piv_file, region=tag, calibrated=ok,
             dt_vel=dt_vel, fps=fps, xscale=xscale, yscale=yscale,
             pts_ROI=np.array(PTS_ROI), nframes=nframes, npoints=npoints,
             window=FFT_WINDOW, detrend=str(FFT_DETREND),
             f=f, amp_u=amp_u, amp_v=amp_v, amp_total=amp_total,
             f_star=f_star, amp_u_star=amp_u_star, amp_v_star=amp_v_star,
             amp_total_star=amp_total_star, U0=U0,
             amp_flib=_peak_amp["flib"], amp_flow=_peak_amp["flow"],
             amp_flib_minus_flow=_peak_amp["flib_minus_flow"],
             amp_flib_plus_flow=_peak_amp["flib_plus_flow"],
             amp_flib_star=_peak_amp_star["flib"],
             amp_flow_star=_peak_amp_star["flow"],
             amp_flib_minus_flow_star=_peak_amp_star["flib_minus_flow"],
             amp_flib_plus_flow_star=_peak_amp_star["flib_plus_flow"],
             powerU=powerU, powerV=powerV, f_peak=f_peak, f_low=f_low,
             f_pol=f_pol, pol_IW=pol_IW, polarization=pol_data,
             powerU_mean=pol["powerU_mean"], powerV_mean=pol["powerV_mean"],
             powerRatio=pol["powerRatio"], powerRatio_std=pol["powerRatio_std"],
             powerRatio_med=pol["powerRatio_med"],
             powerRatio_p25=pol["powerRatio_p25"],
             powerRatio_p75=pol["powerRatio_p75"])
    print("wrote", out_file)
else:
    print("[skip] npz exists (OVERWRITE_NPZ=False):", os.path.basename(out_file))

# Spectrum + polarization figures (raw and normalized), guarded by OVERWRITE_FIG.
fig_stem = os.path.splitext(out_file)[0]
pol_stem = os.path.join(out_dir, "%s_%s%s" % (POLFIG_STEM, tag, BP_TAG))
_dphi = parse_run_name(name)[2]
fig_file = figure_filename(fig_stem, FIG_FORMAT, normalized=False)
for _norm in (False, True):
    _figp = figure_filename(fig_stem, FIG_FORMAT, normalized=_norm)
    _polp = figure_filename(pol_stem, FIG_FORMAT, normalized=_norm)
    if OVERWRITE_FIG or not os.path.isfile(_figp):
        _save_fft_figure(_figp, f, amp_u, amp_v, amp_total, name, ok, f_peak,
                         _frot, _flib, f_low, _dphi, normalize=_norm)
        print("wrote", _figp)
    else:
        print("[skip] exists (OVERWRITE_FIG=False):", os.path.basename(_figp))
    if OVERWRITE_FIG or not os.path.isfile(_polp):
        _save_polarization_v2_figure(_polp, f, pol, parse_run_name(name)[0],
                                     name, normalize=_norm)
        print("wrote", _polp)
    else:
        print("[skip] exists (OVERWRITE_FIG=False):", os.path.basename(_polp))

# Summary row (same schema as batch_Velocity).
row = build_row(name, out_file, True, region=tag, fps=fps, nframes=nframes,
                npoints=npoints, f_peak=f_peak, f_low=f_low, dt_vel=dt_vel,
                ok=ok, peak_amp=_peak_amp, peak_amp_star=_peak_amp_star)
print("  [ok] %s  %s  npts=%d  f_peak=%.4gHz  f_low=%s"
      % (name, tag, npoints, f_peak,
         ("%.4gHz" % f_low) if np.isfinite(f_low) else "nan"))


# ## 5. Summary


# Add/refresh this run's row in the dataset's VelocityFFT summary. The row is
# identical to the one batch_Velocity would write for this run.
if UPDATE_SUMMARY:
    base_dir = os.path.dirname(PATH.rstrip("/"))     # dataset root, where the summary lives
    summary_csv = os.path.join(base_dir, "%s_%s%s.csv" % (SUMMARY_STEM, tag, BP_TAG))
    update_summary(summary_csv, row)
    print("updated %s summary:\n   %s" % (tag, summary_csv))
    print()
    for k, v in row.items():
        print("   %-16s %s" % (k, v))
else:
    print("UPDATE_SUMMARY is False -- summary table left untouched.")


# ## 6. Wavelet analysis (time-frequency)
# 
# Continuous wavelet transform of the region-mean velocity, producing a scalogram (time on x, frequency on y, amplitude in colour). Uses the same `REGION` as the FFT. Writes a raw and a normalized figure plus a dedicated `VelocityWavelet_<region>.npz`; **not** added to the summary.


def _interp_nan_1d(x):
    """Linearly interpolate NaNs in a 1-D time series (endpoints held).
    All-NaN (or <2 finite) input is returned as zeros."""
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.nan_to_num(x, nan=0.0)
    idx = np.arange(x.size)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def wavelet_analysis(Ur, Vr, fps, wavelet=None, fmin=None, fmax=None,
                     nfreq=None, spacing=None, method=None, max_points=None):
    """Per-point continuous wavelet transform, averaged over the region.

    For EVERY grid point of the (already region-cropped) field, the CWT
    amplitude |W U_pt| + |W V_pt| is computed, and the result is averaged over
    all valid points. This is the time-frequency analog of averaged_fft, which
    averages the per-point FFT amplitude spectra. NaNs are interpolated in time
    per point; fully masked points (< 2 finite samples) are skipped. Points are
    processed in memory-bounded chunks using the FFT-based CWT.

    Keyword defaults fall back to the WAVELET_* configuration constants.
    Returns (t, freqs, amp, u_bar, v_bar, npts): t [s] (ntime), freqs [Hz]
    ascending (nfreq), amp (nfreq, ntime) mean over points, u_bar/v_bar the
    region-mean signals [m/s] (reference only), npts = points averaged.
    """
    wavelet = wavelet or WAVELET
    fmin = WAVELET_FMIN if fmin is None else fmin
    fmax = WAVELET_FMAX if fmax is None else fmax
    nfreq = WAVELET_NFREQ if nfreq is None else nfreq
    spacing = WAVELET_SPACING if spacing is None else spacing
    method = WAVELET_METHOD if method is None else method
    max_points = WAVELET_MAX_POINTS if max_points is None else max_points

    ntime = Ur.shape[-1]
    U2 = Ur.reshape(-1, ntime)
    V2 = Vr.reshape(-1, ntime)

    fhi = fmax if fmax else 0.5 * fps                 # None -> Nyquist
    flo = max(fmin, fps / ntime)                      # can't resolve < ~1 cycle
    if spacing == "linear":
        freqs_req = np.linspace(flo, fhi, nfreq)
    else:
        freqs_req = np.logspace(np.log10(flo), np.log10(fhi), nfreq)
    scales = pywt.frequency2scale(wavelet, freqs_req / fps)

    # Valid points: >= 2 finite samples in BOTH U and V (else can't interpolate).
    valid = (np.isfinite(U2).sum(1) >= 2) & (np.isfinite(V2).sum(1) >= 2)
    idx = np.where(valid)[0]
    t = np.arange(ntime) / fps
    u_bar = _interp_nan_1d(np.nanmean(Ur, axis=(0, 1)))
    v_bar = _interp_nan_1d(np.nanmean(Vr, axis=(0, 1)))
    if idx.size == 0:
        freqs = pywt.scale2frequency(wavelet, scales) * fps
        order = np.argsort(freqs)
        print("  [wavelet] no valid points -- empty map")
        return t, freqs[order], np.zeros((nfreq, ntime)), u_bar, v_bar, 0

    if max_points and idx.size > max_points:
        keep = np.unique(np.linspace(0, idx.size - 1, int(max_points))
                         .round().astype(int))
        idx = idx[keep]
        print("  [wavelet] averaging %d of %d valid points (WAVELET_MAX_POINTS)"
              % (idx.size, int(valid.sum())))
    else:
        print("  [wavelet] averaging over all %d valid points" % idx.size)

    # Memory-bounded chunking (~256 MB per complex128 CWT block).
    chunk = max(1, int(256e6 / (nfreq * ntime * 16)))
    acc = np.zeros((nfreq, ntime))
    freqs = None
    done = 0
    for c0 in range(0, idx.size, chunk):
        sel = idx[c0:c0 + chunk]
        uu = np.vstack([_interp_nan_1d(U2[i]) for i in sel])
        cu, freqs = pywt.cwt(uu, scales, wavelet, sampling_period=1.0 / fps,
                             method=method)
        acc += np.abs(cu).sum(axis=1)
        del cu, uu
        vv = np.vstack([_interp_nan_1d(V2[i]) for i in sel])
        cv, _ = pywt.cwt(vv, scales, wavelet, sampling_period=1.0 / fps,
                         method=method)
        acc += np.abs(cv).sum(axis=1)
        del cv, vv
        done += sel.size
        print("  [wavelet] %d / %d points done" % (done, idx.size))

    amp = acc / done                                  # mean over points
    order = np.argsort(freqs)                          # ascending for plotting
    freqs = freqs[order]
    amp = amp[order, :]
    return t, freqs, amp, u_bar, v_bar, done


def _save_wavelet_figure(out_png, t, freqs, amp, name, fps, flib, frot, U0,
                         normalize=False):
    """Wavelet scalogram: time on x, frequency on y, amplitude in colour.

    normalize=False -> dimensional (t [s], f [Hz], amp [m/s]).
    normalize=True  -> non-dimensional (t*f_rot, f/f_rot, amp/U0).
    Dashed white lines mark f_lib and 2 f_lib.
    """
    from matplotlib.colors import LogNorm
    _gf = bool(np.isfinite(frot) and frot)
    _gu = bool(np.isfinite(U0) and U0)
    if normalize:
        x = t * frot if _gf else t
        y = freqs / frot if _gf else freqs
        c = amp / U0 if _gu else amp
        xlabel = (r"$t\,f_{\mathrm{rot}}$ (rotations)" if _gf else "time (s)")
        ylabel = (r"$f / f_{\mathrm{rot}}$" if _gf else "frequency (Hz)")
        clabel = (r"$\langle|Wu|+|Wv|\rangle\,/\,U_0$" if _gu else "amplitude")
    else:
        x, y, c = t, freqs, amp
        xlabel, ylabel = "time (s)", "frequency (Hz)"
        clabel = r"$\langle|Wu|+|Wv|\rangle$  (m/s)"

    finite = c[np.isfinite(c)]
    if WAVELET_LOGC and finite.size and np.any(finite > 0):
        pos = finite[finite > 0]
        norm = LogNorm(vmin=np.nanpercentile(pos, 5),
                       vmax=np.nanpercentile(finite, 99.5))
        vkw = {}
    else:
        norm = None
        vkw = {"vmin": 0.0,
               "vmax": float(np.nanpercentile(finite, 99)) if finite.size else 1.0}

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    pcm = ax.pcolormesh(x, y, np.ma.masked_invalid(c), cmap=WAVELET_CMAP,
                        shading="nearest", norm=norm, **vkw)
    if WAVELET_LOGF:
        ax.set_yscale("log")
    for _fq, _lab in ((flib, r"$f_{\mathrm{lib}}$"),
                      (2.0 * flib if flib else np.nan, r"$2f_{\mathrm{lib}}$")):
        if _fq and np.isfinite(_fq):
            _yy = (_fq / frot if (normalize and _gf) else _fq)
            ax.axhline(_yy, color="w", ls="--", lw=1.0, alpha=0.8)
            ax.text(x[-1], _yy, " " + _lab, color="w", va="center",
                    ha="left", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label(clabel)
    fig.suptitle("%s%s   (%s, $k_0=%g$, top=%s, bottom=%s)%s"
                 % (name, "   (normalized)" if normalize else "", WAVELET, k0,
                    TOP_TOPO, BOTTOM_TOPO, ("   %s" % BP_TAG) if BP_TAG else ""),
                 fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


# Per-point wavelet analysis, averaged over the region (same REGION and BP_TAG
# as the FFT above). Writes a dedicated .npz plus a raw and a normalized
# scalogram figure. These results are NOT added to the summary.
if not PERFORM_WAVELET:
    print("[skip] wavelet analysis (PERFORM_WAVELET=False)")
elif pywt is None:
    print("[skip] wavelet analysis: PyWavelets is not installed "
          "(conda install pywavelets)")
else:
    wav_stem = os.path.join(out_dir, "%s_%s%s" % (WAVELET_STEM, tag, BP_TAG))
    wav_npz = wav_stem + ".npz"
    _wav_figs = [figure_filename(wav_stem, FIG_FORMAT, normalized=_n) for _n in (False, True)]

    _npz_ok = os.path.isfile(wav_npz) and not OVERWRITE_NPZ
    _figs_ok = all(os.path.isfile(f) for f in _wav_figs) and not OVERWRITE_FIG
    if _npz_ok and _figs_ok:
        print("[skip] wavelet outputs exist (OVERWRITE_NPZ/FIG=False):",
              os.path.basename(wav_npz))
    else:
        _wt, _wf, _wamp, _u_bar, _v_bar, _wnpts = wavelet_analysis(Ur, Vr, fps)
        _gf = bool(np.isfinite(_frotN) and _frotN)
        _gu = bool(np.isfinite(U0) and U0)
        if OVERWRITE_NPZ or not os.path.isfile(wav_npz):
            np.savez(wav_npz,
                     run=name, region=tag, wavelet=WAVELET, fps=fps, bp_tag=BP_TAG,
                     flib=_flibN, frot=_frotN, dphi=_dphiN, U0=U0, npoints=_wnpts,
                     t=_wt, freq=_wf, amp=_wamp,
                     t_star=(_wt * _frotN if _gf else np.full_like(_wt, np.nan)),
                     f_star=(_wf / _frotN if _gf else np.full_like(_wf, np.nan)),
                     amp_star=(_wamp / U0 if _gu else np.full_like(_wamp, np.nan)),
                     u_bar=_u_bar, v_bar=_v_bar)
            print("wrote", wav_npz)
        else:
            print("[skip] wavelet npz exists (OVERWRITE_NPZ=False):",
                  os.path.basename(wav_npz))
        for _norm, _wp in zip((False, True), _wav_figs):
            if OVERWRITE_FIG or not os.path.isfile(_wp):
                _save_wavelet_figure(_wp, _wt, _wf, _wamp, name, fps, _flibN, _frotN,
                                     U0, normalize=_norm)
                print("wrote", _wp)
            else:
                print("[skip] exists (OVERWRITE_FIG=False):", os.path.basename(_wp))
        print("  [ok] wavelet: %d points averaged, amp map %s" % (_wnpts, _wamp.shape))
