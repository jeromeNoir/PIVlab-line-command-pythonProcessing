"""Auto-generated .py twin of batch_KineticEnergy.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Batch kinetic-energy post-processing (time series **and** FFT)
# 
# Notebook version of `batch_KineticEnergy.py`. Loops over every sub-folder of
# `BASE_DIR`, loads the PIV `.mat` **once**, calibrates the field, and — for each
# region (`ROI` / `FULL`) — computes in a single pass:
# 
# - the kinetic-energy **time series** `Ek(t) = 0.5*<U^2 + V^2>` (region average),
#   with `<Ek>` and `std(Ek)`;
# - **two FFT spectra** of the kinetic energy, differing only in when the region
#   average is taken:
# 
#   | curve | meaning | order |
#   |---|---|---|
#   | `FFT(<Ek>)` | FFT of the region-averaged series `<Ek>(t)` | **average, then FFT** |
#   | `<FFT(Ek)>` | FFT at each grid point, amplitudes then averaged over the region | **FFT, then average** |
# 
#   Averaging first cancels fluctuations that are incoherent across the region, so
#   `FFT(<Ek>)` is normally the smaller of the two; the gap measures how spatially
#   coherent the signal is at each frequency. **Both are drawn** in the per-run
#   figure, but **only `<FFT(Ek)>` is written to the summary** (peak, and the
#   amplitudes at `f_lib` and `2*f_lib`).
# 
# Per run it writes a single combined `<run>/PostProcessing/KineticEnergy_<region>.npz`
# (time series + both spectra) and a two-panel figure `KineticEnergy_FFT_<region>.png`.
# Across all runs it writes one `KineticEnergy_summary_<region>.csv` at `BASE_DIR`
# (time-series stats **and** the `<FFT(Ek)>` quantities) plus a resonance figure
# `KineticEnergy_vs_fstar_<region>.png`.
# 
# Calibration is read from the last row of `acquisition_log.txt`
# (`dt_vel = pulse_sep`, `fps = cam_fps/2`, PIVlab pairs images). Set the config in
# the next cell, then *Run All*.


import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, amp_at_freq,
                        amp_from_rfft, compute_fft, dimensionless_numbers,
                        fft_axis_limits, figure_filename, libration_ke_scale,
                        libration_velocity_scale, load_piv, parse_run_name,
                        peak_freq, read_acquisition_params, read_paramPostprocessing,
                        region_fields, region_tag, regions_to_run)


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle. (Figures are unaffected.)
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = True
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# --- Configuration --------------------------------------------------------- #
BASE_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly")  # top-level folder containing the runs

# Calibration, ROI, filenames, ... for this dataset: read from
# param_postProcessing.json in BASE_DIR (also rebinds them inside
# piv_postprocessing_lib so its helpers use this dataset's values).
_P = read_paramPostprocessing(BASE_DIR)
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(BASE_DIR)
PIV_FILENAME, LOG_FILENAME = _P.PIV_FILENAME, _P.LOG_FILENAME
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
UNCAL_SCALE = _P.UNCAL_SCALE

# Per-run output + summary stems. The region tag ('ROI'/'FULL') is appended, so
# each run gets one combined KineticEnergy_<region>.npz (time series + both FFT
# spectra) and the batch writes one KineticEnergy_summary_<region>.csv.
RESULT_STEM   = "KineticEnergy"            # per-run npz -> <stem>_<region>.npz
SUMMARY_STEM  = "KineticEnergy_summary"    # summary     -> <stem>_<region>.csv
FFTFIG_STEM   = "KineticEnergy_FFT"        # per-run 2-panel figure beside the npz
RESONFIG_STEM = "KineticEnergy_vs_fstar"   # dataset resonance figure at BASE_DIR

# Which region(s) to process: 'BOTH' (ROI and FULL), 'ROI' only, or 'FULL' only.
REGION = 'ROI'

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'

# Only runs without an existing result are processed unless this is True.
REPROCESS_ALL = True

# --- FFT settings ---------------------------------------------------------- #
# Two spectra are always computed and drawn (the per-point field is already in
# hand, since the .mat is read once for the time series, so both are free):
#   FFT(<Ek>) = FFT of the region-averaged series  (average, then FFT)
#   <FFT(Ek)> = per-point FFT amplitudes averaged   (FFT, then average)
# Only <FFT(Ek)> goes into the summary (f_peak, amp_peak, amp_flib, amp_2flib).
DETREND = True   # remove the mean before the FFT (kills the DC spike)
LOGY = True      # log amplitude axis on the spectrum panel (False -> linear)
LOGX = False     # log frequency axis (False -> linear)

# Restrict which runs appear in the resonance FIGURE (the summary CSV always
# keeps every run). Set a value, list or (lo, hi) range; None -> keep all. A
# filtered figure is shown but NOT saved, so it never overwrites the full one.
SELECT_FROT = None    # Hz,  e.g. 0.5
SELECT_FLIB = None    # Hz,  e.g. 0.44  or  [0.40, 0.44]  or  (0.4, 0.5) range
SELECT_DPHI = None    # deg, e.g. 2.0


def compute_fft_per_point(ek_pt, fps, detrend=DETREND):
    """<FFT(Ek)>: FFT at each grid point, THEN average the amplitudes.

    ek_pt has shape (npoints, nframes) -- the kinetic-energy time series at every
    grid point of the region. Each point is transformed on its own with the same
    normalisation as compute_fft, and |FFT| is then averaged over the points.
    Points that are entirely NaN are dropped; remaining NaNs are filled with that
    point's own mean. Returns (freq, amp_mean, npoints_used).
    """
    A = np.asarray(ek_pt, dtype=float)
    n = A.shape[-1]
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    keep = ~np.all(np.isnan(A), axis=1)
    if not np.any(keep):
        return freq, np.full_like(freq, np.nan), 0
    A = A[keep]
    A = np.where(np.isnan(A), np.nanmean(A, axis=1, keepdims=True), A)
    if detrend:
        A = A - A.mean(axis=1, keepdims=True)
    amp = amp_from_rfft(np.fft.rfft(A, axis=-1), n)
    return freq, np.nanmean(amp, axis=0), int(keep.sum())


# Labels say WHERE the region average sits relative to the FFT.
LBL_AVG_FIRST = r"$|\mathrm{FFT}(\langle E_k \rangle)|$   average, then FFT"
LBL_FFT_FIRST = r"$\langle |\mathrm{FFT}(E_k)| \rangle$   FFT, then average"


def save_fft_figure(out_png, t, ek, freq, amp_avg, amp_mean, run, flib,
                    frot=None, dphi_deg=None, normalize=False):
    """Two-panel per-run figure: time series + both amplitude spectra.

    Panel 2 overlays FFT(<Ek>) (amp_avg) and <FFT(Ek)> (amp_mean). normalize=True
    scales Ek by the libration KE scale U0**2 and the frequency axis by f_rot.
    Closed straight away so a batch does not leave many windows open.
    """
    escale = (libration_ke_scale(dphi_deg, flib) if normalize and flib
              and dphi_deg is not None and np.isfinite(dphi_deg) else 1.0)
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    ek = ek / escale
    amp_avg = amp_avg / escale
    amp_mean = None if amp_mean is None else amp_mean / escale
    freq = freq / fscale
    _flib = flib / fscale if flib else flib
    _frot = frot / fscale if frot else frot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: the region-averaged time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel((r"$\langle E_k \rangle(t) / E_{\mathrm{lib}}$" if normalize
                        else r"$\langle E_k \rangle(t)$  (m$^2$/s$^2$)"), fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: both amplitude spectra
    plot = axes[1].semilogy if LOGY else axes[1].plot
    if amp_mean is not None:
        plot(freq, amp_mean, color="C2", lw=1, label=LBL_FFT_FIRST)
    plot(freq, amp_avg, color="C1", lw=1, label=LBL_AVG_FIRST)
    axes[1].set_xlabel(r"$f / f_{\mathrm{rot}}$" if normalize else "frequency (Hz)",
                       fontsize=13)
    axes[1].set_ylabel((r"$|\widehat{E_k}| / E_{\mathrm{lib}}$" if normalize
                        else r"$|\widehat{E_k}|$  (m$^2$/s$^2$)"), fontsize=13)
    axes[1].set_title("FFT amplitude spectrum" + ("  (mean removed)" if DETREND else ""),
                      fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=9)
    # Frequency axis to [0, max(4*frot,4*flib)]; y to decade bounds.
    curves = [amp_avg] if amp_mean is None else [amp_avg, amp_mean]
    _xmax, _ymin, _ymax = fft_axis_limits(freq, curves, _frot, _flib)
    if LOGX:
        axes[1].set_xscale("log")
        _pos = freq[(freq > 0) & (freq <= _xmax)]
        axes[1].set_xlim(_pos.min() if _pos.size else _xmax / 100.0, _xmax)
    else:
        axes[1].set_xlim(0, _xmax)
    if _ymin and _ymax:
        axes[1].set_ylim(_ymin, _ymax)
    if _flib:
        for fq, lbl in ((_flib, r"$f_{\mathrm{lib}}$"),
                        (2 * _flib, r"$2f_{\mathrm{lib}}$")):
            if fq <= freq.max():
                axes[1].axvline(fq, color="k", ls="--", lw=1, alpha=0.6)
                axes[1].annotate(lbl, xy=(fq, 1),
                                 xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points",
                                 fontsize=10)

    fig.suptitle(run + ("   (normalized)" if normalize else "")
                 + "   ($k_0 = %g$, top=%s, bottom=%s)" % (k0, TOP_TOPO, BOTTOM_TOPO),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_row(name, npz_file, fig_file, processed, region=np.nan,
              mean_Ekin=np.nan, std_Ekin=np.nan, dt_vel=np.nan, fps=np.nan,
              nframes=np.nan, npoints=np.nan, ok=np.nan,
              f_peak=np.nan, amp_peak=np.nan, amp_flib=np.nan, amp_2flib=np.nan):
    """One combined summary row: folder metadata, time-series stats and the
    <FFT(Ek)> quantities (peak + amplitudes at f_lib and 2*f_lib)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    # Session index: the integer following 'SS' in the folder name (SS2 -> 2).
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    # Dimensionless numbers + normalised (_star): frequencies / frot, Ek and its
    # spectral amplitudes / U0**2 (the FFT is of the kinetic energy).
    dl = dimensionless_numbers(frot_hz, flib_hz, dphi_deg)
    fn = frot_hz if np.isfinite(frot_hz) and frot_hz else np.nan
    ke_scale = dl["U0_mps"] ** 2 if np.isfinite(dl["U0_mps"]) and dl["U0_mps"] else np.nan
    row = {"run": name, "run idx": run_idx, "region": region,
           "processed": processed, "frot_Hz": frot_hz, "flib_Hz": flib_hz,
           "dphi_deg": dphi_deg, "fstar": fstar, "flib_star": fstar,
           "calibrated": ok, "dt_vel_s": dt_vel, "fps_Hz": fps,
           "nframes": nframes, "npoints": npoints,
           "mean_Ekin": mean_Ekin, "std_Ekin": std_Ekin,
           "mean_Ekin_star": mean_Ekin / ke_scale,
           "std_Ekin_star": std_Ekin / ke_scale,
           "f_peak_Hz": f_peak, "f_peak_star": f_peak / fn,
           "amp_peak": amp_peak, "amp_peak_star": amp_peak / ke_scale,
           "amp_flib": amp_flib, "amp_flib_star": amp_flib / ke_scale,
           "amp_2flib": amp_2flib, "amp_2flib_star": amp_2flib / ke_scale,
           "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO,
           "npz": npz_file, "figure": fig_file}
    row.update(dl)
    return row


def process_run(run_dir, region, reprocess=False):
    """Process one run folder for one region ('ROI'/'full'). Returns a summary dict.

    Reads the PIV .mat once, computes the Ek time series and BOTH FFT spectra,
    writes <run>/PostProcessing/KineticEnergy_<region>.npz (time series + spectra)
    and the two-panel figure. If the .npz already exists and reprocess is False,
    the row is rebuilt from that cache (the large .mat is not re-read). A folder
    with no PIV file yields processed=False and NaN metrics.
    """
    tag = region_tag(region)
    name = os.path.basename(run_dir.rstrip("/"))
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    piv_file = os.path.join(run_dir, PIV_FILENAME)
    out_dir = os.path.join(run_dir, "PostProcessing")
    out_file = os.path.join(out_dir, "%s_%s.npz" % (RESULT_STEM, tag))
    fig_stem = os.path.join(out_dir, "%s_%s" % (FFTFIG_STEM, tag))
    fig_file = figure_filename(fig_stem, FIG_FORMAT, normalized=False)

    # Reuse an existing result unless a reprocess was requested.
    if os.path.isfile(out_file) and not reprocess:
        try:
            d = np.load(out_file, allow_pickle=True)
            print("  [skip] %-40s %-4s already processed (cached)" % (name, tag))
            _g = lambda k: (float(d[k]) if k in d.files else np.nan)
            return build_row(name, out_file, fig_file, True, region=tag,
                             mean_Ekin=_g("mean_Ekin"), std_Ekin=_g("std_Ekin"),
                             dt_vel=_g("dt_vel"), fps=_g("fps"),
                             nframes=int(d["nframes"]) if "nframes" in d.files else np.nan,
                             npoints=int(d["npoints"]) if "npoints" in d.files else np.nan,
                             ok=bool(d["calibrated"]) if "calibrated" in d.files else np.nan,
                             f_peak=_g("f_peak"), amp_peak=_g("amp_peak"),
                             amp_flib=_g("amp_flib"), amp_2flib=_g("amp_2flib"))
        except Exception as exc:
            print("  [warn] %s: cached result unreadable (%s) -> reprocessing"
                  % (name, exc))

    if not os.path.isfile(piv_file):
        print("  [skip] no %s in %s" % (PIV_FILENAME, name))
        return build_row(name, "", "", False, region=tag)

    dt_vel, fps, ok = read_acquisition_params(os.path.join(run_dir, LOG_FILENAME))
    if ok:
        xscale, yscale = XSCALE, YSCALE
    else:
        print("  [warn] %s: Calibration not possible - all velocities will be "
              "in px/frame" % name)
        xscale = yscale = UNCAL_SCALE   # dt_vel, fps already fell back to 1

    X, Y, U, V, nframes = load_piv(piv_file)

    # Calibrate: positions -> m (or px), velocities -> m/s (or px/frame).
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel

    # Restrict to the region ('ROI' crops to PTS_ROI, 'full' keeps everything).
    Xr, Yr, Ur, Vr = region_fields(X, Y, U, V, PTS_ROI, region)

    # Per-point kinetic energy Ek(t) over the region, then the region average.
    Ek_pt = 0.5 * (Ur ** 2 + Vr ** 2).reshape(-1, nframes)
    npoints = Ek_pt.shape[0]
    Ek_frame = np.nanmean(Ek_pt, axis=0)
    t = np.arange(nframes) / fps
    mean_Ekin = float(np.nanmean(Ek_frame))
    std_Ekin = float(np.nanstd(Ek_frame))

    # Normalised (_star) Ek: Ek / U0**2, U0 the libration wall velocity scale.
    U0 = libration_velocity_scale(flib_hz, dphi_deg)
    ke_scale = U0 ** 2 if np.isfinite(U0) and U0 else np.nan
    Ek_frame_star = Ek_frame / ke_scale
    mean_Ekin_star = mean_Ekin / ke_scale
    std_Ekin_star = std_Ekin / ke_scale

    # (1) FFT(<Ek>): average, then FFT.
    freq, amp_avg, _ = compute_fft(Ek_frame, fps, detrend=DETREND)
    # (2) <FFT(Ek)>: FFT at each point, then average -- kept in the summary.
    freq, amp_mean, _npts = compute_fft_per_point(Ek_pt, fps, detrend=DETREND)
    f_peak, amp_peak = peak_freq(freq, amp_mean)
    amp_flib = amp_at_freq(freq, amp_mean, flib_hz)
    amp_2flib = amp_at_freq(freq, amp_mean, 2.0 * flib_hz)

    # One combined npz: time series + both spectra + kept scalars.
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_file,
             run=name, PIV_file=piv_file, region=tag, calibrated=ok,
             dt_vel=dt_vel, fps=fps, xscale=xscale, yscale=yscale,
             pts_ROI=np.array(PTS_ROI), npoints=npoints, nframes=nframes,
             t=t, Ek_frame=Ek_frame, Ek_frame_star=Ek_frame_star,
             mean_Ekin=mean_Ekin, std_Ekin=std_Ekin,
             mean_Ekin_star=mean_Ekin_star, std_Ekin_star=std_Ekin_star, U0=U0,
             detrend=DETREND, freq=freq,
             amp_fft_of_mean=amp_avg, amp_mean_of_fft=amp_mean,
             f_peak=f_peak, amp_peak=amp_peak,
             amp_flib=amp_flib, amp_2flib=amp_2flib)

    # Two-panel figure (time series + both spectra), raw and normalized.
    _flib = None if not np.isfinite(flib_hz) else flib_hz
    _frot = None if not np.isfinite(frot_hz) else frot_hz
    for _norm, _out in ((False, fig_file),
                        (True, figure_filename(fig_stem, FIG_FORMAT, normalized=True))):
        save_fft_figure(_out, t, Ek_frame, freq, amp_avg, amp_mean,
                        "%s  (%s)" % (name, tag), _flib, _frot, dphi_deg,
                        normalize=_norm)

    print("  [ok] %-40s %-4s nframes=%d npts=%d fps=%.4gHz  <Ek>=%.4e  "
          "std=%.4e  f_peak=%.4gHz"
          % (name, tag, nframes, npoints, fps, mean_Ekin, std_Ekin, f_peak))

    return build_row(name, out_file, fig_file, True, region=tag,
                     mean_Ekin=mean_Ekin, std_Ekin=std_Ekin, dt_vel=dt_vel,
                     fps=fps, nframes=nframes, npoints=npoints, ok=ok,
                     f_peak=f_peak, amp_peak=amp_peak, amp_flib=amp_flib,
                     amp_2flib=amp_2flib)


def _match(series, sel, tol=1e-6):
    """Boolean mask selecting rows of `series`:
      None             -> all rows
      scalar           -> equal to that value (within tol)
      (lo, hi) tuple   -> range  lo <= value <= hi  (inclusive)
      [a, b, ...] list -> equal to any listed value
    """
    if sel is None:
        return pd.Series(True, index=series.index)
    if isinstance(sel, tuple) and len(sel) == 2:
        lo, hi = sel
        return (series >= lo - tol) & (series <= hi + tol)
    vals = sel if isinstance(sel, (list, set)) else [sel]
    mask = pd.Series(False, index=series.index)
    for v in vals:
        mask |= (series - v).abs() <= tol
    return mask


def plot_summary(df, base_dir, region="", normalize=True, fmt=FIG_FORMAT):
    """Two panels: mean(Ek) and std(Ek) versus f* = flib/frot, for one region.

    The main resonance sweep (most common dphi) is a connected curve; runs at
    other dphi are overlaid as squares; repeat acquisitions of an identical
    (frot, flib, dphi) point are shown as open diamonds. Saved to base_dir
    (tagged with the region) and displayed inline.
    """
    tag = region or (str(df["region"].iloc[0]) if "region" in df and len(df) else "")

    dfv = df.dropna(subset=["fstar", "mean_Ekin"]).copy()
    if dfv.empty:
        print("  (figure skipped: no processed runs with valid f*)")
        return None

    _no_filter = all(v is None for v in (SELECT_FROT, SELECT_FLIB, SELECT_DPHI))
    dfv = dfv[_match(dfv["frot_Hz"], SELECT_FROT)
              & _match(dfv["flib_Hz"], SELECT_FLIB)
              & _match(dfv["dphi_deg"], SELECT_DPHI)].copy()
    if dfv.empty:
        print("  (figure skipped: no runs match the SELECT_* filter)")
        return None

    dfv = dfv.sort_values(["frot_Hz", "flib_Hz", "dphi_deg", "run"])
    dfv["is_repeat"] = (dfv.groupby(["frot_Hz", "flib_Hz", "dphi_deg"])
                          .cumcount() > 0)
    dfv = dfv.sort_values("fstar")

    main_dphi = dfv.loc[~dfv["is_repeat"], "dphi_deg"].mode().iloc[0]
    is_main = (dfv["dphi_deg"] == main_dphi) & ~dfv["is_repeat"]

    if normalize:
        scale = libration_ke_scale(dfv["dphi_deg"], dfv["flib_Hz"])
        dfv["mean_Ekin"] = dfv["mean_Ekin"] / scale
        dfv["std_Ekin"] = dfv["std_Ekin"] / scale
        panels = [("mean_Ekin", r"$\langle E_k \rangle / E_{\mathrm{lib}}$",
                   "Mean kinetic energy (normalized)"),
                  ("std_Ekin", r"std $E_k$ / $E_{\mathrm{lib}}$",
                   "Std of kinetic energy (normalized)")]
    else:
        panels = [("mean_Ekin", r"$\langle E_k \rangle$  (m$^2$/s$^2$)",
                   "Mean kinetic energy"),
                  ("std_Ekin", r"std $E_k$  (m$^2$/s$^2$)",
                   "Std of kinetic energy")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, ylabel, title) in zip(axes, panels):
        main = dfv[is_main]
        ax.plot(main["fstar"], main[col], "-o", color="C0",
                label=r"$\delta\varphi=%g^\circ$ (sweep)" % main_dphi)
        for dphi, g in dfv[~is_main & ~dfv["is_repeat"]].groupby("dphi_deg"):
            ax.plot(g["fstar"], g[col], "s", ms=7,
                    label=r"$\delta\varphi=%g^\circ$" % dphi)
        rep = dfv[dfv["is_repeat"]]
        if not rep.empty:
            ax.plot(rep["fstar"], rep[col], "D", ms=8, mfc="none",
                    mec="k", mew=1.5, label="repeat")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f^* = f_{\mathrm{lib}} / f_{\mathrm{rot}}$", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    if tag:
        fig.suptitle("REGION: %s%s   ($k_0 = %g$, top=%s, bottom=%s)"
                     % (tag, "   (normalized)" if normalize else "", k0,
                        TOP_TOPO, BOTTOM_TOPO),
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    stem = ("%s_%s" % (RESONFIG_STEM, tag)) if tag else RESONFIG_STEM
    out = os.path.join(base_dir, figure_filename(stem, fmt, normalized=normalize))
    if _no_filter:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % out)
    else:
        print("  (filtered view -- shown but not saved)")
    return fig


def run_batch(reprocess_all=REPROCESS_ALL):
    """Process every run under BASE_DIR for each region; write one per-region
    summary + resonance figure. Returns {region: DataFrame}."""
    subdirs = sorted(d for d in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(d)
                     and not os.path.basename(d).startswith("."))
    print("Found %d subfolders in %s%s"
          % (len(subdirs), BASE_DIR,
             "  (reprocessing ALL)" if reprocess_all else
             "  (skipping already-processed)"))

    results = {}
    for region in regions_to_run(REGION):
        tag = region_tag(region)
        print("\n========== REGION: %s ==========" % tag)
        rows = []
        for run_dir in subdirs:
            try:
                row = process_run(run_dir, region, reprocess=reprocess_all)
            except Exception as exc:  # keep the batch going
                print("  [error] %s: %s" % (os.path.basename(run_dir), exc))
                row = None
            if row is not None:
                rows.append(row)

        if not rows:
            print("  No runs found for region %s." % tag)
            continue

        df = pd.DataFrame(rows).sort_values(["frot_Hz", "flib_Hz", "dphi_deg"])
        csv_path = os.path.join(BASE_DIR, "%s_%s.csv" % (SUMMARY_STEM, tag))
        df.to_csv(csv_path, index=False)
        n_proc = int(df["processed"].sum())
        print("\n  Summary %s (%d runs: %d processed, %d unprocessed):\n    %s"
              % (tag, len(df), n_proc, len(df) - n_proc, csv_path))
        # Both a raw and a normalized resonance figure for this region.
        for _norm in (False, True):
            plot_summary(df, BASE_DIR, tag, normalize=_norm)
        results[tag] = df

    return results


# ## Run
# 
# Runs the batch with `REPROCESS_ALL` from the config cell. Set it to `True`
# (or call `run_batch(reprocess_all=True)`) to reprocess every run.


# Returns {region: DataFrame}, one combined summary per processed region.
results = run_batch(reprocess_all=REPROCESS_ALL)
next(iter(results.values())).head(30) if results else results
