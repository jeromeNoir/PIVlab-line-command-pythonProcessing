"""Auto-generated .py twin of batch_KineticEnergyFFT.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Batch FFT of the kinetic-energy time series
# 
# Batch version of [`single_KineticEnergyFFT.ipynb`](single_KineticEnergyFFT.ipynb): instead of
# one run, it loops over **every sub-folder of `BASE_DIR`** and, **for both regions**
# (ROI and full field), writes a per-run figure next to that run's
# `KineticEnergy_timeSeries_<region>.npz` (→ `KineticEnergy_FFT_<region>.png`).
# 
# Run `batch_KineticEnergy` first so the region `.npz` files exist.
# 
# **Two spectra are computed, differing only in when the region average is taken:**
# 
# | curve | meaning | order |
# |---|---|---|
# | `fft(Ek_mean)` | FFT of the region-averaged series `<Ek>(t)` (the `Ek_frame` stored in the `.npz`) | **average, then FFT** |
# | `fft(Ek)_mean` | FFT of `Ek(t)` at **each grid point**, amplitudes then averaged over the region | **FFT, then average** |
# 
# They are not the same: averaging first cancels fluctuations that are incoherent
# across the region, so `fft(Ek_mean)` is normally the smaller of the two; the gap
# between them measures how spatially coherent the signal at each frequency is.
# 
# `fft(Ek)_mean` needs the per-point field, which the `.npz` does not store, so the
# run's PIV `.mat` is re-read and the region/calibration recorded in the `.npz` are
# re-applied. If the `.mat` is missing, only `fft(Ek_mean)` is drawn.
# 
# Both use the same normalisation as `single_KineticEnergyFFT`: the one-sided amplitude
# `|FFT| * 2 / N` (DC and Nyquist keep `1/N`), with the mean removed first by
# default.
# 
# The figure has two panels: (1) the `<Ek>(t)` time series; (2) both amplitude
# spectra, with dashed guides at `flib` and `2*flib`. Across all runs it also writes
# `KineticEnergyFFT_summary_ROI.csv` / `.xlsx` **and** `..._FULL.csv` / `.xlsx` at
# `BASE_DIR`.
# 
# Set the config in the next cell, then *Run All*. `ONLY_RUNS` restricts to a few
# folders for testing.


import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, amp_at_freq,
                        amp_from_rfft, compute_fft,
                        dimensionless_numbers, fft_axis_limits, figure_filename,
                        libration_ke_scale, load_piv,
                        parse_run_name, peak_freq, read_paramPostprocessing,
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
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly")   # top-level folder containing all runs

# Dataset parameters from param_postProcessing.json in BASE_DIR (also rebinds
# them inside piv_postprocessing_lib for its helpers).
_P = read_paramPostprocessing(BASE_DIR)
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(BASE_DIR)
PIV_FILENAME = _P.PIV_FILENAME

# Per-run input/output and summary stems. The region tag ('ROI'/'full') is
# appended, so the batch reads KineticEnergy_timeSeries_<region>.npz, writes
# KineticEnergy_FFT_<region>.png beside it, and KineticEnergyFFT_summary_<region>.
NPZ_STEM     = "KineticEnergy_timeSeries"   # per-run input (in PostProcessing/)
FIG_STEM     = "KineticEnergy_FFT"          # per-run figure, written beside it
SUMMARY_STEM = "KineticEnergyFFT_summary"   # summary table at BASE_DIR

# Which region(s) to process: 'BOTH' (ROI and FULL), 'ROI' only, or 'FULL' only.
REGION = 'ROI'

# Runs whose figure already exists are skipped unless this is True.
REPROCESS_ALL = True

# Restrict processing to these run-folder names (for testing). Empty -> all runs.
ONLY_RUNS = []

DETREND = True        # remove the mean before the FFT (kills the DC spike)
LOGY = True           # logarithmic amplitude axis (False -> linear)
LOGX = False          # logarithmic frequency axis (False -> linear)

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'

# Compute the per-point spectrum fft(Ek)_mean (re-reads the PIV .mat, so it is
# the slow part). False -> only fft(Ek_mean), straight from the .npz.
PER_POINT = True

# Folder-name -> physical value conversions (Hz). Folder tokens are integers,
# e.g. 'frot050' -> 0.50 Hz, 'flib0400' -> 0.400 Hz, 'flib1500' -> 1.500 Hz.


def find_npz(run_dir, region):
    """The run's KineticEnergy_timeSeries_<region>.npz, or None if it has none."""
    fname = "%s_%s.npz" % (NPZ_STEM, region_tag(region))
    for cand in (os.path.join(run_dir, "PostProcessing", fname),
                 os.path.join(run_dir, fname)):
        if os.path.isfile(cand):
            return cand
    return None


def compute_fft_per_point(ek_pt, fps, detrend=DETREND):
    """fft(Ek)_mean: FFT at each grid point, THEN average the amplitudes.

    ek_pt has shape (npoints, nframes) - the kinetic-energy time series at every
    grid point of the region. Each point is transformed on its own with the same
    normalisation as compute_fft, and |FFT| is then averaged over the points.
    Points that are entirely NaN are dropped; remaining NaNs are filled with
    that point's own mean.
    Returns (freq, amp_mean, npoints_used).
    """
    A = np.asarray(ek_pt, dtype=float)
    n = A.shape[-1]
    freq = np.fft.rfftfreq(n, d=1.0 / fps)

    keep = ~np.all(np.isnan(A), axis=1)
    if not np.any(keep):
        return freq, np.full_like(freq, np.nan), 0
    A = A[keep]
    rowmean = np.nanmean(A, axis=1, keepdims=True)
    A = np.where(np.isnan(A), rowmean, A)
    if detrend:
        A = A - A.mean(axis=1, keepdims=True)
    amp = amp_from_rfft(np.fft.rfft(A, axis=-1), n)
    return freq, np.nanmean(amp, axis=0), int(keep.sum())


def per_point_Ek(run_dir, npz_data):
    """Rebuild the per-point kinetic energy Ek(t) over the region, (npoints, nframes).

    The .npz only stores the region-averaged series, so the run's PIV .mat is
    re-read and the region + calibration recorded in the .npz are re-applied
    (matching batch_KineticEnergy). The region ('ROI' / 'full') comes from the
    .npz. Returns None if the .mat cannot be found.
    """
    piv_file = str(npz_data["PIV_file"]) if "PIV_file" in npz_data.files else ""
    if not os.path.isfile(piv_file):                    # data may have moved
        piv_file = os.path.join(run_dir, PIV_FILENAME)
    if not os.path.isfile(piv_file):
        return None

    dt_vel = float(npz_data["dt_vel"]) if "dt_vel" in npz_data.files else 1.0
    xscale = float(npz_data["xscale"]) if "xscale" in npz_data.files else 1.0
    yscale = float(npz_data["yscale"]) if "yscale" in npz_data.files else 1.0
    pts_roi = np.asarray(npz_data["pts_ROI"])
    region = str(npz_data["region"]) if "region" in npz_data.files else "ROI"

    X, Y, U, V, nframes = load_piv(piv_file)
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel

    Xr, Yr, Ur, Vr = region_fields(X, Y, U, V, pts_roi, region)

    # Same energy definition as batch_KineticEnergy, kept per point.
    Ek_pt = 0.5 * (Ur ** 2 + Vr ** 2)
    return Ek_pt.reshape(-1, Ek_pt.shape[-1])


# Labels say WHERE the region average sits relative to the FFT.
LBL_AVG_FIRST = r"$|\mathrm{FFT}(\langle E_k \rangle)|$   average, then FFT"
LBL_FFT_FIRST = r"$\langle |\mathrm{FFT}(E_k)| \rangle$   FFT, then average"


def save_run_figure(out_png, t, ek, freq, amp, freq_pt, amp_pt, run, flib,
                    frot=None, dphi_deg=None, normalize=False):
    """Two-panel per-run figure (time series + both amplitude spectra).

    Panel 2 overlays fft(Ek_mean) and fft(Ek)_mean; amp_pt may be None when the
    per-point spectrum is unavailable. normalize=True scales Ek by the libration
    KE scale U0**2 and the frequency axis by f_rot. Closed straight away.
    """
    escale = (libration_ke_scale(dphi_deg, flib) if normalize and flib
              and dphi_deg is not None and np.isfinite(dphi_deg) else 1.0)
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    ek = ek / escale
    amp = amp / escale
    amp_pt = None if amp_pt is None else amp_pt / escale
    freq = freq / fscale
    freq_pt = None if freq_pt is None else freq_pt / fscale
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
    if amp_pt is not None:
        plot(freq_pt, amp_pt, color="C2", lw=1, label=LBL_FFT_FIRST)
    plot(freq, amp, color="C1", lw=1, label=LBL_AVG_FIRST)
    axes[1].set_xlabel(r"$f / f_{\mathrm{rot}}$" if normalize else "frequency (Hz)",
                       fontsize=13)
    axes[1].set_ylabel((r"$|\widehat{E_k}| / E_{\mathrm{lib}}$" if normalize
                        else r"$|\widehat{E_k}|$  (m$^2$/s$^2$)"), fontsize=13)
    ttl = "FFT amplitude spectrum" + ("  (mean removed)" if DETREND else "")
    axes[1].set_title(ttl, fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=9)
    # Frequency axis to [0, max(4*frot,4*flib)]; y to decade bounds.
    curves = [amp] if amp_pt is None else [amp, amp_pt]
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


def build_row(name, fig_file, processed, region=np.nan, fps=np.nan, nframes=np.nan,
              f_peak=np.nan, amp_peak=np.nan, f_peak_pt=np.nan,
              amp_peak_pt=np.nan, npoints=np.nan, amp_flib=np.nan,
              amp_2flib=np.nan):
    """Assemble one summary-table row (folder-name metadata + results/flags)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    # Dimensionless numbers + normalised (_star): frequencies / frot, Ek
    # amplitudes / U0**2 (the FFT is of the kinetic energy).
    dl = dimensionless_numbers(frot_hz, flib_hz, dphi_deg)
    fn = frot_hz if np.isfinite(frot_hz) and frot_hz else np.nan
    ke_scale = dl["U0_mps"] ** 2 if np.isfinite(dl["U0_mps"]) and dl["U0_mps"] else np.nan
    row = {"run": name, "run idx": run_idx, "region": region,
           "processed": processed, "frot_Hz": frot_hz, "flib_Hz": flib_hz,
           "dphi_deg": dphi_deg, "fstar": fstar, "flib_star": fstar,
           "fps_Hz": fps, "nframes": nframes,
           "f_peak_Hz": f_peak, "f_peak_star": f_peak / fn,
           "amp_peak": amp_peak, "amp_peak_star": amp_peak / ke_scale,
           "f_peak_perPoint_Hz": f_peak_pt, "f_peak_perPoint_star": f_peak_pt / fn,
           "amp_peak_perPoint": amp_peak_pt,
           "amp_peak_perPoint_star": amp_peak_pt / ke_scale,
           "amp_flib": amp_flib, "amp_flib_star": amp_flib / ke_scale,
           "amp_2flib": amp_2flib, "amp_2flib_star": amp_2flib / ke_scale,
           "npoints": npoints, "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO,
            "figure": fig_file}
    row.update(dl)
    return row


def process_run(run_dir, region, reprocess=False):
    """FFT one run's Ek(t) both ways for one region; write its figure. Returns a dict."""
    tag = region_tag(region)
    name = os.path.basename(run_dir.rstrip("/"))
    npz_path = find_npz(run_dir, region)
    if npz_path is None:
        print("  [skip] no %s_%s.npz in %s" % (NPZ_STEM, tag, name))
        return build_row(name, "", False, region=tag)

    fig_stem = os.path.join(os.path.dirname(npz_path), "%s_%s" % (FIG_STEM, tag))
    fig_file = figure_filename(fig_stem, FIG_FORMAT, normalized=False)
    fig_norm = figure_filename(fig_stem, FIG_FORMAT, normalized=True)
    if os.path.isfile(fig_file) and os.path.isfile(fig_norm) and not reprocess:
        print("  [skip] %-40s %-4s figures already exist" % (name, tag))
        return build_row(name, fig_file, True, region=tag)

    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    ek = data["Ek_frame"]
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else name

    # (1) average, then FFT  -> fft(Ek_mean)
    freq, amp, _ = compute_fft(ek, fps, detrend=DETREND)
    f_peak, amp_peak = peak_freq(freq, amp)

    # (2) FFT, then average  -> fft(Ek)_mean  (needs the per-point field)
    freq_pt = amp_pt = None
    f_peak_pt = amp_peak_pt = np.nan
    npoints = np.nan
    if PER_POINT:
        try:
            Ek_pt = per_point_Ek(run_dir, data)
        except Exception as exc:
            Ek_pt = None
            print("  [warn] %s: per-point FFT failed (%s)" % (name, exc))
        if Ek_pt is None:
            print("  [warn] %s: no PIV .mat -> fft(Ek)_mean skipped" % name)
        else:
            freq_pt, amp_pt, npoints = compute_fft_per_point(Ek_pt, fps,
                                                             detrend=DETREND)
            f_peak_pt, amp_peak_pt = peak_freq(freq_pt, amp_pt)

    frot, flib, dphi_deg = parse_run_name(name)
    # Amplitude of the Ek spectrum at f_lib and 2*f_lib (raw; build_row also
    # stores /U0**2). Ek ~ velocity**2, so the libration shows up at 2*f_lib.
    amp_flib = amp_at_freq(freq, amp, flib)
    amp_2flib = amp_at_freq(freq, amp, 2.0 * flib)
    flib = None if not np.isfinite(flib) else flib
    frot = None if not np.isfinite(frot) else frot
    for _norm, _out in ((False, fig_file), (True, fig_norm)):
        save_run_figure(_out, t, ek, freq, amp, freq_pt, amp_pt,
                        "%s  (%s)" % (run, tag), flib, frot, dphi_deg, normalize=_norm)

    print("  [ok] %-40s %-4s nframes=%d fps=%.4gHz  f_peak=%.4gHz  "
          "fpeak(per-point)=%.4gHz  npts=%s"
          % (name, tag, np.size(ek), fps, f_peak, f_peak_pt,
             "-" if not np.isfinite(npoints) else int(npoints)))
    return build_row(name, fig_file, True, region=tag, fps=fps,
                     nframes=int(np.size(ek)), f_peak=f_peak, amp_peak=amp_peak,
                     f_peak_pt=f_peak_pt, amp_peak_pt=amp_peak_pt, npoints=npoints,
                     amp_flib=amp_flib, amp_2flib=amp_2flib)


def run_batch(reprocess_all=REPROCESS_ALL, only_runs=None):
    """FFT every run under BASE_DIR for each region; write per-region summary.
    Returns {region: DataFrame}."""
    if only_runs is None:
        only_runs = ONLY_RUNS
    subdirs = sorted(d for d in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(d)
                     and not os.path.basename(d).startswith("."))
    if only_runs:
        wanted = set(only_runs)
        subdirs = [d for d in subdirs if os.path.basename(d) in wanted]

    print("Found %d subfolders to process in %s%s"
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
        n_proc = int((df["processed"] == True).sum())
        print("\n  Summary %s (%d runs: %d processed):\n    %s"
              % (tag, len(df), n_proc, csv_path))
        results[tag] = df

    return results


# ## Run
# 
# `ONLY_RUNS` in the config cell restricts the batch to a few folders for testing.
# Clear it (`ONLY_RUNS = []`) and re-run to process **all** runs. Set
# `REPROCESS_ALL = True` to redraw every figure even where one already exists, and
# `PER_POINT = False` to skip the per-point spectrum (avoids re-reading the PIV
# `.mat`, which is the slow step).


# Returns {region: DataFrame}, one FFT summary per processed region (per REGION).
results = run_batch(reprocess_all=REPROCESS_ALL)
next(iter(results.values())).head(40) if results else results
