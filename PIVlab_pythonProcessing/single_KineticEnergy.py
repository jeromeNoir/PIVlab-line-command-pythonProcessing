"""Auto-generated .py twin of single_KineticEnergy.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Kinetic energy of a single PIV run
# 
# Single-run version of `batch_KineticEnergy`: it runs **exactly** the batch's
# per-run step for one `PATH`, so the outputs are identical to what the batch would
# write for that run. In one pass it reads the PIV `.mat`, computes the ROI/FULL
# kinetic-energy time series `⟨Ek⟩(t)` **and** both FFT spectra of `Ek`
# (`FFT(⟨Ek⟩)` = average-then-FFT, `⟨FFT(Ek)⟩` = FFT-then-average), then writes:
# 
# - `PostProcessing/KineticEnergy_<region>.npz` — time series + both spectra;
# - `PostProcessing/KineticEnergy_FFT_<region>[_normalized].<fmt>` — the two-panel
#   figure (both spectra drawn);
# - one row in `KineticEnergy_summary_<region>.csv` (when `UPDATE_SUMMARY`) — the
#   time-series stats plus the `⟨FFT(Ek)⟩` quantities (peak + amplitudes at `f_lib`
#   and `2·f_lib`). Only `⟨FFT(Ek)⟩` is tabled.
# 
# Calibration is read from the run's `acquisition_log.txt`; all constants come from
# the dataset's `param_postProcessing.json` (same as the batch).


# ## 1. Imports


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, amp_at_freq,
                        amp_from_rfft, compute_fft, dimensionless_numbers,
                        fft_axis_limits, figure_filename, libration_ke_scale,
                        libration_velocity_scale, load_piv, parse_run_name,
                        peak_freq, read_acquisition_params, read_paramPostprocessing,
                        region_fields, region_tag)


# ## 2. Configuration


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle. (Figures are unaffected.)
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = False
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# The single run folder to process (must hold the .mat and the acquisition log).
PATH = ('/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
        'CylinderExperimentsGMA/k20_bottomOnly/frot0.50Hz_flib0.410Hz_dphi2deg_SS1')

# Spatial extent: 'ROI' (crop to PTS_ROI) or 'FULL' (whole field). Written into
# every output filename, so the two never overwrite each other.
REGION = 'ROI'

# Per-run output + summary stems (region tag appended) -- identical to the batch.
RESULT_STEM = 'KineticEnergy'          # per-run npz -> <stem>_<region>.npz
FFTFIG_STEM = 'KineticEnergy_FFT'      # per-run 2-panel figure beside the npz
SUMMARY_STEM = 'KineticEnergy_summary' # summary at the dataset root -> <stem>_<region>.csv

# --- Overwrite / output switches -----------------------------------------
REPROCESS      = True   # recompute + overwrite the npz/figure if they exist
UPDATE_SUMMARY = True   # add/refresh this run's row in KineticEnergy_summary_<region>.csv

# --- FFT ------------------------------------------------------------------
DETREND = True   # remove the mean before the FFT (kills the DC spike)
LOGY    = True   # log amplitude axis on the spectrum panel
LOGX    = False  # log frequency axis (False -> linear)

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'

# ----------------------------------------------------------------------
# Parameters live in the dataset folder (PATH's parent) -- same as the batch.
_P = read_paramPostprocessing(os.path.dirname(PATH.rstrip('/')))
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(PATH)
PIV_FILENAME, LOG_FILENAME = _P.PIV_FILENAME, _P.LOG_FILENAME
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
UNCAL_SCALE = _P.UNCAL_SCALE
tag = region_tag(REGION)


# ## 3. Helper functions


# These three helpers are byte-for-byte the batch_KineticEnergy versions, so the
# per-run npz, figure and summary row are identical to what the batch writes.

def compute_fft_per_point(ek_pt, fps, detrend=DETREND):
    """<FFT(Ek)>: FFT at each grid point, THEN average the amplitudes."""
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


LBL_AVG_FIRST = r"$|\mathrm{FFT}(\langle E_k \rangle)|$   average, then FFT"
LBL_FFT_FIRST = r"$\langle |\mathrm{FFT}(E_k)| \rangle$   FFT, then average"


def make_fft_figure(out_png, t, ek, freq, amp_avg, amp_mean, run, flib,
                    frot=None, dphi_deg=None, normalize=False):
    """Two-panel figure (time series + both spectra). Same drawing/saving as
    batch_KineticEnergy.save_fft_figure; returned (not closed) so it also displays."""
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

    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel((r"$\langle E_k \rangle(t) / E_{\mathrm{lib}}$" if normalize
                        else r"$\langle E_k \rangle(t)$  (m$^2$/s$^2$)"), fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

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
                axes[1].annotate(lbl, xy=(fq, 1), xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points", fontsize=10)

    fig.suptitle(run + ("   (normalized)" if normalize else "")
                 + "   ($k_0 = %g$, top=%s, bottom=%s)" % (k0, TOP_TOPO, BOTTOM_TOPO),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


def build_row(name, npz_file, fig_file, processed, region=np.nan,
              mean_Ekin=np.nan, std_Ekin=np.nan, dt_vel=np.nan, fps=np.nan,
              nframes=np.nan, npoints=np.nan, ok=np.nan,
              f_peak=np.nan, amp_peak=np.nan, amp_flib=np.nan, amp_2flib=np.nan):
    """One combined summary row -- identical schema to batch_KineticEnergy."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
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


def update_summary(csv_path, row):
    """Drop any existing row for this run, append the new one, write the CSV."""
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["run"] != row["run"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["frot_Hz", "flib_Hz", "dphi_deg"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df


# ## 4. Process the run (time series + both FFT spectra + figure)


# The batch_KineticEnergy per-run step, for this one run + region.
name = os.path.basename(PATH.rstrip("/"))
frot_hz, flib_hz, dphi_deg = parse_run_name(name)
out_dir = os.path.join(PATH, "PostProcessing")
out_file = os.path.join(out_dir, "%s_%s.npz" % (RESULT_STEM, tag))
fig_stem = os.path.join(out_dir, "%s_%s" % (FFTFIG_STEM, tag))
fig_file = figure_filename(fig_stem, FIG_FORMAT, normalized=False)

if os.path.isfile(out_file) and not REPROCESS:
    # Reuse the cached result (exactly like the batch's skip path).
    d = np.load(out_file, allow_pickle=True)
    _g = lambda k: (float(d[k]) if k in d.files else np.nan)
    row = build_row(name, out_file, fig_file, True, region=tag,
                    mean_Ekin=_g("mean_Ekin"), std_Ekin=_g("std_Ekin"),
                    dt_vel=_g("dt_vel"), fps=_g("fps"),
                    nframes=int(d["nframes"]) if "nframes" in d.files else np.nan,
                    npoints=int(d["npoints"]) if "npoints" in d.files else np.nan,
                    ok=bool(d["calibrated"]) if "calibrated" in d.files else np.nan,
                    f_peak=_g("f_peak"), amp_peak=_g("amp_peak"),
                    amp_flib=_g("amp_flib"), amp_2flib=_g("amp_2flib"))
    print("[skip] %s exists and REPROCESS is False -- using cached values"
          % os.path.basename(out_file))
else:
    piv_file = os.path.join(PATH, PIV_FILENAME)
    if not os.path.isfile(piv_file):
        raise FileNotFoundError("no %s in %s" % (PIV_FILENAME, PATH))

    dt_vel, fps, ok = read_acquisition_params(os.path.join(PATH, LOG_FILENAME))
    if ok:
        xscale, yscale = XSCALE, YSCALE
    else:
        print("[warn] Calibration not possible - velocities will be in px/frame")
        xscale = yscale = UNCAL_SCALE

    X, Y, U, V, nframes = load_piv(piv_file)
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel
    Xr, Yr, Ur, Vr = region_fields(X, Y, U, V, PTS_ROI, REGION)

    # Per-point Ek(t) over the region, then the region average.
    Ek_pt = 0.5 * (Ur ** 2 + Vr ** 2).reshape(-1, nframes)
    npoints = Ek_pt.shape[0]
    Ek_frame = np.nanmean(Ek_pt, axis=0)
    t = np.arange(nframes) / fps
    mean_Ekin = float(np.nanmean(Ek_frame))
    std_Ekin = float(np.nanstd(Ek_frame))

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
    print("wrote", out_file)

    # Two-panel figure (both spectra), raw and normalized -- identical to batch.
    _flib = None if not np.isfinite(flib_hz) else flib_hz
    _frot = None if not np.isfinite(frot_hz) else frot_hz
    for _norm, _out in ((False, fig_file),
                        (True, figure_filename(fig_stem, FIG_FORMAT, normalized=True))):
        make_fft_figure(_out, t, Ek_frame, freq, amp_avg, amp_mean,
                        "%s  (%s)" % (name, tag), _flib, _frot, dphi_deg,
                        normalize=_norm)
        print("wrote", _out)
        plt.show()

    row = build_row(name, out_file, fig_file, True, region=tag,
                    mean_Ekin=mean_Ekin, std_Ekin=std_Ekin, dt_vel=dt_vel,
                    fps=fps, nframes=nframes, npoints=npoints, ok=ok,
                    f_peak=f_peak, amp_peak=amp_peak, amp_flib=amp_flib,
                    amp_2flib=amp_2flib)

print("run     : %s   (%s)" % (name, tag))
print("<Ek>    : %.4e   std = %.4e   f_peak(<FFT(Ek)>) = %.4g Hz"
      % (row["mean_Ekin"], row["std_Ekin"], row["f_peak_Hz"]))


# ## 5. Summary


# Add/refresh this run's row in the dataset's single KineticEnergy summary.
# The row is identical to the one batch_KineticEnergy would write for this run.
if UPDATE_SUMMARY:
    base_dir = os.path.dirname(PATH.rstrip("/"))     # dataset root, where the summary lives
    ke_summary_csv = os.path.join(base_dir, "%s_%s.csv" % (SUMMARY_STEM, tag))
    update_summary(ke_summary_csv, row)
    print("updated %s summary:\n   %s" % (tag, ke_summary_csv))
    print()
    for k, v in row.items():
        print("   %-16s %s" % (k, v))
else:
    print("UPDATE_SUMMARY is False -- summary table left untouched.")
