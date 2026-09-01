"""Auto-generated .py twin of single_VelocityFFT.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Velocity FFT of a single run
# 
# Interactive companion to `batch_VelocityFFT.ipynb` (velocity analogue of
# [`single_KineticEnergyFFT.ipynb`](single_KineticEnergyFFT.ipynb)).
# 
# It reads a per-run `VelocityFFT_<region>.npz` (produced by
# `batch_VelocityFFT.ipynb`), which already holds the **region-averaged
# single-sided amplitude spectra** of the two velocity components — `amp_u`,
# `amp_v` (and `amp_total = amp_u + amp_v`) — versus frequency `f`, computed with
# `np.fft.rfft` (mean removed, Hann window, amplitude-corrected) and the PIV
# **field** sampling frequency `fps = cam_fps/2`.
# 
# Because the FFT is precomputed by the batch, this notebook only **loads and
# plots** it (there is no time series stored and nothing is re-transformed). It
# draws a **single panel** with the two component spectra `|FFT(U)|` and
# `|FFT(V)|`.
# 
# The frequency axis spans `[0, max(4*frot, 4*flib)]` unless `FMAX` overrides it,
# and the amplitude axis is set to decade bounds whose top is always at or above
# the peak. If the libration frequency can be parsed from the run name, dashed
# guides are drawn at `flib` and `2*flib` (the interior velocity responds mainly
# at `flib`).
# 
# Section 7 additionally draws the polarization figure from the same `.npz`.


# ## 1. Imports


import os
import re
import numpy as np
import matplotlib.pyplot as plt


NPZ_STEM = "VelocityFFT"   # + _<region>.npz

from piv_postprocessing_lib import (topography_arrangement, amp_at_freq, dimensionless_numbers,
                        fft_axis_limits, k0,
                        fft_guide_lines, figure_filename, libration_velocity_scale, parse_flib,
                        parse_frot, parse_run_name,
                        peak_freq, peak_freq_in_band, resolve_npz, region_tag)


# ## 2. Configuration
# 
# Edit these, then run the cells below. `PATH` may be the `.npz` file itself or a
# run folder (its `PostProcessing/VelocityFFT.npz` is used).


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle. (Figures are unaffected.)
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = False
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# --- edit me -------------------------------------------------------------
PATH   = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
          "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom/"
          "frot0.50Hz_flib0.460Hz_dphi2deg_SS1/PostProcessing")   # run folder or .npz
REGION = 'ROI'       # 'ROI' or 'FULL' -- which VelocityFFT_<region>.npz to read
LOGY   = True        # logarithmic amplitude axis (False -> linear)
LOGX   = False       # logarithmic frequency axis (False -> linear)
FMAX   = 2        # upper frequency limit for the plot (Hz); None -> Nyquist
SAVE   = True       # also write a PNG next to the .npz
OVERWRITE_FIG  = True   # (over)write VelocityFFT_<region>[_normalized] even
                        #  if it exists (False -> keep the existing figures)
UPDATE_SUMMARY = False  # insert/replace this run's row in the dataset's
                        #  VelocityFFT_summary_<region>.csv (False -> leave it)

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'
OUTPUT = None        # PNG path; None -> VelocityFFT_<region>.png beside the .npz
OVERWRITE_POL = False  # (over)write Velocity_polarization.png next to the .npz
                       #  even if it exists (False -> keep existing)
F_LOW_FMIN = 0.2    # lower edge (Hz) of the band searched for f_low, the
                     # strongest peak below f_lib/2; keeps the near-DC bins out
THRESHOLD_PEAK = 1.5   # f_low is significant only if its amplitude exceeds
                       # THRESHOLD_PEAK * mean amplitude over the search band;
                       # otherwise f_low (and the sidebands) are set to NaN
# -------------------------------------------------------------------------
tag = region_tag(REGION)
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(PATH)


# ## 3. Helper functions


# ## 4. Load the spectrum


npz_path = resolve_npz(PATH, "%s_%s.npz" % (NPZ_STEM, tag))
print("Reading %s" % npz_path)

data = np.load(npz_path, allow_pickle=True)
f = np.asarray(data["f"], dtype=float)
amp_u = np.asarray(data["amp_u"], dtype=float)
amp_v = np.asarray(data["amp_v"], dtype=float)
amp_total = np.asarray(data["amp_total"], dtype=float)
powerU = np.asarray(data["powerU"], dtype=float) if "powerU" in data.files else None
powerV = np.asarray(data["powerV"], dtype=float) if "powerV" in data.files else None
fps = float(data["fps"]) if "fps" in data.files else 1.0
run = str(data["run"]) if "run" in data.files else os.path.basename(
    os.path.dirname(npz_path))
region = str(data["region"]) if "region" in data.files else tag
window = str(data["window"]) if "window" in data.files else "?"
npoints = int(data["npoints"]) if "npoints" in data.files else -1
nframes = int(data["nframes"]) if "nframes" in data.files else f.size

flib = parse_flib(run)
frot = parse_frot(run)
print("run     = %s  (%s)" % (run, region))
print("fps     = %g Hz   nframes = %d   Nyquist = %g Hz" % (fps, nframes, f[-1]))
print("df      = %.4g Hz   ROI points averaged = %d   window = %s"
      % (f[1] - f[0] if f.size > 1 else np.nan, npoints, window))
print("flib    = %s" % ("%g Hz" % flib if flib else "unknown"))


# ## 5. Peak frequencies


for label, amp in (("U", amp_u), ("V", amp_v), ("total", amp_total)):
    fp, ap = peak_freq(f, amp)
    print("peak |FFT(%-5s)| at %.4f Hz   (amplitude = %.3e m/s)" % (label, fp, ap))

# Dominant peak of the summed spectrum -- shown in the figure title.
f_peak, _ = peak_freq(f, amp_total)
# f_low: strongest peak in [F_LOW_FMIN, flib/2] -- the low-frequency response
# below the libration forcing. Taken on the summed amplitude so it matches the
# f_low_Hz column batch_VelocityFFT writes to its summary table.
f_low, a_low = peak_freq_in_band(f, amp_total, F_LOW_FMIN,
                                 flib / 2.0 if flib else np.nan)
# Significance test: the low-frequency peak must exceed THRESHOLD_PEAK x the mean of the
# summed spectrum over the search band [F_LOW_FMIN, f_lib/2]; otherwise there is
# no meaningful f_low (so f_low and the f_lib -/+ f_low sidebands -- and all
# their amplitudes -- are set to NaN).
if flib and np.isfinite(f_low):
    _band = (f >= F_LOW_FMIN) & (f <= flib / 2.0)
    _bmean = np.nanmean(amp_total[_band]) if np.any(_band) else np.nan
    if not (np.isfinite(_bmean) and a_low >= THRESHOLD_PEAK * _bmean):
        print("f_low: peak %.3e < %.3g x band mean %.3e -> not significant"
              % (a_low, THRESHOLD_PEAK, _bmean))
        f_low, a_low = np.nan, np.nan
if np.isfinite(f_low):
    print("f_low = %.4f Hz   (amplitude = %.3e m/s)   [band %.3g - %.4g Hz]"
          % (f_low, a_low, F_LOW_FMIN, flib / 2.0))
else:
    print("f_low: unavailable (flib unknown, no bin, or peak not significant)")

if flib:
    print("  reference: flib = %.4f Hz,  2*flib = %.4f Hz" % (flib, 2 * flib))
    if np.isfinite(f_low):
        print("  sidebands: flib-f_low = %.4f Hz,  flib+f_low = %.4f Hz"
              % (flib - f_low, flib + f_low))
if frot:
    print("  reference: frot = %.4f Hz" % frot)

# Dimensionless numbers + normalised (_star) frequencies (f / frot).
_dl = dimensionless_numbers(frot if frot else np.nan,
                            flib if flib else np.nan, parse_run_name(run)[2])
print("dimensionless: E=%.3e  E_l=%.3e  Ro=%.3g  Re=%.4g  Re_l=%.4g  "
      "Re_bl=%.3g  U0=%.4g m/s"
      % (_dl["E"], _dl["E_l"], _dl["Ro"], _dl["Re"], _dl["Re_l"],
         _dl["Re_bl"], _dl["U0_mps"]))
if frot and np.isfinite(frot):
    _flow_star = f_low / frot if np.isfinite(f_low) else np.nan
    print("normalised:  flib* = %.4f   f_peak* = %.4f   f_low* = %.4f"
          % (flib / frot, f_peak / frot, _flow_star))

# Amplitude of the summed spectrum at f_lib, f_low and the f_lib -/+ f_low
# sidebands -- raw (m/s) and normalised by U0.
_U0 = _dl["U0_mps"]
_targets = {"f_lib": flib if flib else np.nan, "f_low": f_low,
            "f_lib - f_low": (flib - f_low) if (flib and np.isfinite(f_low)) else np.nan,
            "f_lib + f_low": (flib + f_low) if (flib and np.isfinite(f_low)) else np.nan}
print("peak amplitudes (summed spectrum |FFT(U)|+|FFT(V)|):")
for _lbl, _ft in _targets.items():
    _a = amp_at_freq(f, amp_total, _ft)
    _as = _a / _U0 if (np.isfinite(_a) and np.isfinite(_U0) and _U0) else np.nan
    print("  %-14s at %8s Hz :  %.3e m/s   ( /U0 = %.3e )"
          % (_lbl, ("%.4f" % _ft) if np.isfinite(_ft) else "nan", _a, _as))


# ## 6. Plot


# Frequency axis to [0, max(4*frot, 4*flib)] (FMAX overrides); y to decade
# bounds. Both a raw and a normalized (freq / f_rot, amplitude / U0) figure.
dphi_deg = parse_run_name(run)[2]
U0 = libration_velocity_scale(dphi_deg, flib) if (flib and np.isfinite(dphi_deg)) else np.nan


def _make(normalize):
    ascale = U0 if (normalize and np.isfinite(U0) and U0) else 1.0
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    at = amp_total / ascale
    ff = f / fscale
    _frot = frot / fscale if frot else frot
    _flib = flib / fscale if flib else flib
    _flow = f_low / fscale if np.isfinite(f_low) else f_low
    _fu = "" if fscale != 1.0 else " Hz"

    xlim_max, ymin_c, ymax_c = fft_axis_limits(ff, [at], _frot, _flib)
    fmax = (FMAX / fscale if fscale != 1.0 else FMAX) if FMAX else xlim_max
    sel = ff <= fmax
    if LOGX:
        sel = sel & (ff > 0)
    # y lower bound: the closest power of 10 below the minimum of the visible
    # total amplitude (keeps the whole curve in view on the log axis).
    _vis = at[sel & (at > 0)]
    if _vis.size:
        ymin_c = 10.0 ** np.floor(np.log10(_vis.min()))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot = ax.semilogy if LOGY else ax.plot
    plot(ff[sel], at[sel], color="C0", lw=1.2,
         label=r"$|\widehat{U}|+|\widehat{V}|$")

    ax.set_xlabel(r"$f / f_{\mathrm{rot}}$" if fscale != 1.0 else "frequency (Hz)",
                  fontsize=13)
    ax.set_ylabel(r"amplitude $/\,U_0$" if ascale != 1.0 else r"amplitude  (m/s)",
                  fontsize=13)
    _fpk = f_peak / fscale if np.isfinite(f_peak) else f_peak
    title = ("%s   (%s)%s   ($k_0 = %g$, top=%s, bottom=%s)"
             % (run, region, "   (normalized)" if normalize else "", k0,
                TOP_TOPO, BOTTOM_TOPO))
    if np.isfinite(_fpk) and _fpk > 0:
        title += "\n$f_{\mathrm{peak}}$ = %.4g%s" % (_fpk, _fu)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    for fq, lbl, colr in fft_guide_lines(_frot, _flib, _flow):
        if fq > fmax:
            continue
        ax.axvline(fq, color=colr, ls="--", lw=1, alpha=0.85,
                   label="%s = %.4g%s" % (lbl, fq, _fu))
    # Black diamonds at the detected peaks (f_lib, f_low, f_lib -/+ f_low), at
    # the total amplitude *on the plotted curve* (interpolated at each exact
    # frequency) so every marker sits exactly on the total-amplitude line.
    _pk = []
    if _flib is not None and np.isfinite(_flib):
        _pk.append(_flib)
    if np.isfinite(_flow):
        _pk.append(_flow)
        if _flib is not None and np.isfinite(_flib):
            _pk += [_flib - _flow, _flib + _flow]
    _dlbl = False
    for _pf in _pk:
        if not (0.0 <= _pf <= fmax):
            continue
        _pa = float(np.interp(_pf, ff, at))
        if np.isfinite(_pa):
            ax.plot(_pf, _pa, "D", color="k", ms=6, zorder=5,
                    label=None if _dlbl else "detected peaks")
            _dlbl = True
    ax.legend(fontsize=9, ncol=2)
    if LOGX:
        ax.set_xscale("log")
        _pos = ff[(ff > 0) & (ff <= fmax)]
        ax.set_xlim(_pos.min() if _pos.size else fmax / 100.0, fmax)
    else:
        ax.set_xlim(0, fmax)
    if ymin_c and ymax_c:
        ax.set_ylim(ymin_c, ymax_c)
    fig.tight_layout()
    return fig


if OUTPUT:
    _stem, _ext = os.path.splitext(OUTPUT)
    _fmt = _ext.lstrip(".") or FIG_FORMAT
else:
    _stem = os.path.join(os.path.dirname(npz_path), "VelocityFFT_%s" % tag)
    _fmt = FIG_FORMAT

for _norm in (False, True):
    fig = _make(_norm)
    if SAVE:
        output = figure_filename(_stem, _fmt, normalized=_norm)
        if OVERWRITE_FIG or not os.path.isfile(output):
            fig.savefig(output, dpi=200, bbox_inches="tight")
            print("Figure written to:\n  %s" % output)
        else:
            print("Figure already exists (set OVERWRITE_FIG=True to overwrite):"
                  "\n  %s" % output)
    plt.show()


# ## 6.5 Update the dataset summary (optional)


# When UPDATE_SUMMARY is True, this run's row in VelocityFFT_summary_<region>.csv
# (in the dataset root) is inserted/replaced with the values computed above --
# handy to push a single run's corrected f_low / amplitudes into the summary
# without re-running the whole batch. The row schema and the 'kept'
# de-duplication match batch_VelocityFFT exactly.
if UPDATE_SUMMARY:
    import pandas as pd
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(npz_path)))
    SUMMARY_CSV = os.path.join(BASE_DIR, "VelocityFFT_summary_%s.csv" % tag)

    _dphiN = parse_run_name(run)[2]
    _U0row = libration_velocity_scale(flib, _dphiN) if flib else np.nan
    _uscale = _U0row if (np.isfinite(_U0row) and _U0row) else np.nan
    _amp = {"flib": amp_at_freq(f, amp_total, flib if flib else np.nan),
            "flow": amp_at_freq(f, amp_total, f_low),
            "flib_minus_flow": amp_at_freq(f, amp_total,
                                           (flib - f_low) if flib else np.nan),
            "flib_plus_flow": amp_at_freq(f, amp_total,
                                          (flib + f_low) if flib else np.nan)}
    _amps = {_k: (_a / _uscale) for _k, _a in _amp.items()}

    _fn = frot if (frot and np.isfinite(frot)) else np.nan
    _ss = re.search(r"SS(\d+)", run)
    _dt = float(data["dt_vel"]) if "dt_vel" in data.files else np.nan
    _ok = bool(data["calibrated"]) if "calibrated" in data.files else np.nan
    _row = {"run": run, "run idx": int(_ss.group(1)) if _ss else np.nan,
            "region": tag, "processed": True,
            "frot_Hz": frot if frot else np.nan,
            "flib_Hz": flib if flib else np.nan, "dphi_deg": _dphiN,
            "fstar": (flib / _fn) if flib else np.nan,
            "flib_star": (flib / _fn) if flib else np.nan,
            "calibrated": _ok, "dt_vel_s": _dt, "fps_Hz": fps,
            "nframes": nframes, "npoints": npoints,
            "f_peak_Hz": f_peak, "f_peak_star": f_peak / _fn,
            "f_low_Hz": f_low, "f_low_star": f_low / _fn,
            "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO, "npz": npz_path}
    for _k in ("flib", "flow", "flib_minus_flow", "flib_plus_flow"):
        _row["amp_%s" % _k] = _amp[_k]
        _row["amp_%s_star" % _k] = _amps[_k]
    _row.update(dimensionless_numbers(frot if frot else np.nan,
                                      flib if flib else np.nan, _dphiN))

    if os.path.isfile(SUMMARY_CSV):
        _df = pd.read_csv(SUMMARY_CSV)
        for _c in ("processed", "kept", "calibrated"):
            if _c in _df.columns and _df[_c].dtype == object:
                _df[_c] = _df[_c].astype(str).str.strip().str.lower().isin(
                    ("true", "1", "1.0", "yes"))
    else:
        _df = pd.DataFrame()
        print("  (no existing summary -> creating it)")

    # Replace the row with the same (run, region), else append.
    if not _df.empty and {"run", "region"}.issubset(_df.columns):
        _df = _df[~((_df["run"] == run) & (_df["region"] == tag))]
    _df = pd.concat([_df, pd.DataFrame([_row])], ignore_index=True)

    # Re-derive 'kept': keep one run per (frot, flib, dphi) group -- the highest
    # SS index -- exactly as batch_VelocityFFT's mark_kept does.
    _df["kept"] = False
    _proc = _df[_df["processed"] == True].copy()
    if not _proc.empty:
        _proc["_ss"] = _proc["run idx"].fillna(1)
        for _, _g in _proc.groupby(["frot_Hz", "flib_Hz", "dphi_deg"],
                                   dropna=False):
            _df.loc[_g["_ss"].idxmax(), "kept"] = True

    _df.to_csv(SUMMARY_CSV, index=False)
    print("Summary row updated for %s:\n  %s" % (run, SUMMARY_CSV))
else:
    print("UPDATE_SUMMARY is False -> summary file left unchanged.")


# ## 7. Polarization
# 
# Same figure as `batch_VelocityFFT`: overlays the inertial-wave relation
# `2*((2*frot/f)**2 - 1)` with the measured `powerRatio` (ROI statistics of the
# per-point `powerV/powerU`), read from the `.npz`. Log y-axis, x limited to
# 0.01..1 Hz. The ratio is drawn both as the ROI mean and as the median with a
# 25-75 percentile band. Set `OVERWRITE_POL = True` to overwrite
# `Velocity_polarization.png` next to the `.npz`.


# Polarization - identical to batch_VelocityFFT's figure. The arrays are read
# from the .npz (the raw U/V time series are not stored here, so nothing can be
# recomputed). Both a raw and a normalized (freq / f_rot) figure are written.
if "powerRatio" in data.files:

    def _make_pol(normalize):
        fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
        ff = f / fscale
        m = f > 0
        figp, axp = plt.subplots(figsize=(9, 6))
        if frot:
            theory = 2.0 * ((2.0 * frot / f[m]) ** 2 - 1.0)
            axp.plot(ff[m], theory, "k-", lw=2.2,
                     label=r"$2[(2f_{\mathrm{rot}}/f)^2-1]$  (IW)")
        axp.plot(ff[m], np.asarray(data["powerRatio"], dtype=float)[m], lw=1.2,
                 label=r"mean $P_V/P_U$")
        _pct = ("powerRatio_med", "powerRatio_p25", "powerRatio_p75")
        if all(kk in data.files for kk in _pct):
            med = np.asarray(data["powerRatio_med"], dtype=float)
            p25 = np.asarray(data["powerRatio_p25"], dtype=float)
            p75 = np.asarray(data["powerRatio_p75"], dtype=float)
            line, = axp.plot(ff[m], med[m], lw=1.2, label=r"median $P_V/P_U$")
            axp.fill_between(ff[m], p25[m], p75[m], color=line.get_color(),
                             alpha=0.20, lw=0, label="ROI 25-75%")
        else:
            print("  (no percentile arrays in this .npz -> mean only)")
        axp.set_yscale("log")
        axp.set_xlim(0.01 / fscale, 1.0 / fscale)
        axp.set_xlabel(r"$f / f_{\mathrm{rot}}$" if fscale != 1.0 else "frequency (Hz)",
                       fontsize=13)
        axp.set_ylabel(r"polarization  $P_V/P_U$", fontsize=13)
        axp.set_title("Polarization - %s  (%s)%s   ($k_0 = %g$, top=%s, bottom=%s)"
                      % (run, region, "   (normalized)" if normalize else "", k0,
                         TOP_TOPO, BOTTOM_TOPO),
                      fontsize=13)
        axp.grid(True, which="both", ls=":", alpha=0.4)
        axp.legend(fontsize=9)
        figp.tight_layout()
        return figp

    _pol_stem = os.path.join(os.path.dirname(npz_path), "Velocity_polarization_%s" % tag)
    for _norm in (False, True):
        pol_png = figure_filename(_pol_stem, FIG_FORMAT, normalized=_norm)
        if OVERWRITE_POL or not os.path.isfile(pol_png):
            figp = _make_pol(_norm)
            figp.savefig(pol_png, dpi=200, bbox_inches="tight")
            print("Polarization figure written to:\n  %s" % pol_png)
        else:
            print("Polarization figure already exists (set OVERWRITE_POL=True to "
                  "overwrite):\n  %s" % pol_png)
    plt.show()
else:
    print("powerRatio not in this .npz - reprocess with batch_VelocityFFT "
          "to add it")
