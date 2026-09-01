"""Auto-generated .py twin of single_KineticEnergyFFT.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # FFT of a kinetic-energy time series
# 
# Interactive companion to [`batch_KineticEnergyFFT.ipynb`](batch_KineticEnergyFFT.ipynb).
# 
# It reads a per-run `KineticEnergy_timeSeries.npz` (produced by
# `batch_KineticEnergy.py`, containing `t`, `Ek_frame` and the PIV sampling
# frequency `fps`), computes the **FFT** of `Ek(t)`, and draws two panels:
# 
# 1. the kinetic-energy time series `Ek(t)`;
# 2. its amplitude spectrum `|FFT|` versus frequency.
# 
# The one-sided amplitude is normalised as `|FFT| * 2 / N` so a pure tone reads
# as its physical amplitude (DC and Nyquist bins keep the `1/N` scaling). By
# default the mean is removed first so the DC term does not dwarf the rest. If the
# libration frequency can be parsed from the run name, dashed guides are drawn at
# `flib` and `2*flib`.
# 
# Run with an environment that has **numpy / matplotlib** (e.g. `dpivsoft`).


# ## 1. Imports


import os
import re
import numpy as np
import matplotlib.pyplot as plt


NPZ_STEM = "KineticEnergy_timeSeries"   # + _<region>.npz

from piv_postprocessing_lib import (topography_arrangement, compute_fft, dimensionless_numbers, k0,
                        fft_axis_limits, figure_filename, libration_ke_scale, parse_flib,
                        parse_frot, parse_run_name,
                        resolve_npz, region_tag)


# ## 2. Configuration
# 
# Edit these, then run the cells below. `PATH` may be the `.npz` file itself or a
# run folder (its `PostProcessing/KineticEnergy_timeSeries.npz` is used).


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
PATH    = "/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly/frot0.50Hz_flib0.500Hz_dphi24deg/PostProcessing"   # run folder or .npz
REGION  = 'ROI'       # 'ROI' or 'FULL' -- which KineticEnergy_timeSeries_<region>.npz to read
DETREND = True        # remove the mean before the estimate (kills the DC spike)
LOGY    = True        # logarithmic power axis (False -> linear)
LOGX    = False       # logarithmic frequency axis (False -> linear)
SAVE    = True        # also write a PNG next to the .npz

# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'
OUTPUT  = None        # PNG path; None -> KineticEnergy_FFT_<region>.png beside the .npz
# -------------------------------------------------------------------------
# If PATH is a .npz file it is read directly and REGION only tags the output;
# if PATH is a folder, KineticEnergy_timeSeries_<REGION>.npz inside it is used.
tag = region_tag(REGION)
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(PATH)


# ## 3. Helper functions
# 
# Same logic as the script.


# ## 4. Load the time series


npz_path = resolve_npz(PATH, "%s_%s.npz" % (NPZ_STEM, tag))
print("Reading %s" % npz_path)

data = np.load(npz_path, allow_pickle=True)
t = data["t"]
ek = data["Ek_frame"]
fps = float(data["fps"]) if "fps" in data.files else 1.0
run = str(data["run"]) if "run" in data.files else os.path.basename(
    os.path.dirname(npz_path))
region = str(data["region"]) if "region" in data.files else tag

flib = parse_flib(run)
frot = parse_frot(run)
print("run    = %s" % run)
print("region = %s" % region)
print("fps    = %g Hz   nframes = %d" % (fps, ek.size))
print("flib   = %s" % ("%g Hz" % flib if flib else "unknown"))


# ## 5. Compute the FFT


freq, amp, fft = compute_fft(ek, fps, detrend=DETREND)

# Peak frequency (ignoring the DC bin).
if amp.size > 1:
    k = 1 + int(np.argmax(amp[1:]))
    print("peak at %.4f Hz  (amplitude = %.3e)" % (freq[k], amp[k]))
    if flib:
        print("  flib = %.4f Hz, 2*flib = %.4f Hz" % (flib, 2 * flib))

    # Dimensionless numbers + normalised (_star) frequencies (freq / frot).
    _dl = dimensionless_numbers(frot if frot else np.nan,
                                flib if flib else np.nan, parse_run_name(run)[2])
    print("dimensionless: E=%.3e  E_l=%.3e  Ro=%.3g  Re=%.4g  Re_l=%.4g  "
      "Re_bl=%.3g  U0=%.4g m/s"
      % (_dl["E"], _dl["E_l"], _dl["Ro"], _dl["Re"], _dl["Re_l"],
         _dl["Re_bl"], _dl["U0_mps"]))
    if frot and np.isfinite(frot):
        print("normalised:  flib* = %.4f   f_peak* = %.4f"
              % (flib / frot, freq[k] / frot))


# ## 6. Plot


dphi_deg = parse_run_name(run)[2]


def _make(normalize):
    escale = (libration_ke_scale(dphi_deg, flib) if normalize and flib
              and np.isfinite(dphi_deg) else 1.0)
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    ekn, ampn, freqn = ek / escale, amp / escale, freq / fscale
    _flib = flib / fscale if flib else flib
    _frot = frot / fscale if frot else frot

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(t, ekn, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel(r"$E_k(t) / E_{\mathrm{lib}}$" if normalize
                       else r"$E_k(t)$  (m$^2$/s$^2$)", fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    if LOGY:
        axes[1].semilogy(freqn, ampn, color="C1", lw=1)
    else:
        axes[1].plot(freqn, ampn, color="C1", lw=1)
    axes[1].set_xlabel(r"$f / f_{\mathrm{rot}}$" if normalize else "frequency (Hz)",
                       fontsize=13)
    axes[1].set_ylabel(r"$|\widehat{E_k}| / E_{\mathrm{lib}}$" if normalize
                       else r"$|\widehat{E_k}|$  (m$^2$/s$^2$)", fontsize=13)
    ttl = "FFT amplitude spectrum" + ("  (mean removed)" if DETREND else "")
    axes[1].set_title(ttl, fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    _xmax, _ymin, _ymax = fft_axis_limits(freqn, ampn, _frot, _flib)
    if LOGX:
        axes[1].set_xscale("log")
        _pos = freqn[(freqn > 0) & (freqn <= _xmax)]
        axes[1].set_xlim(_pos.min() if _pos.size else _xmax / 100.0, _xmax)
    else:
        axes[1].set_xlim(0, _xmax)
    if _ymin and _ymax:
        axes[1].set_ylim(_ymin, _ymax)
    if _flib:
        for fq, lbl in ((_flib, r"$f_{\mathrm{lib}}$"),
                        (2 * _flib, r"$2f_{\mathrm{lib}}$")):
            if fq <= freqn.max():
                axes[1].axvline(fq, color="k", ls="--", lw=1, alpha=0.6)
                axes[1].annotate(lbl, xy=(fq, 1), xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points",
                                 fontsize=10)
    fig.suptitle("%s  (%s)%s   ($k_0 = %g$, top=%s, bottom=%s)"
                 % (run, region, "   (normalized)" if normalize else "", k0,
                    TOP_TOPO, BOTTOM_TOPO),
                 fontsize=13)
    fig.tight_layout()
    return fig


if OUTPUT:
    _stem, _ext = os.path.splitext(OUTPUT)
    _fmt = _ext.lstrip(".") or FIG_FORMAT
else:
    _stem = os.path.join(os.path.dirname(npz_path), "KineticEnergy_FFT_%s" % tag)
    _fmt = FIG_FORMAT

# Both a raw and a normalized figure.
for _norm in (False, True):
    fig = _make(_norm)
    if SAVE:
        output = figure_filename(_stem, _fmt, normalized=_norm)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % output)
    plt.show()
