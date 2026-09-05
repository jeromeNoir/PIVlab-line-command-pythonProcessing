"""Auto-generated .py twin of PLOT_FFT.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # PLOT_FFT -- ROI-averaged spectra of velocity and kinetic energy
# 
# Viewer of the ROI-averaged spectra stored in each run's
# `PostProcessing/Velocity.npz` and `KineticEnergy.npz`; **nothing is
# recomputed from the `.mat`** (use `PIV_processing` to reprocess). Draws the
# **single run `RUN_DIR`** (`BATCH = False`: figures saved AND shown) or
# **every run sub-folder of every dataset in `BASE_DIRS`**
# (`BATCH = True`: figures saved, not shown).
# 
# Per run, up to four figures (each written `_DIM` = physical units and
# `_NODIM` = dimensionless with `^*` on all quantities: `f / F_SCALE`,
# velocity amplitude / `U_SCALE`, energy amplitude / `EK_SCALE`):
# 
# | figure | spectrum | from |
# |---|---|---|
# | `PLOT_FFT_velocity_DIM` / `_NODIM` | `FFT_U_ROIaveraged + FFT_V_ROIaveraged` -- the ROI-averaged total velocity amplitude, with the stored detected peaks (black diamonds) | `Velocity.npz` |
# | `PLOT_FFT_energy_DIM` / `_NODIM` | `FFT_EK_ROIaveraged` -- the ROI average of the per-point Ek spectra | `KineticEnergy.npz` |
# 
# Both carry the dashed guides at `f_rot`, `f_lib`, `2 f_lib`, `f_low` (the
# strongest detected velocity peak) and its sidebands. A run missing one of the
# two `.npz` files just skips that figure.


import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from piv_postprocessing_lib import (fft_axis_limits, fft_guide_lines,
                        figure_filename)


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the
# library). Off by default: for a viewer the feedback is the point.
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print

MUTE_PRINT = False

builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print


# ## Configuration
# 
# `RUN_DIR` (used when `BATCH = False`) may be the run folder, its
# `PostProcessing` folder, or either `.npz` in it.


# --- edit me --------------------------------------------------------------- #
# BATCH = True  -> draw the spectra for every run sub-folder of every dataset
#                  in BASE_DIRS that
#                  has a Velocity.npz or KineticEnergy.npz: figures are SAVED
#                  but not shown;
# BATCH = False -> only the single run RUN_DIR: figures saved AND shown.
BATCH = False

RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_topBottom/"
           "frot0.50Hz_flib0.400Hz_dphi2deg_SS1")  # run folder (BATCH = False)

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
        "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets swept when BATCH = True (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

LOGY = True             # log amplitude axis (False -> linear)
LOGX = False            # log frequency axis (False -> linear)
FMAX = 2.0              # frequency limit [Hz]; None -> auto

SAVE = True             # write the figures next to the .npz
OVERWRITE_FIG = True    # False -> keep existing figure files
FIG_FORMAT = "png"      # figure format: png or pdf


# ## Load the `.npz` and calibrate on the fly
# 
# The stored spectra are native (px/frame, px²/frame², 1/frame); physical
# values are the stored arrays times the stored calibration factors,
# dimensionless values divide by the stored `*_SCALE`.


def _post_dir(path):
    """The PostProcessing folder for `path` (run folder, that folder itself,
    or one of its .npz files)."""
    path = path.rstrip("/")
    if os.path.isfile(path):
        return os.path.dirname(path)
    cand = os.path.join(path, "PostProcessing")
    return cand if os.path.isdir(cand) else path


def _scalars(data, keys):
    return tuple(np.asarray(data[k]).item() for k in keys)


def load_run(path):
    """Load one run's spectra -> dict (applied to the module globals by
    process_run). SPECTRA holds one entry per available figure."""
    out_dir = _post_dir(path)
    SPECTRA = {}
    shared = None

    vel_npz = os.path.join(out_dir, "Velocity.npz")
    if os.path.isfile(vel_npz):
        d = np.load(vel_npz, allow_pickle=True)
        UCAL, FCAL, U_SCALE = _scalars(d, ("UCAL", "FCAL", "U_SCALE"))
        f_peaks = np.atleast_1d(np.asarray(d["F_PEAKS"], float)) * FCAL
        SPECTRA["velocity"] = dict(
            f=np.asarray(d["FREQ"], float) * FCAL,
            amp=(np.asarray(d["FFT_U_ROIaveraged"], float)
                 + np.asarray(d["FFT_V_ROIaveraged"], float)) * UCAL,
            scale=U_SCALE,
            label=r"$\langle|\widehat{U}|+|\widehat{V}|\rangle_{\mathrm{ROI}}$",
            unit="m/s",
            peaks_f=f_peaks,
            peaks_a=np.atleast_1d(np.asarray(d["AMP_PEAKS"], float)) * UCAL)
        shared = d

    ke_npz = os.path.join(out_dir, "KineticEnergy.npz")
    if os.path.isfile(ke_npz):
        d = np.load(ke_npz, allow_pickle=True)
        ECAL, FCAL, EK_SCALE = _scalars(d, ("ECAL", "FCAL", "EK_SCALE"))
        SPECTRA["energy"] = dict(
            f=np.asarray(d["FREQ"], float) * FCAL,
            amp=np.asarray(d["FFT_EK_ROIaveraged"], float) * ECAL,
            scale=EK_SCALE,
            label=r"$\langle|\widehat{E_k}|\rangle_{\mathrm{ROI}}$",
            unit=u"m\u00b2/s\u00b2",
            peaks_f=np.array([]), peaks_a=np.array([]))
        if shared is None:
            shared = d

    if shared is None:
        raise FileNotFoundError("no Velocity.npz or KineticEnergy.npz under "
                                "%s" % out_dir)

    run = str(np.asarray(shared["run"]).item())
    calibrated = bool(np.asarray(shared["calibrated"]).item())
    k0, top_topo, bottom_topo, frot, flib, F_SCALE = _scalars(
        shared, ("k0", "top_topo", "bottom_topo", "frot_Hz", "flib_Hz",
                 "F_SCALE"))

    # f_low = strongest detected velocity peak (guides on BOTH figures).
    _pf = SPECTRA.get("velocity", {}).get("peaks_f", np.array([]))
    f_low = float(_pf[0]) if _pf.size else np.nan

    funit = "Hz" if calibrated else "1/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)

    print("run   : %s  (%scalibrated)   spectra: %s"
          % (run, "" if calibrated else "NOT ", ", ".join(sorted(SPECTRA))))
    print("frot = %s Hz   flib = %s Hz   f_low = %s"
          % (frot, flib, "%.4g Hz" % f_low if np.isfinite(f_low) else "none"))

    return dict(out_dir=out_dir, run=run, calibrated=calibrated,
                frot=frot, flib=flib, f_low=f_low, F_SCALE=F_SCALE,
                SPECTRA=SPECTRA, funit=funit, _anno=_anno)


# ## The figure builder and the run driver


def make_fft(kind, normalize):
    """One spectrum figure (kind = 'velocity' or 'energy'). normalize=True
    divides f by F_SCALE and the amplitude by the spectrum's *_SCALE, with
    ^* on all quantities and no units."""
    sp = SPECTRA[kind]
    fscale = F_SCALE if (normalize and np.isfinite(F_SCALE) and F_SCALE) else 1.0
    ascale = (sp["scale"] if (normalize and np.isfinite(sp["scale"])
                              and sp["scale"]) else 1.0)
    ff = sp["f"] / fscale
    at = sp["amp"] / ascale
    _frot = frot / fscale if (np.isfinite(frot) and frot) else None
    _flib = flib / fscale if (np.isfinite(flib) and flib) else None
    _flow = f_low / fscale if np.isfinite(f_low) else None
    _fu = "" if normalize else " %s" % funit
    _star = lambda lbl: lbl[:-1] + r"^{\,*}$" if normalize else lbl

    xlim_max, ymin_c, ymax_c = fft_axis_limits(ff, [at], _frot, _flib)
    fmax = (FMAX / fscale) if FMAX else xlim_max
    sel = ff <= fmax
    if LOGX:
        sel &= ff > 0
    # y limits from the VISIBLE part of the curve (excluding the detrended
    # near-zero DC bin), one decade of margin.
    _vis = at[sel & (ff > 0) & (at > 0)]
    if _vis.size:
        ymin_c = 10.0 ** np.floor(np.log10(_vis.min()))
        ymax_c = 10.0 ** np.ceil(np.log10(_vis.max()))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot = ax.semilogy if LOGY else ax.plot
    plot(ff[sel], at[sel], color="C0", lw=1.2, label=_star(sp["label"]))
    ax.set_xlabel(r"$f^*$" if normalize else "frequency (%s)" % funit,
                  fontsize=13)
    ax.set_ylabel(_star(sp["label"]) if normalize
                  else "%s  (%s)" % (sp["label"], sp["unit"]), fontsize=13)
    ax.set_title("%s%s   (%s)" % (run, "   (normalized)" if normalize else "",
                                  _anno), fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    for fq, lbl, colr in fft_guide_lines(_frot, _flib, _flow):
        if fq > fmax:
            continue
        ax.axvline(fq, color=colr, ls="--", lw=1, alpha=0.85,
                   label="%s = %.4g%s" % (lbl, fq, _fu))
    # Black diamonds at the stored detected peaks, sitting on the curve.
    _dlbl = False
    for _pf in sp["peaks_f"] / fscale:
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


def process_run(path, show):
    """Load one run, draw and save its spectra (_DIM and _NODIM each); show
    them when `show` (single-run mode), close them otherwise (batch mode)."""
    globals().update(load_run(path))
    for kind in ("velocity", "energy"):
        if kind not in SPECTRA:
            print("  [skip] no %s spectrum stored" % kind)
            continue
        for _norm in (False, True):
            _fig = make_fft(kind, _norm)
            if SAVE:
                _out = figure_filename(
                    os.path.join(out_dir, "PLOT_FFT_%s" % kind),
                    FIG_FORMAT, normalized=_norm)
                if OVERWRITE_FIG or not os.path.isfile(_out):
                    _fig.savefig(_out, dpi=200, bbox_inches="tight")
                    print("Figure written to:\n  %s" % _out)
                else:
                    print("Figure exists (OVERWRITE_FIG=False), kept:\n  %s"
                          % _out)
            if show:
                plt.show()
            else:
                plt.close(_fig)


if BATCH:
    for _bd in BASE_DIRS:
        _bd = _bd.rstrip("/")
        _runs = sorted(_d for _d in glob.glob(os.path.join(_bd, "*"))
                       if os.path.isdir(_d)
                       and not os.path.basename(_d).startswith(".")
                       and (os.path.isfile(os.path.join(_d, "PostProcessing",
                                                        "Velocity.npz"))
                            or os.path.isfile(os.path.join(
                                _d, "PostProcessing", "KineticEnergy.npz"))))
        print("=== dataset %s: %d runs with spectra ==="
              % (os.path.basename(_bd), len(_runs)))
        for _rd in _runs:
            try:
                process_run(_rd, show=False)
            except Exception as exc:          # keep the batch going
                print("  [error] %s: %s" % (os.path.basename(_rd), exc))
            print()
else:
    process_run(RUN_DIR, show=True)
