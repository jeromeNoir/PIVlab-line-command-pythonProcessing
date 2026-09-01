"""Auto-generated .py twin of plot_batch_Velocity.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Replot batch_Velocity figures from the `.npz` (no reprocessing)
# 
# Reproduces **exactly** the per-run figures that `batch_Velocity` writes -- the
# spectrum figure `VelocityFFT_<region>[_normalized].<fmt>` and the polarization
# figure `Velocity_polarization_<region>[_normalized].<fmt>` -- but **without
# re-reading the PIV `.mat`**. It loops over every sub-folder of `BASE_DIR` and
# redraws from each run's cached `PostProcessing/VelocityFFT_<region>.npz`, which
# already stores everything the figures need (`f`, `amp_u`, `amp_v`, `amp_total`,
# `f_peak`, `f_low`, and the polarization arrays).
# 
# It reuses `batch_Velocity`'s `_save_fft_figure` and `_save_polarization_v2_figure`
# **verbatim**, so the output is byte-identical to the batch's figures. Only the
# plotting options matter here (`FIG_FORMAT`, `LOGX`, `LOGY`); no FFT is recomputed
# and no summary/`.npz` is written. Use `batch_Velocity` (or
# `process_single_Velocity`) if you actually need to recompute from the `.mat`.
# 
# Set the config, then *Run All*. `ONLY_RUNS` restricts to a few folders;
# `OVERWRITE = False` skips a run whose figures already exist.


# ## 1. Imports


import os
import glob
import numpy as np
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, fft_axis_limits,
                        fft_guide_lines, figure_filename,
                        libration_velocity_scale, parse_run_name,
                        read_paramPostprocessing, region_tag, regions_to_run)


# ## 2. Configuration


# --- Mute switch -----------------------------------------------------------
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = True
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# --- Configuration --------------------------------------------------------- #
BASE_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom")

# k0 / topography for the figure titles come from the dataset's parameter file
# (also rebinds the library globals, matching batch_Velocity).
_P = read_paramPostprocessing(BASE_DIR)
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(BASE_DIR)

# Per-run stems (region tag appended) -- identical to batch_Velocity.
RESULT_STEM = "VelocityFFT"            # per-run npz read from PostProcessing/
POLFIG_STEM = "Velocity_polarization"  # per-run polarization figure

# Which region(s): 'BOTH' (ROI and FULL), 'ROI' only, or 'FULL' only.
REGION = 'ROI'

# Saved-figure format and axis scales -- must match how you want the figures.
FIG_FORMAT = 'png'
LOGX = False
LOGY = True

# Redraw even if a figure already exists (False -> skip those runs).
OVERWRITE = True

# --- Spectrum-figure axis limits ------------------------------------------ #
# Each is a (min, max) tuple, or None -> auto (exactly as batch_Velocity).
# They apply to the SPECTRUM figure only (the polarization figure is unchanged).
# Physical (raw figure): x = frequency [Hz], y = amplitude [m/s].
XLIM_PHYS = None        # e.g. (0.0, 2.0)   Hz
YLIM_PHYS = None        # e.g. (1e-5, 1e-1) m/s   (log axis when LOGY)
# Normalised figure: x = f / f_rot, y = amplitude / U0.
XLIM_NORM = (0.0, 4.0)        # e.g. (0.0, 4.0)
YLIM_NORM = (1e-4, 1e0)        # e.g. (1e-3, 1e1)

# Restrict to these run-folder names (for testing). Empty -> all runs.
ONLY_RUNS = []


# ## 3. Figure functions (verbatim from batch_Velocity)


def _save_fft_figure(out_png, f, amp_u, amp_v, amp_total, name, calibrated,
                     f_peak, frot=None, flib=None, f_low=None, dphi_deg=None,
                     normalize=False, xlim=None, ylim=None):
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
    # Optional user-set axis limits override the auto ones (None -> keep auto).
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
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


# ## 4. Redraw every run from its `.npz`


def redraw_run(run_dir, region):
    """Redraw one run's spectrum + polarization figures from its cached .npz."""
    tag = region_tag(region)
    name = os.path.basename(run_dir.rstrip("/"))
    out_dir = os.path.join(run_dir, "PostProcessing")
    npz = os.path.join(out_dir, "%s_%s.npz" % (RESULT_STEM, tag))
    if not os.path.isfile(npz):
        print("  [skip] %-40s %-4s no %s" % (name, tag, os.path.basename(npz)))
        return False

    d = np.load(npz, allow_pickle=True)
    _g = lambda k, dflt=np.nan: (d[k] if k in d.files else dflt)
    f = np.asarray(d["f"], float)
    amp_u = np.asarray(d["amp_u"], float)
    amp_v = np.asarray(d["amp_v"], float)
    amp_total = np.asarray(d["amp_total"], float)
    calibrated = bool(d["calibrated"]) if "calibrated" in d.files else True
    f_peak = float(_g("f_peak"))
    f_low = float(_g("f_low"))
    frot, flib, dphi = parse_run_name(name)
    frot = None if not np.isfinite(frot) else frot
    flib = None if not np.isfinite(flib) else flib
    # Polarization arrays used by _save_polarization_v2_figure.
    pol = {k: np.asarray(d[k], float) for k in
           ("powerRatio", "powerRatio_med", "powerRatio_p25", "powerRatio_p75")
           if k in d.files}

    fig_stem = os.path.splitext(npz)[0]                 # -> VelocityFFT_<tag>
    pol_stem = os.path.join(out_dir, "%s_%s" % (POLFIG_STEM, tag))
    drew = False
    for _norm in (False, True):
        _spec = figure_filename(fig_stem, FIG_FORMAT, normalized=_norm)
        _polf = figure_filename(pol_stem, FIG_FORMAT, normalized=_norm)
        _xl = XLIM_NORM if _norm else XLIM_PHYS
        _yl = YLIM_NORM if _norm else YLIM_PHYS
        if OVERWRITE or not os.path.isfile(_spec):
            _save_fft_figure(_spec, f, amp_u, amp_v, amp_total, name, calibrated,
                             f_peak, frot, flib, f_low, dphi, normalize=_norm,
                             xlim=_xl, ylim=_yl)
            drew = True
        if pol and (OVERWRITE or not os.path.isfile(_polf)):
            _save_polarization_v2_figure(_polf, f, pol, parse_run_name(name)[0],
                                         name, normalize=_norm)
            drew = True
    print("  [%s] %-40s %-4s" % ("ok" if drew else "skip (exists)", name, tag))
    return drew


def run_batch():
    """Redraw every run under BASE_DIR for each requested region."""
    subdirs = sorted(dd for dd in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(dd)
                     and not os.path.basename(dd).startswith("."))
    if ONLY_RUNS:
        wanted = set(ONLY_RUNS)
        subdirs = [dd for dd in subdirs if os.path.basename(dd) in wanted]
    print("Found %d subfolders in %s" % (len(subdirs), BASE_DIR))
    print("spectrum axis limits  phys: x=%s y=%s | norm: x=%s y=%s"
          % (XLIM_PHYS, YLIM_PHYS, XLIM_NORM, YLIM_NORM))
    n = 0
    for region in regions_to_run(REGION):
        tag = region_tag(region)
        print("\n========== REGION: %s ==========" % tag)
        for run_dir in subdirs:
            try:
                n += bool(redraw_run(run_dir, region))
            except Exception as exc:
                print("  [error] %s: %s" % (os.path.basename(run_dir), exc))
    print("\n%d run(s) redrawn." % n)
    return n


_ = run_batch()
