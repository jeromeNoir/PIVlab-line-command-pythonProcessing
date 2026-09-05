"""Auto-generated .py twin of MAPS_VELOCITY.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # MAPS_VELOCITY -- mean / std velocity maps
# 
# Viewer of the per-point **time mean and std velocity maps** stored in each
# run's `PostProcessing/Velocity.npz`; it never touches the PIV `.mat` (use
# `PIV_processing` to recompute). Draws the **single run `RUN_DIR`**
# (`BATCH = False`: figures saved AND shown) or **every run sub-folder of every
# dataset in `BASE_DIRS`** that has a `Velocity.npz` (`BATCH = True`: figures
# saved, not shown).
# 
# One figure per run (written twice: `_DIM` = physical units, `_NODIM` =
# dimensionless, starred quantities: amplitudes / `U_0`, lengths / `R`):
# `MAP_COMPONENT = 'U'` or `'V'` gives a 2-panel figure (mean + std of that
# component), `'both'` a 4-panel one. Panel titles: $\overline{U}$ /
# $\overline{V}$ for the mean, $U_{\mathrm{std}}$ / $V_{\mathrm{std}}$ for the
# std. The ROI rectangle is overlaid.
# 
# The maps are drawn as stored.
# 
# The dashed +/- theta inertial-wave characteristics
# (`sin(theta) = f/(2 f_rot)`, theta from the z-axis) are overlaid on the
# **std panels only** -- traced at `f_lib`, or at `FREQ` when it is set.
# 
# The FFT plots live in their own viewers (`MAPS_FFT`, `SINGLE_polarization`);
# the detected peaks and dimensionless numbers are still PRINTED per run.


import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from piv_postprocessing_lib import (dimensionless_numbers, figure_filename,
                        peak_freq, read_paramPostprocessing, resolve_npz)


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
# `RUN_DIR` (used when `BATCH = False`) may be the `Velocity.npz` itself,
# the run folder, or its `PostProcessing` folder.


# --- edit me --------------------------------------------------------------- #
# BATCH = True  -> draw the maps for every run sub-folder of every dataset in
#                  BASE_DIRS that
#                  has a Velocity.npz: the figures are SAVED but not shown;
# BATCH = False -> only the single run RUN_DIR: figures saved AND shown.
BATCH = True

RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_topBottom/"
           "frot0.50Hz_flib0.400Hz_dphi2deg_SS1")  # run folder or Velocity.npz
                                                   # (BATCH = False)

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
        "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets swept when BATCH = True (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

REGION = "ROI"          # 'ROI' or 'FULL': which region the peak printout uses

# Which component(s) to draw: 'U' or 'V' -> a 2-panel figure,
# 'both' -> a 4-panel one.
MAP_COMPONENT = "both"

# --- Inertial-wave characteristic lines -------------------------------------
# Dashed +/- theta lines, sin(theta) = f/(2 f_rot), theta from the z-axis.
# FREQ = None -> traced at f_lib; a number [Hz] -> traced at that frequency.
FREQ = None               # None -> use f_lib; a number -> use that frequency
N_LINES = 4               # lines per angle, equally spaced across the domain
SCALING_WIDTH_LINE = 0.7  # line-width scaling for the +/- theta lines

SAVE = True             # write the figures next to the .npz
OVERWRITE_FIG = True    # False -> keep existing figure files
FIG_FORMAT = "png"      # 'png' or 'pdf'; every figure is written twice:
                        # _DIM (physical units) and _NODIM (dimensionless)


# ## Load the `.npz` and calibrate on the fly
# 
# The stored arrays are native (px, px/frame); physical values are the stored
# arrays times the stored calibration factors, dimensionless values divide by
# the stored `*_SCALE`.


def load_run(path):
    """Load one run's Velocity.npz and derive everything the viewer needs.

    Returns a dict -- applied to the module globals by process_run -- with the
    calibrated maps and grid (stored axes), the peak/spectrum info for the
    printout and the run parameters.
    """
    npz_path = resolve_npz(path, "Velocity.npz")
    print("Reading %s" % npz_path)
    data = np.load(npz_path, allow_pickle=True)

    if REGION not in ("ROI", "FULL"):
        raise ValueError("REGION must be 'ROI' or 'FULL', got %r" % (REGION,))

    _g = lambda k: np.asarray(data[k])
    _s = lambda k: np.asarray(data[k]).item()

    run = str(_s("run"))
    calibrated = bool(_s("calibrated"))
    k0, top_topo, bottom_topo = _s("k0"), _s("top_topo"), _s("bottom_topo")
    frot, flib, dphi = _s("frot_Hz"), _s("flib_Hz"), _s("dphi_deg")
    nframes, npoints = int(_s("nframes")), int(_s("npoints_%s" % REGION))

    # Calibration factors (native -> physical), non-dim scales (physical -> *).
    XCAL, YCAL, UCAL, VCAL, FCAL = (_s(k) for k in
                                    ("XCAL", "YCAL", "UCAL", "VCAL", "FCAL"))
    U_SCALE, F_SCALE, LENGTH_SCALE = (_s(k) for k in
                                      ("U_SCALE", "F_SCALE", "LENGTH_SCALE"))

    # Region-averaged total spectrum and stored peaks (for the printout).
    f = _g("FREQ").astype(float) * FCAL
    amp_total = _g("FFT_TOTAL_%saveraged" % REGION).astype(float) * UCAL
    f_peaks = np.atleast_1d(_g("F_PEAKS").astype(float)) * FCAL
    amp_peaks = np.atleast_1d(_g("AMP_PEAKS").astype(float)) * UCAL
    f_low = float(f_peaks[0]) if f_peaks.size else np.nan

    # Maps: grid and per-point time statistics, physical units, STORED axes
    # (the display orientation is resolved in make_maps).
    Xm = _g("X").astype(float) * XCAL              # stored image x [m]
    Zm = _g("Y").astype(float) * YCAL              # stored image y [m]
    Umean = _g("Umean").astype(float) * UCAL
    Ustd = _g("Ustd").astype(float) * UCAL
    Vmean = _g("Vmean").astype(float) * VCAL
    Vstd = _g("Vstd").astype(float) * VCAL
    pts_ROI = np.asarray(_g("pts_ROI"), dtype=float)   # ROI corners [m]

    # Read the dataset's param_postProcessing.json to rebind the library
    # globals (R, H, l, nu) that dimensionless_numbers uses.
    _base = os.path.dirname(os.path.dirname(os.path.dirname(npz_path)))
    read_paramPostprocessing(_base)

    funit = "Hz" if calibrated else "1/frame"
    aunit = "m/s" if calibrated else "px/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)

    print("run     = %s  (%s, %scalibrated)"
          % (run, REGION, "" if calibrated else "NOT "))
    print("fps     = %g Hz   nframes = %d   points (%s) = %d"
          % (_s("PIV_FPS"), nframes, REGION, npoints))
    print("frot = %s Hz   flib = %s Hz   dphi = %s deg" % (frot, flib, dphi))

    return dict(npz_path=npz_path, out_dir=os.path.dirname(npz_path), run=run,
                calibrated=calibrated, k0=k0, top_topo=top_topo,
                bottom_topo=bottom_topo, frot=frot, flib=flib, dphi=dphi,
                U_SCALE=U_SCALE, F_SCALE=F_SCALE, LENGTH_SCALE=LENGTH_SCALE,
                f=f, amp_total=amp_total, f_peaks=f_peaks,
                amp_peaks=amp_peaks, f_low=f_low,
                Xm=Xm, Zm=Zm, Umean=Umean, Ustd=Ustd, Vmean=Vmean, Vstd=Vstd,
                pts_ROI=pts_ROI,
                funit=funit, aunit=aunit, _anno=_anno)


# ## Detected peaks and dimensionless numbers (printed)


def print_summary():
    """Detected peaks + dimensionless numbers of the loaded run."""
    f_peak, a_peak = peak_freq(f, amp_total)
    print("spectrum maximum (excl. DC): f_peak = %.4f %s  (%.3e %s)"
          % (f_peak, funit, a_peak, aunit))
    _fn = frot if (np.isfinite(frot) and frot) else np.nan
    if f_peaks.size:
        print("detected peaks (band F_LOW_MIN .. flib - DELTA_F, strongest "
              "first):")
        for _i, (_fp, _ap) in enumerate(zip(f_peaks, amp_peaks), 1):
            print("  %2d.  f = %.4f %s   f* = %.4f   amp = %.3e %s   amp/U0"
                  " = %.3e" % (_i, _fp, funit, _fp / _fn, _ap, aunit,
                               _ap / U_SCALE))
        print("f_low (strongest detected peak) = %.4f %s" % (f_low, funit))
    else:
        print("no detected peaks stored (uncalibrated run, or nothing "
              "significant)")
    if np.isfinite(flib) and flib:
        print("references: flib = %.4f, 2*flib = %.4f %s"
              % (flib, 2 * flib, funit))
        if np.isfinite(f_low):
            print("sidebands: flib-f_low = %.4f, flib+f_low = %.4f %s"
                  % (flib - f_low, flib + f_low, funit))
    _dl = dimensionless_numbers(frot, flib, dphi)
    print("dimensionless: E=%.3e  E_l=%.3e  Ro=%.3g  Re=%.4g  Re_l=%.4g  "
          "Re_bl=%.3g  U0=%.4g m/s"
          % (_dl["E"], _dl["E_l"], _dl["Ro"], _dl["Re"], _dl["Re_l"],
             _dl["Re_bl"], _dl["U0_mps"]))


# ## Mean / std velocity maps
# 
# Per-point time statistics from the `.npz` (`Umean`, `Ustd`, `Vmean`, `Vstd`),
# on the physical grid, ROI rectangle overlaid. Mean maps use a symmetric diverging colormap
# about 0; std maps a sequential one.


def _angle_lines(axm, f_hz, lab, color="w", legend=False):
    """Dashed +/- theta inertial-wave characteristics on map axis `axm`,
    with sin(theta) = f_hz / (2 f_rot), theta measured from the z-axis (the
    rotation axis). N_LINES parallel lines per sign, equally spaced across
    the domain (N_LINES=1 -> a single centred pair)."""
    if not (np.isfinite(frot) and frot and f_hz is not None
            and np.isfinite(f_hz)):
        return
    r = abs(f_hz) / (2.0 * frot)
    if r > 1.0:
        print("  (f=%g Hz: f/(2 f_rot)=%.3g > 1 -- no real angle)" % (f_hz, r))
        return
    xl, zl = axm.get_xlim(), axm.get_ylim()
    xc = 0.5 * (xl[0] + xl[1]); zc = 0.5 * (zl[0] + zl[1])
    n = max(1, int(N_LINES))
    span, sc = np.array(zl), zc            # lines run across z, spaced along x
    a0s = np.array([xc]) if n == 1 else np.linspace(xl[0], xl[1], n)
    slope = np.tan(np.arcsin(r))
    lw = 1.5 * float(SCALING_WIDTH_LINE)
    labeled = False
    for a0 in a0s:
        for sgn in (+1.0, -1.0):
            off = a0 + sgn * slope * (span - sc)
            axm.plot(off, span, "--", color=color, lw=lw,
                     label=(None if (labeled or not legend) else lab))
            labeled = True
    axm.set_xlim(xl); axm.set_ylim(zl)     # keep the map extent
    if legend:
        leg = axm.legend(loc="upper right", fontsize=7)
        leg.get_frame().set_facecolor("0.25"); leg.get_frame().set_alpha(0.7)
        for _t in leg.get_texts():
            _t.set_color("w")


def _map_panels():
    """[(title, 2-D map, is_mean)] per MAP_COMPONENT."""
    comp = str(MAP_COMPONENT).lower()
    if comp not in ("u", "v", "both"):
        raise ValueError("MAP_COMPONENT must be 'U', 'V' or 'both', got %r"
                         % (MAP_COMPONENT,))
    disp = {"u": (Umean, Ustd), "v": (Vmean, Vstd)}
    panels = []
    for c in ("u", "v"):
        if comp in (c, "both"):
            m, s = disp[c]
            C = c.upper()
            panels += [(r"$\overline{%s}$" % C, m, True),
                       (r"$%s_{\mathrm{std}}$" % C, s, False)]
    return panels


def make_maps(normalize):
    """The mean/std maps figure. normalize=True divides lengths by
    LENGTH_SCALE and velocities by U_0 (= U_SCALE)."""
    lscale = (LENGTH_SCALE if (normalize and np.isfinite(LENGTH_SCALE)
                               and LENGTH_SCALE) else 1.0)
    ascale = U_SCALE if (normalize and np.isfinite(U_SCALE) and U_SCALE) else 1.0
    # Characteristics frequency: FREQ overrides f_lib when set.
    _fchar = flib if FREQ is None else float(FREQ)
    if normalize and np.isfinite(F_SCALE) and F_SCALE:
        _alab = ((r"$f^*_{\mathrm{lib}}$" if FREQ is None else r"$f^*$")
                 + " = %.2g" % (_fchar / F_SCALE))
    else:
        _alab = r"%g Hz ($f/f_{\mathrm{rot}}$=%.2g)" % (_fchar, _fchar / frot)
    Xp, Zp = Xm / lscale, Zm / lscale
    roi_x, roi_z = pts_ROI[:, 0], pts_ROI[:, 1]
    # NODIM: starred quantities (x*, U*, ...) with no unit; DIM: units.
    if normalize:
        xlab, zlab = r"$x^*$", r"$z^*$"
        vunit = ""
        _star = lambda lbl: lbl[:-1] + r"^{\,*}$"   # $\overline{U}$ -> $\overline{U}^{\,*}$
    else:
        _u = " (m)" if calibrated else " (px)"
        xlab, zlab = "x" + _u, "z" + _u
        vunit = "  (%s)" % aunit
        _star = lambda lbl: lbl

    _alegended = False              # characteristics legend once, first std panel
    panels = _map_panels()
    n = len(panels)
    ncols = 2
    nrows = n // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (lbl, Z, is_mean) in zip(axes.ravel(), panels):
        Z = np.asarray(Z, dtype=float) / ascale
        if is_mean:                      # symmetric about 0, robust limit
            vmax = np.nanpercentile(np.abs(Z), 99.0)
            pcm = ax.pcolormesh(Xp, Zp, Z, shading="auto", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
        else:
            vmax = np.nanpercentile(Z, 99.0)
            pcm = ax.pcolormesh(Xp, Zp, Z, shading="auto", cmap="viridis",
                                vmin=0.0, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=_star(lbl) + vunit)
        ax.set_title(_star(lbl), fontsize=12)
        ax.set_aspect("equal")
        # ROI rectangle (pts_ROI is stored in metres, stored axes).
        if calibrated:
            xs = np.sort(np.asarray(roi_x, dtype=float)) / lscale
            zs = np.sort(np.asarray(roi_z, dtype=float)) / lscale
            ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]],
                    [zs[0], zs[0], zs[1], zs[1], zs[0]],
                    "k--", lw=1.2, alpha=0.8)
        # +/- theta inertial-wave characteristics (FREQ or f_lib), on the
        # std panels only -- the mean maps stay clean.
        if not is_mean:
            _angle_lines(ax, _fchar, _alab, color="w",
                         legend=not _alegended)
            _alegended = True
    for ax in axes[-1, :]:
        ax.set_xlabel(xlab, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(zlab, fontsize=12)
    fig.suptitle("%s%s   (%s)" % (run,
                                  "   (non-dimensional)" if normalize else "",
                                  _anno), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def process_run(path, show):
    """Load one run, print its summary, draw and save the maps (_DIM and
    _NODIM); show them when `show` (single-run mode), close them otherwise."""
    globals().update(load_run(path))
    print_summary()
    _stem = os.path.join(out_dir, "MAPS_VELOCITY")
    for _norm in (False, True):
        _fig = make_maps(_norm)
        if SAVE:
            _out = figure_filename(_stem, FIG_FORMAT, normalized=_norm)
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
                       and os.path.isfile(os.path.join(_d, "PostProcessing",
                                                       "Velocity.npz")))
        print("=== dataset %s: %d runs with a Velocity.npz ==="
              % (os.path.basename(_bd), len(_runs)))
        for _rd in _runs:
            try:
                process_run(_rd, show=False)
            except Exception as exc:          # keep the batch going
                print("  [error] %s: %s" % (os.path.basename(_rd), exc))
            print()
else:
    process_run(RUN_DIR, show=True)
