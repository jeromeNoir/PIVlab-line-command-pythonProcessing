"""Auto-generated .py twin of MAPS_FFT.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # FFT maps -- viewer
# 
# Draws the maps for the **single run `RUN_DIR`** (`BATCH = False`: the figures
# are saved AND shown) or for **every run sub-folder of every dataset in
# `BASE_DIRS`** that has a `Velocity.npz` (`BATCH = True`: the figures are
# saved but not shown).
# Successor of `colormaps_single_FFTVelocity`: it **reads** one per-run
# `PostProcessing/Velocity.npz` -- the per-point spectra `FFT_U` / `FFT_V` are
# already computed and stored there -- and never touches the PIV `.mat` (use
# `PIV_processing` with `BATCH = False` to recompute). Everything (grid, spectra, calibration
# factors, non-dim scales, run parameters) comes from the `.npz`.
# 
# Two figures are written (`_DIM` = physical units, `_NODIM` = dimensionless:
# lengths / `LENGTH_SCALE`, amplitudes / `U_SCALE`, frequencies / `F_SCALE`),
# each with two panels of `|FFT(U)| + |FFT(V)|` per grid point:
# 
# 1. the **amplitude at `f_lib`** (max within +/- `HALFWIDTH_BINS` bins), with
#    the dashed +/- theta inertial-wave characteristics
#    (`sin(theta) = f / (2 f_rot)`, traced at `f_lib`, or at `FREQ` when set);
# 2. the **integral over the band `[FMIN, FMAX]` normalised by `FMAX - FMIN`**
#    (the mean amplitude each point carries in the band) -- no characteristics
#    on this one.
# 
# The band is set by `PEAK_SELECT`: `True` uses the `FMIN` / `FMAX` values from
# the configuration, `False` uses `FMIN = DELTA_F`, `FMAX = f_lib - DELTA_F`.
# 
# The `pts_ROI` rectangle (stored in the `.npz`, in metres) is overlaid dashed
# on every map. Each panel carries the marginal profile of the map averaged along the
# coordinate perpendicular to the vertical: the +/- theta characteristics are
# measured from `z` (the rotation axis) and the marginal (mean over `x`,
# versus `z`) sits on the RIGHT of the map.


import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from piv_postprocessing_lib import figure_filename, resolve_npz


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the
# library). Off by default: for a single-run viewer the feedback is the point.
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
           "frot0.50Hz_flib0.430Hz_dphi2deg_SS1")  # run folder or Velocity.npz
                                                   # (BATCH = False)

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
        "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets swept when BATCH = True (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

# --- Frequency band of the integral map [Hz] --------------------------------
#   PEAK_SELECT = True  -> the band is [FMIN, FMAX] as set below;
#   PEAK_SELECT = False -> the band is [DELTA_F, flib - DELTA_F].
PEAK_SELECT = False
FMIN = 0.288             # Hz, used only when PEAK_SELECT = True
FMAX = 0.345             # Hz, used only when PEAK_SELECT = True
DELTA_F = 0.05           # Hz, sets the band when PEAK_SELECT = False

# --- Inertial-wave characteristic lines (panel 1 only) ----------------------
# FREQ = None -> the characteristics are traced at f_lib; a number [Hz]
# traces them at that frequency instead.
FREQ = None               # None -> use f_lib; a number -> use that frequency
N_LINES = 4               # lines per angle, equally spaced across the domain
SCALING_WIDTH_LINE = 0.7  # line-width scaling for the +/- theta lines

# --- Options ----------------------------------------------------------------
HALFWIDTH_BINS = 2       # peak-at-flib map = max over +/- this many FFT bins
CMAP = "viridis"
LOGC = False             # log colour scale on the maps (False -> linear)
FIG_FORMAT = "png"       # 'png' or 'pdf'
SAVE = True              # write the figures next to the .npz


# ## Load the `.npz` and calibrate on the fly
# 
# The stored arrays are native (px, px/frame, 1/frame); physical values are the
# stored arrays times the stored calibration factors, dimensionless values
# divide by the stored `*_SCALE`.


def load_run(path):
    """Load one run's Velocity.npz and derive everything the maps need.

    Returns a dict -- applied to the module globals by process_run -- with the
    grid, the calibrated per-point total spectrum, the run parameters, the
    frequency band.
    """
    npz_path = resolve_npz(path, "Velocity.npz")
    print("Reading %s" % npz_path)
    data = np.load(npz_path, allow_pickle=True)

    _g = lambda k: np.asarray(data[k])
    _s = lambda k: np.asarray(data[k]).item()

    run = str(_s("run"))
    calibrated = bool(_s("calibrated"))
    k0, top_topo, bottom_topo = _s("k0"), _s("top_topo"), _s("bottom_topo")
    frot, flib, dphi = _s("frot_Hz"), _s("flib_Hz"), _s("dphi_deg")

    # Calibration factors (native -> physical) and non-dim scales (physical -> *).
    XCAL, YCAL, UCAL, VCAL, FCAL = (_s(k) for k in
                                    ("XCAL", "YCAL", "UCAL", "VCAL", "FCAL"))
    U_SCALE, F_SCALE, LENGTH_SCALE = (_s(k) for k in
                                      ("U_SCALE", "F_SCALE", "LENGTH_SCALE"))

    # Grid and per-point total spectrum, calibrated to physical units.
    x = _g("X").astype(float) * XCAL                         # m
    z = _g("Y").astype(float) * YCAL                         # m
    freq = _g("FREQ").astype(float) * FCAL                   # Hz
    amp_total = (_g("FFT_U").astype(float) * UCAL
                 + _g("FFT_V").astype(float) * VCAL)         # (ny, nx, nf), m/s
    pts_ROI = np.asarray(_g("pts_ROI"), dtype=float)         # ROI corners [m]

    if not (np.isfinite(flib) and flib > 0):
        raise ValueError("flib unknown from the run name -- cannot place the "
                         "f_lib map or the default band")

    # Frequency band of the integral map.
    if PEAK_SELECT:
        fmin, fmax = float(FMIN), float(FMAX)
    else:
        fmin, fmax = float(DELTA_F), float(flib) - float(DELTA_F)
    _band = (freq >= fmin) & (freq <= fmax)
    if not (fmax > fmin) or _band.sum() < 2:
        raise ValueError("band [%.4g, %.4g] Hz invalid (need FMAX > FMIN and at "
                         "least 2 FFT bins)" % (fmin, fmax))

    funit = "Hz" if calibrated else "1/frame"
    aunit = "m/s" if calibrated else "px/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)

    print("run   : %s  (%scalibrated)   grid %d x %d   %d frequency bins"
          % (run, "" if calibrated else "NOT ", x.shape[0], x.shape[1], freq.size))
    print("flib  : %g %s   df = %.4g %s   Nyquist = %.4g %s"
          % (flib, funit, freq[1] - freq[0], funit, freq[-1], funit))
    print("band  : [%.4g, %.4g] %s  (%s)"
          % (fmin, fmax, funit,
             "PEAK_SELECT" if PEAK_SELECT else "DELTA_F .. flib - DELTA_F"))

    return dict(npz_path=npz_path, out_dir=os.path.dirname(npz_path), run=run,
                calibrated=calibrated, k0=k0, top_topo=top_topo,
                bottom_topo=bottom_topo, frot=frot, flib=flib, dphi=dphi,
                U_SCALE=U_SCALE, F_SCALE=F_SCALE, LENGTH_SCALE=LENGTH_SCALE,
                x=x, z=z, freq=freq, amp_total=amp_total, pts_ROI=pts_ROI,
                fmin=fmin, fmax=fmax, _band=_band,
                funit=funit, aunit=aunit, _anno=_anno)


# ## The two maps
# 
# 1. amplitude at `f_lib`: max of `|FFT(U)|+|FFT(V)|` within
#    +/- `HALFWIDTH_BINS` bins of `f_lib`;
# 2. band average: integral over `[FMIN, FMAX]` / `(FMAX - FMIN)`.


_trapz = getattr(np, "trapezoid", np.trapz)   # np.trapz deprecated in numpy>=2


def compute_maps():
    """The two maps of the loaded run (module globals set by load_run)."""
    # (1) Amplitude at f_lib (max over +/- HALFWIDTH_BINS bins around it).
    _k = int(np.argmin(np.abs(freq - flib)))
    _lo = max(0, _k - HALFWIDTH_BINS)
    _hi = min(freq.size, _k + HALFWIDTH_BINS + 1)
    peak_flib = np.max(amp_total[:, :, _lo:_hi], axis=2)  # NaN where all-NaN

    # (2) Integral over [fmin, fmax] normalised by the bandwidth.
    bandavg = (_trapz(amp_total[:, :, _band], freq[_band], axis=2)
               / (fmax - fmin))
    return dict(peak_flib=peak_flib, bandavg=bandavg)


# ## Figure builders and the two figures
# 
# One figure per unit system (`_DIM`, `_NODIM`), two panels each, every panel
# with its marginal profile (mean along the coordinate perpendicular to the
# reference direction) and colorbar.


def _stack_profile(fld, xs, zs):
    """(coord, profile): `fld` averaged over x, versus z."""
    return np.nanmean(zs, axis=1), np.nanmean(fld, axis=1)


def _angle_lines(axm, f_hz, lab):
    """Dashed +/- theta inertial-wave characteristics on map axis `axm`,
    with sin(theta) = f_hz / (2 f_rot), theta measured from the z-axis (the
    rotation axis) and `lab` as the legend label. N_LINES parallel lines per
    sign, equally spaced across the domain (N_LINES=1 -> a centred pair)."""
    if not (np.isfinite(frot) and frot and np.isfinite(f_hz)):
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
            axm.plot(off, span, "--", color="w", lw=lw,
                     label=(None if labeled else lab))
            labeled = True
    axm.set_xlim(xl); axm.set_ylim(zl)          # keep the map extent
    leg = axm.legend(loc="upper right", fontsize=7)
    leg.get_frame().set_facecolor("0.25"); leg.get_frame().set_alpha(0.7)
    for _t in leg.get_texts():
        _t.set_color("w")


def _panel(fig, axm, xs, zs, fld, title, cbar_label, xlab, zlab, plab,
           angle_freq=None, angle_lab=None, roi=None):
    """One map panel: pcolormesh + ROI rectangle + optional characteristics +
    marginal + colorbar, appended with make_axes_locatable so they match the
    map's box. All labels arrive ready-made (make_maps builds the DIM and
    starred NODIM variants). `roi` is the ((x, z), (x, z)) corner pair,
    already scaled."""
    finite = fld[np.isfinite(fld)]
    if LOGC:
        pos = finite[finite > 0]
        vmin = np.nanpercentile(pos, 5) if pos.size else None
        vmax = np.nanpercentile(finite, 99.5)
        norm = LogNorm(vmin=vmin, vmax=vmax) if (vmin and vmin > 0) else None
        vkw = {} if norm else {"vmin": 0.0, "vmax": vmax}
    else:
        norm = None
        vkw = {"vmin": 0.0, "vmax": float(np.nanpercentile(finite, 99))}
    pcm = axm.pcolormesh(xs, zs, np.ma.masked_invalid(fld), cmap=CMAP,
                         shading="nearest", norm=norm, **vkw)
    axm.set_aspect("equal")
    axm.set_title(title, fontsize=10)
    axm.set_xlabel(xlab)
    axm.set_ylabel(zlab)
    if roi is not None:
        (_xa, _za), (_xb, _zb) = roi
        _rx = sorted((_xa, _xb)); _rz = sorted((_za, _zb))
        axm.plot([_rx[0], _rx[1], _rx[1], _rx[0], _rx[0]],
                 [_rz[0], _rz[0], _rz[1], _rz[1], _rz[0]],
                 "r--", lw=1.2, alpha=0.9)
    if angle_freq is not None:
        _angle_lines(axm, angle_freq, angle_lab)

    coord, prof = _stack_profile(fld, xs, zs)
    div = make_axes_locatable(axm)
    axp = div.append_axes("right", size="26%", pad=0.15, sharey=axm)
    axp.plot(prof, coord, color="C3", lw=1.3)
    axp.set_xlabel(plab)
    axp.tick_params(labelleft=False)              # z labels stay on the map
    axp.grid(True, alpha=0.3)
    cax = div.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(pcm, cax=cax).set_label(cbar_label)


def make_maps(normalize):
    """The two-panel figure: amplitude at f_lib (with characteristics) and the
    band average (without). normalize=True divides lengths by LENGTH_SCALE,
    amplitudes by U_SCALE, frequencies by F_SCALE, and simply stars every
    label (x*, z*, f*, amplitude*) with no units or /SCALE suffixes."""
    lscale = (LENGTH_SCALE if (normalize and np.isfinite(LENGTH_SCALE)
                               and LENGTH_SCALE) else 1.0)
    ascale = U_SCALE if (normalize and np.isfinite(U_SCALE) and U_SCALE) else 1.0
    fscale = F_SCALE if (normalize and np.isfinite(F_SCALE) and F_SCALE) else 1.0
    # Characteristics frequency: FREQ overrides f_lib when set.
    _fchar = flib if FREQ is None else float(FREQ)
    xs, zs = x / lscale, z / lscale
    # NODIM: every quantity simply starred (x*, z*, f*, amplitude*), no units
    # and no /U_SCALE-style suffixes; DIM: physical units.
    if normalize:
        xlab, zlab = r"$x^*$", r"$z^*$"
        p1_title = (r"$(|\widehat{U}|+|\widehat{V}|)^{\,*}$ at "
                    r"$f^*_{\mathrm{lib}}$ = %.4g" % (flib / fscale))
        p1_cbar = r"$(|\widehat{U}|+|\widehat{V}|)^{\,*}$"
        p2_title = (r"$\langle|\widehat{U}|+|\widehat{V}|\rangle^{\,*}"
                    r"_{[%.3g,\,%.3g]}$" % (fmin / fscale, fmax / fscale))
        p2_cbar = r"$\langle|\widehat{U}|+|\widehat{V}|\rangle^{\,*}$"
        plab = r"$\langle\cdot\rangle_x^{\,*}$"
        angle_lab = ((r"$f^*_{\mathrm{lib}}$" if FREQ is None else r"$f^*$")
                     + " = %.2g" % (_fchar / fscale))
    else:
        _lu = "[m]" if calibrated else "[px]"
        _fum = r"\mathrm{Hz}" if calibrated else r"\mathrm{1/frame}"
        xlab, zlab = "x %s" % _lu, "z %s" % _lu
        p1_title = (r"$|\widehat{U}|+|\widehat{V}|$ at $f_{\mathrm{lib}}$"
                    " = %.4g %s" % (flib, funit))
        p1_cbar = r"$|\widehat{U}|+|\widehat{V}|$  (%s)" % aunit
        p2_title = (r"$\langle|\widehat{U}|+|\widehat{V}|\rangle"
                    r"_{[%.3g,\,%.3g]\,%s}$" % (fmin, fmax, _fum))
        p2_cbar = r"$\langle|\widehat{U}|+|\widehat{V}|\rangle$  (%s)" % aunit
        plab = r"$\langle\cdot\rangle_x$  (%s)" % aunit
        angle_lab = r"%g Hz ($f/f_{\mathrm{rot}}$=%.2g)" % (_fchar,
                                                              _fchar / frot)

    # pts_ROI is stored in metres -- only meaningful on calibrated axes.
    _roi = pts_ROI / lscale if calibrated else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    _panel(fig, ax1, xs, zs, peak_flib / ascale, p1_title, p1_cbar,
           xlab, zlab, plab, angle_freq=_fchar, angle_lab=angle_lab,
           roi=_roi)
    _panel(fig, ax2, xs, zs, bandavg / ascale, p2_title, p2_cbar,
           xlab, zlab, plab, roi=_roi)
    fig.suptitle("%s%s   (%s)" % (run,
                                  "   (non-dimensional)" if normalize else "",
                                  _anno), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def process_run(path, show):
    """Load one run, draw and save its two figures (_DIM and _NODIM); show
    them when `show` (single-run mode), close them otherwise (batch mode)."""
    globals().update(load_run(path))
    globals().update(compute_maps())
    for _norm in (False, True):
        _fig = make_maps(_norm)
        if SAVE:
            _out = figure_filename(os.path.join(out_dir, "MAPS_FFT"),
                                   FIG_FORMAT, normalized=_norm)
            _fig.savefig(_out, dpi=200, bbox_inches="tight")
            print("Figure written to:\n  %s" % _out)
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
