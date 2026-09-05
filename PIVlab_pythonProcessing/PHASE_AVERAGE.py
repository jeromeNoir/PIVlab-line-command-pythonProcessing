"""Auto-generated .py twin of PHASE_AVERAGE.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # PHASE_AVERAGE -- phase averaging at the libration period
# 
# Phase-averages each run's stored velocity fields `U(x, z, t)`, `V(x, z, t)`
# (`Velocity.npz` -- nothing is recomputed from the `.mat`) at the libration
# period `T = 1/f_lib`: for every phase `t` in `[0, T)` the frames at
# `t_n = t + n/f_lib`, `n = 0..n_max` (all the COMPLETE periods that fit in the
# time series) are averaged. Because `T` is generally a non-integer number of
# frames, each frame is assigned to the phase bin its `t mod T` falls in (bin
# width = one frame interval), which implements exactly that rule with no phase
# drift and uses every frame of the complete periods.
# 
# Draws the **single run `RUN_DIR`** (`BATCH = False`: figure saved AND shown)
# or **every run sub-folder of every dataset in `BASE_DIRS`** that has a
# `Velocity.npz` (`BATCH = True`: figures saved, not shown).
# 
# Per run it writes into `<run>/PostProcessing/`:
# 
# - **`Velocity_phaseAveraged.npz`** -- the phase-averaged fields `UPA`, `VPA`
#   `(ny, nx, n_phase)` in NATIVE units, the phase axes (`PHASE` in fractions
#   of the period, `TIME_PHASE` in frames), the per-bin sample count
#   `N_SAMPLES`, and the usual shared header (calibration factors, non-dim
#   scales, grid, ROI ...);
# - **`PHASE_AVERAGE_DIM` / `_NODIM`** -- the FIRST phase-averaged frame
#   (phase 0): the velocity magnitude as background, the quiver on top, the
#   ROI rectangle and the dashed +/- theta inertial-wave characteristics at
#   `f_lib` (`sin(theta) = f_lib/(2 f_rot)`, theta from the z-axis) overlaid
#   (`_NODIM`: starred quantities).


import glob
import os

import warnings

import numpy as np
import matplotlib.pyplot as plt

from piv_postprocessing_lib import figure_filename, resolve_npz

# nan-mean over an all-NaN pixel (masked grid points) is expected -- the NaN
# result is the right answer, so silence the noise.
warnings.filterwarnings("ignore", message="Mean of empty slice")


# --- Mute switch -----------------------------------------------------------
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
# BATCH = True  -> phase-average every run sub-folder of every dataset in
#                  BASE_DIRS that has a Velocity.npz: figures SAVED, not shown;
# BATCH = False -> only the single run RUN_DIR: figure saved AND shown.
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

# --- Inertial-wave characteristic lines (at f_lib) --------------------------
N_LINES = 4               # lines per angle, equally spaced across the domain
SCALING_WIDTH_LINE = 0.7  # line-width scaling for the +/- theta lines

# --- Figure (first phase-averaged frame) ------------------------------------
QUIVER_SKIP = 1          # draw every n-th vector in each direction
QUIVER_SCALE = None      # None -> matplotlib autoscale
QUIVER_COLOR = "black"
CMAP = "viridis"         # background: velocity magnitude

SAVE = True              # write npz + figures next to the source .npz
OVERWRITE_FIG = True     # False -> keep existing figure files
FIG_FORMAT = "png"       # figure format: png or pdf


# ## Load one run, phase-average, write `Velocity_phaseAveraged.npz`


# Header keys copied verbatim from Velocity.npz into the phase-averaged file
# (everything except the big time/frequency arrays and the FFT results).
_SKIP_KEYS = {"U", "V", "Umean", "Vmean", "Ustd", "Vstd", "FREQ", "TIME",
              "F_PEAKS", "AMP_PEAKS", "N_PEAKS", "window", "detrend"}


def load_run(path):
    """Phase-average one run's Velocity.npz, write Velocity_phaseAveraged.npz
    and return everything the figure needs (dict applied to the module
    globals by process_run)."""
    npz_path = resolve_npz(path, "Velocity.npz")
    print("Reading %s" % npz_path)
    data = np.load(npz_path, allow_pickle=True)
    _g = lambda k: np.asarray(data[k])
    _s = lambda k: np.asarray(data[k]).item()

    run = str(_s("run"))
    calibrated = bool(_s("calibrated"))
    k0, top_topo, bottom_topo = _s("k0"), _s("top_topo"), _s("bottom_topo")
    frot, flib = _s("frot_Hz"), _s("flib_Hz")
    fps = float(_s("PIV_FPS"))
    XCAL, YCAL, UCAL, VCAL = (_s(k) for k in ("XCAL", "YCAL", "UCAL", "VCAL"))
    U_SCALE, LENGTH_SCALE = _s("U_SCALE"), _s("LENGTH_SCALE")
    F_SCALE = _s("F_SCALE")

    if not (np.isfinite(flib) and flib > 0):
        raise ValueError("flib unknown from the run name -- cannot define "
                         "the phase-averaging period")

    U = _g("U").astype(np.float32)
    V = _g("V").astype(np.float32)
    nframes = U.shape[2]

    # Period in FRAMES (generally non-integer) and the complete periods.
    T_frames = fps / float(flib)
    n_periods = int(np.floor(nframes / T_frames))
    if n_periods < 1:
        raise ValueError("time series shorter than one libration period "
                         "(%d frames < %.1f)" % (nframes, T_frames))
    n_used = int(np.floor(n_periods * T_frames))
    n_phase = int(np.floor(T_frames))          # one bin per frame interval

    # Phase of every used frame and its bin: t_n = t + n/flib <-> t mod T.
    k = np.arange(n_used)
    phase = (k % T_frames) / T_frames                       # in [0, 1)
    bins = np.minimum((phase * n_phase).astype(int), n_phase - 1)

    UPA = np.empty(U.shape[:2] + (n_phase,), np.float32)
    VPA = np.empty(V.shape[:2] + (n_phase,), np.float32)
    N_SAMPLES = np.zeros(n_phase, int)
    for i in range(n_phase):
        sel = k[bins == i]
        N_SAMPLES[i] = sel.size
        UPA[:, :, i] = np.nanmean(U[:, :, sel], axis=2)
        VPA[:, :, i] = np.nanmean(V[:, :, sel], axis=2)
    del U, V

    out_dir = os.path.dirname(npz_path)
    pa_path = os.path.join(out_dir, "Velocity_phaseAveraged.npz")
    payload = {kk: np.asarray(data[kk]) for kk in data.files
               if kk not in _SKIP_KEYS and not kk.startswith("FFT_")}
    payload.update(UPA=UPA, VPA=VPA,
                   PHASE=np.arange(n_phase) / float(n_phase),   # of the period
                   TIME_PHASE=np.arange(n_phase) * T_frames / n_phase,  # frames
                   N_SAMPLES=N_SAMPLES, n_phase=n_phase,
                   n_periods=n_periods, nframes_used=n_used,
                   frames_per_period=T_frames)
    np.savez_compressed(pa_path, **payload)

    funit = "Hz" if calibrated else "1/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)
    print("run   : %s  (%scalibrated)   flib = %s %s"
          % (run, "" if calibrated else "NOT ", flib, funit))
    print("phase : T = %.2f frames   %d complete periods   %d bins   "
          "%d-%d samples/bin" % (T_frames, n_periods, n_phase,
                                 N_SAMPLES.min(), N_SAMPLES.max()))
    print("Phase-averaged fields written to:\n  %s" % pa_path)

    return dict(npz_path=npz_path, out_dir=out_dir, run=run,
                calibrated=calibrated, frot=frot, flib=flib,
                U_SCALE=U_SCALE, LENGTH_SCALE=LENGTH_SCALE, F_SCALE=F_SCALE,
                x=_g("X").astype(float) * XCAL,
                z=_g("Y").astype(float) * YCAL,
                u0=UPA[:, :, 0].astype(float) * UCAL,
                v0=VPA[:, :, 0].astype(float) * VCAL,
                pts_ROI=np.asarray(_g("pts_ROI"), float),
                n_periods=n_periods, _anno=_anno)


# ## The figure (first phase-averaged frame) and the run driver


def _angle_lines(axm, f_hz, lab):
    """Dashed +/- theta inertial-wave characteristics on map axis `axm`,
    with sin(theta) = f_hz / (2 f_rot), theta measured from the z-axis.
    N_LINES parallel lines per sign, equally spaced across the domain."""
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
    axm.set_xlim(xl); axm.set_ylim(zl)     # keep the map extent
    leg = axm.legend(loc="upper right", fontsize=7)
    leg.get_frame().set_facecolor("0.25"); leg.get_frame().set_alpha(0.7)
    for _t in leg.get_texts():
        _t.set_color("w")


def make_phase_figure(normalize):
    """Phase-0 field: |velocity| background + quiver + ROI rectangle."""
    lscale = (LENGTH_SCALE if (normalize and np.isfinite(LENGTH_SCALE)
                               and LENGTH_SCALE) else 1.0)
    ascale = U_SCALE if (normalize and np.isfinite(U_SCALE) and U_SCALE) else 1.0
    xs, zs = x / lscale, z / lscale
    us, vs = u0 / ascale, v0 / ascale
    speed = np.hypot(us, vs)
    if normalize:
        xlab, zlab = r"$x^*$", r"$z^*$"
        clab = r"$|\mathbf{u}|^{\,*}$"
    else:
        _lu = "(m)" if calibrated else "(px)"
        xlab, zlab = "x " + _lu, "z " + _lu
        clab = r"$|\mathbf{u}|$  (%s)" % ("m/s" if calibrated else "px/frame")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    pcm = ax.pcolormesh(xs, zs, np.ma.masked_invalid(speed), cmap=CMAP,
                        shading="auto", vmin=0.0,
                        vmax=float(np.nanpercentile(speed, 99)))
    fig.colorbar(pcm, ax=ax, label=clab)
    sk = max(1, int(QUIVER_SKIP))
    ax.quiver(xs[::sk, ::sk], zs[::sk, ::sk], us[::sk, ::sk], vs[::sk, ::sk],
              color=QUIVER_COLOR, pivot="mid", width=0.0025,
              scale=QUIVER_SCALE, zorder=3)
    # ROI rectangle (pts_ROI is stored in metres).
    if calibrated:
        (xa, za), (xb, zb) = pts_ROI / lscale
        rx, rz = sorted((xa, xb)), sorted((za, zb))
        ax.plot([rx[0], rx[1], rx[1], rx[0], rx[0]],
                [rz[0], rz[0], rz[1], rz[1], rz[0]],
                "r--", lw=1.2, alpha=0.9)
    # +/- theta inertial-wave characteristics at f_lib.
    if normalize and np.isfinite(F_SCALE) and F_SCALE:
        _alab = r"$f^*_{\mathrm{lib}}$ = %.2g" % (flib / F_SCALE)
    else:
        _alab = r"%g Hz ($f/f_{\mathrm{rot}}$=%.2g)" % (flib, flib / frot)
    _angle_lines(ax, flib, _alab)
    ax.set_aspect("equal")
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(zlab, fontsize=12)
    ax.set_title("%s%s\nphase-averaged over %d periods -- phase 0   (%s)"
                 % (run, "   (non-dimensional)" if normalize else "",
                    n_periods, _anno), fontsize=10)
    fig.tight_layout()
    return fig


def process_run(path, show):
    """Phase-average one run, save the npz and the phase-0 figure (_DIM and
    _NODIM); show when `show` (single-run mode)."""
    globals().update(load_run(path))
    for _norm in (False, True):
        _fig = make_phase_figure(_norm)
        if SAVE:
            _out = figure_filename(os.path.join(out_dir, "PHASE_AVERAGE"),
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
