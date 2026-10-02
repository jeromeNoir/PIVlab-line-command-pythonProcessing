"""Auto-generated .py twin of PHASE_AVERAGE.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # PHASE_AVERAGE -- phase averaging of the velocity fields
# 
# Phase-averages each run's stored velocity fields `U(x, z, t)`, `V(x, z, t)`
# (`Velocity.npz` -- nothing is recomputed from the `.mat`) at the period
# `T = 1/f_avg`, where `f_avg` is `PHASE_FREQ_HZ` (Hz) or, when that is
# `None`, the run's libration frequency `f_lib`: for every phase `t` in
# `[0, T)` the frames at `t_n = t + n T`, `n = 0..n_max` (all the COMPLETE
# periods that fit in the time series) are averaged.
# 
# With `SUBTRACT_MEAN = True` (default) each point's time-averaged velocity
# (per component, over the used frames -- an integer number of periods) is
# subtracted BEFORE the phase averaging, so `UPA`/`VPA` hold the phase-locked
# **fluctuations**; the subtracted maps are stored as `UMEAN_T`/`VMEAN_T`.
# 
# Because `T` is generally a non-integer number of frames, two methods are
# offered (`PHASE_METHOD`):
# 
# - **`"bin"`** -- each frame is assigned to the phase bin its `t mod T`
#   falls in (bin width = one frame interval): no phase drift, every frame of
#   the complete periods is used, no interpolation error;
# - **`"interp"`** -- the velocity is linearly interpolated in time at each
#   grid point at `n_phase` regular phases of every period, then averaged
#   across the periods: every phase sample sits exactly on the requested
#   phase (no binning jitter), at the cost of linear-interpolation smoothing.
# 
# Draws the **single run `RUN_DIR`** (`BATCH = False`: figure saved AND shown)
# or **every run sub-folder of every dataset in `BASE_DIRS`** that has a
# `Velocity.npz` (`BATCH = True`: figures saved, not shown).
# 
# Per run it writes into `<run>/PostProcessing/` (the averaging frequency is
# part of the figure/movie file names):
# 
# - **`Velocity_phaseAveraged.npz`** -- EVERY phase-averaged frame: `UPA`,
#   `VPA` `(ny, nx, n_phase)` in NATIVE units, the phase axes (`PHASE` in
#   fractions of the period, `TIME_PHASE` in frames), the per-bin sample
#   count `N_SAMPLES`, the averaging settings (`phase_freq_Hz`,
#   `phase_method`, `mean_subtracted`, `UMEAN_T`/`VMEAN_T` when subtracted),
#   and the usual shared header (calibration factors, non-dim scales, grid,
#   ROI ...);
# - **`PHASE_AVERAGE_favg<f>Hz_DIM` / `_NODIM`** -- the FIRST phase-averaged
#   frame (phase 0): the velocity magnitude as background, the quiver on top,
#   the ROI rectangle and the dashed +/- theta inertial-wave characteristics
#   at `f_avg` (`sin(theta) = f_avg/(2 f_rot)`, theta from the z-axis)
#   overlaid (`_NODIM`: starred quantities);
# - with `MOVIE = True`, **`PHASE_AVERAGE_favg<f>Hz_MOVIE_DIM` / `_NODIM`**
#   (`.mp4` or `.gif`, `MOVIE_FORMAT`) -- the same layout cycling through ALL
#   the `n_phase` frames at `MOVIE_FPS`, with ONE color scale across the
#   whole cycle (99th percentile of |u| over all phases).


import glob
import os
import pickle

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
BATCH = False

RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_topBottom_fullView_spacer64mm/"
           "frot0.50Hz_flib0.400Hz_dphi0deg_SS3")  # run folder or Velocity.npz
                                                   # (BATCH = False)

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets swept when BATCH = True (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

# --- Phase averaging ---------------------------------------------------------
PHASE_FREQ_HZ = 0.5      # averaging frequency in Hz; None -> the run's f_lib
PHASE_METHOD = "interp"      # "bin"    -> each frame goes into the phase bin its
                          #             t mod T falls in (one bin per frame
                          #             interval; no interpolation);
                          # "interp" -> the velocity is linearly interpolated
                          #             in time at each grid point at regular
                          #             phases of every period, then averaged
                          #             across the periods.
SUBTRACT_MEAN = True      # subtract each point's time-averaged velocity (per
                          # component, over the used frames) BEFORE the phase
                          # averaging -> UPA/VPA hold the phase-locked
                          # FLUCTUATIONS (the subtracted means are stored in
                          # the npz as UMEAN_T/VMEAN_T)

# --- Movie of the N phase-averaged frames ------------------------------------
MOVIE = True             # also write a movie cycling through ALL the phases
                         # (same layout as the figure; _DIM and _NODIM files)
MOVIE_FPS = 5           # movie playback rate (phase frames per second)
MOVIE_FORMAT = "mp4"     # "mp4" (needs ffmpeg) or "gif"; mp4 falls back to
                         # gif when ffmpeg is not available

# --- Inertial-wave characteristic lines (at the averaging frequency) --------
N_LINES = 4               # lines per angle, equally spaced across the domain
SCALING_WIDTH_LINE = 0.7  # line-width scaling for the +/- theta lines

# --- Figure (first phase-averaged frame) ------------------------------------
SUBTRACT_SPACE_MEAN = True  # subtract each phase frame's SPATIAL-mean U and
                            # V from the displayed field (background |u|
                            # color map AND quiver) -- display only, the npz
                            # keeps the full phase-averaged fields
BKG_COLOR = "amplitude"  # background color map: "amplitude" -> |u| (>= 0),
                         # "U" or "V" -> that signed component, SYMMETRIC
                         # color scale around 0 (a diverging CMAP such as
                         # "RdBu_r" reads best for U/V)
QUIVER = True            # draw the velocity quiver on top (False -> none)
QUIVER_SKIP = 1          # draw every n-th vector in each direction
QUIVER_SCALE = None      # None -> matplotlib autoscale
QUIVER_COLOR = "black"
CMAP = "viridis"         # background colormap

SAVE = True              # write npz + figures next to the source .npz
SAVE_PICKLE = True       # ALSO save each figure as <name>.fig.pickle: reopen
                         # it fully interactive (Qt zoom/cursor) with
                         #   fig = pickle.load(open(path, "rb")); fig.show()
OVERWRITE_FIG = True     # False -> keep existing figure/movie files
FIG_FORMAT = "png"       # figure format: png or pdf


# ## Load one run, phase-average, write `Velocity_phaseAveraged.npz`


# Header keys copied verbatim from Velocity.npz into the phase-averaged file
# (everything except the big time/frequency arrays and the FFT results).
_SKIP_KEYS = {"U", "V", "Umean", "Vmean", "Ustd", "Vstd", "FREQ", "TIME",
              "F_PEAKS", "AMP_PEAKS", "N_PEAKS", "window", "detrend"}


def load_run(path):
    """Phase-average one run's Velocity.npz at f_avg (PHASE_FREQ_HZ, or the
    run's f_lib when None) with the "bin" or "interp" method -- after
    subtracting each point's time-averaged velocity when SUBTRACT_MEAN --
    write EVERY phase-averaged frame to Velocity_phaseAveraged.npz and
    return everything the figure and the movie need (dict applied to the
    module globals by process_run)."""
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

    # The averaging frequency: PHASE_FREQ_HZ, or the run's f_lib when None.
    f_avg = float(PHASE_FREQ_HZ) if PHASE_FREQ_HZ else flib
    if not (np.isfinite(f_avg) and f_avg > 0):
        raise ValueError("no valid averaging frequency (PHASE_FREQ_HZ=%r, "
                         "flib=%r)" % (PHASE_FREQ_HZ, flib))

    U = _g("U").astype(np.float32)
    V = _g("V").astype(np.float32)
    nframes = U.shape[2]

    # Period in FRAMES (generally non-integer) and the complete periods.
    T_frames = fps / f_avg
    n_periods = int(np.floor(nframes / T_frames))
    if n_periods < 1:
        raise ValueError("time series shorter than one averaging period "
                         "(%d frames < %.1f)" % (nframes, T_frames))
    n_used = int(np.floor(n_periods * T_frames))
    n_phase = int(np.floor(T_frames))          # one phase per frame interval

    # Subtract each point's time-averaged velocity (per component, over the
    # USED frames -- an integer number of periods, so a pure oscillation
    # averages to zero) -> the phase average holds the FLUCTUATIONS.
    UMEAN_T = VMEAN_T = None
    if SUBTRACT_MEAN:
        UMEAN_T = np.nanmean(U[:, :, :n_used], axis=2).astype(np.float32)
        VMEAN_T = np.nanmean(V[:, :, :n_used], axis=2).astype(np.float32)
        U = U - UMEAN_T[:, :, None]
        V = V - VMEAN_T[:, :, None]

    UPA = np.empty(U.shape[:2] + (n_phase,), np.float32)
    VPA = np.empty(V.shape[:2] + (n_phase,), np.float32)
    N_SAMPLES = np.zeros(n_phase, int)

    if PHASE_METHOD == "bin":
        # Each used frame goes into the bin its t mod T falls in (bin width
        # = one frame interval; the fractional tail [n_phase, T) joins the
        # last bin). Binning on k % T directly keeps an exactly-integer T
        # free of float round-off (no empty bins).
        k = np.arange(n_used)
        bins = np.minimum(np.floor(k % T_frames).astype(int), n_phase - 1)
        for i in range(n_phase):
            sel = k[bins == i]
            N_SAMPLES[i] = sel.size
            UPA[:, :, i] = np.nanmean(U[:, :, sel], axis=2)
            VPA[:, :, i] = np.nanmean(V[:, :, sel], axis=2)
    elif PHASE_METHOD == "interp":
        # Each grid point's velocity is linearly interpolated in time at the
        # exact instants t = (n + i/n_phase) T, then averaged over periods n.
        pn = np.arange(n_periods)
        for i in range(n_phase):
            t = (pn + i / float(n_phase)) * T_frames
            t = t[t <= nframes - 1.0]
            f0 = np.floor(t).astype(int)
            f1 = np.minimum(f0 + 1, nframes - 1)
            w = (t - f0).astype(np.float32)
            N_SAMPLES[i] = t.size
            UPA[:, :, i] = np.nanmean(U[:, :, f0] * (1.0 - w)
                                      + U[:, :, f1] * w, axis=2)
            VPA[:, :, i] = np.nanmean(V[:, :, f0] * (1.0 - w)
                                      + V[:, :, f1] * w, axis=2)
    else:
        raise ValueError("PHASE_METHOD must be 'bin' or 'interp', got %r"
                         % (PHASE_METHOD,))
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
                   frames_per_period=T_frames,
                   phase_freq_Hz=f_avg, phase_method=PHASE_METHOD,
                   mean_subtracted=bool(SUBTRACT_MEAN))
    if SUBTRACT_MEAN:
        payload.update(UMEAN_T=UMEAN_T, VMEAN_T=VMEAN_T)  # native units
    np.savez_compressed(pa_path, **payload)

    funit = "Hz" if calibrated else "1/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)
    print("run   : %s  (%scalibrated)   flib = %s %s"
          % (run, "" if calibrated else "NOT ", flib, funit))
    print("phase : f_avg = %g %s (%s)   T = %.2f frames   %d complete "
          "periods   %d phases   %d-%d samples/phase"
          % (f_avg, funit,
             "PHASE_FREQ_HZ" if PHASE_FREQ_HZ else "f_lib",
             T_frames, n_periods, n_phase,
             N_SAMPLES.min(), N_SAMPLES.max()))
    print("method: %s   time mean: %s" % (
        PHASE_METHOD,
        "subtracted per point/component" if SUBTRACT_MEAN else "kept"))
    print("Phase-averaged fields (all %d frames) written to:\n  %s"
          % (n_phase, pa_path))

    # Calibrated phase-averaged fields: every phase for the movie, phase 0
    # for the figure.
    uall = UPA * np.float32(UCAL)
    vall = VPA * np.float32(VCAL)   

    return dict(npz_path=npz_path, out_dir=out_dir, run=run,
                calibrated=calibrated, frot=frot, flib=flib, f_avg=f_avg,
                U_SCALE=U_SCALE, LENGTH_SCALE=LENGTH_SCALE, F_SCALE=F_SCALE,
                x=_g("X").astype(float) * XCAL,
                z=_g("Y").astype(float) * YCAL,
                uall=uall, vall=vall, n_phase=n_phase,
                u0=uall[:, :, 0].astype(float),
                v0=vall[:, :, 0].astype(float),
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


def _remove_space_mean(us, vs):
    """SUBTRACT_SPACE_MEAN: subtract each phase frame's spatial-mean U and V
    (per component) from the DISPLAYED field -- background and quiver only,
    the stored npz is untouched. Works on one frame (2D) or on the whole
    (nz, nx, n_phase) stack."""
    if not SUBTRACT_SPACE_MEAN:
        return us, vs
    ax = (0, 1)
    return (us - np.nanmean(us, axis=ax, keepdims=True),
            vs - np.nanmean(vs, axis=ax, keepdims=True))


def _bkg_field(us, vs):
    """The background quantity per BKG_COLOR: (field, signed, LaTeX symbol).
    Works on one frame (2D) or on the whole (nz, nx, n_phase) stack."""
    q = str(BKG_COLOR).lower()
    if q.startswith("u"):
        return us, True, "U"
    if q.startswith("v"):
        return vs, True, "V"
    return np.hypot(us, vs), False, r"|\mathbf{u}|"


def _bkg_vmax(us, vs):
    """One color-scale bound for the figure/movie: the 99th percentile of
    |background quantity| over everything passed in."""
    field, _signed, _sym = _bkg_field(us, vs)
    return float(np.nanpercentile(np.abs(field), 99))


def _phase_axes(normalize, us, vs, vmax):
    """The shared figure layout (BKG_COLOR background + optional quiver +
    ROI + characteristics) for the velocity field (us, vs); vmax fixes the
    color scale (symmetric around 0 for a signed background). Returns
    (fig, ax, pcm, quiver-or-None, title_fmt) -- title_fmt % phase_text
    completes the title."""
    lscale = (LENGTH_SCALE if (normalize and np.isfinite(LENGTH_SCALE)
                               and LENGTH_SCALE) else 1.0)
    xs, zs = x / lscale, z / lscale
    field, signed, sym = _bkg_field(us, vs)
    if normalize:
        xlab, zlab = r"$x^*$", r"$z^*$"
        clab = r"$%s^{\,*}$" % sym
    else:
        _lu = "(m)" if calibrated else "(px)"
        xlab, zlab = "x " + _lu, "z " + _lu
        clab = r"$%s$  (%s)" % (sym, "m/s" if calibrated else "px/frame")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    pcm = ax.pcolormesh(xs, zs, np.ma.masked_invalid(field), cmap=CMAP,
                        shading="auto",
                        vmin=(-vmax if signed else 0.0), vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=clab)
    if QUIVER:
        sk = max(1, int(QUIVER_SKIP))
        qv = ax.quiver(xs[::sk, ::sk], zs[::sk, ::sk], us[::sk, ::sk],
                       vs[::sk, ::sk], color=QUIVER_COLOR, pivot="mid",
                       width=0.0025, scale=QUIVER_SCALE, zorder=3)
    else:
        qv = None
    # ROI rectangle (pts_ROI is stored in metres).
    if calibrated:
        (xa, za), (xb, zb) = pts_ROI / lscale
        rx, rz = sorted((xa, xb)), sorted((za, zb))
        ax.plot([rx[0], rx[1], rx[1], rx[0], rx[0]],
                [rz[0], rz[0], rz[1], rz[1], rz[0]],
                "r--", lw=1.2, alpha=0.9)
    # +/- theta inertial-wave characteristics at the averaging frequency.
    if normalize and np.isfinite(F_SCALE) and F_SCALE:
        _alab = r"$f^*_{\mathrm{avg}}$ = %.2g" % (f_avg / F_SCALE)
    else:
        _alab = r"%g Hz ($f/f_{\mathrm{rot}}$=%.2g)" % (f_avg, f_avg / frot)
    _angle_lines(ax, f_avg, _alab)
    ax.set_aspect("equal")
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(zlab, fontsize=12)
    _method = PHASE_METHOD + (", space mean removed"
                              if SUBTRACT_SPACE_MEAN else "")
    title_fmt = ("%s%s\nphase-averaged at %g Hz (%s) over %d periods -- "
                 "%%s   (%s)"
                 % (run, "   (non-dimensional)" if normalize else "",
                    f_avg, _method, n_periods, _anno))
    return fig, ax, pcm, qv, title_fmt


def make_phase_figure(normalize):
    """Phase-0 field: BKG_COLOR background (+ optional quiver + ROI)."""
    ascale = U_SCALE if (normalize and np.isfinite(U_SCALE) and U_SCALE) else 1.0
    us, vs = _remove_space_mean(u0 / ascale, v0 / ascale)
    vmax = _bkg_vmax(us, vs)
    fig, ax, pcm, qv, title_fmt = _phase_axes(normalize, us, vs, vmax)
    ax.set_title(title_fmt % "phase 0", fontsize=10)
    fig.tight_layout()
    return fig


def make_phase_movie(normalize, out_stem):
    """Movie of ALL the phase-averaged frames, built from still images:
    every phase is rendered to its own PNG (the exact phase figure, with ONE
    color scale across the whole cycle), then the images are assembled --
    never a live matplotlib canvas.

    mp4: ffmpeg places each image, centered, on a black background padded to
    the next multiple of 16 in width and height (H.264's macroblock size,
    the dimensions every player handles), encoded H.264 / yuv420p.
    gif: the PNGs are assembled with Pillow (no size constraint)."""
    import shutil
    import subprocess
    import tempfile

    ascale = U_SCALE if (normalize and np.isfinite(U_SCALE) and U_SCALE) else 1.0
    us, vs = _remove_space_mean(np.asarray(uall, float) / ascale,
                                np.asarray(vall, float) / ascale)
    vmax = _bkg_vmax(us, vs)                 # one scale for all the phases

    fmt = str(MOVIE_FORMAT).lower().lstrip(".")
    if fmt == "mp4" and shutil.which("ffmpeg") is None:
        print("  [movie] ffmpeg not found -- falling back to gif")
        fmt = "gif"
    out = "%s_%s.%s" % (out_stem, "NODIM" if normalize else "DIM", fmt)
    if not OVERWRITE_FIG and os.path.isfile(out):
        print("Movie exists (OVERWRITE_FIG=False), kept:\n  %s" % out)
        return out

    tmp = tempfile.mkdtemp(prefix="PHASE_AVERAGE_movie_")
    try:
        # 1) One PNG per phase. savefig WITHOUT bbox trimming: every image
        #    is exactly figsize x dpi pixels, so the frames are identical.
        #    plt.ioff(): no window pops up per frame (qt backend).
        with plt.ioff():
            for i in range(n_phase):
                fig, ax, _pcm, _qv, title_fmt = _phase_axes(
                    normalize, us[:, :, i], vs[:, :, i], vmax)
                ax.set_title(title_fmt % ("phase %d/%d" % (i, n_phase)),
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(tmp, "frame_%04d.png" % i), dpi=150)
                plt.close(fig)

        # 2) Assemble the stills.
        if fmt == "mp4":
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(MOVIE_FPS),
                 "-i", os.path.join(tmp, "frame_%04d.png"),
                 "-vf", ("pad=ceil(iw/16)*16:ceil(ih/16)*16:"
                         "(ow-iw)/2:(oh-ih)/2:black"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out],
                check=True)
        else:
            from PIL import Image
            frames = [Image.open(os.path.join(tmp, "frame_%04d.png" % i))
                      for i in range(n_phase)]
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           duration=int(round(1000.0 / MOVIE_FPS)), loop=0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("Movie (%d phases) written to:\n  %s" % (n_phase, out))
    return out


def process_run(path, show):
    """Phase-average one run, save the npz, the phase-0 figure (_DIM and
    _NODIM; as png/pdf and, with SAVE_PICKLE, as a reopenable interactive
    .fig.pickle) and, when MOVIE, the movie over all phases; show the
    figures when `show` (single-run mode). The averaging frequency is part
    of the output file names (PHASE_AVERAGE_favg<f>Hz_...)."""
    globals().update(load_run(path))
    stem = "PHASE_AVERAGE_favg%gHz" % f_avg
    for _norm in (False, True):
        _fig = make_phase_figure(_norm)
        if SAVE:
            _out = figure_filename(os.path.join(out_dir, stem),
                                   FIG_FORMAT, normalized=_norm)
            if OVERWRITE_FIG or not os.path.isfile(_out):
                _fig.savefig(_out, dpi=200, bbox_inches="tight")
                print("Figure written to:\n  %s" % _out)
            else:
                print("Figure exists (OVERWRITE_FIG=False), kept:\n  %s"
                      % _out)
            if SAVE_PICKLE:
                _pkl = os.path.splitext(_out)[0] + ".fig.pickle"
                if OVERWRITE_FIG or not os.path.isfile(_pkl):
                    with open(_pkl, "wb") as _fh:
                        pickle.dump(_fig, _fh)
                    print("Interactive figure pickled to:\n  %s" % _pkl)
                else:
                    print("Pickle exists (OVERWRITE_FIG=False), kept:\n  %s"
                          % _pkl)
        if show:
            plt.show()
        else:
            plt.close(_fig)
        if MOVIE:
            make_phase_movie(_norm,
                             os.path.join(out_dir, stem + "_MOVIE"))


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
