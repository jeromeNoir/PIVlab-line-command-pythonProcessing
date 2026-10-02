"""Auto-generated .py twin of K_SPECTRUM_VELOCITY.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # K_SPECTRUM_VELOCITY -- spatial (wavenumber) spectra of the velocity
# 
# From each run's stored (PIVlab-filtered) velocity fields `U(z, x, t)`,
# `V(z, x, t)` in `Velocity.npz` -- **nothing is recomputed from the `.mat`**.
# The calculation is **limited to the ROI** with the stored `MASK_ROI`
# (1 inside, NaN outside), as in the other notebooks: the fields are cropped
# to the ROI bounding box and points outside the mask are NaN'd.
# 
# - at each vertical position `z`, the FFT of `U` and `V` along the
#   HORIZONTAL coordinate `x`, for every frame, time-averaged
#   -> **`mean_FFT_HORIZONTAL`** `(nz_roi, nkh)`;
# - at each horizontal position `x`, the FFT along the VERTICAL coordinate
#   `z`, for every frame, time-averaged
#   -> **`mean_FFT_VERTICAL`** `(nkv, nx_roi)`.
# 
# The amplitude is `|FFT U| + |FFT V|` (2/n amplitude normalization); before
# each FFT the line's spatial mean is removed and NaN gaps are zero-filled.
# Wavenumbers `kh`, `kv` are in **1/m** (cycles per metre); the figures can
# show them as wavelengths `lambda = 1/k` (m) instead (`X_AXIS` for the 1D
# figure, `MAP_K_AXIS` for the maps) and with log axes (`LOGX`/`LOGY` for the
# 1D figure, `MAP_LOGK` for the maps' k axis, `MAP_LOG` for their color
# scale).
# 
# Draws the **single run `RUN_DIR`** (`BATCH = False`: figures saved AND
# shown) or **every run sub-folder of every dataset in `BASE_DIRS`** that has
# a `Velocity.npz` (`BATCH = True`: figures saved, not shown).
# 
# Per run it writes into `<run>/PostProcessing/`:
# 
# | output | content |
# |---|---|
# | `K_SpectrumVelocity.npz` | `KH`, `KV` (1/m), `mean_FFT_HORIZONTAL`, `mean_FFT_VERTICAL` (total and per component `_U`/`_V`), the ROI `x_pos`/`z_pos` axes (m), the ROI box and the run header |
# | `K_SPECTRUM_VELOCITY_1D` | `mean_FFT_HORIZONTAL` averaged over all `z` vs `kh`, and `mean_FFT_VERTICAL` averaged over all `x` vs `kv`, on the same axes |
# | `K_SPECTRUM_VELOCITY_MAP_KH` | map of `mean_FFT_HORIZONTAL`: `kh` (or `lambda_h`) on x, `z` on y, amplitude as color |
# | `K_SPECTRUM_VELOCITY_MAP_KV` | map of `mean_FFT_VERTICAL`: `x` on x, `kv` (or `lambda_v`) on y, amplitude as color |


import glob
import os

import warnings

import numpy as np
import matplotlib.pyplot as plt

from piv_postprocessing_lib import pickle_figure, resolve_npz

# nan-mean over an all-NaN line (masked grid points) is expected -- the NaN
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
# BATCH = True  -> process every run sub-folder of every dataset in BASE_DIRS
#                  that has a Velocity.npz: figures SAVED, not shown;
# BATCH = False -> only the single run RUN_DIR: figures saved AND shown.
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

# --- Spectra -----------------------------------------------------------------
VELOCITY_SOURCE = "full"  # "full"          -> the native filtered U, V of
                          #                    Velocity.npz (every frame);
                          # "phase_average" -> the phase-averaged UPA/VPA of
                          #                    Velocity_phaseAveraged.npz
                          #                    (run PHASE_AVERAGE first);
                          #                    the time average then runs
                          #                    over the phases of one cycle.
                          # With "phase_average" the outputs carry a
                          # _phaseAveraged / _PA tag (no overwrite of the
                          # full-velocity results).
KMAX = None             # wavenumber-axis limit in 1/m; None -> Nyquist
                        # (in "lambda" mode it bounds the SHORTEST
                        # wavelength shown: lambda > 1/KMAX)
X_AXIS = "k"            # 1D figure x-axis: "k" -> wavenumber in 1/m,
                        # "lambda" -> wavelength in m
LOGX = True            # log x-axis on the 1D figure
LOGY = True             # log amplitude axis on the 1D figure
MAP_K_AXIS = "lambda"        # maps: k axis as "k" -> wavenumber in 1/m,
                        # "lambda" -> wavelength in m
MAP_LOGK = False        # maps: log scale on the k (or lambda) axis
MAP_LOG = False         # True -> logarithmic COLOR scale on the two maps
T_CHUNK = 200           # frames FFT'd per block (memory only, no effect on
                        # the result)

# --- Figures -----------------------------------------------------------------
CMAP = "viridis"        # maps: amplitude colormap
SAVE = True             # write npz + figures next to the source .npz
SAVE_PICKLE = True      # ALSO save each figure as <name>.fig.pickle:
                        # reopen it fully interactive (Qt zoom/cursor)
                        # with openFigure.py
OVERWRITE_FIG = True    # False -> keep existing figure files
FIG_FORMAT = "png"      # figure format: png or pdf


# ## Load one run, compute the wavenumber spectra, write the npz
# 
# The stored `U`, `V` are PIVlab's FILTERED velocity components (native
# units); amplitudes are calibrated with `UCAL`/`VCAL` (m/s) and the grid
# with `XCAL`/`YCAL` (m), so `kh`, `kv` come out in 1/m.


def _line_fft_timeMean(A, axis, dstep):
    """Time-averaged amplitude |rfft| of A (nz, nx, nt) along the spatial
    `axis` (0 = vertical/z, 1 = horizontal/x), sampled every `dstep` metres.
    Each line's spatial mean is removed and NaN gaps (masked points) are
    zero-filled before the FFT; 2/n normalization -> amplitude per mode,
    same unit as A. Returns (k_1perm, meanAmp) with the time axis averaged
    out."""
    n = A.shape[axis]
    m = np.nanmean(A, axis=axis, keepdims=True)         # line mean (per frame)
    B = A - m
    B = np.where(np.isfinite(B), B, 0.0).astype(np.float32)
    k = np.fft.rfftfreq(n, d=dstep)                     # 1/m
    acc = 0.0
    nt = A.shape[2]
    for t0 in range(0, nt, max(1, int(T_CHUNK))):
        blk = B[:, :, t0:t0 + max(1, int(T_CHUNK))]
        acc = acc + np.abs(np.fft.rfft(blk, axis=axis)).sum(axis=2)
    return k, (2.0 / n) * acc / float(nt)


def load_run(path):
    """Compute one run's spatial spectra -- from the native filtered U/V of
    Velocity.npz or, with VELOCITY_SOURCE='phase_average', from the UPA/VPA
    of Velocity_phaseAveraged.npz -- restricted to the ROI with the stored
    MASK_ROI (1 inside, NaN outside), as in the other notebooks: the fields
    are cropped to the ROI bounding box and points outside the mask are
    NaN'd (zero-filled in the FFT). Writes K_SpectrumVelocity<tag>.npz and
    returns everything the figures need (dict applied to the module globals
    by process_run)."""
    npz_path = resolve_npz(path, "Velocity.npz")
    print("Reading %s" % npz_path)
    data = np.load(npz_path, allow_pickle=True)
    _g = lambda k: np.asarray(data[k])
    _s = lambda k: np.asarray(data[k]).item()

    run = str(_s("run"))
    calibrated = bool(_s("calibrated"))
    k0, top_topo, bottom_topo = _s("k0"), _s("top_topo"), _s("bottom_topo")
    frot, flib = _s("frot_Hz"), _s("flib_Hz")
    XCAL, YCAL, UCAL, VCAL = (_s(k) for k in ("XCAL", "YCAL", "UCAL", "VCAL"))

    # The ROI mask (1 inside, NaN outside) and its bounding box: ALL the
    # calculation is limited to the ROI.
    mask = _g("MASK_ROI").astype(float)
    inside = np.isfinite(mask) & (mask > 0)
    if not inside.any():
        raise ValueError("MASK_ROI has no points inside the ROI")
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    mroi = mask[r0:r1, c0:c1]

    x = _g("X").astype(float)[r0:r1, c0:c1] * XCAL      # ROI box, metres
    z = _g("Y").astype(float)[r0:r1, c0:c1] * YCAL
    x_pos = np.nanmean(x, axis=0)                       # (nx_roi,)
    z_pos = np.nanmean(z, axis=1)                       # (nz_roi,)
    dx = float(np.nanmean(np.abs(np.diff(x_pos))))      # grid steps, metres
    dz = float(np.nanmean(np.abs(np.diff(z_pos))))

    # The velocity source: the native filtered fields (every frame), or the
    # phase-averaged fields written by PHASE_AVERAGE (the "time" average
    # then runs over the phases of one cycle). Both are NATIVE units.
    if str(VELOCITY_SOURCE).lower().startswith("p"):
        pa_path = os.path.join(os.path.dirname(npz_path),
                               "Velocity_phaseAveraged.npz")
        if not os.path.isfile(pa_path):
            raise FileNotFoundError(
                "VELOCITY_SOURCE='phase_average' but there is no %s -- run "
                "PHASE_AVERAGE on this run first" % pa_path)
        pa = np.load(pa_path, allow_pickle=True)
        Uraw = np.asarray(pa["UPA"], np.float32)
        Vraw = np.asarray(pa["VPA"], np.float32)
        src = ("phase-averaged UPA/VPA: %d phases at %g Hz (%s)"
               % (int(np.asarray(pa["n_phase"]).item()),
                  float(np.asarray(pa["phase_freq_Hz"]).item()),
                  str(np.asarray(pa["phase_method"]).item())))
        src_tag = "_PA"
        npz_tag = "_phaseAveraged"
    else:
        Uraw = _g("U").astype(np.float32)
        Vraw = _g("V").astype(np.float32)
        src = "full filtered velocity: %d frames" % Uraw.shape[2]
        src_tag = ""
        npz_tag = ""

    # Cropped to the ROI box, calibrated, NaN outside the mask.
    U = Uraw[r0:r1, c0:c1] * np.float32(UCAL) * mroi[:, :, None].astype(
        np.float32)
    V = Vraw[r0:r1, c0:c1] * np.float32(VCAL) * mroi[:, :, None].astype(
        np.float32)
    del Uraw, Vraw

    # FFT along x at each z (axis=1) and along z at each x (axis=0),
    # per component, time-averaged.
    KH, ampU_h = _line_fft_timeMean(U, 1, dx)
    _, ampV_h = _line_fft_timeMean(V, 1, dx)
    KV, ampU_v = _line_fft_timeMean(U, 0, dz)
    _, ampV_v = _line_fft_timeMean(V, 0, dz)
    del U, V

    mean_FFT_HORIZONTAL = ampU_h + ampV_h               # (nz_roi, nkh)
    mean_FFT_VERTICAL = ampU_v + ampV_v                 # (nkv, nx_roi)

    out_dir = os.path.dirname(npz_path)
    ks_path = os.path.join(out_dir, "K_SpectrumVelocity%s.npz" % npz_tag)
    if SAVE:
        np.savez_compressed(
            ks_path,
            KH=KH, KV=KV, x_pos=x_pos, z_pos=z_pos, dx=dx, dz=dz,
            mean_FFT_HORIZONTAL=mean_FFT_HORIZONTAL,
            mean_FFT_VERTICAL=mean_FFT_VERTICAL,
            mean_FFT_HORIZONTAL_U=ampU_h, mean_FFT_HORIZONTAL_V=ampV_h,
            mean_FFT_VERTICAL_U=ampU_v, mean_FFT_VERTICAL_V=ampV_v,
            velocity_source=str(VELOCITY_SOURCE),
            roi_only=True, pts_ROI=_g("pts_ROI"),
            roi_box=np.array([r0, r1, c0, c1]),
            run=run, calibrated=calibrated, frot_Hz=frot, flib_Hz=flib,
            dphi_deg=_s("dphi_deg"), run_idx=_s("run_idx"), k0=k0,
            top_topo=top_topo, bottom_topo=bottom_topo,
            UCAL=UCAL, VCAL=VCAL, XCAL=XCAL, YCAL=YCAL,
            U_SCALE=_s("U_SCALE"), LENGTH_SCALE=_s("LENGTH_SCALE"))
        print("Wavenumber spectra written to:\n  %s" % ks_path)

    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)
    if src_tag:
        _anno += ", phase-averaged input"
    print("run   : %s  (%scalibrated)   frot = %s Hz   flib = %s Hz"
          % (run, "" if calibrated else "NOT ", frot, flib))
    print("input : %s" % src)
    print("ROI   : %d x %d of %d x %d grid points (rows %d-%d, cols %d-%d)"
          % (r1 - r0, c1 - c0, mask.shape[0], mask.shape[1],
             r0, r1 - 1, c0, c1 - 1))
    print("grid  : dx = %.4g m  dz = %.4g m   kh < %.4g 1/m   kv < %.4g 1/m"
          % (dx, dz, KH[-1], KV[-1]))

    return dict(npz_path=npz_path, out_dir=out_dir, run=run,
                calibrated=calibrated, frot=frot, flib=flib,
                KH=KH, KV=KV, x_pos=x_pos, z_pos=z_pos, src_tag=src_tag,
                mean_FFT_HORIZONTAL=mean_FFT_HORIZONTAL,
                mean_FFT_VERTICAL=mean_FFT_VERTICAL, _anno=_anno)


# ## The figures and the run driver


_ALAB = r"$\langle|\widehat{U}|+|\widehat{V}|\rangle_t$  (%s)"


def _save_or_skip(fig, out_path, show):
    if SAVE:
        if OVERWRITE_FIG or not os.path.isfile(out_path):
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print("Figure written to:\n  %s" % out_path)
            if SAVE_PICKLE:
                pickle_figure(fig, out_path)
        else:
            print("Figure exists (OVERWRITE_FIG=False), kept:\n  %s"
                  % out_path)
    if show:
        fig.show()
    else:
        plt.close(fig)


def make_1d_figure():
    """Figure 1: mean_FFT_HORIZONTAL averaged over z, and mean_FFT_VERTICAL
    averaged over x, on the same axes -- vs the wavenumber k (1/m) or the
    wavelength lambda = 1/k (m), X_AXIS; log axes with LOGX/LOGY."""
    spec_h = np.nanmean(mean_FFT_HORIZONTAL, axis=0)[1:]    # DC dropped
    spec_v = np.nanmean(mean_FFT_VERTICAL, axis=1)[1:]
    as_lambda = str(X_AXIS).lower().startswith("l")
    if as_lambda:
        xh, xv = 1.0 / KH[1:], 1.0 / KV[1:]
        xlab = r"$\lambda$  (m)"
        lab_h = r"horizontal: $\langle\cdot\rangle_z$ vs $\lambda_h$"
        lab_v = r"vertical: $\langle\cdot\rangle_x$ vs $\lambda_v$"
    else:
        xh, xv = KH[1:], KV[1:]
        xlab = r"$k$  (1/m)"
        lab_h = r"horizontal: $\langle\cdot\rangle_z$ vs $k_h$"
        lab_v = r"vertical: $\langle\cdot\rangle_x$ vs $k_v$"
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(xh, spec_h, color="C0", lw=1.3, label=lab_h)
    ax.plot(xv, spec_v, color="C1", lw=1.3, label=lab_v)
    if LOGX:
        ax.set_xscale("log")
    if LOGY:
        ax.set_yscale("log")
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(_ALAB % ("m/s" if calibrated else "px/frame"), fontsize=12)
    if KMAX:
        if as_lambda:
            ax.set_xlim(1.0 / KMAX, float(max(xh.max(), xv.max())))
        else:
            ax.set_xlim(xh[0] if LOGX else 0, KMAX)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title("%s -- space-averaged %s spectra   (%s)"
                 % (run, "wavelength" if as_lambda else "wavenumber",
                    _anno), fontsize=10)
    fig.tight_layout()
    return fig


def _map_figure(kk, pos, amp, xlab, ylab, title):
    """One amplitude map (pcolormesh); color = time-averaged amplitude."""
    from matplotlib.colors import LogNorm
    amp = np.ma.masked_invalid(amp)
    vmax = float(np.nanpercentile(amp.compressed(), 99.5))
    if MAP_LOG:
        vmin = max(float(np.nanpercentile(amp.compressed(), 5)),
                   vmax * 1e-6)
        kw = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        kw = dict(vmin=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pcm = ax.pcolormesh(kk, pos, amp, cmap=CMAP, shading="auto", **kw)
    fig.colorbar(pcm, ax=ax,
                 label=_ALAB % ("m/s" if calibrated else "px/frame"))
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title("%s -- %s   (%s)" % (run, title, _anno), fontsize=10)
    fig.tight_layout()
    return fig


def _map_k_axis(K, sub):
    """The maps' k coordinate (DC dropped) per MAP_K_AXIS: values, the
    axis label (sub = 'h' or 'v'), the word for the title and whether the
    axis is a wavelength."""
    as_lambda = str(MAP_K_AXIS).lower().startswith("l")
    if as_lambda:
        return (1.0 / K[1:], r"$\lambda_%s$  (m)" % sub, "wavelength", True)
    return (K[1:], r"$k_%s$  (1/m)" % sub, "wavenumber", False)


def _map_k_limits(ax, which, kk, as_lambda):
    """MAP_LOGK log scale and the KMAX bound on the map's k/lambda axis
    (`which` = 'x' or 'y')."""
    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    if MAP_LOGK:
        set_scale("log")
    if KMAX:
        if as_lambda:
            set_lim(1.0 / KMAX, float(np.max(kk)))
        else:
            set_lim(float(np.min(kk)) if MAP_LOGK else 0, KMAX)


def make_map_kh():
    """Figure 2: kh (or lambda_h) on x, vertical position z on y."""
    kk, xlab, word, as_lambda = _map_k_axis(KH, "h")
    fig = _map_figure(kk, z_pos, mean_FFT_HORIZONTAL[:, 1:], xlab, "z  (m)",
                      "horizontal %s spectrum vs z" % word)
    _map_k_limits(fig.axes[0], "x", kk, as_lambda)
    return fig


def make_map_kv():
    """Figure 3: x on x, kv (or lambda_v) on y."""
    kk, ylab, word, as_lambda = _map_k_axis(KV, "v")
    fig = _map_figure(x_pos, kk, mean_FFT_VERTICAL[1:, :], "x  (m)", ylab,
                      "vertical %s spectrum vs x" % word)
    _map_k_limits(fig.axes[0], "y", kk, as_lambda)
    return fig


def process_run(path, show):
    """Compute one run's spatial spectra, save the npz and the 3 figures;
    keep the windows open when `show` (single-run mode). With
    VELOCITY_SOURCE='phase_average' the figure names carry a _PA tag."""
    globals().update(load_run(path))
    for _make, _name in ((make_1d_figure, "K_SPECTRUM_VELOCITY_1D"),
                         (make_map_kh, "K_SPECTRUM_VELOCITY_MAP_KH"),
                         (make_map_kv, "K_SPECTRUM_VELOCITY_MAP_KV")):
        _fig = _make()
        _save_or_skip(_fig, os.path.join(out_dir, "%s%s.%s"
                                         % (_name, src_tag, FIG_FORMAT)),
                      show)


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
