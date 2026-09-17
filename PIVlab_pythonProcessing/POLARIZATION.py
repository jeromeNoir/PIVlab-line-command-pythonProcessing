"""Auto-generated .py twin of POLARIZATION.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # POLARIZATION -- viewer
# 
# Polarization of the velocity field from each run's stored per-point spectra
# (`Velocity.npz`; nothing is recomputed from the `.mat`). Draws the **single
# run `RUN_DIR`** (`BATCH = False`: figures saved AND shown) or **every run
# sub-folder of every dataset in `BASE_DIRS`** that has a `Velocity.npz`
# (`BATCH = True`: figures saved, not shown).
# 
# Per grid point the power ratio `(FFT_V / FFT_U)**2` is computed from the
# stored spectra (calibrated: x `UCAL` / `VCAL`), with a **threshold**: any
# `FFT_U` or `FFT_V` amplitude below `MIN_FFT_AMP` (physical units, m/s) is
# replaced by NaN, so the ratio there is NaN and drops out of the statistics.
# The figure shows the `REGION` statistics per frequency -- mean, median and
# the 25-75% band -- against the inertial-wave relation
# `2[(2 f_rot/f)**2 - 1]`.
# 
# Per run, `POLARIZATION_DIM` / `_NODIM` are written into
# `<run>/PostProcessing/` (`_NODIM`: frequency / `F_SCALE`).


import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from piv_postprocessing_lib import figure_filename, resolve_npz


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
# BATCH = True  -> draw the polarization for every run sub-folder of every
#                  dataset in BASE_DIRS that has a Velocity.npz: figures are
#                  SAVED but not shown;
# BATCH = False -> only the single run RUN_DIR: figures saved AND shown.
BATCH = False

RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly/"
           "frot0.50Hz_flib0.400Hz_dphi2deg_SS1")  # run folder or Velocity.npz
                                                   # (BATCH = False)

ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
        "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets swept when BATCH = True (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

REGION = "ROI"          # 'ROI' or 'FULL': which region the statistics use

# Threshold on the spectra, PHYSICAL units [m/s]: any FFT_U or FFT_V
# amplitude below this is set to NaN, so the ratio (FFT_V/FFT_U)**2 there is
# NaN and drops out of the mean/median/percentiles.
MIN_FFT_AMP = 1e-5

SAVE = True             # write the figures next to the .npz
OVERWRITE_FIG = True    # False -> keep existing figure files
FIG_FORMAT = "png"      # figure format: png or pdf


# ## Load one run and build the thresholded polarization statistics


def load_run(path):
    """Load one run's Velocity.npz and build the thresholded polarization
    statistics (dict applied to the module globals by process_run)."""
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
    UCAL, VCAL, FCAL = (_s(k) for k in ("UCAL", "VCAL", "FCAL"))
    F_SCALE = _s("F_SCALE")

    f = _g("FREQ").astype(float) * FCAL                  # Hz

    # Per-point spectra in PHYSICAL units, thresholded: amplitudes below
    # MIN_FFT_AMP become NaN, so the ratio there is NaN too.
    FU = _g("FFT_U").astype(float) * UCAL
    FV = _g("FFT_V").astype(float) * VCAL
    FU[FU < MIN_FFT_AMP] = np.nan
    FV[FV < MIN_FFT_AMP] = np.nan
    n_cut = int(np.isnan(FU).sum() + np.isnan(FV).sum())
    if REGION == "ROI":
        _m = _g("MASK_ROI").astype(float)[:, :, None]
        FU = FU * _m
        FV = FV * _m
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = (FV / FU) ** 2
    rp[~np.isfinite(rp)] = np.nan
    rp = rp.reshape(-1, f.size)
    pol_mean = np.nanmean(rp, axis=0)
    pol_med = np.nanmedian(rp, axis=0)
    pol_p25 = np.nanpercentile(rp, 25, axis=0)
    pol_p75 = np.nanpercentile(rp, 75, axis=0)
    del rp, FU, FV

    funit = "Hz" if calibrated else "1/frame"
    _anno = "$k_0 = %g$, top=%s, bottom=%s" % (k0, top_topo, bottom_topo)

    print("run   : %s  (%s, %scalibrated)   frot = %s Hz   flib = %s Hz"
          % (run, REGION, "" if calibrated else "NOT ", frot, flib))
    print("threshold: MIN_FFT_AMP = %g m/s -> %d spectrum samples NaNed"
          % (MIN_FFT_AMP, n_cut))

    return dict(npz_path=npz_path, out_dir=os.path.dirname(npz_path),
                run=run, calibrated=calibrated, frot=frot, flib=flib,
                dphi=dphi, F_SCALE=F_SCALE, f=f,
                pol_mean=pol_mean, pol_med=pol_med,
                pol_p25=pol_p25, pol_p75=pol_p75,
                funit=funit, _anno=_anno)


# ## The figure and the run driver


def make_polarization(normalize):
    """The polarization figure; normalize=True divides frequencies by
    F_SCALE (labels f*)."""
    fscale = F_SCALE if (normalize and np.isfinite(F_SCALE) and F_SCALE) else 1.0
    ff = f / fscale
    m = f > 0
    fig, ax = plt.subplots(figsize=(9, 6))
    if np.isfinite(frot) and frot:
        theory = 2.0 * ((2.0 * frot / f[m]) ** 2 - 1.0)
        ax.plot(ff[m], theory, "k-", lw=2.2,
                label=r"$2[(2f_{\mathrm{rot}}/f)^2-1]$  (IW)")
    ax.plot(ff[m], pol_mean[m], lw=1.2, label=r"mean $P_V/P_U$")
    line, = ax.plot(ff[m], pol_med[m], lw=1.2, label=r"median $P_V/P_U$")
    ax.fill_between(ff[m], pol_p25[m], pol_p75[m], color=line.get_color(),
                    alpha=0.20, lw=0, label="%s 25-75%%" % REGION)
    ax.set_yscale("log")
    ax.set_xlim(0.01 / fscale, 1.0 / fscale)
    ax.set_xlabel(r"$f^*$" if fscale != 1.0 else "frequency (%s)" % funit,
                  fontsize=13)
    ax.set_ylabel(r"polarization  $P_V/P_U$", fontsize=13)
    ax.set_title("%s  (%s)%s   (%s)"
                 % (run, REGION,
                    "   (non-dimensional)" if normalize else "", _anno),
                 fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def process_run(path, show):
    """Load one run, draw and save POLARIZATION_DIM / _NODIM into its
    PostProcessing folder; show when `show` (single-run mode)."""
    globals().update(load_run(path))
    for _norm in (False, True):
        _fig = make_polarization(_norm)
        if SAVE:
            _out = figure_filename(os.path.join(out_dir, "POLARIZATION"),
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
