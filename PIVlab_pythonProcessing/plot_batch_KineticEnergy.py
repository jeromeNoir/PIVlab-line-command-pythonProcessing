"""Auto-generated .py twin of plot_batch_KineticEnergy.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Replot batch_KineticEnergy figures from the `.npz` (no reprocessing)
# 
# Reproduces **exactly** the per-run two-panel figure that `batch_KineticEnergy`
# writes -- `KineticEnergy_FFT_<region>[_normalized].<fmt>` (panel 1: `<Ek>(t)`;
# panel 2: both spectra `FFT(<Ek>)` and `<FFT(Ek)>`) -- but **without re-reading the
# PIV `.mat`**. It loops over every sub-folder of `BASE_DIR` and redraws from each
# run's cached `PostProcessing/KineticEnergy_<region>.npz`, which already stores
# `t`, `Ek_frame`, `freq`, `amp_fft_of_mean` and `amp_mean_of_fft`.
# 
# It reuses `batch_KineticEnergy`'s `save_fft_figure` **verbatim**, so with the
# axis limits left at `None` the output is byte-identical to the batch figures.
# 
# The four axis-limit constants set the **spectrum panel** (panel 2) limits, in
# physical units (raw figure: frequency [Hz], amplitude [m^2/s^2]) and normalized
# units (`f/f_rot`, amplitude `/E_lib`), exactly as in `plot_batch_Velocity`. The
# time-series panel is left auto. Nothing is recomputed and no `.npz`/summary is
# written.


# ## 1. Imports


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, fft_axis_limits,
                        figure_filename, libration_ke_scale, parse_run_name,
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
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly")

# k0 / topography for the figure titles come from the dataset's parameter file.
_P = read_paramPostprocessing(BASE_DIR)
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(BASE_DIR)

# Per-run stems (region tag appended) -- identical to batch_KineticEnergy.
RESULT_STEM = "KineticEnergy"       # per-run npz read from PostProcessing/
FFTFIG_STEM = "KineticEnergy_FFT"   # per-run 2-panel figure

# Which region(s): 'BOTH' (ROI and FULL), 'ROI' only, or 'FULL' only.
REGION = 'ROI'

FIG_FORMAT = 'png'
DETREND = True   # only affects the "(mean removed)" title tag when redrawing
LOGX = False
LOGY = True

# Redraw even if a figure already exists (False -> skip those runs).
OVERWRITE = True

# Restrict to these run-folder names (for testing). Empty -> all runs.
ONLY_RUNS = []

# --- Spectrum-panel axis limits (panel 2) --------------------------------- #
# Each is a (min, max) tuple, or None -> auto (exactly as batch_KineticEnergy).
# Physical (raw figure): x = frequency [Hz], y = amplitude [m^2/s^2].
XLIM_PHYS = None        # e.g. (0.0, 2.0)   Hz
YLIM_PHYS = None        # e.g. (1e-8, 1e-4) m^2/s^2   (log axis when LOGY)
# Normalised figure: x = f / f_rot, y = amplitude / E_lib.
XLIM_NORM = None        # e.g. (0.0, 4.0)
YLIM_NORM = (1e-6, 1e-1)        # e.g. (1e-3, 1e1)

# --- Resonance summary figure (mean/std Ek vs frequency) ------------------- #
# Redrawn from each region's KineticEnergy_summary_<region>.csv (no reprocessing).
# Raw figure: x = f_lib [Hz].  Normalized figure: x = f* = f_lib / f_rot.
REDRAW_RESONANCE = True                    # also replot the dataset resonance figure
SUMMARY_STEM  = "KineticEnergy_summary"    # <stem>_<region>.csv read from BASE_DIR
RESONFIG_STEM = "KineticEnergy_vs_fstar"   # output figure stem at BASE_DIR
# Optional run filters for the resonance figure (None -> keep all runs):
SELECT_FROT = None    # Hz,  e.g. 0.5
SELECT_FLIB = None    # Hz,  e.g. 0.44  or  [0.40, 0.44]  or  (0.4, 0.5) range
SELECT_DPHI = None    # deg, e.g. 2.0


# ## 3. Figure function (verbatim from batch_KineticEnergy)


# Labels say WHERE the region average sits relative to the FFT.
LBL_AVG_FIRST = r"$|\mathrm{FFT}(\langle E_k \rangle)|$   average, then FFT"
LBL_FFT_FIRST = r"$\langle |\mathrm{FFT}(E_k)| \rangle$   FFT, then average"


def save_fft_figure(out_png, t, ek, freq, amp_avg, amp_mean, run, flib,
                    frot=None, dphi_deg=None, normalize=False,
                    xlim=None, ylim=None):
    """Two-panel per-run figure: time series + both amplitude spectra.

    Panel 2 overlays FFT(<Ek>) (amp_avg) and <FFT(Ek)> (amp_mean). normalize=True
    scales Ek by the libration KE scale U0**2 and the frequency axis by f_rot.
    Closed straight away so a batch does not leave many windows open.
    """
    escale = (libration_ke_scale(dphi_deg, flib) if normalize and flib
              and dphi_deg is not None and np.isfinite(dphi_deg) else 1.0)
    fscale = frot if (normalize and frot and np.isfinite(frot)) else 1.0
    ek = ek / escale
    amp_avg = amp_avg / escale
    amp_mean = None if amp_mean is None else amp_mean / escale
    freq = freq / fscale
    _flib = flib / fscale if flib else flib
    _frot = frot / fscale if frot else frot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: the region-averaged time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel((r"$\langle E_k \rangle(t) / E_{\mathrm{lib}}$" if normalize
                        else r"$\langle E_k \rangle(t)$  (m$^2$/s$^2$)"), fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: both amplitude spectra
    plot = axes[1].semilogy if LOGY else axes[1].plot
    if amp_mean is not None:
        plot(freq, amp_mean, color="C2", lw=1, label=LBL_FFT_FIRST)
    plot(freq, amp_avg, color="C1", lw=1, label=LBL_AVG_FIRST)
    axes[1].set_xlabel(r"$f / f_{\mathrm{rot}}$" if normalize else "frequency (Hz)",
                       fontsize=13)
    axes[1].set_ylabel((r"$|\widehat{E_k}| / E_{\mathrm{lib}}$" if normalize
                        else r"$|\widehat{E_k}|$  (m$^2$/s$^2$)"), fontsize=13)
    axes[1].set_title("FFT amplitude spectrum" + ("  (mean removed)" if DETREND else ""),
                      fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=9)
    # Frequency axis to [0, max(4*frot,4*flib)]; y to decade bounds.
    curves = [amp_avg] if amp_mean is None else [amp_avg, amp_mean]
    _xmax, _ymin, _ymax = fft_axis_limits(freq, curves, _frot, _flib)
    if LOGX:
        axes[1].set_xscale("log")
        _pos = freq[(freq > 0) & (freq <= _xmax)]
        axes[1].set_xlim(_pos.min() if _pos.size else _xmax / 100.0, _xmax)
    else:
        axes[1].set_xlim(0, _xmax)
    if _ymin and _ymax:
        axes[1].set_ylim(_ymin, _ymax)
    if _flib:
        for fq, lbl in ((_flib, r"$f_{\mathrm{lib}}$"),
                        (2 * _flib, r"$2f_{\mathrm{lib}}$")):
            if fq <= freq.max():
                axes[1].axvline(fq, color="k", ls="--", lw=1, alpha=0.6)
                axes[1].annotate(lbl, xy=(fq, 1),
                                 xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points",
                                 fontsize=10)

    fig.suptitle(run + ("   (normalized)" if normalize else "")
                 + "   ($k_0 = %g$, top=%s, bottom=%s)" % (k0, TOP_TOPO, BOTTOM_TOPO),
                 fontsize=13)
    # Optional user-set limits on the spectrum panel (None -> auto).
    if xlim is not None:
        axes[1].set_xlim(xlim)
    if ylim is not None:
        axes[1].set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ## 4. Redraw every run from its `.npz`


def redraw_run(run_dir, region):
    """Redraw one run's two-panel KE figure from its cached .npz."""
    tag = region_tag(region)
    name = os.path.basename(run_dir.rstrip("/"))
    out_dir = os.path.join(run_dir, "PostProcessing")
    npz = os.path.join(out_dir, "%s_%s.npz" % (RESULT_STEM, tag))
    if not os.path.isfile(npz):
        print("  [skip] %-40s %-4s no %s" % (name, tag, os.path.basename(npz)))
        return False

    d = np.load(npz, allow_pickle=True)
    if not all(k in d.files for k in ("t", "Ek_frame", "freq", "amp_fft_of_mean")):
        print("  [skip] %-40s %-4s npz lacks the FFT arrays (old format)" % (name, tag))
        return False
    t = np.asarray(d["t"], float)
    ek = np.asarray(d["Ek_frame"], float)
    freq = np.asarray(d["freq"], float)
    amp_avg = np.asarray(d["amp_fft_of_mean"], float)
    amp_mean = (np.asarray(d["amp_mean_of_fft"], float)
                if "amp_mean_of_fft" in d.files else None)
    if amp_mean is not None and np.all(np.isnan(amp_mean)):
        amp_mean = None
    frot, flib, dphi = parse_run_name(name)
    flib = None if not np.isfinite(flib) else flib
    frot = None if not np.isfinite(frot) else frot

    fig_stem = os.path.join(out_dir, "%s_%s" % (FFTFIG_STEM, tag))
    drew = False
    for _norm in (False, True):
        _out = figure_filename(fig_stem, FIG_FORMAT, normalized=_norm)
        _xl = XLIM_NORM if _norm else XLIM_PHYS
        _yl = YLIM_NORM if _norm else YLIM_PHYS
        if OVERWRITE or not os.path.isfile(_out):
            save_fft_figure(_out, t, ek, freq, amp_avg, amp_mean,
                            "%s  (%s)" % (name, tag), flib, frot, dphi,
                            normalize=_norm, xlim=_xl, ylim=_yl)
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


# ## 5. Redraw the resonance summary from the CSV
# 
# Reads each region's `KineticEnergy_summary_<region>.csv` at `BASE_DIR` and redraws the raw (x = `f_lib` [Hz]) and normalized (x = `f*`) `mean/std Ek` figure. Nothing is recomputed.


def _match(series, sel, tol=1e-6):
    """Boolean mask selecting rows of `series` (mirrors batch_KineticEnergy):
      None -> all; scalar -> equal (within tol); (lo, hi) -> range; [..] -> any."""
    if sel is None:
        return pd.Series(True, index=series.index)
    if isinstance(sel, tuple) and len(sel) == 2:
        lo, hi = sel
        return (series >= lo - tol) & (series <= hi + tol)
    vals = sel if isinstance(sel, (list, set)) else [sel]
    mask = pd.Series(False, index=series.index)
    for v in vals:
        mask |= (series - v).abs() <= tol
    return mask


def plot_summary(df, base_dir, region="", normalize=True, fmt=FIG_FORMAT):
    """Two panels: mean(Ek) and std(Ek) for one region. The x-axis is
    f* = flib/frot on the normalized figure and f_lib in Hz on the raw one.

    The main resonance sweep (most common dphi) is a connected curve; runs at
    other dphi are overlaid as squares; repeat acquisitions of an identical
    (frot, flib, dphi) point are shown as open diamonds. Saved to base_dir
    (tagged with the region) and displayed inline.
    """
    tag = region or (str(df["region"].iloc[0]) if "region" in df and len(df) else "")

    # Normalized figure: x = f* = flib/frot. Raw figure: x = f_lib in Hz.
    xcol = "fstar" if normalize else "flib_Hz"
    xlabel = (r"$f^* = f_{\mathrm{lib}} / f_{\mathrm{rot}}$" if normalize
              else r"$f_{\mathrm{lib}}$  (Hz)")

    dfv = df.dropna(subset=[xcol, "mean_Ekin"]).copy()
    if dfv.empty:
        print("  (figure skipped: no processed runs with a valid %s)" % xcol)
        return None

    _no_filter = all(v is None for v in (SELECT_FROT, SELECT_FLIB, SELECT_DPHI))
    dfv = dfv[_match(dfv["frot_Hz"], SELECT_FROT)
              & _match(dfv["flib_Hz"], SELECT_FLIB)
              & _match(dfv["dphi_deg"], SELECT_DPHI)].copy()
    if dfv.empty:
        print("  (figure skipped: no runs match the SELECT_* filter)")
        return None

    dfv = dfv.sort_values(["frot_Hz", "flib_Hz", "dphi_deg", "run"])
    dfv["is_repeat"] = (dfv.groupby(["frot_Hz", "flib_Hz", "dphi_deg"])
                          .cumcount() > 0)
    dfv = dfv.sort_values(xcol)

    main_dphi = dfv.loc[~dfv["is_repeat"], "dphi_deg"].mode().iloc[0]
    is_main = (dfv["dphi_deg"] == main_dphi) & ~dfv["is_repeat"]

    if normalize:
        scale = libration_ke_scale(dfv["dphi_deg"], dfv["flib_Hz"])
        dfv["mean_Ekin"] = dfv["mean_Ekin"] / scale
        dfv["std_Ekin"] = dfv["std_Ekin"] / scale
        panels = [("mean_Ekin", r"$\langle E_k \rangle / E_{\mathrm{lib}}$",
                   "Mean kinetic energy (normalized)"),
                  ("std_Ekin", r"std $E_k$ / $E_{\mathrm{lib}}$",
                   "Std of kinetic energy (normalized)")]
    else:
        panels = [("mean_Ekin", r"$\langle E_k \rangle$  (m$^2$/s$^2$)",
                   "Mean kinetic energy"),
                  ("std_Ekin", r"std $E_k$  (m$^2$/s$^2$)",
                   "Std of kinetic energy")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, ylabel, title) in zip(axes, panels):
        main = dfv[is_main]
        ax.plot(main[xcol], main[col], "-o", color="C0",
                label=r"$\delta\varphi=%g^\circ$ (sweep)" % main_dphi)
        for dphi, g in dfv[~is_main & ~dfv["is_repeat"]].groupby("dphi_deg"):
            ax.plot(g[xcol], g[col], "s", ms=7,
                    label=r"$\delta\varphi=%g^\circ$" % dphi)
        rep = dfv[dfv["is_repeat"]]
        if not rep.empty:
            ax.plot(rep[xcol], rep[col], "D", ms=8, mfc="none",
                    mec="k", mew=1.5, label="repeat")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    if tag:
        fig.suptitle("REGION: %s%s   ($k_0 = %g$, top=%s, bottom=%s)"
                     % (tag, "   (normalized)" if normalize else "", k0,
                        TOP_TOPO, BOTTOM_TOPO),
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    stem = ("%s_%s" % (RESONFIG_STEM, tag)) if tag else RESONFIG_STEM
    out = os.path.join(base_dir, figure_filename(stem, fmt, normalized=normalize))
    if _no_filter:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % out)
    else:
        print("  (filtered view -- shown but not saved)")
    return fig


def replot_resonance():
    """Redraw the raw + normalized resonance figure for each region from its
    KineticEnergy_summary_<region>.csv at BASE_DIR (no reprocessing)."""
    n = 0
    for region in regions_to_run(REGION):
        tag = region_tag(region)
        csv = os.path.join(BASE_DIR, "%s_%s.csv" % (SUMMARY_STEM, tag))
        if not os.path.isfile(csv):
            print("  [skip] no %s" % os.path.basename(csv)); continue
        df = pd.read_csv(csv)
        print("\n========== RESONANCE: %s  (%d runs) ==========" % (tag, len(df)))
        for _norm in (False, True):          # raw (x=f_lib) and normalized (x=f*)
            plot_summary(df, BASE_DIR, tag, normalize=_norm)
            n += 1
    print("\n%d resonance figure(s) redrawn." % n)
    return n


if REDRAW_RESONANCE:
    _ = replot_resonance()
