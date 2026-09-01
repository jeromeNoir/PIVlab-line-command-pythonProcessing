"""Auto-generated .py twin of batch_KineticEnergy.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # <a id='toc1_'></a>[Batch kinetic-energy post-processing of PIVlab runs](#toc0_)
# 
# Notebook version of `batch_KineticEnergy.py`.
# 
# Loops over every subfolder of `BASE_DIR`, loads `PIVlab_results_uncalibrated.mat`
# (the *wienerwurst* / `scipy.io.loadmat` branch), calibrates the field, and computes
# the kinetic-energy time series `Ek(t) = 0.5·⟨U² + V²⟩` — **for both regions**: the
# fixed ROI and the full field.
# 
# - Per run it writes `<run>/PostProcessing/KineticEnergy_timeSeries_ROI.npz`
#   **and** `..._FULL.npz`.
# - Across all runs it writes `KineticEnergy_summary_ROI.csv` / `.xlsx` **and**
#   `KineticEnergy_summary_FULL.csv` / `.xlsx` at `BASE_DIR`, plus a resonance figure
#   `KineticEnergy_vs_fstar_<region>.png` for each.
# 
# Calibration is read from the **last row** of `acquisition_log.txt`
# (`dt_vel = pulse_sep`, `fps = cam_fps/2` — PIVlab pairs images, so a
# velocity field is produced every 2 camera frames); `xscale = yscale = 1.2323e-4 m/px`.
# 
# Set the config in the next cell, then *Run All*.


# **Table of contents**<a id='toc0_'></a>    
# - [Batch kinetic-energy post-processing of PIVlab runs](#toc1_)    
#   - [Run](#toc1_1_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->


import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


from piv_postprocessing_lib import (topography_arrangement, dimensionless_numbers, figure_filename, libration_ke_scale,
                        libration_velocity_scale, load_piv, parse_run_name,
                        read_acquisition_params, read_paramPostprocessing,
                        region_fields, region_tag, regions_to_run)


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle. (Figures are unaffected.)
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = True
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# --- Configuration --------------------------------------------------------- #
BASE_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly")  # top-level folder containing the runs

# Calibration, ROI, filenames, ... for this dataset: read from
# param_postProcessing.json in BASE_DIR (also rebinds them inside
# piv_postprocessing_lib so its helpers use this dataset's values).
_P = read_paramPostprocessing(BASE_DIR)
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(BASE_DIR)
PIV_FILENAME, LOG_FILENAME = _P.PIV_FILENAME, _P.LOG_FILENAME
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
UNCAL_SCALE = _P.UNCAL_SCALE

# Per-run output and summary stems. The region tag ('ROI' / 'FULL') is appended,
# so each run gets KineticEnergy_timeSeries_ROI.npz AND _FULL.npz, and the batch
# writes KineticEnergy_summary_ROI.csv AND _FULL.csv.
RESULT_STEM  = "KineticEnergy_timeSeries"      # -> <stem>_<region>.npz
SUMMARY_STEM = "KineticEnergy_summary"         # -> <stem>_<region>.csv

# Which region(s) to process: 'BOTH' (ROI and FULL), 'ROI' only, or 'FULL' only.
REGION = 'ROI'


# Saved-figure format: 'png' or 'pdf'. Every figure is written twice
# -- a raw version and a '_normalized' one.
FIG_FORMAT = 'png'

# Only runs without an existing result are processed unless this is True.
REPROCESS_ALL = True

# Restrict which runs appear in the resonance FIGURE (the summary CSV always
# keeps every run). Set a value or list of values; None -> keep all. A filtered
# figure is shown but NOT saved, so it never overwrites the full one.
SELECT_FROT = None    # Hz,  e.g. 0.5
SELECT_FLIB = None    # Hz,  e.g. 0.44  or  [0.40, 0.44]  or  (0.4, 0.5) range
SELECT_DPHI = None    # deg, e.g. 2.0


# Fixed ROI, calibrated (metres): [(x0, y0), (x1, y1)] (opposite corners).

# PIVlab pairs images (1+2, 3+4, ...), so every velocity field consumes this
# many camera frames. The PIV field sampling frequency is therefore
# f_piv = cam_fps / FRAMES_PER_FIELD (cam_fps in the log is the camera frame
# rate, i.e. TWICE the rate at which PIV fields are produced).

# If the acquisition log cannot be read, the run stays UNCALIBRATED: velocity
# dt, PIV sampling and both spatial scales fall back to 1, so velocities are in
# px/frame, positions in px, and timestamps in frame index.

# Folder-name -> physical value conversions (Hz). Folder tokens are integers,
# e.g. 'frot050' -> 0.50 Hz, 'flib0400' -> 0.400 Hz, 'flib1500' -> 1.500 Hz.


def build_row(name, out_file, processed, region=np.nan, mean_Ekin=np.nan,
              std_Ekin=np.nan, dt_vel=np.nan, fps=np.nan, nframes=np.nan,
              npoints=np.nan, ok=np.nan):
    """Assemble one summary-table row (folder-name metadata + results/flags)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    # Session index: the integer following 'SS' in the folder name (SS2 -> 2).
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    # Dimensionless numbers (E, Ro, Re, ...) and normalised (_star) quantities.
    dl = dimensionless_numbers(frot_hz, flib_hz, dphi_deg)
    ke_scale = dl["U0_mps"] ** 2 if np.isfinite(dl["U0_mps"]) and dl["U0_mps"] else np.nan
    row = {"run": name, "run idx": run_idx, "region": region,
           "processed": processed, "frot_Hz": frot_hz, "flib_Hz": flib_hz,
           "dphi_deg": dphi_deg, "fstar": fstar, "flib_star": fstar,
           "calibrated": ok, "dt_vel_s": dt_vel, "fps_Hz": fps,
           "nframes": nframes, "npoints": npoints,
           "mean_Ekin": mean_Ekin, "std_Ekin": std_Ekin,
           "mean_Ekin_star": mean_Ekin / ke_scale,
           "std_Ekin_star": std_Ekin / ke_scale,
           "top_topo": TOP_TOPO, "bottom_topo": BOTTOM_TOPO, "npz": out_file}
    row.update(dl)
    return row


def process_run(run_dir, region, reprocess=False):
    """Process one run folder for one region ('ROI' or 'full'). Returns a summary dict.

    Writes <run>/PostProcessing/KineticEnergy_timeSeries_<region>.npz. If it already
    exists and reprocess is False, the row is rebuilt from that cache (the large .mat
    is not re-read). A folder with no PIV file yields processed=False and NaN metrics.
    """
    tag = region_tag(region)
    name = os.path.basename(run_dir.rstrip("/"))
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    piv_file = os.path.join(run_dir, PIV_FILENAME)
    out_dir = os.path.join(run_dir, "PostProcessing")
    out_file = os.path.join(out_dir, "%s_%s.npz" % (RESULT_STEM, tag))

    # Reuse an existing result unless a reprocess was requested.
    if os.path.isfile(out_file) and not reprocess:
        try:
            d = np.load(out_file, allow_pickle=True)
            print("  [skip] %-40s %-4s already processed (cached)" % (name, tag))
            return build_row(name, out_file, True, region=tag,
                             mean_Ekin=float(d["mean_Ekin"]),
                             std_Ekin=float(d["std_Ekin"]),
                             dt_vel=float(d["dt_vel"]), fps=float(d["fps"]),
                             nframes=int(d["nframes"]),
                             npoints=int(d["npoints"]) if "npoints" in d.files else np.nan,
                             ok=bool(d["calibrated"]))
        except Exception as exc:
            print("  [warn] %s: cached result unreadable (%s) -> reprocessing"
                  % (name, exc))

    if not os.path.isfile(piv_file):
        print("  [skip] no %s in %s" % (PIV_FILENAME, name))
        return build_row(name, "", False, region=tag)

    dt_vel, fps, ok = read_acquisition_params(os.path.join(run_dir, LOG_FILENAME))
    if ok:
        xscale, yscale = XSCALE, YSCALE
    else:
        print("  [warn] %s: Calibration not possible - all velocities will be "
              "in px/frame" % name)
        xscale = yscale = UNCAL_SCALE   # dt_vel, fps already fell back to 1

    X, Y, U, V, nframes = load_piv(piv_file)

    # Calibrate: positions -> m (or px), velocities -> m/s (or px/frame).
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel

    # Restrict to the region ('ROI' crops to PTS_ROI, 'full' keeps everything).
    Xr, Yr, Ur, Vr = region_fields(X, Y, U, V, PTS_ROI, region)
    npoints = Ur[..., 0].size

    # Kinetic energy time series over the region. Each PIV field is time-stamped
    # from the PIV field sampling frequency (cam_fps/2): t[i] = i / fps.
    speed2 = Ur ** 2 + Vr ** 2
    Ek_frame = 0.5 * np.nanmean(speed2.reshape(-1, nframes), axis=0)
    t = np.arange(nframes) / fps

    mean_Ekin = float(np.nanmean(Ek_frame))
    std_Ekin = float(np.nanstd(Ek_frame))

    # Normalised (_star) Ek: Ek / U0**2, U0 the libration wall velocity scale.
    U0 = libration_velocity_scale(flib_hz, dphi_deg)
    ke_scale = U0 ** 2 if np.isfinite(U0) and U0 else np.nan
    Ek_frame_star = Ek_frame / ke_scale
    mean_Ekin_star = mean_Ekin / ke_scale
    std_Ekin_star = std_Ekin / ke_scale

    # Save per-run npz (out_dir / out_file were defined at the top).
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_file,
             run=name, PIV_file=piv_file, region=tag, calibrated=ok,
             dt_vel=dt_vel, fps=fps, xscale=xscale, yscale=yscale,
             pts_ROI=np.array(PTS_ROI), npoints=npoints,
             nframes=nframes, t=t, Ek_frame=Ek_frame,
             mean_Ekin=mean_Ekin, std_Ekin=std_Ekin, U0=U0,
             Ek_frame_star=Ek_frame_star, mean_Ekin_star=mean_Ekin_star,
             std_Ekin_star=std_Ekin_star)

    print("  [ok] %-40s %-4s nframes=%d npts=%d fps=%.4gHz  <Ek>=%.4e  std=%.4e"
          % (name, tag, nframes, npoints, fps, mean_Ekin, std_Ekin))

    return build_row(name, out_file, True, region=tag, mean_Ekin=mean_Ekin,
                     std_Ekin=std_Ekin, dt_vel=dt_vel, fps=fps, nframes=nframes,
                     npoints=npoints, ok=ok)


def _match(series, sel, tol=1e-6):
    """Boolean mask selecting rows of `series`:
      None             -> all rows
      scalar           -> equal to that value (within tol)
      (lo, hi) tuple   -> range  lo <= value <= hi  (inclusive)
      [a, b, ...] list -> equal to any listed value
    """
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
    """Two panels: mean(Ek) and std(Ek) versus f* = flib/frot, for one region.

    The main resonance sweep (most common dphi) is a connected curve; runs at
    other dphi are overlaid as squares; repeat acquisitions of an identical
    (frot, flib, dphi) point (e.g. _SS1/_SS2) are shown as open diamonds.
    The figure is saved to base_dir (tagged with the region) and displayed inline.
    """
    tag = region or (str(df["region"].iloc[0]) if "region" in df and len(df) else "")

    # Only plot processed runs (unprocessed rows have NaN metrics).
    dfv = df.dropna(subset=["fstar", "mean_Ekin"]).copy()
    if dfv.empty:
        print("  (figure skipped: no processed runs with valid f*)")
        return None

    # Optional run selection (SELECT_FROT / SELECT_FLIB / SELECT_DPHI).
    _no_filter = all(v is None for v in (SELECT_FROT, SELECT_FLIB, SELECT_DPHI))
    dfv = dfv[_match(dfv["frot_Hz"], SELECT_FROT)
              & _match(dfv["flib_Hz"], SELECT_FLIB)
              & _match(dfv["dphi_deg"], SELECT_DPHI)].copy()
    if dfv.empty:
        print("  (figure skipped: no runs match the SELECT_* filter)")
        return None

    # Flag repeats: 2nd+ run sharing the same (frot, flib, dphi).
    dfv = dfv.sort_values(["frot_Hz", "flib_Hz", "dphi_deg", "run"])
    dfv["is_repeat"] = (dfv.groupby(["frot_Hz", "flib_Hz", "dphi_deg"])
                          .cumcount() > 0)
    dfv = dfv.sort_values("fstar")

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
        ax.plot(main["fstar"], main[col], "-o", color="C0",
                label=r"$\delta\varphi=%g^\circ$ (sweep)" % main_dphi)
        for dphi, g in dfv[~is_main & ~dfv["is_repeat"]].groupby("dphi_deg"):
            ax.plot(g["fstar"], g[col], "s", ms=7,
                    label=r"$\delta\varphi=%g^\circ$" % dphi)
        rep = dfv[dfv["is_repeat"]]
        if not rep.empty:
            ax.plot(rep["fstar"], rep[col], "D", ms=8, mfc="none",
                    mec="k", mew=1.5, label="repeat")
        ax.set_yscale("log")
        ax.set_xlabel(r"$f^* = f_{\mathrm{lib}} / f_{\mathrm{rot}}$", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    # Region in a figure-level heading: with REGION='BOTH' two figures are
    # produced (ROI and FULL) and they must be tellable apart at a glance.
    if tag:
        fig.suptitle("REGION: %s%s   ($k_0 = %g$, top=%s, bottom=%s)"
                     % (tag, "   (normalized)" if normalize else "", k0,
                        TOP_TOPO, BOTTOM_TOPO),
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    stem = ("KineticEnergy_vs_fstar_%s" % tag) if tag else "KineticEnergy_vs_fstar"
    out = os.path.join(base_dir, figure_filename(stem, fmt, normalized=normalize))
    if _no_filter:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % out)
    else:
        print("  (filtered view -- shown but not saved)")
    return fig


def run_batch(reprocess_all=REPROCESS_ALL):
    """Process every run under BASE_DIR for each region; write per-region summary
    + figure. Returns {region: DataFrame}."""
    subdirs = sorted(d for d in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(d)
                     and not os.path.basename(d).startswith("."))
    print("Found %d subfolders in %s%s"
          % (len(subdirs), BASE_DIR,
             "  (reprocessing ALL)" if reprocess_all else
             "  (skipping already-processed)"))

    results = {}
    for region in regions_to_run(REGION):
        tag = region_tag(region)
        print("\n========== REGION: %s ==========" % tag)
        rows = []
        for run_dir in subdirs:
            try:
                row = process_run(run_dir, region, reprocess=reprocess_all)
            except Exception as exc:  # keep the batch going
                print("  [error] %s: %s" % (os.path.basename(run_dir), exc))
                row = None
            if row is not None:
                rows.append(row)

        if not rows:
            print("  No runs found for region %s." % tag)
            continue

        df = pd.DataFrame(rows).sort_values(["frot_Hz", "flib_Hz", "dphi_deg"])
        csv_path = os.path.join(BASE_DIR, "%s_%s.csv" % (SUMMARY_STEM, tag))
        df.to_csv(csv_path, index=False)
        n_proc = int(df["processed"].sum())
        print("\n  Summary %s (%d runs: %d processed, %d unprocessed):\n    %s"
              % (tag, len(df), n_proc, len(df) - n_proc, csv_path))
        # Both a raw and a normalized figure for every region.
    for _norm in (False, True):
        plot_summary(df, BASE_DIR, tag, normalize=_norm)
        results[tag] = df

    return results


# ## <a id='toc1_1_'></a>[Run](#toc0_)
# 
# Runs the batch with the `REPROCESS_ALL` set in the config cell above. Set it to
# `True` (or call `run_batch(reprocess_all=True)`) to reprocess every run.


# Returns {region: DataFrame}, one summary per processed region (per REGION).
results = run_batch(reprocess_all=REPROCESS_ALL)
next(iter(results.values())).head(30) if results else results
