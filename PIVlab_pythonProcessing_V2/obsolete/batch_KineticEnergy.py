
"""
Batch post-processing of PIVlab results: mean kinetic energy per run.

Loops over every subfolder of BASE_DIR, loads 'PIVlab_results_uncalibrated.mat'
(the wienerwurst / scipy-loadmat branch of processPIVLab_results.ipynb),
calibrates the field, crops to a fixed ROI, and computes the kinetic-energy
time series Ek(t) = 0.5 * <U^2 + V^2>_ROI.

Per run it writes  <run>/PostProcessing/KineticEnergy_timeSeries.npz.
Across all runs it writes a summary table (CSV + XLSX) at BASE_DIR, with
frot / flib / dphi parsed from the folder name -> feeds the resonance curve.

Decisions baked in (see conversation):
  - Quantity      : kinetic energy only.
  - ROI           : fixed rectangle (notebook default, calibrated metres).
  - Calibration   : read from the LAST row of acquisition_log.txt --
                      dt_vel = pulse_sep (us -> s)  : displacement -> velocity
                      fps    = cam_fps/2 (Hz)       : PIV field timestamps t=i/fps
                                                      (image pairing halves cam_fps)
                    xscale = yscale = 1.2323e-4 m/px (notebook value).
                    If the log can't be read, the run stays uncalibrated
                    (scales=dt=fps=1 -> velocities in px/frame).
Run with the dpivsoft (or any numpy/scipy/pandas) environment, e.g.
  /Users/jeromenoir/anaconda3/bin/python batch_KineticEnergy.py
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from piv_common import (LOG_FILENAME, PIV_FILENAME, PTS_ROI, UNCAL_SCALE, XSCALE,
                        create_mask, extract_roi, load_piv, parse_run_name,
                        read_acquisition_params)

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom")

RESULT_FILENAME = "KineticEnergy_timeSeries.npz"   # per-run output (in PostProcessing/)

# By default only runs without an existing result are processed. Set True (or
# pass --reprocess-all on the command line) to reprocess every run.
REPROCESS_ALL = True

# Pixel calibration (notebook value). Same in x and y.
YSCALE = XSCALE

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
# Adjust these divisors if the naming convention of a dataset differs.

# --------------------------------------------------------------------------- #
#  Data loading (wienerwurst branch of processPIVLab_results.ipynb)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  Metadata helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  Per-run processing
# --------------------------------------------------------------------------- #
def build_row(name, out_file, processed, mean_Ekin=np.nan, std_Ekin=np.nan,
              dt_vel=np.nan, fps=np.nan, nframes=np.nan, ok=np.nan):
    """Assemble one summary-table row (folder-name metadata + results/flags)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    # Session index: the integer following 'SS' in the folder name (SS2 -> 2).
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    return {"run": name, "run idx": run_idx, "processed": processed,
            "frot_Hz": frot_hz, "flib_Hz": flib_hz, "dphi_deg": dphi_deg,
            "fstar": fstar, "calibrated": ok, "dt_vel_s": dt_vel, "fps_Hz": fps,
            "nframes": nframes, "mean_Ekin": mean_Ekin, "std_Ekin": std_Ekin,
            "npz": out_file}


def process_run(run_dir, reprocess=False):
    """Process one run folder. Returns a summary dict.

    If a per-run result .npz already exists and reprocess is False, the row is
    rebuilt from that cache (the large .mat is not re-read). A folder with no
    PIV file yields a row with processed=False and NaN metrics.
    """
    name = os.path.basename(run_dir.rstrip("/"))
    piv_file = os.path.join(run_dir, PIV_FILENAME)
    out_dir = os.path.join(run_dir, "PostProcessing")
    out_file = os.path.join(out_dir, RESULT_FILENAME)

    # Reuse an existing result unless a reprocess was requested.
    if os.path.isfile(out_file) and not reprocess:
        try:
            d = np.load(out_file, allow_pickle=True)
            print("  [skip] %-40s already processed (cached)" % name)
            return build_row(name, out_file, True,
                             mean_Ekin=float(d["mean_Ekin"]),
                             std_Ekin=float(d["std_Ekin"]),
                             dt_vel=float(d["dt_vel"]), fps=float(d["fps"]),
                             nframes=int(d["nframes"]), ok=bool(d["calibrated"]))
        except Exception as exc:
            print("  [warn] %s: cached result unreadable (%s) -> reprocessing"
                  % (name, exc))

    if not os.path.isfile(piv_file):
        print("  [skip] no %s in %s" % (PIV_FILENAME, name))
        return build_row(name, "", False)

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

    # Crop to the fixed ROI.
    mask = create_mask(X, Y, PTS_ROI)
    Xr, Yr, Ur, Vr = extract_roi(X, Y, U, V, mask)
    if Ur.size == 0:
        print("  [warn] ROI empty for %s -> using full field" % name)
        Xr, Yr, Ur, Vr = X, Y, U, V

    # Kinetic energy time series over the ROI. Each PIV field is time-stamped
    # from the PIV field sampling frequency (cam_fps/2): t[i] = i / fps.
    speed2 = Ur ** 2 + Vr ** 2
    Ek_frame = 0.5 * np.nanmean(speed2.reshape(-1, nframes), axis=0)
    t = np.arange(nframes) / fps

    mean_Ekin = float(np.nanmean(Ek_frame))
    std_Ekin = float(np.nanstd(Ek_frame))

    # Save per-run npz (out_dir / out_file were defined at the top).
    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_file,
             run=name, PIV_file=piv_file, calibrated=ok,
             dt_vel=dt_vel, fps=fps, xscale=xscale, yscale=yscale,
             pts_ROI=np.array(PTS_ROI),
             nframes=nframes, t=t, Ek_frame=Ek_frame,
             mean_Ekin=mean_Ekin, std_Ekin=std_Ekin)

    print("  [ok] %-40s nframes=%d dt_vel=%.5gs fps=%.4gHz  <Ek>=%.4e  std=%.4e"
          % (name, nframes, dt_vel, fps, mean_Ekin, std_Ekin))

    return build_row(name, out_file, True, mean_Ekin, std_Ekin,
                     dt_vel, fps, nframes, ok)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(reprocess_all=REPROCESS_ALL):
    subdirs = sorted(d for d in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(d)
                     and not os.path.basename(d).startswith("."))
    print("Found %d subfolders in %s%s"
          % (len(subdirs), BASE_DIR,
             "  (reprocessing ALL)" if reprocess_all else
             "  (skipping already-processed)"))

    rows = []
    for run_dir in subdirs:
        try:
            row = process_run(run_dir, reprocess=reprocess_all)
        except Exception as exc:  # keep the batch going
            print("  [error] %s: %s" % (os.path.basename(run_dir), exc))
            row = None
        if row is not None:
            rows.append(row)

    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows).sort_values(["frot_Hz", "flib_Hz", "dphi_deg"])
    csv_path = os.path.join(BASE_DIR, "KineticEnergy_summary.csv")
    df.to_csv(csv_path, index=False)
    n_proc = int(df["processed"].sum())
    print("\nSummary (%d runs: %d processed, %d unprocessed) written to:\n  %s"
          % (len(df), n_proc, len(df) - n_proc, csv_path))
    try:
        xlsx_path = os.path.join(BASE_DIR, "KineticEnergy_summary.xlsx")
        df.to_excel(xlsx_path, index=False)
        print("  %s" % xlsx_path)
    except Exception as exc:
        print("  (xlsx skipped: %s)" % exc)

    plot_summary(df, BASE_DIR)


# --------------------------------------------------------------------------- #
#  Summary figure: kinetic energy vs f* = flib/frot
# --------------------------------------------------------------------------- #
def plot_summary(df, base_dir):
    """Two panels: mean(Ek) and std(Ek) versus f* = flib/frot.

    The main resonance sweep (most common dphi, i.e. 2 deg here) is drawn as a
    connected curve; runs at other dphi (the flib1500 dphi sweep, all at f*=3)
    are overlaid as separate markers. Repeat acquisitions of an identical
    (frot, flib, dphi) point (e.g. _SS1/_SS2) are shown with a distinct marker
    rather than folded into the curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only plot processed runs (unprocessed rows have NaN metrics).
    dfv = df.dropna(subset=["fstar", "mean_Ekin"]).copy()
    if dfv.empty:
        print("  (figure skipped: no processed runs with valid f*)")
        return

    # Flag repeats: 2nd+ run sharing the same (frot, flib, dphi).
    dfv = dfv.sort_values(["frot_Hz", "flib_Hz", "dphi_deg", "run"])
    dfv["is_repeat"] = (dfv.groupby(["frot_Hz", "flib_Hz", "dphi_deg"])
                          .cumcount() > 0)
    dfv = dfv.sort_values("fstar")

    main_dphi = dfv.loc[~dfv["is_repeat"], "dphi_deg"].mode().iloc[0]
    is_main = (dfv["dphi_deg"] == main_dphi) & ~dfv["is_repeat"]

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

    fig.tight_layout()
    out = os.path.join(base_dir, "KineticEnergy_vs_fstar.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figure written to:\n  %s" % out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch kinetic-energy post-processing of PIVlab runs.")
    parser.add_argument("--reprocess-all", action="store_true",
                        default=REPROCESS_ALL,
                        help="Reprocess every run, ignoring existing results "
                             "(default: only process runs not yet processed).")
    args = parser.parse_args()
    main(reprocess_all=args.reprocess_all)
