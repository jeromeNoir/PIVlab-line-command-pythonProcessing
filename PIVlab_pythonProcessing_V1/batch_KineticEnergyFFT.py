"""
# Batch FFT of the kinetic-energy time series

Batch version of [`fft_KineticEnergy.ipynb`](fft_KineticEnergy.ipynb): instead of
one run, it loops over **every sub-folder of `BASE_DIR`** and writes a per-run
figure next to that run's `KineticEnergy_timeSeries.npz`
(`KineticEnergy_FFT.png`).

**Two spectra are computed, differing only in when the ROI average is taken:**

| curve | meaning | order |
|---|---|---|
| `fft(Ek_mean)` | FFT of the ROI-averaged series `<Ek>_ROI(t)` (the `Ek_frame` stored in the `.npz`) | **average, then FFT** |
| `fft(Ek)_mean` | FFT of `Ek(t)` at **each ROI grid point**, amplitudes then averaged over the ROI | **FFT, then average** |

They are not the same: averaging first cancels fluctuations that are incoherent
across the ROI, so `fft(Ek_mean)` is normally the smaller of the two; the gap
between them measures how spatially coherent the signal at each frequency is.

`fft(Ek)_mean` needs the per-point field, which the `.npz` does not store, so the
run's PIV `.mat` is re-read and the ROI/calibration recorded in the `.npz` are
re-applied. If the `.mat` is missing, only `fft(Ek_mean)` is drawn.

Both use the same normalisation as `fft_KineticEnergy`: the one-sided amplitude
`|FFT| * 2 / N` (DC and Nyquist keep `1/N`), with the mean removed first by
default.

The figure has two panels: (1) the `<Ek>_ROI(t)` time series; (2) both amplitude
spectra, with dashed guides at `flib` and `2*flib`. Across all runs it also
writes `KineticEnergyFFT_summary.csv` / `.xlsx` at `BASE_DIR`.

Set the config in the next cell, then *Run All*. To test on a couple of folders
first, list their names in `ONLY_RUNS`.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


# --- Configuration --------------------------------------------------------- #
BASE_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_bottomOnly")

NPZ_NAME = "KineticEnergy_timeSeries.npz"   # per-run input (in PostProcessing/)
FIG_NAME = "KineticEnergy_FFT.png"          # per-run figure, written beside it
PIV_FILENAME = "PIVlab_results_uncalibrated.mat"   # re-read for the per-point FFT

# Runs whose figure already exists are skipped unless this is True.
REPROCESS_ALL = True

# Restrict processing to these run-folder names (for testing). Empty -> all runs.
ONLY_RUNS = []

DETREND = True        # remove the mean before the FFT (kills the DC spike)
LOGY = True           # logarithmic amplitude axis (False -> linear)

# Compute the per-point spectrum fft(Ek)_mean (re-reads the PIV .mat, so it is
# the slow part). False -> only fft(Ek_mean), straight from the .npz.
PER_POINT = True

# Folder-name -> physical value conversions (Hz). Folder tokens are integers,
# e.g. 'frot050' -> 0.50 Hz, 'flib0400' -> 0.400 Hz, 'flib1500' -> 1.500 Hz.
FROT_DIVISOR = 100.0
FLIB_DIVISOR = 1000.0


def parse_freq_token(name, tag, divisor):
    """Frequency [Hz] carried by a folder-name token.

    The current naming states the value in Hz: 'frot0.50Hz' -> 0.5. The legacy
    naming used a zero-padded integer: 'frot050' -> 050/divisor -> 0.5. The Hz
    form is tried first, and the legacy form is still accepted so that older
    summary tables (which store the old run names) keep parsing.

    NaN if the token is absent. Note the legacy pattern would misread an Hz name
    ('frot0.50Hz' -> 'frot0' -> 0.0), which is why order matters here.
    """
    m = re.search(r"%s([\d.]+)Hz" % tag, name)
    if m:
        return float(m.group(1))
    m = re.search(r"%s(\d+)" % tag, name)
    return float(m.group(1)) / divisor if m else np.nan


def parse_run_name(name):
    """Parse frot/flib (Hz) and dphi (deg) from a folder name.

    e.g. 'frot0.50Hz_flib0.400Hz_dphi2.5deg_SS1' -> (0.5, 0.4, 2.5), and the legacy
    'frot050_flib0400_dphi2.5deg_SS1' gives the same.
    Missing tokens come back as NaN.
    """
    frot_hz = parse_freq_token(name, "frot", FROT_DIVISOR)
    flib_hz = parse_freq_token(name, "flib", FLIB_DIVISOR)
    dphi = re.search(r"dphi([\d.]+)deg", name)
    dphi_deg = float(dphi.group(1)) if dphi else np.nan
    return frot_hz, flib_hz, dphi_deg


def find_npz(run_dir):
    """The run's KineticEnergy_timeSeries.npz, or None if it has none."""
    for cand in (os.path.join(run_dir, "PostProcessing", NPZ_NAME),
                 os.path.join(run_dir, NPZ_NAME)):
        if os.path.isfile(cand):
            return cand
    return None


def load_piv(file_path):
    """Load a PIVlab wienerwurst .mat file and return calibrated-ready fields.

    Returns X, Y (2D grids), U, V (validated velocity, NaN where invalid),
    and nframes. Identical to batch_KineticEnergy so the ROI/energy match.
    """
    mat = loadmat(file_path,
                  variable_names=["x", "y", "u", "v", "u_filt", "v_filt"])
    if "u_filt" not in mat:
        raise ValueError("Not a wienerwurst PIV file: %s" % file_path)

    X_original = mat["x"][:, :, 0]
    Y_original = mat["y"][:, :, 0]

    U_filtered = mat["u_filt"]
    V_filtered = mat["v_filt"]
    U_org = mat["u"]           # velocity prior to validation
    V_org = mat["v"]

    # Flip axes (match notebook)
    X = X_original[:, ::-1].astype(float, copy=False)
    Y = Y_original[::-1, :].astype(float, copy=False)
    U = U_filtered[::-1, ::-1, :].astype(float, copy=False)
    V = V_filtered[::-1, ::-1, :].astype(float, copy=False)
    U_original = U_org[::-1, ::-1, :]
    V_original = V_org[::-1, ::-1, :]

    nframes = U_filtered.shape[2]

    # Mask wherever the un-validated vectors are NaN
    mask = np.isnan(U_original) | np.isnan(V_original)
    U[mask] = np.nan
    V[mask] = np.nan

    Y = Y.max() - Y
    V = -V

    return X, Y, U, V, nframes


def create_mask(x, y, pts_roi):
    """Boolean mask of grid points inside the ROI rectangle."""
    pts = np.asarray(pts_roi).reshape(2, 2)
    (x1, y1), (x2, y2) = pts
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def extract_roi(x, y, u, v, mask):
    """Crop x, y (2D) and u, v (3D, frames last) to the ROI bounding box."""
    if not np.any(mask):
        return (np.array([]),) * 4
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    x_roi = x[np.ix_(rows, cols)]
    y_roi = y[np.ix_(rows, cols)]
    u_roi = u[np.ix_(rows, cols, np.arange(u.shape[2]))]
    v_roi = v[np.ix_(rows, cols, np.arange(v.shape[2]))]
    return x_roi, y_roi, u_roi, v_roi


def _amp_from_rfft(fft, n):
    """One-sided amplitude |FFT|*2/N along the last axis (DC/Nyquist keep 1/N)."""
    amp = np.abs(fft) * 2.0 / n
    amp[..., 0] = np.abs(fft[..., 0]) / n        # DC term is not doubled
    if n % 2 == 0:
        amp[..., -1] = np.abs(fft[..., -1]) / n  # Nyquist is not doubled either
    return amp


def compute_fft(ek, fps, detrend=DETREND):
    """fft(Ek_mean): FFT of the ROI-averaged series -> average, THEN FFT.

    Returns (freq, amp, fft) for the positive-frequency half of Ek(t), with the
    same normalisation as fft_KineticEnergy.
    """
    ek = np.asarray(ek, dtype=float)
    # Fill any NaNs (e.g. fully-masked frames) with the series mean.
    if np.any(np.isnan(ek)):
        ek = np.where(np.isnan(ek), np.nanmean(ek), ek)
    if detrend:
        ek = ek - ek.mean()
    n = ek.size
    fft = np.fft.rfft(ek)
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    return freq, _amp_from_rfft(fft, n), fft


def compute_fft_per_point(ek_pt, fps, detrend=DETREND):
    """fft(Ek)_mean: FFT at each ROI point, THEN average the amplitudes.

    ek_pt has shape (npoints, nframes) - the kinetic-energy time series at every
    ROI grid point. Each point is transformed on its own with the same
    normalisation as compute_fft, and |FFT| is then averaged over the ROI.
    Points that are entirely NaN are dropped; remaining NaNs are filled with
    that point's own mean.
    Returns (freq, amp_mean, npoints_used).
    """
    A = np.asarray(ek_pt, dtype=float)
    n = A.shape[-1]
    freq = np.fft.rfftfreq(n, d=1.0 / fps)

    keep = ~np.all(np.isnan(A), axis=1)
    if not np.any(keep):
        return freq, np.full_like(freq, np.nan), 0
    A = A[keep]
    rowmean = np.nanmean(A, axis=1, keepdims=True)
    A = np.where(np.isnan(A), rowmean, A)
    if detrend:
        A = A - A.mean(axis=1, keepdims=True)
    amp = _amp_from_rfft(np.fft.rfft(A, axis=-1), n)
    return freq, np.nanmean(amp, axis=0), int(keep.sum())


def per_point_Ek(run_dir, npz_data):
    """Rebuild the per-point kinetic energy Ek(t) over the ROI, (npoints, nframes).

    The .npz only stores the ROI-averaged series, so the run's PIV .mat is
    re-read and the ROI + calibration recorded in the .npz are re-applied
    (matching batch_KineticEnergy). Returns None if the .mat cannot be found.
    """
    piv_file = str(npz_data["PIV_file"]) if "PIV_file" in npz_data.files else ""
    if not os.path.isfile(piv_file):                    # data may have moved
        piv_file = os.path.join(run_dir, PIV_FILENAME)
    if not os.path.isfile(piv_file):
        return None

    dt_vel = float(npz_data["dt_vel"]) if "dt_vel" in npz_data.files else 1.0
    xscale = float(npz_data["xscale"]) if "xscale" in npz_data.files else 1.0
    yscale = float(npz_data["yscale"]) if "yscale" in npz_data.files else 1.0
    pts_roi = np.asarray(npz_data["pts_ROI"])

    X, Y, U, V, nframes = load_piv(piv_file)
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel

    mask = create_mask(X, Y, pts_roi)
    Xr, Yr, Ur, Vr = extract_roi(X, Y, U, V, mask)
    if Ur.size == 0:
        Ur, Vr = U, V

    # Same energy definition as batch_KineticEnergy, kept per point.
    Ek_pt = 0.5 * (Ur ** 2 + Vr ** 2)
    return Ek_pt.reshape(-1, Ek_pt.shape[-1])


# Labels say WHERE the ROI average sits relative to the FFT.
LBL_AVG_FIRST = r"$|\mathrm{FFT}(\langle E_k \rangle_{\mathrm{ROI}})|$   average, then FFT"
LBL_FFT_FIRST = r"$\langle |\mathrm{FFT}(E_k)| \rangle_{\mathrm{ROI}}$   FFT, then average"


def save_run_figure(out_png, t, ek, freq, amp, freq_pt, amp_pt, run, flib):
    """Two-panel per-run figure (time series + both amplitude spectra).

    Panel 2 overlays fft(Ek_mean) and fft(Ek)_mean; amp_pt may be None when the
    per-point spectrum is unavailable. Closed straight away so a long batch does
    not leave dozens of windows open.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: the ROI-averaged time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel(r"$\langle E_k \rangle_{\mathrm{ROI}}(t)$  (m$^2$/s$^2$)",
                       fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: both amplitude spectra
    plot = axes[1].semilogy if LOGY else axes[1].plot
    if amp_pt is not None:
        plot(freq_pt, amp_pt, color="C2", lw=1, label=LBL_FFT_FIRST)
    plot(freq, amp, color="C1", lw=1, label=LBL_AVG_FIRST)
    axes[1].set_xlabel("frequency (Hz)", fontsize=13)
    axes[1].set_ylabel(r"$|\widehat{E_k}|$  (m$^2$/s$^2$)", fontsize=13)
    ttl = "FFT amplitude spectrum" + ("  (mean removed)" if DETREND else "")
    axes[1].set_title(ttl, fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=9)
    if flib:
        for fq, lbl in ((flib, r"$f_{\mathrm{lib}}$"),
                        (2 * flib, r"$2f_{\mathrm{lib}}$")):
            if fq <= freq.max():
                axes[1].axvline(fq, color="k", ls="--", lw=1, alpha=0.6)
                axes[1].annotate(lbl, xy=(fq, 1),
                                 xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points",
                                 fontsize=10)

    fig.suptitle(run, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _peak(freq, amp):
    """(frequency, amplitude) of the spectrum maximum, ignoring the DC bin."""
    if amp is None or np.size(amp) < 2 or np.all(np.isnan(amp)):
        return np.nan, np.nan
    k = 1 + int(np.nanargmax(amp[1:]))
    return float(freq[k]), float(amp[k])


def build_row(name, fig_file, processed, fps=np.nan, nframes=np.nan,
              f_peak=np.nan, amp_peak=np.nan, f_peak_pt=np.nan,
              amp_peak_pt=np.nan, npoints=np.nan):
    """Assemble one summary-table row (folder-name metadata + results/flags)."""
    frot_hz, flib_hz, dphi_deg = parse_run_name(name)
    fstar = flib_hz / frot_hz if frot_hz else np.nan
    ss = re.search(r"SS(\d+)", name)
    run_idx = int(ss.group(1)) if ss else np.nan
    return {"run": name, "run idx": run_idx, "processed": processed,
            "frot_Hz": frot_hz, "flib_Hz": flib_hz, "dphi_deg": dphi_deg,
            "fstar": fstar, "fps_Hz": fps, "nframes": nframes,
            "f_peak_Hz": f_peak, "amp_peak": amp_peak,
            "f_peak_perPoint_Hz": f_peak_pt, "amp_peak_perPoint": amp_peak_pt,
            "npoints": npoints, "figure": fig_file}


def process_run(run_dir, reprocess=False):
    """FFT one run's Ek(t) both ways and write its figure. Returns a summary dict."""
    name = os.path.basename(run_dir.rstrip("/"))
    npz_path = find_npz(run_dir)
    if npz_path is None:
        print("  [skip] no %s in %s" % (NPZ_NAME, name))
        return build_row(name, "", False)

    fig_file = os.path.join(os.path.dirname(npz_path), FIG_NAME)
    if os.path.isfile(fig_file) and not reprocess:
        print("  [skip] %-40s figure already exists" % name)
        return build_row(name, fig_file, True)

    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    ek = data["Ek_frame"]
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else name

    # (1) average, then FFT  -> fft(Ek_mean)
    freq, amp, _ = compute_fft(ek, fps, detrend=DETREND)
    f_peak, amp_peak = _peak(freq, amp)

    # (2) FFT, then average  -> fft(Ek)_mean  (needs the per-point field)
    freq_pt = amp_pt = None
    f_peak_pt = amp_peak_pt = np.nan
    npoints = np.nan
    if PER_POINT:
        try:
            Ek_pt = per_point_Ek(run_dir, data)
        except Exception as exc:
            Ek_pt = None
            print("  [warn] %s: per-point FFT failed (%s)" % (name, exc))
        if Ek_pt is None:
            print("  [warn] %s: no PIV .mat -> fft(Ek)_mean skipped" % name)
        else:
            freq_pt, amp_pt, npoints = compute_fft_per_point(Ek_pt, fps,
                                                             detrend=DETREND)
            f_peak_pt, amp_peak_pt = _peak(freq_pt, amp_pt)

    flib = parse_run_name(name)[1]
    flib = None if not np.isfinite(flib) else flib
    save_run_figure(fig_file, t, ek, freq, amp, freq_pt, amp_pt, run, flib)

    print("  [ok] %-40s nframes=%d fps=%.4gHz  f_peak=%.4gHz  "
          "f_peak(per-point)=%.4gHz  npts=%s"
          % (name, np.size(ek), fps, f_peak, f_peak_pt,
             "-" if not np.isfinite(npoints) else int(npoints)))
    return build_row(name, fig_file, True, fps=fps, nframes=int(np.size(ek)),
                     f_peak=f_peak, amp_peak=amp_peak, f_peak_pt=f_peak_pt,
                     amp_peak_pt=amp_peak_pt, npoints=npoints)


def run_batch(reprocess_all=REPROCESS_ALL, only_runs=None):
    """FFT every run under BASE_DIR, write the summary, return the DataFrame."""
    if only_runs is None:
        only_runs = ONLY_RUNS
    subdirs = sorted(d for d in glob.glob(os.path.join(BASE_DIR, "*"))
                     if os.path.isdir(d)
                     and not os.path.basename(d).startswith("."))
    if only_runs:
        wanted = set(only_runs)
        subdirs = [d for d in subdirs if os.path.basename(d) in wanted]

    print("Found %d subfolders to process in %s%s"
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
        return None

    df = pd.DataFrame(rows).sort_values(["frot_Hz", "flib_Hz", "dphi_deg"])
    csv_path = os.path.join(BASE_DIR, "KineticEnergyFFT_summary.csv")
    df.to_csv(csv_path, index=False)
    n_proc = int((df["processed"] == True).sum())
    print("\nSummary (%d runs: %d processed) written to:\n  %s"
          % (len(df), n_proc, csv_path))
    try:
        xlsx_path = os.path.join(BASE_DIR, "KineticEnergyFFT_summary.xlsx")
        df.to_excel(xlsx_path, index=False)
        print("  %s" % xlsx_path)
    except Exception as exc:
        print("  (xlsx skipped: %s)" % exc)
    return df


def main():
    # --- Run ---
    df = run_batch(reprocess_all=REPROCESS_ALL)
    print(df.head(40).to_string())


if __name__ == "__main__":
    main()
