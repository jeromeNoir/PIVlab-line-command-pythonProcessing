#!/usr/bin/env python
"""
Power spectral density (PSD) of a kinetic-energy time series produced by
batch_KineticEnergy.py.

Reads a per-run KineticEnergy_timeSeries.npz (containing `t`, `Ek_frame` and
the PIV sampling frequency `fps`), estimates the PSD of Ek(t) with Welch's
method (scipy.signal.welch), and draws two panels:

  1. the kinetic-energy time series Ek(t);
  2. its PSD versus frequency.

Welch averages the periodograms of overlapping windowed segments, trading
frequency resolution for a smoother, lower-variance estimate than the raw
periodogram (see periodogram_KineticEnergy.py). By default the mean is removed
per segment (the DC term otherwise dwarfs everything); pass --no-detrend to keep
it. The frequency axis uses the PIV sampling frequency fps stored in the file.
If the libration frequency can be parsed from the run name, dashed guides are
drawn at flib and 2*flib (kinetic energy is quadratic in velocity, so it
typically responds at 2*flib).

Usage:
    python psd_KineticEnergy.py PATH [-o OUT.png] [--no-detrend]
        [--nperseg N] [--window WIN] [--linear] [--show]

PATH may be the .npz file itself or a run folder (its PostProcessing/
KineticEnergy_timeSeries.npz is used).
Run with an env that has numpy / scipy / matplotlib (e.g. dpivsoft).
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

NPZ_NAME = "KineticEnergy_timeSeries.npz"
FLIB_DIVISOR = 1000.0   # folder token 'flib0400' -> 0.400 Hz (matches batch)


def resolve_npz(path):
    """Accept the .npz path directly or a run folder containing it."""
    if os.path.isdir(path):
        cand = os.path.join(path, "PostProcessing", NPZ_NAME)
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(path, NPZ_NAME)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError("No %s under %s" % (NPZ_NAME, path))
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def parse_flib(run_name):
    """Libration frequency (Hz) from a run name, or None."""
    m = re.search(r"flib(\d+)", str(run_name))
    return float(m.group(1)) / FLIB_DIVISOR if m else None


def compute_psd(ek, fps, detrend=True, nperseg=256, window="hann"):
    """Return (freq, pxx) for the Welch PSD estimate of Ek(t).

    Thin wrapper around scipy.signal.welch: it averages the periodograms of
    overlapping windowed segments (length `nperseg`, 50% overlap by default)
    for a smoother, lower-variance one-sided PSD than the raw periodogram.
    Units are (Ek units)^2 / Hz. `nperseg` is clamped to the series length.
    """
    ek = np.asarray(ek, dtype=float)
    # Fill any NaNs (e.g. fully-masked frames) with the series mean.
    if np.any(np.isnan(ek)):
        ek = np.where(np.isnan(ek), np.nanmean(ek), ek)
    nperseg = min(int(nperseg), ek.size)
    freq, pxx = welch(
        ek, fs=fps, window=window, nperseg=nperseg,
        detrend=("constant" if detrend else False),
        scaling="density", return_onesided=True)
    return freq, pxx


def plot(npz_path, output, detrend=True, nperseg=256, window="hann", logy=True):
    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    ek = data["Ek_frame"]
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else os.path.basename(
        os.path.dirname(npz_path))

    freq, pxx = compute_psd(ek, fps, detrend=detrend, nperseg=nperseg,
                            window=window)
    flib = parse_flib(run)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel(r"$E_k(t)$  (m$^2$/s$^2$)", fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: PSD (Welch)
    if logy:
        axes[1].semilogy(freq, pxx, color="C1", lw=1)
    else:
        axes[1].plot(freq, pxx, color="C1", lw=1)
    axes[1].set_xlabel("frequency (Hz)", fontsize=13)
    axes[1].set_ylabel(r"PSD  (m$^4$/s$^4$/Hz)", fontsize=13)
    ttl = "Welch PSD" + ("  (mean removed)" if detrend else "")
    axes[1].set_title(ttl, fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    if flib:
        for f, lbl in ((flib, r"$f_{\mathrm{lib}}$"),
                       (2 * flib, r"$2f_{\mathrm{lib}}$")):
            if f <= freq.max():
                axes[1].axvline(f, color="k", ls="--", lw=1, alpha=0.6)
                axes[1].annotate(lbl, xy=(f, 1), xycoords=("data", "axes fraction"),
                                 xytext=(2, -12), textcoords="offset points",
                                 fontsize=10)

    fig.suptitle(run, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print("Figure written to:\n  %s" % output)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path",
                        help="KineticEnergy_timeSeries.npz file or a run folder.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: next to the .npz).")
    parser.add_argument("--no-detrend", dest="detrend", action="store_false",
                        help="Keep the mean (do not subtract) before the estimate.")
    parser.add_argument("--nperseg", type=int, default=256,
                        help="Welch segment length in samples (default: 256; "
                             "clamped to the series length).")
    parser.add_argument("--window", default="hann",
                        help="Window passed to scipy.signal.welch (default: hann).")
    parser.add_argument("--linear", dest="logy", action="store_false",
                        help="Use a linear power axis (default is log).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure window in addition to saving.")
    args = parser.parse_args()

    npz_path = resolve_npz(args.path)
    output = args.output or os.path.join(
        os.path.dirname(npz_path), "KineticEnergy_PSD.png")

    print("Reading %s" % npz_path)
    plot(npz_path, output, detrend=args.detrend, nperseg=args.nperseg,
         window=args.window, logy=args.logy)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
