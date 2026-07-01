#!/usr/bin/env python
"""
FFT of a kinetic-energy time series produced by batch_KineticEnergy.py.

Reads a per-run KineticEnergy_timeSeries.npz (containing `t`, `Ek_frame` and
the PIV sampling frequency `fps`), computes the FFT of Ek(t), and draws two
panels:

  1. the kinetic-energy time series Ek(t);
  2. the real part of its FFT versus frequency.

By default the mean is removed before the FFT (the DC term otherwise dwarfs
everything); pass --no-detrend to keep it. The frequency axis uses the PIV
sampling frequency fps stored in the file. If the libration frequency can be
parsed from the run name, dashed guides are drawn at flib and 2*flib (kinetic
energy is quadratic in velocity, so it typically responds at 2*flib).

Usage:
    python fft_KineticEnergy.py PATH [-o OUT.png] [--no-detrend] [--show]

PATH may be the .npz file itself or a run folder (its PostProcessing/
KineticEnergy_timeSeries.npz is used).
Run with an env that has numpy / matplotlib (e.g. dpivsoft).
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def compute_fft(ek, fps, detrend=True):
    """Return (freq, fft_complex) for the positive-frequency half of Ek(t)."""
    ek = np.asarray(ek, dtype=float)
    # Fill any NaNs (e.g. fully-masked frames) with the series mean.
    if np.any(np.isnan(ek)):
        ek = np.where(np.isnan(ek), np.nanmean(ek), ek)
    if detrend:
        ek = ek - ek.mean()
    n = ek.size
    fft = np.fft.fft(ek)
    freq = np.fft.fftfreq(n, d=1.0 / fps)
    pos = freq >= 0
    return freq[pos], fft[pos]


def plot(npz_path, output, detrend=True):
    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    ek = data["Ek_frame"]
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else os.path.basename(
        os.path.dirname(npz_path))

    freq, fft = compute_fft(ek, fps, detrend=detrend)
    flib = parse_flib(run)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel(r"$E_k(t)$  (m$^2$/s$^2$)", fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: real part of the FFT
    axes[1].plot(freq, fft.real, color="C1", lw=1)
    axes[1].set_xlabel("frequency (Hz)", fontsize=13)
    axes[1].set_ylabel(r"Re$\{\widehat{E_k}\}$", fontsize=13)
    ttl = "Real part of FFT" + ("  (mean removed)" if detrend else "")
    axes[1].set_title(ttl, fontsize=14)
    axes[1].grid(True, alpha=0.3)
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
                        help="Keep the mean (do not subtract) before the FFT.")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure window in addition to saving.")
    args = parser.parse_args()

    npz_path = resolve_npz(args.path)
    output = args.output or os.path.join(
        os.path.dirname(npz_path), "KineticEnergy_FFT.png")

    print("Reading %s" % npz_path)
    plot(npz_path, output, detrend=args.detrend)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
