#!/usr/bin/env python
"""
FFT of a kinetic-energy time series produced by batch_KineticEnergy.py.

Reads a per-run KineticEnergy_timeSeries.npz (containing `t`, `Ek_frame` and
the PIV sampling frequency `fps`), computes the FFT of Ek(t), and draws two
panels:

  1. the kinetic-energy time series Ek(t);
  2. the amplitude spectrum |FFT| versus frequency.

By default the mean is removed before the FFT (the DC term otherwise dwarfs
everything); pass --no-detrend to keep it. The frequency axis uses the PIV
sampling frequency fps stored in the file. If the libration frequency can be
parsed from the run name, dashed guides are drawn at flib and 2*flib (kinetic
energy is quadratic in velocity, so it typically responds at 2*flib).

Usage:
    python fft_KineticEnergy.py PATH [-o OUT.png] [--no-detrend] [--linear] [--show]

PATH may be the .npz file itself or a run folder (its PostProcessing/
KineticEnergy_timeSeries.npz is used).
Run with an env that has numpy / matplotlib (e.g. dpivsoft).
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from piv_common import (compute_fft, parse_flib, resolve_npz)

NPZ_NAME = "KineticEnergy_timeSeries.npz"


def plot(npz_path, output, detrend=True, logy=True):
    data = np.load(npz_path, allow_pickle=True)
    t = data["t"]
    ek = data["Ek_frame"]
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else os.path.basename(
        os.path.dirname(npz_path))

    freq, amp, _ = compute_fft(ek, fps, detrend=detrend)
    flib = parse_flib(run)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: time series
    axes[0].plot(t, ek, color="C0", lw=1)
    axes[0].set_xlabel("time (s)", fontsize=13)
    axes[0].set_ylabel(r"$E_k(t)$  (m$^2$/s$^2$)", fontsize=13)
    axes[0].set_title("Kinetic energy time series", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: amplitude spectrum
    if logy:
        axes[1].semilogy(freq, amp, color="C1", lw=1)
    else:
        axes[1].plot(freq, amp, color="C1", lw=1)
    axes[1].set_xlabel("frequency (Hz)", fontsize=13)
    axes[1].set_ylabel(r"$|\widehat{E_k}|$  (m$^2$/s$^2$)", fontsize=13)
    ttl = "FFT amplitude spectrum" + ("  (mean removed)" if detrend else "")
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
                        help="Keep the mean (do not subtract) before the FFT.")
    parser.add_argument("--linear", dest="logy", action="store_false",
                        help="Use a linear amplitude axis (default is log).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure window in addition to saving.")
    args = parser.parse_args()

    npz_path = resolve_npz(args.path, NPZ_NAME)
    output = args.output or os.path.join(
        os.path.dirname(npz_path), "KineticEnergy_FFT.png")

    print("Reading %s" % npz_path)
    plot(npz_path, output, detrend=args.detrend, logy=args.logy)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
