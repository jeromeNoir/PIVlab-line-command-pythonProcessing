"""
# FFT of the ROI-averaged velocity spectrum

Interactive companion to `batch_VelocityFFT.ipynb` (velocity analogue of
[`fft_KineticEnergy.ipynb`](fft_KineticEnergy.ipynb)).

It reads a per-run `VelocityFFT.npz` (produced by `batch_VelocityFFT.ipynb`),
which already holds the **ROI-averaged single-sided amplitude spectra** of the
two velocity components — `amp_u`, `amp_v` and `amp_total = amp_u + amp_v` —
versus frequency `f`, computed with `np.fft.rfft` (mean removed, Hann window,
amplitude-corrected) and the PIV **field** sampling frequency `fps = cam_fps/2`.

Because the FFT is precomputed by the batch, this notebook only **loads and
plots** it (there is no time series stored and nothing is re-transformed). It
draws two panels:

1. the component spectra `|FFT(U)|` and `|FFT(V)|`;
2. the total amplitude spectrum `|FFT(U)| + |FFT(V)|`.

If the libration frequency can be parsed from the run name, dashed guides are
drawn at `flib` and `2*flib` (the interior velocity responds mainly at `flib`).
"""

# --- 1. Imports ---
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from piv_common import (parse_flib, parse_frot, peak_freq, resolve_npz)


NPZ_NAME = "VelocityFFT.npz"


# --- 2. Configuration ---
# --- edit me -------------------------------------------------------------
PATH   = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
          "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom/"
          "frot0.50Hz_flib1.500Hz_dphi8deg_SS1/PostProcessing")   # run folder or .npz
LOGY   = True        # logarithmic amplitude axis (False -> linear)
FMAX   = None        # upper frequency limit for the plot (Hz); None -> Nyquist
SAVE   = True       # also write a PNG next to the .npz
OUTPUT = None        # PNG path; None -> Velocity_FFT.png beside the .npz
OVERWRITE_POL = False  # (over)write Velocity_polarization.png next to the .npz
                       #  even if it exists (False -> keep existing)
# -------------------------------------------------------------------------


# --- 3. Helper functions ---


def main():
    # --- 4. Load the spectrum ---
    npz_path = resolve_npz(PATH, NPZ_NAME)
    print("Reading %s" % npz_path)

    data = np.load(npz_path, allow_pickle=True)
    f = np.asarray(data["f"], dtype=float)
    amp_u = np.asarray(data["amp_u"], dtype=float)
    amp_v = np.asarray(data["amp_v"], dtype=float)
    amp_total = np.asarray(data["amp_total"], dtype=float)
    powerU = np.asarray(data["powerU"], dtype=float) if "powerU" in data.files else None
    powerV = np.asarray(data["powerV"], dtype=float) if "powerV" in data.files else None
    fps = float(data["fps"]) if "fps" in data.files else 1.0
    run = str(data["run"]) if "run" in data.files else os.path.basename(
        os.path.dirname(npz_path))
    window = str(data["window"]) if "window" in data.files else "?"
    npoints = int(data["npoints"]) if "npoints" in data.files else -1
    nframes = int(data["nframes"]) if "nframes" in data.files else f.size

    flib = parse_flib(run)
    frot = parse_frot(run)
    print("run     = %s" % run)
    print("fps     = %g Hz   nframes = %d   Nyquist = %g Hz" % (fps, nframes, f[-1]))
    print("df      = %.4g Hz   ROI points averaged = %d   window = %s"
          % (f[1] - f[0] if f.size > 1 else np.nan, npoints, window))
    print("flib    = %s" % ("%g Hz" % flib if flib else "unknown"))

    # --- 5. Peak frequencies ---
    for label, amp in (("U", amp_u), ("V", amp_v), ("total", amp_total)):
        fp, ap = peak_freq(f, amp)
        print("peak |FFT(%-5s)| at %.4f Hz   (amplitude = %.3e m/s)" % (label, fp, ap))
    if flib:
        print("  reference: flib = %.4f Hz,  2*flib = %.4f Hz" % (flib, 2 * flib))

    # --- 6. Plot ---
    fmax = FMAX if FMAX else f[-1]
    sel = f <= fmax


    def add_guides(ax):
        if not flib:
            return
        for fq, lbl in ((flib, r"$f_{\mathrm{lib}}$"),
                        (2 * flib, r"$2f_{\mathrm{lib}}$")):
            if fq <= fmax:
                ax.axvline(fq, color="k", ls="--", lw=1, alpha=0.6)
                ax.annotate(lbl, xy=(fq, 1), xycoords=("data", "axes fraction"),
                            xytext=(2, -12), textcoords="offset points", fontsize=10)


    plot = (lambda ax, *a, **k: ax.semilogy(*a, **k)) if LOGY else \
           (lambda ax, *a, **k: ax.plot(*a, **k))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: component amplitude spectra
    plot(axes[0], f[sel], amp_u[sel], color="C0", lw=1, label=r"$|\widehat{U}|$")
    plot(axes[0], f[sel], amp_v[sel], color="C1", lw=1, label=r"$|\widehat{V}|$")
    axes[0].set_xlabel("frequency (Hz)", fontsize=13)
    axes[0].set_ylabel(r"amplitude  (m/s)", fontsize=13)
    axes[0].set_title("Velocity component spectra", fontsize=14)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=11)
    add_guides(axes[0])

    # Panel 2: total amplitude spectrum
    plot(axes[1], f[sel], amp_total[sel], color="C3", lw=1)
    axes[1].set_xlabel("frequency (Hz)", fontsize=13)
    axes[1].set_ylabel(r"$|\widehat{U}| + |\widehat{V}|$  (m/s)", fontsize=13)
    axes[1].set_title("Total amplitude spectrum", fontsize=14)
    axes[1].grid(True, alpha=0.3, which="both")
    add_guides(axes[1])

    fig.suptitle(run, fontsize=13)
    fig.tight_layout()

    if SAVE:
        output = OUTPUT or os.path.join(os.path.dirname(npz_path), "Velocity_FFT.png")
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % output)

    plt.show()

    # --- 7. Polarization ---
    # Polarization - identical to batch_VelocityFFT's figure. The arrays are read
    # from the .npz (the raw U/V time series are not stored here, so nothing can be
    # recomputed).
    if "powerRatio" in data.files:
        m = f > 0
        figp, axp = plt.subplots(figsize=(9, 6))
        if frot:
            theory = 2.0 * ((2.0 * frot / f[m]) ** 2 - 1.0)
            axp.plot(f[m], theory, "k-", lw=2.2,
                     label=r"$2[(2f_{\mathrm{rot}}/f)^2-1]$  (IW)")
        # powerV/powerU over the ROI: the ROI mean, plus the median with a shaded
        # 25-75 percentile band when the batch saved the percentile arrays.
        axp.plot(f[m], np.asarray(data["powerRatio"], dtype=float)[m], lw=1.2,
                 label=r"mean $P_V/P_U$")
        _pct = ("powerRatio_med", "powerRatio_p25", "powerRatio_p75")
        if all(k in data.files for k in _pct):
            med = np.asarray(data["powerRatio_med"], dtype=float)
            p25 = np.asarray(data["powerRatio_p25"], dtype=float)
            p75 = np.asarray(data["powerRatio_p75"], dtype=float)
            line, = axp.plot(f[m], med[m], lw=1.2, label=r"median $P_V/P_U$")
            axp.fill_between(f[m], p25[m], p75[m], color=line.get_color(),
                             alpha=0.20, lw=0, label="ROI 25-75%")
        else:
            print("  (no percentile arrays in this .npz -> mean only)")
        axp.set_yscale("log")
        axp.set_xlim(0.01, 1.0)
        axp.set_xlabel("frequency (Hz)", fontsize=13)
        axp.set_ylabel(r"polarization  $P_V/P_U$", fontsize=13)
        axp.set_title("Polarization - %s" % run, fontsize=13)
        axp.grid(True, which="both", ls=":", alpha=0.4)
        axp.legend(fontsize=9)
        figp.tight_layout()

        pol_png = os.path.join(os.path.dirname(npz_path),
                               "Velocity_polarization.png")
        if OVERWRITE_POL or not os.path.isfile(pol_png):
            figp.savefig(pol_png, dpi=200, bbox_inches="tight")
            print("Polarization figure written to:\n  %s" % pol_png)
        else:
            print("Polarization figure already exists (set OVERWRITE_POL=True to "
                  "overwrite):\n  %s" % pol_png)
        plt.show()
    else:
        print("powerRatio not in this .npz - reprocess with batch_VelocityFFT "
              "to add it")


if __name__ == "__main__":
    main()
