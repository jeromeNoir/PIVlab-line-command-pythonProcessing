#!/usr/bin/env python
"""
Plot the kinetic-energy resonance figures from a summary file.

Standalone companion to batch_KineticEnergy.py: it reads the summary produced
there (KineticEnergy_summary.xlsx by default, or a .csv) and draws two panels,
mean(Ek) and std(Ek) versus f* = flib/frot.

  - The main resonance sweep (most common dphi) is a connected curve.
  - Runs at other dphi are overlaid as distinct square markers.
  - Repeat acquisitions of an identical (frot, flib, dphi) point (e.g.
    _SS1/_SS2) are shown as open diamonds rather than folded into the curve.

Usage:
    python plot_KineticEnergy_summary.py [SUMMARY_FILE] [-o OUTPUT.png] [--show]

If SUMMARY_FILE is omitted, DEFAULT_SUMMARY below is used. The figure is saved
next to the summary file (KineticEnergy_vs_fstar.png) unless -o is given.
Run with an environment that has numpy / pandas / matplotlib (e.g. dpivsoft).
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_SUMMARY = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
                   "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom/"
                   "KineticEnergy_summary.xlsx")


def load_summary(path):
    """Read the summary table from .xlsx or .csv into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported summary format: %s" % ext)


def plot_summary(df, output_path):
    """Draw mean(Ek) and std(Ek) versus f* and save to output_path."""
    dfv = df.dropna(subset=["fstar"]).copy()
    if dfv.empty:
        print("Nothing to plot: no valid f* values.")
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Figure written to:\n  %s" % output_path)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", nargs="?", default=DEFAULT_SUMMARY,
                        help="Path to KineticEnergy_summary.xlsx (or .csv).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: next to the summary).")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure window in addition to saving.")
    args = parser.parse_args()

    if not os.path.isfile(args.summary):
        parser.error("Summary file not found: %s" % args.summary)

    output = args.output or os.path.join(os.path.dirname(args.summary),
                                         "KineticEnergy_vs_fstar.png")

    df = load_summary(args.summary)
    print("Loaded %d runs from %s" % (len(df), args.summary))
    plot_summary(df, output)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
