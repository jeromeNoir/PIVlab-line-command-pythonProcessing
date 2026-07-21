#!/usr/bin/env python
"""
Kinetic-energy post-processing of a SINGLE run, merged into the shared summary.

Same computation as batch_KineticEnergy.py (it reuses that module's functions,
so calibration / ROI / Ek are identical), but for one run folder. The resulting
row is inserted into the shared `KineticEnergy_summary.xlsx` (and `.csv`) that
lives in the run's parent directory: an existing row for the same run is
replaced, otherwise the row is appended. The resonance figure is refreshed too.

Usage:
    python process_one_KineticEnergy.py RUN_DIR [--reprocess] [--no-plot]

    RUN_DIR     path to one run folder (containing PIVlab_results_uncalibrated.mat).
    --reprocess re-read the .mat even if a cached result .npz already exists.
    --no-plot   do not redraw KineticEnergy_vs_fstar.png.

Run with an env that has numpy / scipy / pandas / matplotlib (e.g. dpivsoft):
    /Users/jeromenoir/anaconda3/envs/dpivsoft/bin/python process_one_KineticEnergy.py RUN_DIR
"""

import os
import sys
import argparse
import pandas as pd

# Reuse the batch module (same folder) so the processing is byte-for-byte identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_KineticEnergy as B

SUMMARY_CSV = "KineticEnergy_summary.csv"
SUMMARY_XLSX = "KineticEnergy_summary.xlsx"


def load_existing(summary_dir, columns):
    """Load the shared summary (prefer .csv, fall back to .xlsx, else empty)."""
    csv_path = os.path.join(summary_dir, SUMMARY_CSV)
    xlsx_path = os.path.join(summary_dir, SUMMARY_XLSX)
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    if os.path.isfile(xlsx_path):
        return pd.read_excel(xlsx_path)
    return pd.DataFrame(columns=columns)


def main():
    parser = argparse.ArgumentParser(
        description="Process one PIVlab run and add it to the shared summary.")
    parser.add_argument("run_dir", help="Path to a single run folder.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-read the .mat even if a cached result exists.")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Do not redraw the resonance figure.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    if not os.path.isdir(run_dir):
        parser.error("Run folder not found: %s" % run_dir)

    # The shared summary lives in the run's parent directory (= batch BASE_DIR).
    summary_dir = os.path.dirname(run_dir)
    csv_path = os.path.join(summary_dir, SUMMARY_CSV)
    xlsx_path = os.path.join(summary_dir, SUMMARY_XLSX)

    # Process this one run (writes its PostProcessing/ .npz, returns a row dict).
    print("Processing %s" % os.path.basename(run_dir))
    row = B.process_run(run_dir, reprocess=args.reprocess)
    if not row.get("processed"):
        print("  [warn] run has no PIV results; adding an unprocessed row.")

    # Merge into the shared summary: replace any existing row for this run.
    df = load_existing(summary_dir, list(row.keys()))
    if "run" in df.columns:
        df = df[df["run"] != row["run"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Keep the canonical column order, then sort like the batch does.
    ordered = [c for c in row.keys() if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    df = df.sort_values(["frot_Hz", "flib_Hz", "dphi_deg"]).reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    print("\nSummary now has %d runs (%d processed). Updated:\n  %s"
          % (len(df), int(df["processed"].sum()), csv_path))
    try:
        df.to_excel(xlsx_path, index=False)
        print("  %s" % xlsx_path)
    except Exception as exc:
        print("  (xlsx skipped: %s)" % exc)

    if args.plot:
        B.plot_summary(df, summary_dir)


if __name__ == "__main__":
    main()
