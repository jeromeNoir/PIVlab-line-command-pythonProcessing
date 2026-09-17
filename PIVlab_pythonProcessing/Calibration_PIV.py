#!/usr/bin/env python
"""Interactive calibration: measure XSCALE (m/px) on a calibration image.

Shows a calibration image ROTATED by the dataset's ROTATE (param_postProcessing
.json; 0, 90, 180 or -90 deg -- the same frame select_ROI_quiver and the batch
tools use; a 90-deg rotation preserves pixel distances, so the measurement is
unaffected), asks you to CLICK THE TWO ENDS of a segment of known length, then
asks for that length in millimetres and turns it into the geometric calibration

    XSCALE = (distance_mm / 1000) / distance_px      [m/px],   YSCALE = XSCALE

which is written into the dataset's param_postProcessing.json.

This tool is INTERACTIVE: run it from a terminal (it opens a window and waits),
not through ipynb_to_py.py's Agg twins.

    python Calibration_PIV.py                      # uses BASE_DIR below
    python Calibration_PIV.py <dataset_folder>     # or override it here

The calibration image file name is asked in the terminal (absolute path, or
relative to the dataset folder). Plain images (tif/png/jpg/bmp) and .mat files
holding bg_img_A/bg_img_B/bg_img/background are accepted.

Controls -- in the figure window:
    left click        set the segment ends (two clicks)
    toolbar zoom/pan  zoom in first if you want precision: while a toolbar
                      tool is ACTIVE, clicks are ignored (deactivate it --
                      press its button again, or the o/p key -- to place
                      the points on the zoomed view)
    enter             accept the segment
    r                 clear the points and start again
    escape / close    quit without saving
then in the terminal:
    the segment length in millimetres, followed by
    enter             validate: write XSCALE (and YSCALE = XSCALE)
    r                 redo: pick the segment again
    esc               cancel: nothing written
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.io import loadmat

import matplotlib
# Interactive backend required (the clicks need a real window). Agg would
# silently draw nothing and exit, so fall forward to whatever is present.
if matplotlib.get_backend().lower().startswith("agg"):
    for _bk in ("macosx", "qtagg", "qt5agg", "tkagg"):
        try:
            matplotlib.use(_bk)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt

from piv_postprocessing_lib import read_paramPostprocessing

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# The dataset folder (holds param_postProcessing.json). A command-line argument
# overrides it. The calibration image is asked in the terminal at start-up.
BASE_DIR = ('/Users/jeromenoir/Documents/MyDocuments/'
            'TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k20_topBottom_centerTight')

# Default answer of the image prompt (empty -> no default). Absolute, or
# relative to BASE_DIR.
IMAGE_FILE = 'PIVlab_calibration.tif'

BG_CMAP = 'gray'
# Contrast stretch, as (low, high) intensity percentiles -- raw camera frames
# are nearly black, clipping the tails makes the target visible. None -> none.
BG_CLIP = (1.0, 99.5)

# Write the measured XSCALE (and YSCALE = XSCALE) into the dataset's
# param_postProcessing.json. False -> the value is only printed.
SAVE_TO_PARAM = True
# Mirror it into param_postProcessing_default.json as well. Left False on
# purpose: the calibration is genuinely per-dataset, like PTS_ROI.
SAVE_TO_DEFAULT = False

# Also save a figure with the accepted segment and the numbers drawn on it,
# next to param_postProcessing.json -- it documents where XSCALE comes from.
SAVE_FIGURE = True
FIG_NAME = 'Calibration_selection.png'      # in the dataset folder


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_image(path):
    """The calibration image as a 2-D (or RGB) array, from an image file or a
    .mat holding one of the usual background keys."""
    if path.lower().endswith(".mat"):
        mat = loadmat(path)
        for key in ("bg_img_A", "bg_img_B", "bg_img", "background"):
            if key in mat:
                return np.asarray(mat[key]), "%s:%s" % (os.path.basename(path), key)
        raise SystemExit("%s holds none of bg_img_A/bg_img_B/bg_img/background"
                         % path)
    return plt.imread(path), os.path.basename(path)


def rotate_image_px(img, rotate):
    """The image rotated into the dataset's analysis frame, pixel coordinates.

    Same np.rot90 mapping as select_ROI_quiver's rotate_image (origin='upper');
    no extent handling is needed here because only a pixel DISTANCE is
    measured, and 90-deg rotations preserve it.
    """
    rotate = int(rotate)
    if rotate == 0:
        return img
    if rotate == 180:
        return np.rot90(img, 2)
    if rotate == -90:
        return np.rot90(img, 1)
    if rotate == 90:
        return np.rot90(img, 3)
    raise ValueError("ROTATE must be 0, 90, 180 or -90 deg, got %r" % rotate)


def write_scales(param_path, xscale):
    """Replace XSCALE and YSCALE in an existing param_postProcessing.json,
    keeping every other key (and the key order) untouched."""
    with open(param_path) as fh:
        data = json.load(fh)
    data["XSCALE"] = float(xscale)
    data["YSCALE"] = float(xscale)
    tmp = param_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    shutil.move(tmp, param_path)


def pick_segment(img, title):
    """Open the image and let the user click the two ends of a segment.

    Returns [(x0, y0), (x1, y1)] in pixels of the ROTATED image, or None when
    cancelled (escape / window closed without enter).
    """
    vmin = vmax = None
    if BG_CLIP is not None and img.ndim == 2:
        vmin, vmax = np.percentile(img[np.isfinite(img)], BG_CLIP)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=110)
    ax.imshow(img, cmap=BG_CMAP, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title(title + "\nzoom/pan with the toolbar (clicks are ignored while "
                 "a tool is active), then click the TWO ends of the reference "
                 "segment and press ENTER (r = redo, esc = quit)", fontsize=10)

    state = {"pts": [], "accepted": False, "artists": []}

    def _clear():
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        state["pts"] = []
        fig.canvas.draw_idle()

    def on_click(event):
        # Toolbar zoom/pan active: the click belongs to the tool (it is the
        # first corner of the zoom rectangle), NOT a segment point.
        tb = getattr(fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        if event.inaxes != ax or event.button != 1:
            return
        if len(state["pts"]) >= 2:
            print("\nsegment already has two points -- press r to redo, "
                  "enter to accept")
            return
        state["pts"].append((float(event.xdata), float(event.ydata)))
        state["artists"] += ax.plot(event.xdata, event.ydata, "+", color="red",
                                    ms=14, mew=2, zorder=5)
        if len(state["pts"]) == 2:
            (x0, y0), (x1, y1) = state["pts"]
            state["artists"] += ax.plot([x0, x1], [y0, y1], "-", color="red",
                                        lw=1.6, zorder=4)
            print("\rsegment: (%.1f, %.1f) -> (%.1f, %.1f)   length %.2f px   "
                  % (x0, y0, x1, y1, np.hypot(x1 - x0, y1 - y0)),
                  end="", flush=True)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "escape":
            state["pts"] = []
            state["accepted"] = False
            plt.close(fig)
        elif event.key in ("r", "R"):
            _clear()
            print("\npoints cleared -- click the two ends again")
        elif event.key in ("enter", "return"):
            if len(state["pts"]) < 2:
                print("\nneed TWO points -- click the segment ends first")
                return
            state["accepted"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n>>> Click the two ends of the reference segment, then press ENTER "
          "to accept (r to redo, esc to quit).")
    plt.show(block=True)
    return state["pts"] if (state["accepted"] and len(state["pts"]) == 2) else None


def ask_distance_mm():
    """The segment's physical length [mm] from the terminal; None to cancel."""
    while True:
        raw = input("distance of the segment in millimetres "
                    "(esc to cancel): ").strip()
        if raw.lower() in ("esc", "escape", "q", "quit"):
            return None
        try:
            val = float(raw)
        except ValueError:
            print("  not a number -- try again")
            continue
        if val <= 0:
            print("  the distance must be positive -- try again")
            continue
        return val


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(base_dir):
    base_dir = base_dir.rstrip("/")
    P = read_paramPostprocessing(base_dir)
    ROTATE = int(P.ROTATE)

    # -- the calibration image, asked in the terminal ----------------------
    prompt = "calibration image (absolute, or relative to the dataset folder)"
    prompt += " [%s]: " % IMAGE_FILE if IMAGE_FILE else ": "
    name = input(prompt).strip() or IMAGE_FILE
    if not name:
        raise SystemExit("no image given -- nothing done.")
    img_path = name if os.path.isabs(name) else os.path.join(base_dir, name)
    if not os.path.isfile(img_path):
        raise SystemExit("no such file: %s" % img_path)

    img_raw, img_name = load_image(img_path)
    img = rotate_image_px(img_raw, ROTATE)

    print("dataset      : %s" % os.path.basename(base_dir))
    print("image        : %s  %s  (shown ROTATED by %g deg)"
          % (img_name, img_raw.shape, ROTATE))
    print("XSCALE now   : %.9g m/px" % P.XSCALE)

    title = "%s  --  %s  (ROTATE = %g deg)" % (os.path.basename(base_dir),
                                               img_name, ROTATE)

    # -- pick / measure / confirm loop -------------------------------------
    while True:
        pts = pick_segment(img, title)
        if pts is None:
            print("\naborted -- nothing written.")
            return

        (x0, y0), (x1, y1) = pts
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        print("\n\nsegment length : %.3f px" % length_px)

        dist_mm = ask_distance_mm()
        if dist_mm is None:
            print("cancelled -- nothing written.")
            return

        xscale = (dist_mm / 1000.0) / length_px          # m/px
        print("\n  %.6g mm over %.3f px" % (dist_mm, length_px))
        print("  XSCALE = YSCALE = %.9g m/px   (was %.9g)" % (xscale, P.XSCALE))

        raw = input("ENTER to validate, r to redo, esc to cancel: ").strip()
        if raw == "":
            break                                        # validated
        if raw.lower() in ("r",):
            print("redo -- pick the segment again")
            continue
        print("cancelled -- nothing written.")
        return

    # -- write the parameter file ------------------------------------------
    if SAVE_TO_PARAM:
        param_path = os.path.join(base_dir, "param_postProcessing.json")
        write_scales(param_path, xscale)
        print("\nXSCALE = YSCALE = %.9g written to %s" % (xscale, param_path))
        if SAVE_TO_DEFAULT:
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "param_postProcessing_default.json")
            write_scales(default_path, xscale)
            print("mirrored into %s" % default_path)
        else:
            print("(param_postProcessing_default.json left alone -- the "
                  "calibration is per-dataset; set SAVE_TO_DEFAULT=True to "
                  "mirror it anyway.)")
    else:
        print("\nSAVE_TO_PARAM = False -- value only printed, nothing written.")

    # -- document the measurement ------------------------------------------
    if SAVE_FIGURE:
        vmin = vmax = None
        if BG_CLIP is not None and img.ndim == 2:
            vmin, vmax = np.percentile(img[np.isfinite(img)], BG_CLIP)
        fig, ax = plt.subplots(figsize=(11, 7), dpi=110)
        ax.imshow(img, cmap=BG_CMAP, origin="upper", vmin=vmin, vmax=vmax)
        ax.plot([x0, x1], [y0, y1], "-", color="red", lw=1.8, zorder=4)
        ax.plot([x0, x1], [y0, y1], "+", color="red", ms=14, mew=2, zorder=5)
        ax.annotate("%.6g mm  =  %.2f px\nXSCALE = %.9g m/px"
                    % (dist_mm, length_px, xscale),
                    xy=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                    xytext=(12, -12), textcoords="offset points",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="red",
                              alpha=0.85), zorder=6)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_title(title, fontsize=10)
        fig_path = os.path.join(base_dir, FIG_NAME)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("figure written to %s" % fig_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BASE_DIR)
