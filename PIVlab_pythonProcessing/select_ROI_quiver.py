#!/usr/bin/env python
"""Interactive ROI picker: velocity quiver (physical units) on the background image.

Draws one PIV run's velocity field as a quiver in metres / metres-per-second on
top of the run's background camera image, then asks you to DRAG A RECTANGLE over
the region you want to analyse. The two corners are reported -- and optionally
written into the dataset's param_postProcessing.json as PTS_ROI -- in physical
units, as

    PTS_ROI = [[x_left, y_top], [x_right, y_bottom]]     # top-left, bottom-right

which is exactly the order create_mask / region_fields expect. The figure, with
the rectangle and its corner coordinates drawn on it, is saved next to that same
parameter file.

This tool is INTERACTIVE: run it from a terminal (it opens a window and waits),
not through ipynb_to_py.py's Agg twins.

    python select_ROI_quiver.py                       # uses RUN_DIR below
    python select_ROI_quiver.py <run_folder>          # or override it here

Controls
    drag left mouse   draw / redraw the rectangle
    enter             accept: print the corners (and save if SAVE_TO_PARAM)
    r                 clear the rectangle and start again
    escape / close    quit without saving

Coordinate convention (identical to every other tool here)
    x = x_px * XSCALE                     metres, 0 at the left image edge
    y = (y_px_max - y_px) * YSCALE        metres, y up, 0 at the BOTTOM row of
                                          the PIV grid (not the image edge)
so the image extent is computed from the PIV grid's own pixel coordinates rather
than from the image size -- that is what keeps the picture and the vectors, and
hence the saved ROI, on the same axes as load_piv's output.
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.io import loadmat

import matplotlib
# Interactive backend required (the RectangleSelector needs a real window). Agg
# would silently draw nothing and exit, so fall forward to whatever is present.
if matplotlib.get_backend().lower().startswith("agg"):
    for _bk in ("macosx", "qtagg", "qt5agg", "tkagg"):
        try:
            matplotlib.use(_bk)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from piv_postprocessing_lib import (load_piv, read_acquisition_params,
                                    read_paramPostprocessing)

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# The run folder to look at (holds the .mat, the acquisition log and, ideally,
# background.mat). A command-line argument overrides it.
RUN_DIR = ('/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
           'CylinderExperimentsGMA/k20_topBottom/frot0.50Hz_flib0.400Hz_dphi2deg_SS1')

# Which velocity field to draw:
#   an int  -> that frame (0-based)
#   'mean'  -> time-average of U and V (steady streaming; small arrows)
#   'rms'   -> root-mean-square field, sign taken from the mean (shows where the
#              motion is, which is usually what you want when choosing an ROI)
FRAME = 'rms'

# Which velocity load_piv returns: True -> the filtered/validated field,
# False -> the original one with the rejected vectors set to NaN. None -> the
# dataset's VALIDATE_VELOCITY from param_postProcessing.json (the usual choice:
# the ROI is then picked on the same field the batch tools will analyse).
VALIDATE_VELOCITY = None

QUIVER_SKIP = 3          # draw every n-th vector in each direction
QUIVER_SCALE = None      # None -> matplotlib autoscale; a number -> m/s per axis-width
QUIVER_COLOR = 'yellow'  # arrows sit on a grey image, so keep them bright
QUIVER_WIDTH = 0.0022    # shaft width, in axes fractions

# Background image. None -> <RUN_DIR>/background.mat (bg_img_A). A path to any
# image file (or another .mat) is used instead. Runs recorded without a
# background get a plain field and a warning -- the vectors and the picking are
# unaffected, only the picture is missing.
BG_FILE = None
BG_CMAP = 'gray'
# Contrast stretch, as (low, high) intensity percentiles. Raw PIV frames are
# nearly black -- clipping the tails is what makes the walls and the topography
# visible enough to place the rectangle against. None -> no stretch.
BG_CLIP = (1.0, 99.5)

# Write the picked rectangle into the dataset's param_postProcessing.json
# (RUN_DIR's parent) as PTS_ROI. False -> corners are only printed.
SAVE_TO_PARAM = True
# Mirror the new PTS_ROI into param_postProcessing_default.json as well. Left
# False on purpose: the ROI is the one parameter that is genuinely per-dataset,
# so copying one dataset's rectangle into the template would mislead the next
# dataset seeded from it. Any OTHER key you edit by hand should be mirrored.
SAVE_TO_DEFAULT = False

# Also save the figure with the accepted rectangle -- and its corner
# coordinates -- drawn on it. It goes NEXT TO param_postProcessing.json, in the
# dataset folder, so the picture documents the PTS_ROI the file now holds. One
# ROI per dataset, so one picture: re-picking overwrites it.
SAVE_FIGURE = True
FIG_NAME = 'ROI_selection.png'      # in the dataset folder (RUN_DIR's parent)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pixel_grid(piv_file):
    """The PIV grid's raw pixel coordinates (x_px, y_px), as 2-D arrays.

    load_piv returns the grid already flipped and about to be scaled; the
    background image has to be placed with the ORIGINAL pixel numbering, so it
    is read separately here. Both the 3-D-stack and cell-array .mat layouts are
    handled, same as the library does.
    """
    mat = loadmat(piv_file, variable_names=["x", "y"])
    out = []
    for key in ("x", "y"):
        a = np.asarray(mat[key])
        if a.dtype == object:
            a = np.asarray(a.ravel()[0], dtype=float)
        elif a.ndim == 3:
            a = a[:, :, 0]
        out.append(np.asarray(a, dtype=float))
    return out[0], out[1]


def background_image(run_dir, bg_file=None):
    """(image, source_name) for the run, or (None, reason) when there is none."""
    path = bg_file or os.path.join(run_dir, "background.mat")
    if not os.path.isfile(path):
        return None, "no background image (%s)" % os.path.basename(path)
    if path.lower().endswith(".mat"):
        mat = loadmat(path)
        for key in ("bg_img_A", "bg_img_B", "bg_img", "background"):
            if key in mat:
                return np.asarray(mat[key]), "%s:%s" % (os.path.basename(path), key)
        return None, "%s holds no bg_img_A/B" % os.path.basename(path)
    return plt.imread(path), os.path.basename(path)


def image_extent(img_shape, x_px, y_px, xscale, yscale):
    """imshow extent (left, right, bottom, top) in metres for a full-frame image.

    Pixel p (MATLAB's 1-based numbering, which is what PIVlab writes into x/y)
    has its CENTRE at x = p*xscale and y = (y_px.max() - p)*yscale, so its edges
    sit half a pixel either side -- hence the 0.5 offsets. Getting these wrong
    shifts the picture under the vectors by half an interrogation window.
    """
    ny, nx = img_shape[:2]
    y_top_px = float(y_px.max())
    left = 0.5 * xscale
    right = (nx + 0.5) * xscale
    top = (y_top_px - 0.5) * yscale
    bottom = (y_top_px - (ny + 0.5)) * yscale
    return left, right, bottom, top


def select_frame(U, V, frame):
    """Reduce the (ny, nx, nframes) velocity stack to the single field to draw."""
    if frame == 'mean':
        return np.nanmean(U, axis=2), np.nanmean(V, axis=2), 'time-mean'
    if frame == 'rms':
        # RMS magnitude carrying the sign of the mean: the arrows then point the
        # way the flow mostly goes, but are as long as the actual motion -- a
        # plain time-mean of an oscillating flow nearly cancels and looks empty.
        rms_u = np.sqrt(np.nanmean(U ** 2, axis=2))
        rms_v = np.sqrt(np.nanmean(V ** 2, axis=2))
        sgn_u = np.sign(np.nanmean(U, axis=2))
        sgn_v = np.sign(np.nanmean(V, axis=2))
        sgn_u[sgn_u == 0] = 1.0
        sgn_v[sgn_v == 0] = 1.0
        return rms_u * sgn_u, rms_v * sgn_v, 'RMS (sign of the mean)'
    i = int(frame)
    return U[:, :, i], V[:, :, i], 'frame %d' % i


def write_pts_roi(param_path, pts_roi):
    """Replace PTS_ROI in an existing param_postProcessing.json, keeping the rest.

    The file is re-read and re-dumped rather than rebuilt from the defaults, so
    every other key (and the key order) survives untouched.
    """
    with open(param_path) as fh:
        data = json.load(fh)
    data["PTS_ROI"] = [[float(pts_roi[0][0]), float(pts_roi[0][1])],
                       [float(pts_roi[1][0]), float(pts_roi[1][1])]]
    tmp = param_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    shutil.move(tmp, param_path)


# ----------------------------------------------------------------------
# 1. Load the run
# ----------------------------------------------------------------------
def main(run_dir):
    run_dir = run_dir.rstrip("/")
    base_dir = os.path.dirname(run_dir)
    P = read_paramPostprocessing(base_dir)
    piv_file = os.path.join(run_dir, P.PIV_FILENAME)
    if not os.path.isfile(piv_file):
        raise SystemExit("No %s in %s" % (P.PIV_FILENAME, run_dir))

    dt_vel, fps, ok = read_acquisition_params(os.path.join(run_dir, P.LOG_FILENAME))
    if ok:
        xscale, yscale = P.XSCALE, P.YSCALE
        vel_unit, pos_unit = "m/s", "m"
    else:
        print("[warn] no usable acquisition log: velocities stay in px/frame and "
              "positions in px -- the ROI would be saved in those units too.")
        xscale = yscale = P.UNCAL_SCALE
        vel_unit, pos_unit = "px/frame", "px"

    x_px, y_px = pixel_grid(piv_file)
    X, Y, U, V, nframes = load_piv(piv_file, validate_velocity=VALIDATE_VELOCITY)

    # Calibrate exactly as the batch tools do: positions -> m, velocities -> m/s.
    X = xscale * X
    Y = yscale * Y
    U = xscale * U / dt_vel
    V = yscale * V / dt_vel

    u, v, frame_label = select_frame(U, V, FRAME)
    speed = np.hypot(u, v)

    validated = P.VALIDATE_VELOCITY if VALIDATE_VELOCITY is None else VALIDATE_VELOCITY

    print("run          : %s" % os.path.basename(run_dir))
    print("velocity     : %s" % ("filtered / validated" if validated else
                                 "original, rejected vectors NaN"))
    print("frames       : %d at %.4g Hz   (dt_pulse = %.4g s)" % (nframes, fps, dt_vel))
    print("field shown  : %s" % frame_label)
    print("x range      : %.4f .. %.4f %s" % (np.nanmin(X), np.nanmax(X), pos_unit))
    print("y range      : %.4f .. %.4f %s" % (np.nanmin(Y), np.nanmax(Y), pos_unit))
    print("speed        : median %.4g, max %.4g %s"
          % (np.nanmedian(speed), np.nanmax(speed), vel_unit))

    # ------------------------------------------------------------------
    # 2. Figure: background image + quiver, both in physical units
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7), dpi=110)

    img, bg_name = background_image(run_dir, BG_FILE)
    if img is None:
        # No picture: fall back to the extent of the velocity grid itself.
        extent = (float(np.nanmin(X)), float(np.nanmax(X)),
                  float(np.nanmin(Y)), float(np.nanmax(Y)))
        print("[warn] %s -- drawing the vectors on a blank field" % bg_name)
    else:
        extent = image_extent(img.shape, x_px, y_px, xscale, yscale)
        vmin = vmax = None
        if BG_CLIP is not None and img.ndim == 2:
            vmin, vmax = np.percentile(img[np.isfinite(img)], BG_CLIP)
        ax.imshow(img, cmap=BG_CMAP, origin="upper", zorder=0, vmin=vmin, vmax=vmax,
                  extent=extent)
        print("background   : %s  %s" % (bg_name, img.shape))

    sk = max(1, int(QUIVER_SKIP))
    q = ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], u[::sk, ::sk], v[::sk, ::sk],
                  color=QUIVER_COLOR, pivot="mid", width=QUIVER_WIDTH,
                  scale=QUIVER_SCALE, zorder=3)
    # Reference arrow at a round-ish value near the median speed.
    ref = float(np.nanmedian(speed))
    if np.isfinite(ref) and ref > 0:
        ref = 10.0 ** np.floor(np.log10(ref))
        ax.quiverkey(q, 0.90, 1.03, ref, "%g %s" % (ref, vel_unit),
                     labelpos="E", coordinates="axes")

    # The ROI currently in the parameter file, for reference.
    (xa, ya), (xb, yb) = np.asarray(P.PTS_ROI, dtype=float).reshape(2, 2)
    ax.add_patch(plt.Rectangle((min(xa, xb), min(ya, yb)), abs(xb - xa), abs(yb - ya),
                               fill=False, ec="deepskyblue", ls="--", lw=1.2, zorder=4,
                               label="PTS_ROI in the parameter file"))

    # Show exactly the background image, and nothing below zero. The physical
    # origin sits on the BOTTOM ROW OF THE PIV GRID, which is a few pixels above
    # the bottom of the frame, so the raw image extent dips slightly negative --
    # a sliver the tools cannot address anyway (create_mask would never select
    # it), so it is clipped away rather than shown.
    left, right, bottom, top = extent
    xlim = (max(0.0, min(left, right)), max(left, right))
    ylim = (max(0.0, min(bottom, top)), max(bottom, top))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    print("axes         : x %.4f .. %.4f, y %.4f .. %.4f %s"
          % (xlim[0], xlim[1], ylim[0], ylim[1], pos_unit))

    ax.set_xlabel("x [%s]" % pos_unit)
    ax.set_ylabel("y [%s]" % pos_unit)
    ax.set_title("%s  --  %s\ndrag a rectangle, then press ENTER to accept "
                 "(r = redo, esc = quit)" % (os.path.basename(run_dir), frame_label),
                 fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)

    # ------------------------------------------------------------------
    # 3. Ask for the rectangle
    # ------------------------------------------------------------------
    state = {"pts": None, "saved": False}

    def on_select(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        # A drag that runs off the axes still reports coordinates outside them,
        # so clamp: the ROI can never be negative, nor reach past the image.
        x0, x1 = np.clip([x0, x1], *xlim)
        y0, y1 = np.clip([y0, y1], *ylim)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return
        # Stored top-left first, bottom-right second -- the PTS_ROI convention.
        state["pts"] = [[x0, y1], [x1, y0]]
        print("\rrectangle: top-left (%.6f, %.6f)  bottom-right (%.6f, %.6f) %s   "
              % (x0, y1, x1, y0, pos_unit), end="", flush=True)

    selector = RectangleSelector(ax, on_select, useblit=False, button=[1],
                                 minspanx=5, minspany=5, spancoords="pixels",
                                 interactive=True,
                                 props=dict(facecolor="none", edgecolor="red", lw=1.6))

    def on_key(event):
        if event.key == "escape":
            state["pts"] = None
            plt.close(fig)
        elif event.key in ("r", "R"):
            state["pts"] = None
            selector.set_visible(False)
            selector.update()
            fig.canvas.draw_idle()
            print("\nrectangle cleared -- drag a new one")
        elif event.key in ("enter", "return"):
            if state["pts"] is None:
                print("\nno rectangle drawn yet -- drag one first")
                return
            state["saved"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n>>> Drag a rectangle over the region you want, then press ENTER to "
          "accept it (r to redo, esc to quit).")
    plt.show(block=True)

    # ------------------------------------------------------------------
    # 4. Report / save the corners
    # ------------------------------------------------------------------
    if not state["saved"] or state["pts"] is None:
        print("\naborted -- nothing written.")
        return

    (x_left, y_top), (x_right, y_bottom) = state["pts"]
    print("\n\nSelected ROI [%s]" % pos_unit)
    print("  top-left     : (%.9g, %.9g)" % (x_left, y_top))
    print("  bottom-right : (%.9g, %.9g)" % (x_right, y_bottom))
    print("  width x height: %.6g x %.6g %s" % (x_right - x_left, y_top - y_bottom, pos_unit))
    print("  PTS_ROI = [[%.17g, %.17g], [%.17g, %.17g]]"
          % (x_left, y_top, x_right, y_bottom))

    if not ok and SAVE_TO_PARAM:
        print("\n[warn] the field is UNCALIBRATED (px): refusing to write PTS_ROI, "
              "which the tools read as metres. Fix the acquisition log first.")
    elif SAVE_TO_PARAM:
        param_path = os.path.join(base_dir, "param_postProcessing.json")
        write_pts_roi(param_path, state["pts"])
        print("\nPTS_ROI written to %s" % param_path)
        if SAVE_TO_DEFAULT:
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "param_postProcessing_default.json")
            write_pts_roi(default_path, state["pts"])
            print("PTS_ROI mirrored into %s" % default_path)
        else:
            print("(param_postProcessing_default.json left alone -- the ROI is "
                  "per-dataset; set SAVE_TO_DEFAULT=True to mirror it anyway.)")

    if SAVE_FIGURE:
        ax.add_patch(plt.Rectangle((x_left, y_bottom), x_right - x_left,
                                   y_top - y_bottom, fill=False, ec="red", lw=1.8,
                                   zorder=5, label="selected ROI"))

        # The corners, written on the picture itself: the figure then carries
        # the numbers that went into PTS_ROI, so it can be read on its own.
        label_box = dict(boxstyle="round,pad=0.3", fc="black", ec="red", alpha=0.85)
        ax.annotate("top-left  (%.5f, %.5f) %s" % (x_left, y_top, pos_unit),
                    xy=(x_left, y_top), xytext=(5, -5), textcoords="offset points",
                    ha="left", va="top", fontsize=9, color="white",
                    bbox=label_box, zorder=6)
        ax.annotate("bottom-right  (%.5f, %.5f) %s" % (x_right, y_bottom, pos_unit),
                    xy=(x_right, y_bottom), xytext=(-5, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9, color="white",
                    bbox=label_box, zorder=6)

        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.set_title("%s  --  %s\nPTS_ROI = [[%.6f, %.6f], [%.6f, %.6f]]  %s"
                     % (os.path.basename(run_dir), frame_label,
                        x_left, y_top, x_right, y_bottom, pos_unit), fontsize=10)

        # Next to param_postProcessing.json, not in the run's PostProcessing/:
        # the ROI belongs to the dataset, not to the run it happened to be
        # picked on.
        fig_path = os.path.join(base_dir, FIG_NAME)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print("figure  written to %s" % fig_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)
