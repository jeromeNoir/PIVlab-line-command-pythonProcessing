"""Auto-generated .py twin of make_velocity_movie.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Velocity-field movie from a PIVlab `.mat`
# 
# Reads the velocity field of one run straight from a PIV `.mat` and writes an
# `.mp4` of the field -- the velocity norm `|u|` as a colour map -- over a time
# window.
# 
# - `PIV_FILENAME` is a **local** constant here (not read from
#   `param_postProcessing.json`) -- point it at a raw or a band-passed
#   (`..._bp<lo>-<hi>Hz.mat`) file. Calibration (`XSCALE`/`YSCALE`), the log name
#   and the ROI still come from the dataset's `param_postProcessing.json`.
# - `T_START` / `T_END` are in **seconds** (from the first PIV field); `None` -> first / last frame.
# - `OUTPUT_FPS` sets the playback rate of the `.mp4`.
# - If `PIV_FILENAME` carries a `_bp<lo>-<hi>Hz` tag, that tag is appended to the
#   movie file name (same rule as `process_single_Velocity`), so a filtered
#   movie never overwrites the raw one.
# 
# The velocity honours `VALIDATE_VELOCITY` from the dataset parameters (filtered
# field when `True`, original-with-NaN when `False`), since it goes through
# `load_piv`.


# ## 1. Imports


import os
import re
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter


from piv_postprocessing_lib import (topography_arrangement, load_piv, parse_run_name,
                        read_acquisition_params, read_paramPostprocessing, region_tag)

# A Jupyter kernel may not inherit the shell PATH -- point matplotlib at ffmpeg.
matplotlib.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"


# ## 2. User settings


# --- Mute switch -----------------------------------------------------------
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = False
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# Run folder holding the .mat and the acquisition log.
RUN_DIR = ('/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
           'CylinderExperimentsGMA/k6_TopBottom/frot0.50Hz_flib1.500Hz_dphi2deg_SS1')

# LOCAL PIV .mat filename -- NOT taken from param_postProcessing.json. Point it at
# a raw file or a band-passed one (..._bp<lo>-<hi>Hz.mat).
PIV_FILENAME = 'PIVlab_results_uncalibrated_bp0.62-0.88Hz.mat'

# --- Time window [s], counted from the first PIV field -- None -> first / last frame
T_START = None
T_END   = None

# --- Movie ----------------------------------------------------------------
OUTPUT_FPS  = 20        # playback frame rate of the .mp4
MAX_FRAMES  = 600       # cap; a longer window is strided down to this many frames
MAG_CLIM    = None      # speed colour scale (vmin, vmax); None -> autoscale (0..p99)

# --- View -----------------------------------------------------------------
REGION       = 'FULL'   # 'FULL' whole field, or 'ROI' to zoom the view to PTS_ROI
SHOW_ROI_BOX = True      # overlay the ROI rectangle from PTS_ROI

# ----------------------------------------------------------------------
# Calibration / ROI / log name come from the dataset's param file (RUN_DIR's parent).
_P = read_paramPostprocessing(os.path.dirname(RUN_DIR.rstrip('/')))
k0 = _P.k0
TOP_TOPO, BOTTOM_TOPO = topography_arrangement(RUN_DIR)
XSCALE, YSCALE = _P.XSCALE, _P.YSCALE
PTS_ROI = _P.PTS_ROI
LOG_FILENAME = _P.LOG_FILENAME
UNCAL_SCALE = _P.UNCAL_SCALE
tag = region_tag(REGION)


# ## 3. Load the velocity field


name = os.path.basename(RUN_DIR.rstrip("/"))
piv_file = os.path.join(RUN_DIR, PIV_FILENAME)
if not os.path.isfile(piv_file):
    raise FileNotFoundError("no %s in %s" % (PIV_FILENAME, RUN_DIR))

dt_vel, fps, ok = read_acquisition_params(os.path.join(RUN_DIR, LOG_FILENAME))
if ok:
    xscale, yscale = XSCALE, YSCALE
else:
    print("[warn] calibration not possible -- axes in px, velocity in px/frame")
    xscale = yscale = UNCAL_SCALE       # dt_vel, fps already fell back to 1

# load_piv honours VALIDATE_VELOCITY and the flips/sign convention (z up).
X, Y, U, V, nframes = load_piv(piv_file)
x = xscale * X                          # coordinates -> m (or px)
z = yscale * Y
u = xscale * U / dt_vel                 # velocity -> m/s (or px/frame)
w = yscale * V / dt_vel

# Time window [s] -> frame indices (nearest frame). None -> first / last frame.
_tmax = (nframes - 1) / fps
i0 = 0 if T_START is None else int(round(T_START * fps))
i1 = (nframes - 1) if T_END is None else int(round(T_END * fps))
if not (0 <= i0 <= i1 <= nframes - 1):
    raise ValueError("time window [%s, %s] s -> frames [%d, %d], outside "
                     "[0, %.4g] s ([0, %d])" % (T_START, T_END, i0, i1, _tmax, nframes - 1))
frames_all = np.arange(i0, i1 + 1)
t0, t1 = i0 / fps, i1 / fps

# Band-pass tag from the PIV filename -> appended to the movie name (never
# overwrites the raw movie). Same rule as process_single_Velocity.
_bpm = re.search(r"_bp[-0-9.]+Hz", PIV_FILENAME)
BP_TAG = _bpm.group(0) if _bpm else ""

print("run     : %s" % name)
print("grid    : %d x %d vectors   frames %d" % (x.shape[0], x.shape[1], nframes))
print("fps     : %.4g Hz   dt = %.4g s   calibrated = %s" % (fps, dt_vel, ok))
print("window  : t %.4g..%.4g s  -> frames %d..%d  (%d frames)  BP_TAG=%r"
      % (t0, t1, i0, i1, frames_all.size, BP_TAG))


# ## 4. Write the movie


out_dir = os.path.join(RUN_DIR, "PostProcessing")
os.makedirs(out_dir, exist_ok=True)
movie_path = os.path.join(out_dir,
                          "velocity_movie_%s_t%g-%gs%s.mp4" % (tag, t0, t1, BP_TAG))

# Speed and the shared colour scale.
mag = np.sqrt(u ** 2 + w ** 2)
_win = mag[:, :, frames_all]
vmin, vmax = ((0.0, float(np.nanpercentile(_win, 99))) if MAG_CLIM is None else MAG_CLIM)

# Stride the window down to MAX_FRAMES if needed.
stride = max(1, int(np.ceil(frames_all.size / MAX_FRAMES)))
fr = frames_all[::stride]
if stride > 1:
    print("window > MAX_FRAMES (%d): keeping every %dth frame -> %d frames"
          % (MAX_FRAMES, stride, fr.size))

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
pcm = ax.pcolormesh(x, z, mag[:, :, fr[0]], cmap="viridis", shading="gouraud",
                    vmin=vmin, vmax=vmax)
if SHOW_ROI_BOX and PTS_ROI is not None:
    (xa, za), (xb, zb) = PTS_ROI
    ax.add_patch(Rectangle((min(xa, xb), min(za, zb)), abs(xb - xa), abs(zb - za),
                           fill=False, ec="red", lw=1.2))
if tag == "ROI" and PTS_ROI is not None:
    (xa, za), (xb, zb) = PTS_ROI
    ax.set_xlim(min(xa, xb), max(xa, xb))
    ax.set_ylim(min(za, zb), max(za, zb))
ax.set_aspect("equal")
_ulab = "m/s" if ok else "px/frame"
_xlab = "x [m]" if ok else "x [px]"
_zlab = "z [m]" if ok else "z [px]"
ax.set_xlabel(_xlab)
ax.set_ylabel(_zlab)
fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02).set_label(r"$|u|$  [%s]" % _ulab)
fig.suptitle("%s   (%s)   ($k_0=%g$, top=%s, bottom=%s)%s"
             % (name, tag, k0, TOP_TOPO, BOTTOM_TOPO,
                ("   %s" % BP_TAG) if BP_TAG else ""), fontsize=10)
ttl = ax.set_title("")
fig.tight_layout()

writer = FFMpegWriter(fps=OUTPUT_FPS, bitrate=4000, metadata={"title": name})
with writer.saving(fig, movie_path, dpi=150):
    for k in fr:
        pcm.set_array(mag[:, :, k])
        ttl.set_text("frame %d   t = %.3f s" % (k, k / fps))
        writer.grab_frame()
plt.close(fig)
print("movie  ->", movie_path)
