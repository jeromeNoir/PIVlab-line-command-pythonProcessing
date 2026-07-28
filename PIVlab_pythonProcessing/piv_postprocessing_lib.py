"""
Shared helpers for the PIVlab post-processing tools.

Every tool -- both the .py scripts and their .ipynb twins -- imports from here,
so a fix lands once instead of being copy-pasted into each file. Before this
module existed the same load_piv/parse_run_name/... bodies were duplicated across
five tools; a single change (e.g. teaching the parsers the 'frot0.50Hz' folder
naming) meant editing nine separate places and hoping none were missed.

Layout
------
  configuration    dataset paths, calibration, ROI, folder-name conventions
  data loading     load_piv, create_mask, extract_roi
  metadata         read_acquisition_params, parse_run_name, parse_flib/frot,
                   resolve_npz
  spectra          amp_from_rfft, compute_fft, peak_freq
  physics          compute_polarization

Importing
---------
The tools live next to this file, so a plain `import piv_postprocessing_lib` resolves:
Python puts a script's own directory on sys.path, and Jupyter runs a notebook
with its folder as the working directory. Run a notebook from somewhere else and
the import will fail -- start it from this folder.

Anything genuinely specific to one tool (REPROCESS_ALL, DETREND, output file
names, plotting) deliberately stays in that tool.
"""

import os
import re
import json
import shutil

import numpy as np
from scipy.io import loadmat

# --------------------------------------------------------------------------- #
#  Configuration -- per-dataset parameter files
#
#  This module holds NO hardcoded parameter values. The DEFAULTS live in
#  param_postProcessing_default.json next to this file; every dataset folder has
#  its own param_postProcessing.json (calibration, ROI, k0, ...) overriding them.
#  A tool loads its dataset's file with
#      P = read_paramPostprocessing(BASE_DIR)
#  which returns a namespace (P.XSCALE, P.PTS_ROI, ...) AND rebinds the module
#  globals, so the library's own helpers -- read_acquisition_params
#  (FRAMES_PER_FIELD), parse_run_name (the divisors), region_tag (REGIONS),
#  libration_ke_scale / dimensionless_numbers (R, H, l, nu) -- use the dataset's
#  values. The default file is loaded once at import (bottom of this block) so
#  those globals exist even before, or without, a dataset file being read.
#
#  If a dataset folder has no param_postProcessing.json, read_paramPostprocessing
#  copies the default one there and stops, so the user can edit it and relaunch.
#
#  Deliberately NOT in a parameter file:
#    BASE_DIR  - names WHERE the file is read from, so it cannot be read from it.
#    NPZ_NAME  - per-tool ('KineticEnergy_timeSeries.npz' vs 'VelocityFFT.npz'),
#                so resolve_npz takes it as an argument.
# --------------------------------------------------------------------------- #
PARAM_FILENAME = "param_postProcessing.json"
DEFAULT_PARAM_FILENAME = "param_postProcessing_default.json"
_DEFAULT_PARAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   DEFAULT_PARAM_FILENAME)

# The keys stored in a parameter file. k (= k0*pi/R) and l (= 2*pi/k) are DERIVED
# from k0 and R on load, so only k0 is stored.
_PARAM_KEYS = ("PIV_FILENAME", "LOG_FILENAME", "XSCALE", "YSCALE", "PTS_ROI",
               "FRAMES_PER_FIELD", "UNCAL_DT", "UNCAL_FPS", "UNCAL_SCALE",
               "FROT_DIVISOR", "FLIB_DIVISOR", "REGIONS", "R", "H", "k0", "nu")


class _Param(dict):
    """A dict whose entries are also attributes, so P.XSCALE == P['XSCALE'].

    Returned by read_paramPostprocessing; the attribute form reads best in the
    notebooks (P.PTS_ROI), the dict form is handy for looping over the keys.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:                     # -> AttributeError, not KeyError
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _read_default_params():
    """The default parameter dict from param_postProcessing_default.json.

    This file is the single source of default values (the module no longer holds
    any). It sits next to this module and must not be deleted -- it is the
    fallback template both for filling missing keys and for seeding a dataset
    that has no parameter file of its own.
    """
    if not os.path.isfile(_DEFAULT_PARAM_PATH):
        raise FileNotFoundError(
            "default parameter file missing: %s" % _DEFAULT_PARAM_PATH)
    with open(_DEFAULT_PARAM_PATH) as fh:
        return json.load(fh)


def _params_from_raw(raw):
    """Typed _Param from a raw dict: fill missing keys from the defaults, restore
    the in-module types (JSON has no tuples) and derive k and l from k0 and R."""
    defaults = _read_default_params()
    P = _Param()
    for key in _PARAM_KEYS:
        P[key] = raw.get(key, defaults[key])
    P["PTS_ROI"] = [tuple(p) for p in P["PTS_ROI"]]
    P["REGIONS"] = tuple(P["REGIONS"])
    # k and l are DERIVED from k0 and R -- recomputed so a stale value can never
    # be used (only k0 is stored).
    P["k"] = P["k0"] * np.pi / P["R"]
    P["l"] = 2.0 * np.pi / P["k"]
    return P


def _apply_params(P):
    """Rebind the module globals to the values in P (k and l included, even
    though they are derived rather than stored)."""
    globals().update({key: P[key] for key in _PARAM_KEYS})
    globals()["k"] = P["k"]
    globals()["l"] = P["l"]


def write_paramPostprocessing(base_dir, params=None, overwrite=False):
    """Write param_postProcessing.json into base_dir (from the default template);
    return its path.

    params (a dict) overrides individual default values. An existing file is
    preserved unless overwrite=True, so a hand-tuned calibration is never
    silently clobbered.
    """
    path = os.path.join(base_dir, PARAM_FILENAME)
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError("%s already exists (pass overwrite=True)" % path)
    data = _read_default_params()
    if params:
        data.update(params)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return path


def read_paramPostprocessing(base_dir, apply=True):
    """Load base_dir/param_postProcessing.json into a namespace.

    base_dir is the dataset folder that holds the run sub-folders (BASE_DIR in
    the batch tools; the parent of a single RUN_DIR elsewhere). Returns a _Param
    with every key in _PARAM_KEYS, each missing key filled from the default file,
    plus the derived k and l. apply=True (the default) also rebinds the module
    globals, so the library's own helpers use this dataset's values.

    If the dataset folder has no parameter file, the default one is copied there
    and the program stops (SystemExit), so the user can edit it and relaunch.
    """
    path = os.path.join(base_dir, PARAM_FILENAME)
    if not os.path.isfile(path):
        shutil.copyfile(_DEFAULT_PARAM_PATH, path)
        print("No parameter file found, a default one has been created, please "
              "edit and relaunch the notebook\n  %s" % path)
        raise SystemExit
    with open(path) as fh:
        raw = json.load(fh)
    P = _params_from_raw(raw)
    if apply:
        _apply_params(P)
    return P


# Populate the module globals (XSCALE, PTS_ROI, FRAMES_PER_FIELD, the divisors,
# REGIONS, R, H, k0, k, l, nu, ...) from the default file at import, so the
# helpers have parameter values even before -- or without -- a dataset file being
# read. read_paramPostprocessing(BASE_DIR) later overrides them per dataset.
_apply_params(_params_from_raw({}))


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
# PIVlab writes the velocity under different names depending on the export.
# Filtered (validated) fields are preferred; the un-validated ones are the
# fallback and also supply the NaN validity pattern.
FILTERED_NAMES = (("u_filt", "v_filt"), ("u_filtered", "v_filtered"))
UNFILTERED_NAMES = (("u", "v"), ("u_original", "v_original"))

_VELOCITY_VARS = ["x", "y"] + [n for pair in FILTERED_NAMES + UNFILTERED_NAMES
                               for n in pair]


def _first_present(mat, pairs):
    """First (u_name, v_name) pair of `pairs` that is present in `mat`, else None."""
    for u_name, v_name in pairs:
        if u_name in mat and v_name in mat:
            return u_name, v_name
    return None


def _as_frames(arr):
    """(ny, nx, nframes) float stack from a 3-D array or a cell array of 2-D frames.

    PIVlab's batch export stores the velocity as one 3-D array; its session
    export stores a MATLAB cell array with one 2-D frame per cell (scipy loads
    that as an object array), so both are normalised here.
    """
    a = np.asarray(arr)
    if a.dtype == object:                       # cell array: one frame per cell
        return np.stack([np.asarray(c, dtype=float) for c in a.ravel()], axis=-1)
    return a.astype(float, copy=False)


def _as_grid(arr):
    """2-D coordinate grid from a 3-D stack, a cell array, or a plain 2-D array."""
    a = np.asarray(arr)
    if a.dtype == object:                       # cell array: grid in the first cell
        return np.asarray(a.ravel()[0], dtype=float)
    if a.ndim == 3:                             # stored per frame; grid is constant
        return a[:, :, 0].astype(float, copy=False)
    return a.astype(float, copy=False)


def load_piv(file_path):
    """Load a PIVlab .mat file and return calibrated-ready fields.

    Returns X, Y (2D grids), U, V (velocity, NaN where invalid) and nframes.
    Axis flips / sign flips reproduce the original notebook exactly.

    The velocity is taken from the first pair present, in this order:
      u_filt / v_filt        -- filtered, batch export
      u_filtered / v_filtered-- filtered, session export
      u / v  or  u_original / v_original  -- unfiltered fallback, with a message
    Raises ValueError (after printing) when the file holds no velocity at all.
    Both the 3-D-array and cell-array layouts are handled.
    """
    mat = loadmat(file_path, variable_names=_VELOCITY_VARS)

    names = _first_present(mat, FILTERED_NAMES)
    unfiltered = _first_present(mat, UNFILTERED_NAMES)
    if names is None:
        if unfiltered is None:
            print("No velocity field found")
            raise ValueError("No velocity field found in %s" % file_path)
        print("no filtered velocity found using the unfiltered velocities")
        names = unfiltered

    X_original = _as_grid(mat["x"])
    Y_original = _as_grid(mat["y"])
    U_sel = _as_frames(mat[names[0]])
    V_sel = _as_frames(mat[names[1]])

    # Flip axes (match notebook)
    X = X_original[:, ::-1]
    Y = Y_original[::-1, :]
    U = U_sel[::-1, ::-1, :]
    V = V_sel[::-1, ::-1, :]

    nframes = U.shape[2]

    # Mask wherever the un-validated vectors are NaN. Skipped when the
    # unfiltered fields are absent, or when they are what we just loaded.
    if unfiltered is not None and unfiltered != names:
        U_original = _as_frames(mat[unfiltered[0]])[::-1, ::-1, :]
        V_original = _as_frames(mat[unfiltered[1]])[::-1, ::-1, :]
        if U_original.shape == U.shape:
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


def calibrate(x_px, y_px, u_px, v_px, xscale, yscale, dt_pulse):
    """Pixels/pulse -> metres and m/s, with z pointing up.

    PIVlab works in image coordinates: y grows downward and v is positive
    downward. Here z = y_max - y and w = -v, and axis 0 is reversed so the row
    index grows with z. Flipping coordinates and data together is a pure
    reindexing -- no physics changes, the arrays just read bottom-to-top.

    Rank-agnostic: the grids are 2-D and the velocities may be 2-D (one frame) or
    3-D (frames last). Reversing only axis 0 with [::-1] is correct for both.
    """
    x = (x_px * xscale)[::-1]
    z = ((y_px.max() - y_px) * yscale)[::-1]
    u = (u_px * xscale / dt_pulse)[::-1]
    w = (-v_px * yscale / dt_pulse)[::-1]
    return x, z, u, w


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





def region_tag(region):
    """Validate a region choice and return it as the filename tag 'ROI'/'FULL'."""
    if region not in REGIONS:
        raise ValueError("region must be one of %s, got %r" % (REGIONS, region))
    return region


def region_fields(x, y, u, v, pts_roi, region):
    """Return (x, y, u, v) restricted to the chosen region.

    region='FULL' -> the whole field unchanged. region='ROI' -> cropped to the
    PTS_ROI rectangle, falling back to the full field if the ROI selects nothing
    (matching the batch tools' long-standing behaviour).
    """
    region_tag(region)                       # validate
    if region == "FULL":
        return x, y, u, v
    xr, yr, ur, vr = extract_roi(x, y, u, v, create_mask(x, y, pts_roi))
    return (x, y, u, v) if ur.size == 0 else (xr, yr, ur, vr)


def regions_to_run(region):
    """Expand a batch's REGION setting into the tuple of regions to process.

    'BOTH' -> ('ROI', 'FULL'); 'ROI' or 'FULL' -> just that one. Any other value
    raises, so a typo fails loudly instead of silently doing nothing.
    """
    if region == "BOTH":
        return REGIONS
    return (region_tag(region),)             # validates ROI/FULL, else raises


# --------------------------------------------------------------------------- #
#  Metadata
# --------------------------------------------------------------------------- #
def read_acquisition_log(log_path):
    """The last data row of a tab-separated PIVlab acquisition log, as a dict.

    The general reader: keys are the header columns (cam_fps, pulse_sep,
    imageamount, ...), values are strings. read_acquisition_params is the numeric
    convenience wrapper over the two columns the batch tools need.
    """
    import csv
    with open(log_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError("no data rows in " + log_path)
    return rows[-1]


def read_acquisition_params(log_path):
    """Return (dt_vel_s, fps_Hz, ok) from the run's acquisition log.

    The log is tab-separated with a header; the recording row is the LAST data
    row. From it:
      - dt_vel = pulse_sep * 1e-6            (s, image separation within a pair
                                              -> velocity magnitude)
      - fps    = cam_fps / FRAMES_PER_FIELD  (Hz, PIV field sampling frequency ->
                                              timestamps). cam_fps is the camera
                                              frame rate; PIV pairs images, so a
                                              field is produced every 2 frames.
    Returns ok=False if the log / its last row cannot be read.
    """
    try:
        with open(log_path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        header = lines[0].split("\t")
        i_pulse = header.index("pulse_sep")
        i_fps = header.index("cam_fps")
        for ln in reversed(lines[1:]):
            cols = ln.split("\t")
            try:
                return (float(cols[i_pulse]) * 1e-6,
                        float(cols[i_fps]) / FRAMES_PER_FIELD, True)
            except (IndexError, ValueError):
                continue
    except (OSError, ValueError):
        pass
    return UNCAL_DT, UNCAL_FPS, False


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

    e.g. 'frot0.50Hz_flib0.400Hz_dphi2.5deg_SS1' -> (0.5, 0.4, 2.5), and the
    legacy 'frot050_flib0400_dphi2.5deg_SS1' gives the same.
    Missing tokens come back as NaN.
    """
    frot_hz = parse_freq_token(name, "frot", FROT_DIVISOR)
    flib_hz = parse_freq_token(name, "flib", FLIB_DIVISOR)
    dphi = re.search(r"dphi([\d.]+)deg", name)
    dphi_deg = float(dphi.group(1)) if dphi else np.nan
    return frot_hz, flib_hz, dphi_deg


def parse_flib(run_name):
    """Libration frequency (Hz) from a run name, or None if absent."""
    v = parse_freq_token(str(run_name), "flib", FLIB_DIVISOR)
    return None if np.isnan(v) else v


def parse_frot(run_name):
    """Rotation frequency (Hz) from a run name, or None if absent."""
    v = parse_freq_token(str(run_name), "frot", FROT_DIVISOR)
    return None if np.isnan(v) else v


def resolve_npz(path, npz_name):
    """Accept the .npz path directly, or a run folder containing it.

    npz_name is explicit rather than a module constant: the tools look for
    different files ('KineticEnergy_timeSeries.npz' vs 'VelocityFFT.npz'), so a
    shared default here would silently send one of them after the wrong file.
    """
    if os.path.isdir(path):
        cand = os.path.join(path, "PostProcessing", npz_name)
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(path, npz_name)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError("No %s under %s" % (npz_name, path))
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def _load_npz(filepath, kind):
    """Load an .npz, print its stored variables (name, dtype, shape/value), and
    return them as a plain dict {name: value}."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)
    data = np.load(filepath, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    print("%s  <-  %s" % (kind, os.path.basename(filepath)))
    print("  %d stored variables:" % len(out))
    for k in data.files:
        v = np.asarray(out[k])
        if v.ndim == 0:                       # scalar: run name, fps, region, ...
            print("    %-16s = %s" % (k, v.item()))
        else:                                 # array: t, Ek_frame, spectra, ...
            print("    %-16s  %-8s shape %s" % (k, v.dtype, v.shape))
    return out


def read_KineticEnergy(filepath):
    """Read a KineticEnergy_timeSeries_<region>.npz and return its variables.

    Loads everything written by batch_KineticEnergy / reprocess_single_KineticEnergy --
    run, region, calibrated, dt_vel, fps, xscale, yscale, pts_ROI, npoints,
    nframes, t, Ek_frame, mean_Ekin, std_Ekin (older files may lack a few) --
    returns them as a dict {name: value}, and prints the list of what it found.
    """
    return _load_npz(filepath, "KineticEnergy")


def read_VelocityFFT(filepath):
    """Read a VelocityFFT_<region>.npz and return its variables.

    Loads everything written by batch_Velocity -- run, region, calibrated,
    dt_vel, fps, xscale, yscale, pts_ROI, nframes, npoints, window, detrend, f,
    amp_u, amp_v, amp_total, powerU, powerV, f_peak, f_pol, pol_IW, polarization,
    powerU_mean, powerV_mean, powerRatio(+_std/_med/_p25/_p75) -- returns them as
    a dict {name: value}, and prints the list of what it found.
    """
    return _load_npz(filepath, "VelocityFFT")


# --------------------------------------------------------------------------- #
#  Spectra
# --------------------------------------------------------------------------- #
def amp_from_rfft(fft, n):
    """One-sided amplitude |FFT|*2/N along the last axis (DC/Nyquist keep 1/N)."""
    amp = np.abs(fft) * 2.0 / n
    amp[..., 0] = np.abs(fft[..., 0]) / n        # DC term is not doubled
    if n % 2 == 0:
        amp[..., -1] = np.abs(fft[..., -1]) / n  # Nyquist is not doubled either
    return amp


def compute_fft(ek, fps, detrend=True):
    """Return (freq, amp, fft) for the positive-frequency half of a series.

    `fft` is the complex one-sided spectrum from numpy.fft.rfft and `amp` is its
    normalised amplitude (|FFT| * 2 / N, so a pure tone reads as its physical
    amplitude; the DC bin keeps the 1/N scaling).
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
    return freq, amp_from_rfft(fft, n), fft


def peak_freq(f, amp):
    """(frequency, amplitude) of the spectrum maximum, ignoring the DC bin.

    (nan, nan) if there is nothing to pick.
    """
    if np.size(amp) < 2 or np.size(f) < 2 or np.all(np.isnan(amp)):
        return np.nan, np.nan
    k = 1 + int(np.nanargmax(amp[1:]))
    return float(f[k]), float(amp[k])


def peak_freq_in_band(f, amp, fmin, fmax):
    """(frequency, amplitude) of the spectrum maximum inside [fmin, fmax].

    Used for f_low, the strongest low-frequency response below the libration
    forcing (band 0.01 Hz .. f_lib/2). (nan, nan) if the band is empty, out of
    range, or holds no finite amplitude.
    """
    f = np.asarray(f, dtype=float)
    amp = np.asarray(amp, dtype=float)
    if not (np.isfinite(fmin) and np.isfinite(fmax)) or fmax <= fmin:
        return np.nan, np.nan
    m = np.isfinite(amp) & (f >= fmin) & (f <= fmax)
    if not m.any():
        return np.nan, np.nan
    idx = np.flatnonzero(m)
    k = idx[int(np.argmax(amp[idx]))]
    return float(f[k]), float(amp[k])


GUIDE_COLOR_ROT = "tab:blue"    # rotation
GUIDE_COLOR_LIB = "tab:red"     # libration forcing and its harmonic
GUIDE_COLOR_LOW = "tab:green"   # low-frequency peak and its sidebands


def amp_at_freq(f, amp, f_target, halfwidth=None):
    """Peak amplitude of `amp` near f_target.

    Returns the maximum of `amp` within [f_target - halfwidth, f_target +
    halfwidth]; halfwidth defaults to two frequency bins, so a peak that leaks a
    bin off the exact forcing frequency is still captured. nan if f_target is
    nan/non-finite or the window holds no finite sample.
    """
    f = np.asarray(f, dtype=float)
    amp = np.asarray(amp, dtype=float)
    if not np.isfinite(f_target):
        return np.nan
    if halfwidth is None:
        df = (f[1] - f[0]) if f.size > 1 else 0.0
        halfwidth = 2.0 * df
    m = np.isfinite(amp) & (f >= f_target - halfwidth) & (f <= f_target + halfwidth)
    if not m.any():
        return np.nan
    return float(np.nanmax(amp[m]))


def fft_guide_lines(frot=None, flib=None, f_low=None):
    """Reference frequencies to mark on an FFT plot: [(frequency, label, colour)].

    2*f_rot, f_lib and 2*f_lib are the forcing frequencies (2*f_rot is where an
    inertial-wave response is expected, so it is marked rather than f_rot itself);
    f_low is the strongest low-frequency peak (see peak_freq_in_band), and
    f_lib -/+ f_low are the sidebands it would produce by beating with the
    libration. Colour-coded by family -- rotation blue, libration red,
    low-frequency response green -- so a legend groups them at a glance. Entries
    that are unknown, non-finite or non-positive are dropped, so the caller can
    just iterate the result.

    Sorted by frequency, which is also the order the legend entries appear in.

    Returns tuples only -- no plotting here, so this module stays free of a
    matplotlib dependency and both notebooks mark the same set.
    """
    ok = lambda v: v is not None and np.isfinite(v) and v > 0
    out = []
    if ok(frot):
        out.append((2.0 * float(frot), r"$2f_{\mathrm{rot}}$", GUIDE_COLOR_ROT))
    if ok(flib):
        out.append((float(flib), r"$f_{\mathrm{lib}}$", GUIDE_COLOR_LIB))
        out.append((2.0 * float(flib), r"$2f_{\mathrm{lib}}$", GUIDE_COLOR_LIB))
    if ok(f_low):
        out.append((float(f_low), r"$f_{\mathrm{low}}$", GUIDE_COLOR_LOW))
        if ok(flib):
            out.append((float(flib) - float(f_low),
                        r"$f_{\mathrm{lib}}-f_{\mathrm{low}}$", GUIDE_COLOR_LOW))
            out.append((float(flib) + float(f_low),
                        r"$f_{\mathrm{lib}}+f_{\mathrm{low}}$", GUIDE_COLOR_LOW))
    out = [t for t in out if np.isfinite(t[0]) and t[0] > 0]
    return sorted(out, key=lambda t: t[0])


def fft_axis_limits(freq, amp, frot=None, flib=None):
    """Axis limits for an FFT amplitude plot, keyed to the forcing frequencies.

    Returns (xmax, ymin, ymax):
      xmax = max(4*frot, 4*flib)           -- freq.max() if neither is known.
      ymax = power of 10 closest to the largest amplitude over [0, xmax].
      ymin = power of 10 closest to the smallest amplitude over the band
             [max(2*frot, 2*flib), xmax]   -- the high-frequency tail, so the
             huge low-frequency / DC values do not pull the floor down.

    `amp` is one amplitude array, or a list/tuple of them sharing `freq`
    (e.g. [amp_u, amp_v] for a two-curve panel). ymin/ymax come back None when
    they cannot be formed (no positive data, or both land on the same decade),
    so the caller can skip set_ylim.

    ymax uses ceil(log10), i.e. the first decade at or above the largest
    amplitude, so the peak of the spectrum is always inside the axis. ymin uses
    the closest decade below the band minimum, which may sit just inside the
    smallest values -- the floor is a readability choice, the top is not.
    """
    freq = np.asarray(freq, dtype=float)
    amps = amp if isinstance(amp, (list, tuple)) else [amp]
    refs = [r for r in (frot, flib) if r is not None and np.isfinite(r) and r > 0]
    xmax = 4.0 * max(refs) if refs else float(np.nanmax(freq))
    fband = 2.0 * max(refs) if refs else 0.0

    full_max, band_min = [], []
    for a in amps:
        a = np.asarray(a, dtype=float)
        pos = np.isfinite(a) & (a > 0)
        f_full = pos & (freq >= 0) & (freq <= xmax)
        f_band = pos & (freq >= fband) & (freq <= xmax)
        if f_full.any():
            full_max.append(float(np.max(a[f_full])))
            band_min.append(float(np.min(a[f_band] if f_band.any() else a[f_full])))
    if not full_max:
        return xmax, None, None
    # ceil, not round: the first decade at or above the peak, so the maximum of
    # the spectrum is always inside the axis (round could land below it).
    ymax = 10.0 ** float(np.ceil(np.log10(max(full_max))))
    ymin = 10.0 ** round(np.log10(min(band_min)))
    if not (ymin < ymax):                    # collapsed to one decade
        ymin = None
    return xmax, ymin, ymax


def libration_ke_scale(dphi_deg, flib_hz):
    """Characteristic libration kinetic-energy scale for normalising Ek.

    (dphi[rad] * 2*pi*flib * R)**2 -- the square of the peak libration wall
    velocity (angular amplitude dphi, angular frequency 2*pi*flib, radius R).
    dphi_deg and flib_hz may be scalars or arrays; the result has units of
    (m/s)**2, matching Ek, so Ek / libration_ke_scale(...) is dimensionless.
    """
    return (dphi_deg * np.pi / 180.0 * flib_hz * R * 2.0 * np.pi) ** 2


FIG_FORMATS = ("png", "pdf")


def figure_filename(stem, fmt="png", normalized=False):
    """Build a figure path: '<stem>[_normalized].<fmt>'.

    Central so every tool tags the normalised twin and honours the png/pdf choice
    the same way. `stem` is the path WITHOUT extension and WITHOUT the
    _normalized suffix. fmt is 'png' or 'pdf' (a leading dot is tolerated).
    """
    fmt = str(fmt).lower().lstrip(".")
    if fmt not in FIG_FORMATS:
        raise ValueError("fmt must be one of %s, got %r" % (FIG_FORMATS, fmt))
    return "%s%s.%s" % (stem, "_normalized" if normalized else "", fmt)


def topography_arrangement(path):
    """(top_topo, bottom_topo) for a dataset, from its folder name in `path`.

    'TopBottom' -> (True, True); 'bottomOnly' -> (False, True); otherwise
    (nan, nan). `path` may be the dataset dir, a run dir, or any path that
    contains the dataset folder name.
    """
    low = str(path).lower()
    if "topbottom" in low:
        return True, True
    if "bottomonly" in low:
        return False, True
    return float("nan"), float("nan")


def dataset_annotation(k0, top_topo=None, bottom_topo=None):
    """Per-dataset figure annotation, e.g. '$k_0$ = 6, top=True, bottom=True'.

    top_topo/bottom_topo are appended only when given (both). Used in figure
    titles and legends so k0 and the topography arrangement travel together.
    """
    s = r"$k_0$ = %g" % k0
    if top_topo is not None and bottom_topo is not None:
        s += ", top=%s, bottom=%s" % (top_topo, bottom_topo)
    return s


def libration_velocity_scale(flib_hz, dphi_deg):
    """Peak libration wall velocity U0 = 2*pi*flib * R * dphi[rad] [m/s].

    The velocity scale that normalises U and V (U_star = U / U0). Its square is
    libration_ke_scale(dphi_deg, flib_hz), the Ek scale -- the two stay
    consistent because both use R (a module global, per-dataset).
    """
    return 2.0 * np.pi * flib_hz * R * (dphi_deg * np.pi / 180.0)


def dimensionless_numbers(frot_hz, flib_hz, dphi_deg):
    """Dimensionless numbers for one run, keyed by their summary-column name.

    Built from the run's forcing (frot, flib, dphi) and the container/fluid
    constants H, l, R, nu -- all module globals, so they follow the dataset once
    read_paramPostprocessing has run. Returned dict:

      U0_mps      libration wall velocity   2*pi*flib*R*dphi[rad]        [m/s]
      E           Ekman (height)            nu / (2*pi*frot*H**2)
      E_l         Ekman (wavelength)        nu / (2*pi*frot*l**2)
      delta_nu_m  viscous BL thickness      sqrt(E)*H                   [m]
      Ro          Rossby                    U0 / (2*pi*frot*R)
      Re          Reynolds (radius)         U0*R / nu
      Re_l        Reynolds (wavelength)     U0*l / nu
      Re_bl       Reynolds (BL thickness)   U0*delta_nu / nu

    Anything requiring frot (E, E_l, delta_nu, Ro, and the BL-based Re_bl) is NaN
    when frot is unknown or zero; U0/Re/Re_l need only flib and dphi.
    """
    frot = float(frot_hz)
    flib = float(flib_hz)
    dphi = float(dphi_deg)
    good_frot = np.isfinite(frot) and frot != 0.0

    U0 = 2.0 * np.pi * flib * R * (dphi * np.pi / 180.0)
    E = nu / (2.0 * np.pi * frot * H ** 2) if good_frot else np.nan
    E_l = nu / (2.0 * np.pi * frot * l ** 2) if good_frot else np.nan
    delta_nu = np.sqrt(E) * H if good_frot else np.nan
    Ro = U0 / (2.0 * np.pi * frot * R) if good_frot else np.nan
    Re = U0 * R / nu
    Re_l = U0 * l / nu
    Re_bl = U0 * delta_nu / nu

    return {"U0_mps": U0, "E": E, "E_l": E_l, "delta_nu_m": delta_nu,
            "Ro": Ro, "Re": Re, "Re_l": Re_l, "Re_bl": Re_bl}


# --------------------------------------------------------------------------- #
#  Physics
# --------------------------------------------------------------------------- #
def compute_polarization(f, powerU, powerV, frot):
    """Polarization curve for one run.

    Returns (f_pol, pol_IW, pol_data):
      f_pol    : 0.01..1 Hz, 100 points
      pol_IW   : 2*((2*frot/f_pol)**2 - 1), the inertial-wave prediction
                 (NaN if frot is unknown)
      pol_data : (powerV/powerU)**2 on the FFT frequency grid f
                 (NaN where powerU <= 0)
    """
    f_pol = np.linspace(0.01, 1.0, 100)
    if frot:
        pol_IW = 2.0 * ((2.0 * frot / f_pol) ** 2 - 1.0)
    else:
        pol_IW = np.full_like(f_pol, np.nan)
    pu = np.asarray(powerU, dtype=float)
    pv = np.asarray(powerV, dtype=float)
    pol_data = np.full_like(pu, np.nan)
    good = pu > 0
    pol_data[good] = (pv[good] / pu[good]) ** 2
    return f_pol, pol_IW, pol_data
