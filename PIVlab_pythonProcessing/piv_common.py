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
The tools live next to this file, so a plain `import piv_common` resolves:
Python puts a script's own directory on sys.path, and Jupyter runs a notebook
with its folder as the working directory. Run a notebook from somewhere else and
the import will fail -- start it from this folder.

Anything genuinely specific to one tool (REPROCESS_ALL, DETREND, output file
names, plotting) deliberately stays in that tool.
"""

import os
import re

import numpy as np
from scipy.io import loadmat

# --------------------------------------------------------------------------- #
#  Configuration shared by every tool
#
#  Deliberately NOT here:
#    BASE_DIR  - each tool points at whichever dataset it is working on, and they
#                genuinely differ (batch_KineticEnergy on k6_TopBottom, the FFT
#                batches on k20_bottomOnly). Centralising it would silently move
#                a tool to another dataset, so it stays in each tool.
#    NPZ_NAME  - per-tool as well ('KineticEnergy_timeSeries.npz' vs
#                'VelocityFFT.npz'), which is why resolve_npz takes it as an
#                argument rather than reading a module constant.
# --------------------------------------------------------------------------- #
PIV_FILENAME = "PIVlab_results_uncalibrated.mat"
LOG_FILENAME = "acquisition_log.txt"

# Pixel calibration. Same in x and y.
XSCALE = 1.2323e-4   # m/px
YSCALE = XSCALE

# Fixed ROI, calibrated (metres): [(x0, y0), (x1, y1)] (opposite corners).
PTS_ROI = [(0.02002585173778408, 0.13110555228124998),
           (0.19561960104659090, 0.00659505254715910)]

# PIVlab pairs images (1+2, 3+4, ...), so every velocity field consumes this many
# camera frames: the PIV field rate is cam_fps / FRAMES_PER_FIELD.
FRAMES_PER_FIELD = 2

# If the acquisition log cannot be read, the run stays UNCALIBRATED: velocity dt,
# PIV sampling and both spatial scales fall back to 1, so velocities are in
# px/frame, positions in px, and timestamps in frame index.
UNCAL_DT = 1.0
UNCAL_FPS = 1.0
UNCAL_SCALE = 1.0

# Legacy folder-name divisors. The current naming states frequencies in Hz
# ('frot0.50Hz'); the legacy naming used zero-padded integers where 'frot050'
# meant 0.50 Hz and 'flib0400' meant 0.400 Hz. Both are still parsed.
FROT_DIVISOR = 100.0
FLIB_DIVISOR = 1000.0

# The two spatial extents every tool can process. 'ROI' crops to PTS_ROI, 'FULL'
# keeps the whole field. Output files are tagged with the chosen tag (_ROI/_FULL).
# Each notebook defines its own REGION locally (not imported from here): the batch
# tools accept 'BOTH' too, the single-run / plotting notebooks pick one.
REGIONS = ("ROI", "FULL")

# Cylinder radius [m]. Used to normalise kinetic energy by the libration velocity
# scale -- see libration_ke_scale().
R = 138e-3

# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_piv(file_path):
    """Load a PIVlab wienerwurst .mat file and return calibrated-ready fields.

    Returns X, Y (2D grids), U, V (validated velocity, NaN where invalid),
    and nframes. Axis flips / sign flips reproduce the notebook exactly.
    """
    mat = loadmat(file_path,
                  variable_names=["x", "y", "u", "v", "u_filt", "v_filt"])
    if "u_filt" not in mat:
        raise ValueError("Not a wienerwurst PIV file: %s" % file_path)

    X_original = mat["x"][:, :, 0]
    Y_original = mat["y"][:, :, 0]

    U_filtered = mat["u_filt"]
    V_filtered = mat["v_filt"]
    U_org = mat["u"]           # velocity prior to validation
    V_org = mat["v"]

    # Flip axes (match notebook)
    X = X_original[:, ::-1].astype(float, copy=False)
    Y = Y_original[::-1, :].astype(float, copy=False)
    U = U_filtered[::-1, ::-1, :].astype(float, copy=False)
    V = V_filtered[::-1, ::-1, :].astype(float, copy=False)
    U_original = U_org[::-1, ::-1, :]
    V_original = V_org[::-1, ::-1, :]

    nframes = U_filtered.shape[2]

    # Mask wherever the un-validated vectors are NaN
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

    Loads everything written by batch_KineticEnergy / single_KineticEnergy --
    run, region, calibrated, dt_vel, fps, xscale, yscale, pts_ROI, npoints,
    nframes, t, Ek_frame, mean_Ekin, std_Ekin (older files may lack a few) --
    returns them as a dict {name: value}, and prints the list of what it found.
    """
    return _load_npz(filepath, "KineticEnergy")


def read_VelocityFFT(filepath):
    """Read a VelocityFFT_<region>.npz and return its variables.

    Loads everything written by batch_VelocityFFT -- run, region, calibrated,
    dt_vel, fps, xscale, yscale, pts_ROI, nframes, npoints, window, detrend, f,
    amp_u, amp_v, amp_total, powerU, powerV, f_peak, f_pol, pol_IW, polarization,
    powerU_mean, powerV_mean, powerRatio(+_std/_med/_p25/_p75) -- returns them as
    a dict {name: value}, and prints the list of what it found.
    """
    return _load_npz(filepath, "VelocityFFT")


def write_xlsx(df, path):
    """Write a summary DataFrame to .xlsx with boolean columns stringified.

    Some pandas/openpyxl versions write numpy-bool cells as blank, so columns
    like 'processed' / 'calibrated' / 'kept' come out empty in the .xlsx while
    the .csv is fine. Converting booleans to their 'True'/'False' text first
    keeps those columns populated and matching the .csv.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype(str)
    out.to_excel(path, index=False)


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
    so the caller can skip set_ylim. "Closest power of 10" is round(log10), which
    can sit just inside the data extreme -- decade-clean bounds by design.
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
    ymax = 10.0 ** round(np.log10(max(full_max))+1)
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
