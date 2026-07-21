"""
Temporal band-pass filtering of a single PIVlab run, point by point.

Filters the two velocity components at EVERY grid point along time, keeping only
the content between two user-defined cut-off frequencies [f_low, f_high], and
writes the result next to the original file.

    <run>/PIVlab_results_uncalibrated.mat
      ->  <run>/PIVlab_results_uncalibrated_bp0.50-2.00Hz.mat

Units
-----
The source .mat is ALREADY uncalibrated: u/v are pixel displacements per laser
pulse pair and x/y are pixels. Filtering is a linear operation, so scaling and
filtering commute -- there is no need to convert to m/s and back, and this
script never does. Everything stays in the native PIVlab units, so the output is
a drop-in replacement for the input (same variables, same shapes, same dtypes)
and any downstream tool calibrates it exactly as it calibrates the original.

The only calibrated quantity a temporal filter needs is TIME, taken from
acquisition_log.txt:
    fps = cam_fps / FRAMES_PER_FIELD   (PIVlab pairs images, so a velocity field
                                        is produced every 2 camera frames)
Frequencies are therefore in Hz of the PIV field rate, with Nyquist = fps/2.

Filter
------
Zero-phase Butterworth (scipy sosfiltfilt) -- no phase shift, so filtered
structures stay where they were in time. The effective order is twice
FILTER_ORDER because the series is passed forwards and backwards.

  f_low <= 0        -> low-pass at f_high
  f_high >= Nyquist -> high-pass at f_low
  otherwise         -> band-pass [f_low, f_high]

NaNs are interpolated along time before filtering (filtfilt would otherwise
smear a single NaN across the whole series) and restored afterwards, so the
validity pattern of u/v is preserved exactly.

Edge transients
---------------
A Butterworth band-pass settles over roughly 1/f_low seconds, and sosfiltfilt's
default padding (a few dozen samples) is far shorter than that for the low
cut-offs used here -- which leaks rejected content into the first and last
seconds of the record. The padding is therefore stretched to
PAD_SETTLING_TIMES / f_low (capped at the record length); on a 50 s record with
f_low = 0.04 Hz this cuts the edge leakage of an out-of-band tone by ~65x.

It cannot be removed entirely: if the record is shorter than a few 1/f_low, the
first and last ~1/f_low seconds stay unreliable no matter the padding. The
script warns when that is the case -- lengthen the record or raise f_low.

Usage
-----
  python filter_velocity_bandpass.py --run <run_dir> --f-low 0.5 --f-high 2.0
  python filter_velocity_bandpass.py            # uses the RUN_DIR / F_LOW / F_HIGH below
"""

import os
import argparse
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat, whosmat

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
RUN_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
           "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/k6_TopBottom/"
           "frot0.50Hz_flib0.500Hz_dphi2deg_SS1")

PIV_FILENAME = "PIVlab_results_uncalibrated.mat"
LOG_FILENAME = "acquisition_log.txt"

# Cut-off frequencies [Hz], in the PIV field rate (NOT the camera rate).
F_LOW = 0.04
F_HIGH = 0.2

FILTER_ORDER = 4          # per direction; sosfiltfilt doubles it (zero-phase)

# Padding used by sosfiltfilt, expressed in filter settling times (~1/f_low).
# The scipy default is a few dozen samples -- orders of magnitude too short for
# sub-Hz cut-offs, which leaves large transients at both ends of the record.
PAD_SETTLING_TIMES = 3.0

# Velocity variables to filter. u/v are the pre-validation fields (they carry the
# NaN validity pattern), u_filt/v_filt the validated ones the analyses read.
VEL_VARS = ("u", "v", "u_filt", "v_filt")

# correlation_map describes the raw image correlation, not the filtered velocity,
# so it is meaningless in a filtered dataset -- and it is ~108 MB. Set True to
# copy it across anyway.
KEEP_CORRELATION_MAP = False

# PIVlab pairs images (1+2, 3+4, ...): one velocity field per 2 camera frames.
FRAMES_PER_FIELD = 2

# --------------------------------------------------------------------------- #
#  Metadata
# --------------------------------------------------------------------------- #
def read_fps(log_path, frames_per_field=FRAMES_PER_FIELD):
    """PIV field sampling rate [Hz] from the LAST data row of the acquisition log.

    Returns (fps, ok). Frequencies are meaningless without it, so ok=False is a
    hard error upstream rather than a silent fallback.
    """
    try:
        with open(log_path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        i_fps = lines[0].split("\t").index("cam_fps")
        for ln in reversed(lines[1:]):
            try:
                return float(ln.split("\t")[i_fps]) / frames_per_field, True
            except (IndexError, ValueError):
                continue
    except (OSError, ValueError):
        pass
    return np.nan, False


# --------------------------------------------------------------------------- #
#  Filtering
# --------------------------------------------------------------------------- #
def design_filter(fps, f_low, f_high, order=FILTER_ORDER):
    """Butterworth SOS for the requested band. Returns (sos, kind, lo, hi)."""
    nyq = fps / 2.0
    lo = None if (f_low is None or f_low <= 0) else float(f_low)
    hi = None if (f_high is None or f_high >= nyq) else float(f_high)

    if lo is not None and lo >= nyq:
        raise ValueError("f_low (%g Hz) is at or above Nyquist (%g Hz): "
                         "nothing would survive" % (lo, nyq))
    if lo is None and hi is None:
        raise ValueError("band [%s, %s] Hz spans the whole spectrum at fps=%g Hz: "
                         "nothing to filter" % (f_low, f_high, fps))
    if lo is not None and hi is not None:
        if lo >= hi:
            raise ValueError("f_low (%g) must be below f_high (%g)" % (lo, hi))
        return signal.butter(order, [lo, hi], btype="bandpass", fs=fps,
                             output="sos"), "bandpass", lo, hi
    if hi is not None:
        return signal.butter(order, hi, btype="lowpass", fs=fps,
                             output="sos"), "lowpass", None, hi
    return signal.butter(order, lo, btype="highpass", fs=fps,
                         output="sos"), "highpass", lo, None


def _fill_nans_time(A):
    """Interpolate NaNs along the last (time) axis. Returns (filled, nan_mask).

    filtfilt propagates a single NaN over the entire series, so gaps are bridged
    first and masked back out afterwards. Points that are NaN for all time are
    zero-filled and fully re-masked.
    """
    nan_mask = np.isnan(A)
    if not nan_mask.any():
        return A, nan_mask

    filled = A.copy()
    nt = A.shape[-1]
    idx = np.arange(nt)
    all_nan = nan_mask.all(axis=-1)

    for i, j in np.argwhere(nan_mask.any(axis=-1)):
        if all_nan[i, j]:
            filled[i, j, :] = 0.0
            continue
        s = filled[i, j, :]
        m = nan_mask[i, j, :]
        s[m] = np.interp(idx[m], idx[~m], s[~m])
    return filled, nan_mask


def settling_padlen(sos, nt, fps, lo, hi, n_settle=PAD_SETTLING_TIMES):
    """Padding matched to the filter's settling time, in samples.

    A Butterworth filter settles over ~1/f_ref seconds, f_ref being its lowest
    cut-off. scipy's default padding is 3*(2*n_sections+1) samples, which for a
    sub-Hz cut-off is orders of magnitude too short and leaves the first and last
    seconds of the output full of transient. Capped at nt-1, which is the most
    sosfiltfilt accepts.
    """
    minimum = 3 * (2 * len(sos) + 1)
    f_ref = lo if lo is not None else hi
    want = minimum if not f_ref else int(np.ceil(n_settle * fps / f_ref))
    return int(min(max(want, minimum), nt - 1))


def filter_field(A, sos, padlen=None):
    """Zero-phase filter a (ny, nx, nt) field along time, preserving NaNs."""
    A = np.asarray(A, dtype=float)
    filled, nan_mask = _fill_nans_time(A)
    kw = {} if padlen is None else {"padlen": padlen}
    try:
        out = signal.sosfiltfilt(sos, filled, axis=-1, **kw)
    except ValueError as exc:
        raise ValueError("filtering failed (%s). The record is probably too "
                         "short for this filter order -- lower FILTER_ORDER." % exc)
    out[nan_mask] = np.nan
    return out


# --------------------------------------------------------------------------- #
#  Naming
# --------------------------------------------------------------------------- #
def output_filename(orig_name, kind, lo, hi):
    """Original name + a tag describing the band, e.g. ..._bp0.50-2.00Hz.mat."""
    stem, ext = os.path.splitext(orig_name)
    if kind == "bandpass":
        tag = "bp%.2f-%.2fHz" % (lo, hi)
    elif kind == "lowpass":
        tag = "lp%.2fHz" % hi
    else:
        tag = "hp%.2fHz" % lo
    return "%s_%s%s" % (stem, tag, ext)


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def process_run(run_dir, f_low=F_LOW, f_high=F_HIGH, order=FILTER_ORDER,
                keep_correlation_map=KEEP_CORRELATION_MAP, overwrite=False):
    """Filter one run and write the result beside the original. Returns the path."""
    piv_path = os.path.join(run_dir, PIV_FILENAME)
    log_path = os.path.join(run_dir, LOG_FILENAME)
    if not os.path.isfile(piv_path):
        raise FileNotFoundError(piv_path)

    fps, ok = read_fps(log_path)
    if not ok:
        raise RuntimeError("could not read cam_fps from %s -- cut-off frequencies "
                           "cannot be interpreted without the sampling rate" % log_path)

    sos, kind, lo, hi = design_filter(fps, f_low, f_high, order)
    out_path = os.path.join(run_dir, output_filename(PIV_FILENAME, kind, lo, hi))
    if os.path.isfile(out_path) and not overwrite:
        raise FileExistsError("%s already exists (use --overwrite)" % out_path)

    # Copy every original variable except the ones we deliberately drop, so the
    # output keeps the PIV settings and stays a drop-in for the loaders.
    names = [n for n, _, _ in whosmat(piv_path)]
    skip = set() if keep_correlation_map else {"correlation_map"}
    wanted = [n for n in names if n not in skip]

    print("run      : %s" % os.path.basename(os.path.normpath(run_dir)))
    print("source   : %s" % PIV_FILENAME)
    print("fps      : %g Hz (PIV field rate, Nyquist %g Hz)" % (fps, fps / 2))
    print("filter   : %s %s, Butterworth order %d, zero-phase (effective %d)"
          % (kind,
             "[%.3f, %.3f] Hz" % (lo, hi) if kind == "bandpass" else
             "%.3f Hz" % (hi if kind == "lowpass" else lo),
             order, 2 * order))

    mat = loadmat(piv_path, variable_names=wanted)
    mat = {k: v for k, v in mat.items() if not k.startswith("__")}

    nt = mat["u_filt"].shape[2]
    T = nt / fps
    print("frames   : %d  (%.2f s, resolves down to ~%.3f Hz)" % (nt, T, fps / nt))

    padlen = settling_padlen(sos, nt, fps, lo, hi)
    # Lowest cut-off present: design_filter guarantees at least one of lo/hi.
    f_ref = lo if lo is not None else hi
    if f_ref is None:
        raise RuntimeError("no cut-off frequency to size the padding from")
    print("padding  : %d samples (%.1f s) -- filter settles over ~1/f_ref = %.1f s"
          % (padlen, padlen / fps, 1.0 / f_ref))
    if T < PAD_SETTLING_TIMES / f_ref:
        print()
        print("  WARNING: the record (%.1f s) is shorter than %g settling times"
              % (T, PAD_SETTLING_TIMES))
        print("           (%.1f s) for f_ref = %.4f Hz. The first and last ~%.0f s"
              % (PAD_SETTLING_TIMES / f_ref, f_ref, 1.0 / f_ref))
        print("           of the filtered record carry edge transients and should")
        print("           not be trusted. Raise the low cut-off or record longer.")
        print()

    for var in VEL_VARS:
        if var not in mat:
            print("  ! %s absent, skipped" % var)
            continue
        before = mat[var]
        after = filter_field(before, sos, padlen=padlen)
        rms_in = float(np.sqrt(np.nanmean(before ** 2)))
        rms_out = float(np.sqrt(np.nanmean(after ** 2)))
        print("  %-7s rms %.4f -> %.4f px/pulse  (%.1f%% of the variance kept)"
              % (var, rms_in, rms_out, 100 * (rms_out / rms_in) ** 2 if rms_in else np.nan))
        mat[var] = after

    # Provenance, so a filtered file is never mistaken for a raw one.
    mat.update({
        "filter_type": kind,
        "filter_f_low_Hz": np.nan if lo is None else lo,
        "filter_f_high_Hz": np.nan if hi is None else hi,
        "filter_order": order,
        "filter_zero_phase": 1,
        "filter_padlen": padlen,
        "filter_fps_Hz": fps,
        "filter_source_file": PIV_FILENAME,
        "filter_units": "unchanged from source: px displacement per pulse pair",
    })
    if not keep_correlation_map and "correlation_map" in names:
        mat["filter_dropped"] = "correlation_map"

    savemat(out_path, mat, do_compression=True)
    print("wrote    : %s  (%.1f MB)"
          % (os.path.basename(out_path), os.path.getsize(out_path) / 1e6))
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", default=RUN_DIR, help="run folder holding the .mat")
    p.add_argument("--f-low", type=float, default=F_LOW, help="low cut-off [Hz], <=0 for low-pass")
    p.add_argument("--f-high", type=float, default=F_HIGH, help="high cut-off [Hz], >=Nyquist for high-pass")
    p.add_argument("--order", type=int, default=FILTER_ORDER, help="Butterworth order per direction")
    p.add_argument("--keep-correlation-map", action="store_true",
                   help="copy correlation_map across (~108 MB) instead of dropping it")
    p.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
    a = p.parse_args()
    process_run(a.run, a.f_low, a.f_high, a.order, a.keep_correlation_map, a.overwrite)


if __name__ == "__main__":
    main()
