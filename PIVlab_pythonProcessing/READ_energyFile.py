#!/usr/bin/env python
"""Read a run's KineticEnergy.npz and return every stored variable under its
in-file name (see FILE_STRUCTURE_ENERGY.md for what each one is).

As a module:

    from READ_energyFile import read_energy_file
    d = read_energy_file("<run>/PostProcessing/KineticEnergy.npz")
    d["EK_ROIaveraged"], d["ECAL"], d["TIME"], ...

    # or unpack straight into the workspace (notebook style):
    globals().update(d)          # -> TIME, FREQ, EK_ROIaveraged, ... as plain variables

From the command line, prints the list of variables (name, shape, dtype,
value for scalars):

    python READ_energyFile.py <run>/PostProcessing/KineticEnergy.npz
"""
import sys

import numpy as np


def read_energy_file(path):
    """Return {name: value} for every variable stored in a KineticEnergy.npz.

    Arrays are returned as numpy arrays, scalars (0-d entries: run, calibrated,
    PIV_FPS, *CAL, *_SCALE, ...) unwrapped to plain Python/numpy scalars.
    Everything stays in native PIV units — apply the *CAL factors yourself.
    """
    data = {}
    with np.load(path, allow_pickle=False) as npz:
        for name in npz.files:
            value = npz[name]
            data[name] = value[()] if value.ndim == 0 else value
    return data


def print_contents(data):
    """One line per variable: name, shape/dtype for arrays, value for scalars."""
    for name in sorted(data):
        value = data[name]
        if isinstance(value, np.ndarray):
            print("  %-24s %-18s %s" % (name, value.shape, value.dtype))
        else:
            print("  %-24s = %s" % (name, value))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python READ_energyFile.py <path/to/KineticEnergy.npz>")
    data = read_energy_file(sys.argv[1])
    print("%s — %d variables" % (sys.argv[1], len(data)))
    print_contents(data)
