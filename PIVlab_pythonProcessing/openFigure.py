#!/usr/bin/env python
"""openFigure.py -- reopen pickled matplotlib figures, fully interactive.

Opens a file-picker dialog to select one or several pickled figures
(`*.fig.pickle`, as written by PHASE_AVERAGE with SAVE_PICKLE = True) and
shows each one in its own Qt window with the full toolbar: zoom, pan and
the live x/y readout under the cursor.

Usage:
    python openFigure.py                # pick the file(s) in a dialog
    python openFigure.py fig1 [fig2 ..] # or pass the paths directly

Only open pickles you created yourself (unpickling executes code), and with
the same matplotlib generation that wrote them.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets

# Where the file picker starts: the data root (falls back to the home dir).
START_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
             "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")


def pick_files():
    """Qt file-picker dialog; returns the selected paths (possibly many)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    start = START_DIR if os.path.isdir(START_DIR) else os.path.expanduser("~")
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        None, "Open pickled figure(s)", start,
        "Pickled figures (*.fig.pickle *.pickle *.pkl);;All files (*)")
    return list(paths)


def open_figure(path):
    """Load one pickled figure and show it in its own interactive window."""
    with open(path, "rb") as fh:
        fig = pickle.load(fh)
    try:
        fig.canvas.manager.set_window_title(os.path.basename(path))
    except Exception:
        pass                                   # cosmetic only
    fig.show()
    print("opened %s" % path)
    return fig


def main(argv):
    paths = argv[1:] or pick_files()
    if not paths:
        print("nothing selected")
        return
    figs = []
    for p in paths:
        try:
            figs.append(open_figure(p))
        except Exception as exc:
            print("  [error] %s: %s" % (os.path.basename(p), exc))
    if figs:
        plt.show()                             # keep the windows alive


if __name__ == "__main__":
    main(sys.argv)
