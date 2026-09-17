"""Auto-generated .py twin of MAKE_thumbmails.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # MAKE_thumbmails -- contact sheets of the PLOT_FFT figures
# 
# Collects the per-run `PLOT_FFT_velocity_DIM` and `PLOT_FFT_energy_DIM`
# figures scattered across every `<run>/PostProcessing/` folder and tiles them
# into one **contact sheet per figure type and dataset**, so a whole sweep can
# be scanned at a glance (successor of the obsolete `make_thumbmail_summary`).
# 
# Each sheet is a **multi-page A4 PDF** written at the dataset root, named
# `Thumbnails_<stem>.pdf`. Tiles are laid out `NCOLS` per row, each `TILE_FRAC`
# of the page width, ordered **physically** (by `frot`, then `flib`, then
# `dphi` parsed from the run name) and captioned with the run name.


import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

from piv_postprocessing_lib import (parse_run_name, read_paramPostprocessing,
                        topography_arrangement)


# --- Mute switch -----------------------------------------------------------
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print

MUTE_PRINT = False

builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print


# ## Configuration


# --- edit me --------------------------------------------------------------- #
ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets to build sheets for (each holds the run sub-folders).
# BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
#     "FullCylinder", "k20_bottomOnly", "k20_topBottom",
#     "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "k20_topBottom_centerTight_spacer64mm",)]

# Per-run figure STEM (inside <run>/PostProcessing/) -> title of its sheet.
GROUPS = [
    ("PLOT_FFT_velocity_DIM", "Velocity FFT  --  ROI averaged"),
    ("PLOT_FFT_energy_DIM",   "Energy FFT  --  ROI averaged"),
    ("MAPS_FFT_DIM",          "Spatial FFT maps"),
    ("MAPS_VELOCITY_DIM",     "Mean / std velocity maps"),
    ("PHASE_AVERAGE_DIM",     "Phase-averaged field  --  phase 0"),
]
SOURCE_EXT = "png"       # source figure format to tile: 'png' or 'pdf' (pdf
                         # needs pymupdf / pypdfium2 / pdf2image installed)

# Run folders whose name contains any of these are ignored.
EXCLUDE = ["From Jerome", "obsolete"]

# Sheet name = OUT_PREFIX + <stem> + OUT_EXT, at the dataset root.
OUT_PREFIX = "Thumbnails_"
OUT_EXT = ".pdf"         # multi-page A4 container -- must be '.pdf'

# --- page geometry ----------------------------------------------------------
A4_W_IN = 210.0 / 25.4
A4_H_IN = 297.0 / 25.4
PAGE_ORIENTATION = "portrait"    # 'portrait' or 'landscape'
PAGE_W, PAGE_H = ((A4_W_IN, A4_H_IN) if PAGE_ORIENTATION == "portrait"
                  else (A4_H_IN, A4_W_IN))
TILE_FRAC = 0.4          # tile width as a fraction of the page width
NCOLS = 2                # tiles per row
DPI = 200                # resolution the tile images are rendered at
WSPACE = 0.12            # gap between columns (fraction of the tile width)
HSPACE = 0.38            # gap between rows (leaves room for the caption)
TITLE_FS = 5             # per-tile caption font size
OVERSAMPLE = 1.0         # extra detail kept before the final resize

SHOW_INLINE = False      # also display each sheet in the notebook (slow)
OVERWRITE = True         # False -> keep an existing sheet


# ## Functions (layout machinery from the retired make_thumbmail_summary)


def sort_key(run_name):
    """Physical ordering: frot, then flib, then dphi; unparsed names go last."""
    frot, flib, dphi = parse_run_name(run_name)
    inf = float("inf")
    val = lambda v: inf if not np.isfinite(v) else v
    return (val(frot), val(flib), val(dphi), run_name)


def collect(base_dir, fname):
    """[(run_name, path)] for every <base>/<run>/PostProcessing/<fname>."""
    hits = []
    for run in os.listdir(base_dir):
        if any(x in run for x in EXCLUDE):
            continue
        p = os.path.join(base_dir, run, "PostProcessing", fname)
        if os.path.isfile(p):
            hits.append((run, p))
    return sorted(hits, key=lambda t: sort_key(t[0]))


def tile_w_in():
    return TILE_FRAC * PAGE_W


def tile_px():
    return int(tile_w_in() * DPI * OVERSAMPLE)


def _pdf_backend():
    for mod in ("fitz", "pypdfium2", "pdf2image"):
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _rasterize_pdf(path, target_px):
    """First page of a PDF as a uint8 RGB array (None if no backend)."""
    backend = _pdf_backend()
    if backend == "fitz":
        import fitz
        page = fitz.open(path).load_page(0)
        zoom = max(1.0, target_px / float(page.rect.width))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        return np.ascontiguousarray(arr[..., :3])
    if backend == "pypdfium2":
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(path)[0]
        scale = max(1.0, target_px / float(page.get_size()[0]))
        return np.asarray(page.render(scale=scale).to_pil().convert("RGB"),
                          dtype=np.uint8)
    if backend == "pdf2image":
        from pdf2image import convert_from_path
        return np.asarray(convert_from_path(path, dpi=DPI)[0].convert("RGB"),
                          dtype=np.uint8)
    return None


def _read(path, target_px=None):
    """Load a PNG (imread) or a PDF page (rasterized), decimated to tile size."""
    target = tile_px() if target_px is None else int(target_px)
    if os.path.splitext(path)[1].lower() == ".pdf":
        img = _rasterize_pdf(path, target)
        if img is None:
            raise RuntimeError("no PDF raster backend (pymupdf / pypdfium2 / "
                               "pdf2image) to tile %s" % path)
    else:
        img = mpimg.imread(path)
    step = max(1, img.shape[1] // max(1, target))
    img = img[::step, ::step]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def page_layout(first_path, ncols=NCOLS):
    """Tile size and how many rows/tiles fit on one A4 page."""
    tile_w = tile_w_in()
    h, w = _read(first_path).shape[:2]
    cell_h = tile_w * h / float(w)

    header_in, bottom_in = 0.85, 0.30
    avail = PAGE_H - header_in - bottom_in
    if cell_h > avail:
        tile_w = tile_w * avail / cell_h
        cell_h = avail

    nrows = int(np.floor((avail / cell_h + HSPACE) / (1.0 + HSPACE)))
    nrows = max(1, nrows)
    per_page = ncols * nrows

    content_w = ncols * tile_w + (ncols - 1) * WSPACE * tile_w
    left = (PAGE_W - content_w) / 2.0 / PAGE_W
    block_h = cell_h * (nrows + (nrows - 1) * HSPACE)
    top = 1.0 - header_in / PAGE_H
    bottom = top - block_h / PAGE_H
    return tile_w, cell_h, nrows, per_page, left, top, bottom, header_in


def contact_sheet(items, title, anno, out_path, ncols=NCOLS):
    """Tile `items` [(caption, path)] onto A4 pages -> ONE multi-page PDF."""
    n = len(items)
    (tile_w, cell_h, nrows, per_page,
     left, top, bottom, header_in) = page_layout(items[0][1], ncols)
    npages = int(np.ceil(n / float(per_page)))

    figs = []
    with PdfPages(out_path) as pdf:
        for pg in range(npages):
            chunk = items[pg * per_page:(pg + 1) * per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_W, PAGE_H))
            axes = np.atleast_1d(axes).ravel()
            fig.subplots_adjust(left=left, right=1.0 - left, top=top,
                                bottom=bottom, wspace=WSPACE, hspace=HSPACE)
            for ax, (caption, path) in zip(axes, chunk):
                ax.imshow(_read(path), interpolation="antialiased")
                ax.set_title(caption, fontsize=TITLE_FS)
                ax.axis("off")
            for ax in axes[len(chunk):]:
                ax.axis("off")

            fig.suptitle("%s\n(%d runs)   %s" % (title, n, anno),
                         fontsize=9, fontweight="bold",
                         y=1.0 - 0.22 * header_in / PAGE_H)
            fig.text(0.5, 0.012, "page %d / %d" % (pg + 1, npages),
                     ha="center", va="bottom", fontsize=7)
            pdf.savefig(fig, dpi=DPI)   # no bbox_inches: keep the exact A4 page
            figs.append(fig)
    return figs, npages, nrows, per_page


# ## Build the sheets
# 
# One sheet per (dataset x figure type); a group with no matching figures is
# reported and skipped.


written, skipped = [], []
_ext = SOURCE_EXT.lower().lstrip(".")
print("source: %s figures  |  A4 %s page %.2f x %.2f in, tile %.0f px @ %d "
      "dpi, %d/row" % (_ext.upper(), PAGE_ORIENTATION, PAGE_W, PAGE_H,
                       tile_px(), DPI, NCOLS))
if OUT_EXT.lower() != ".pdf":
    print("[warn] multi-page output needs OUT_EXT = \".pdf\" (got %r)" % OUT_EXT)

if _ext == "pdf" and _pdf_backend() is None:
    print("\n[abort] SOURCE_EXT = 'pdf' but no PDF raster backend is installed "
          "(pymupdf, pypdfium2 or pdf2image).")
else:
    for base_dir in BASE_DIRS:
        label = os.path.basename(os.path.normpath(base_dir))
        if not os.path.isdir(base_dir):
            print("[skip] %s : not a directory" % label)
            continue
        print("=" * 70)
        print(label)
        # Dataset annotation from ITS OWN parameter file / folder name.
        _k0 = read_paramPostprocessing(base_dir).k0
        _tt, _bt = topography_arrangement(base_dir)
        anno = "($k_0 = %g$, top=%s, bottom=%s)" % (_k0, _tt, _bt)

        for stem, title in GROUPS:
            fname = "%s.%s" % (stem, _ext)
            items = collect(base_dir, fname)
            out_path = os.path.join(base_dir, OUT_PREFIX + stem + OUT_EXT)

            if not items:
                print("  [skip] %-26s no \"%s\" found in any run" % (stem, fname))
                skipped.append((label, stem, "none: " + fname))
                continue
            if os.path.isfile(out_path) and not OVERWRITE:
                print("  [skip] %-26s sheet exists (OVERWRITE=False)" % stem)
                skipped.append((label, stem, "exists"))
                continue

            ttl = "%s  --  %s" % (label, title)
            figs, npages, nrows, per_page = contact_sheet(items, ttl, anno,
                                                          out_path)
            src_w = _read(items[0][1]).shape[1]
            print("  [ok]   %-26s %2d tiles | %d page(s) A4 (%dx%d/page) | "
                  "source %4d px -> %3.0f px tile | %s"
                  % (stem, len(items), npages, nrows, NCOLS, src_w, tile_px(),
                     os.path.basename(out_path)))
            written.append(out_path)
            for _f in figs:
                if SHOW_INLINE:
                    plt.show()
                else:
                    plt.close(_f)

print("\n%d sheet(s) written, %d skipped" % (len(written), len(skipped)))
for p in written:
    print("   %.1f MB  %s" % (os.path.getsize(p) / 1e6, p))
if skipped:
    print("skipped:")
    for lab, st, why in skipped:
        print("   %-14s %-26s %s" % (lab, st, why))
