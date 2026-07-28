"""Auto-generated .py twin of make_thumbmail_summary.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # Thumbnail summary sheets
# 
# Collects the per-run figures scattered across every `<run>/PostProcessing/` folder and tiles them
# into one **contact sheet per figure type**, so a whole sweep can be scanned at a glance.
# 
# For each dataset (`k6_TopBottom`, `k20_bottomOnly`) it builds four sheets:
# 
# | sheet | source figure in each run |
# |---|---|
# | Kinetic energy FFT — FULL | `KineticEnergy_FFT_FULL.png` |
# | Kinetic energy FFT — ROI  | `KineticEnergy_FFT_ROI.png` |
# | Velocity FFT — FULL       | `VelocityFFT_FULL.png` |
# | Velocity FFT — ROI        | `VelocityFFT_ROI.png` |
# 
# Each sheet is written **next to that dataset's summary tables** (the dataset root), named
# `Thumbnails_<figure>.pdf` — a **multi-page PDF with true A4 pages** (210 x 297 mm). As many
# tiles as fit are placed on each page and the run list continues onto the next page; the tile
# size is identical on every page.
# 
# Tiles are ordered physically — by `frot`, then `flib`, then `dphi` (parsed from the run name) — not
# alphabetically, so a resonance sweep reads left-to-right in frequency order. Each tile is captioned
# with its run name.
# 
# Run the settings cell, then *Run All*.


import os
import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


from piv_postprocessing_lib import k0, parse_run_name, topography_arrangement


# ## 1. Settings


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the library),
# so a running batch stays quiet while you edit other files. Re-run this cell to
# toggle. (Figures are unaffected.)
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print
MUTE_PRINT = True
builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------

# Dataset roots. Each holds the run folders AND the summary tables; the sheets
# are written here, beside the summaries.
BASE_DIRS = [
    '/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
    'CylinderExperimentsGMA/k6_TopBottom',
    '/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
    'CylinderExperimentsGMA/k20_bottomOnly',
    '/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/'
    'CylinderExperimentsGMA/k6_bottomOnly'
    ]

# Per-run figure STEM (inside <run>/PostProcessing/) -> title of its contact
# sheet. The suffix ('_normalized') and extension ('.png'/'.pdf') are added from
# the source selection below, so the stem carries neither.
GROUPS = [
    # ('KineticEnergy_FFT_FULL', 'Kinetic energy FFT  --  FULL field'),
    ('KineticEnergy_FFT_ROI',  'Kinetic energy FFT  --  ROI'),
    # ('VelocityFFT_FULL',       'Velocity FFT  --  FULL field'),
    ('VelocityFFT_ROI',        'Velocity FFT  --  ROI'),
]

# --- which per-run figures to tile ------------------------------------
# Each analysis notebook writes every figure twice (a raw and a '_normalized'
# version) in PNG or PDF (its FIG_FORMAT). Pick which set to tile here.
SOURCE_NORMALIZED = True   # False -> raw figures; True -> the '_normalized' ones
SOURCE_EXT        = 'png'   # source figure format to tile: 'png' or 'pdf'
#   PDF sources must be rasterized to be tiled, which needs one of pymupdf,
#   pypdfium2 or pdf2image installed; without it the build reports and stops.

# Run folders whose name contains any of these are ignored (not experiment runs).
EXCLUDE = ['From Jerome']

# Sheet name = OUT_PREFIX + <stem> + ('_normalized' if SOURCE_NORMALIZED) + OUT_EXT
OUT_PREFIX = 'Thumbnails_'
OUT_EXT    = '.pdf'          # multi-page A4 container -- must be '.pdf'

# --- page geometry ----------------------------------------------------
# Real A4 pages (210 x 297 mm). Tiles are laid out NCOLS per row, each
# TILE_FRAC of the page width; as many rows as fit on one page are used and the
# run list is split over as many pages as needed -> ONE MULTI-PAGE PDF.
A4_W_IN = 210.0 / 25.4       # 8.268 in
A4_H_IN = 297.0 / 25.4       # 11.693 in
PAGE_ORIENTATION = 'portrait'    # 'portrait' or 'landscape'
PAGE_W, PAGE_H = ((A4_W_IN, A4_H_IN) if PAGE_ORIENTATION == 'portrait'
                  else (A4_H_IN, A4_W_IN))
TILE_FRAC = 0.4              # tile width as a fraction of the page width
NCOLS     = 2                # tiles per row (2 x 0.4 = 0.8 of the page)
DPI       = 200              # resolution the tile images are rendered at

WSPACE   = 0.12   # gap between columns, as a fraction of the tile width
HSPACE   = 0.38   # gap between rows (leaves room for the per-tile caption)
TITLE_FS = 5      # per-tile caption font size

# Extra detail kept before the final resize. Each tile occupies
# TILE_FRAC*PAGE_W*DPI px (~660 px here), so both the ~2575 px energy figures
# and the ~1180 px velocity ones are genuinely downsampled -- neither is blown
# up, which is what made the velocity sheet look soft before.
OVERSAMPLE = 1.0

SHOW_INLINE = False   # also display each sheet in the notebook (slow for big sheets)
OVERWRITE   = True    # False -> skip a sheet whose file already exists


# ## 2. Functions


def sort_key(run_name):
    """Physical ordering: frot, then flib, then dphi; unparsed names go last."""
    frot, flib, dphi = parse_run_name(run_name)
    inf = float('inf')
    val = lambda v: inf if not np.isfinite(v) else v
    return (val(frot), val(flib), val(dphi), run_name)


def source_name(stem):
    """Per-run figure filename for the current SOURCE_NORMALIZED / SOURCE_EXT.

    e.g. 'VelocityFFT_ROI' -> 'VelocityFFT_ROI.png' (raw) or
    'VelocityFFT_ROI_normalized.pdf' (normalized, pdf).
    """
    suffix = '_normalized' if SOURCE_NORMALIZED else ''
    return '%s%s.%s' % (stem, suffix, SOURCE_EXT.lower().lstrip('.'))


def collect(base_dir, fname):
    """[(run_name, path)] for every <base>/<run>/PostProcessing/<fname>."""
    hits = []
    for run in os.listdir(base_dir):
        if any(x in run for x in EXCLUDE):
            continue
        p = os.path.join(base_dir, run, 'PostProcessing', fname)
        if os.path.isfile(p):
            hits.append((run, p))
    return sorted(hits, key=lambda t: sort_key(t[0]))


def tile_w_in():
    """Width of one tile, in inches (TILE_FRAC of the A4 page width)."""
    return TILE_FRAC * PAGE_W


def tile_px():
    """Width in pixels one tile occupies once rendered at DPI."""
    return int(tile_w_in() * DPI * OVERSAMPLE)


def _pdf_backend():
    """Name of an importable PDF-raster backend, or None."""
    for mod in ('fitz', 'pypdfium2', 'pdf2image'):
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _rasterize_pdf(path, target_px):
    """First page of a PDF as a uint8 RGB array ~target_px wide (None if no backend)."""
    backend = _pdf_backend()
    if backend == 'fitz':
        import fitz
        page = fitz.open(path).load_page(0)
        zoom = max(1.0, target_px / float(page.rect.width))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        return np.ascontiguousarray(arr[..., :3])
    if backend == 'pypdfium2':
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(path)[0]
        scale = max(1.0, target_px / float(page.get_size()[0]))
        return np.asarray(page.render(scale=scale).to_pil().convert('RGB'),
                          dtype=np.uint8)
    if backend == 'pdf2image':
        from pdf2image import convert_from_path
        return np.asarray(convert_from_path(path, dpi=DPI)[0].convert('RGB'),
                          dtype=np.uint8)
    return None


def _read(path, target_px=None):
    """Load a PNG (imread) or a PDF page (rasterized), decimated to tile size.

    The step comes from the image's OWN width against the tile's rendered width,
    so a low-resolution source is never thrown away. Returned as uint8 -- a
    37-tile sheet of float32 RGBA would otherwise hold well over a gigabyte.
    """
    target = tile_px() if target_px is None else int(target_px)
    if os.path.splitext(path)[1].lower() == '.pdf':
        img = _rasterize_pdf(path, target)
        if img is None:
            raise RuntimeError('no PDF raster backend (pymupdf / pypdfium2 / '
                               'pdf2image) to tile %s' % path)
    else:
        img = mpimg.imread(path)
    step = max(1, img.shape[1] // max(1, target))
    img = img[::step, ::step]
    if img.dtype != np.uint8:                       # PNGs load as float32 0..1
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def page_layout(first_path, ncols=NCOLS):
    """Tile size and how many rows/tiles fit on one A4 page.

    Returns (tile_w, cell_h, nrows, per_page, left, top, bottom) in inches /
    figure fractions. The tile keeps the source aspect ratio; if a single tile
    would be taller than the printable area it is scaled down to fit.
    """
    tile_w = tile_w_in()
    h, w = _read(first_path).shape[:2]
    cell_h = tile_w * h / float(w)

    header_in, bottom_in = 0.85, 0.30
    avail = PAGE_H - header_in - bottom_in
    if cell_h > avail:                      # one tile taller than the page
        tile_w = tile_w * avail / cell_h
        cell_h = avail

    # rows that fit:  cell_h * (n + (n-1)*HSPACE) <= avail
    nrows = int(np.floor((avail / cell_h + HSPACE) / (1.0 + HSPACE)))
    nrows = max(1, nrows)
    per_page = ncols * nrows

    content_w = ncols * tile_w + (ncols - 1) * WSPACE * tile_w
    left = (PAGE_W - content_w) / 2.0 / PAGE_W
    block_h = cell_h * (nrows + (nrows - 1) * HSPACE)
    top = 1.0 - header_in / PAGE_H
    bottom = top - block_h / PAGE_H
    return tile_w, cell_h, nrows, per_page, left, top, bottom, header_in


def contact_sheet(items, title, out_path, ncols=NCOLS):
    """Tile `items` onto A4 pages and write them as ONE multi-page PDF.

    items is [(caption, image_path)]. Every page is exactly A4 (PAGE_W x PAGE_H);
    each tile is TILE_FRAC*PAGE_W wide with its height set by the source aspect
    ratio, so figures are never distorted and the tile size is identical on every
    page. Returns (figs, npages, nrows, per_page).
    """
    n = len(items)
    (tile_w, cell_h, nrows, per_page,
     left, top, bottom, header_in) = page_layout(items[0][1], ncols)
    npages = int(np.ceil(n / float(per_page)))
    _tt, _bt = topography_arrangement(title)

    figs = []
    with PdfPages(out_path) as pdf:
        for pg in range(npages):
            chunk = items[pg * per_page:(pg + 1) * per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_W, PAGE_H))
            axes = np.atleast_1d(axes).ravel()
            fig.subplots_adjust(left=left, right=1.0 - left, top=top,
                                bottom=bottom, wspace=WSPACE, hspace=HSPACE)
            for ax, (caption, path) in zip(axes, chunk):
                ax.imshow(_read(path), interpolation='antialiased')
                ax.set_title(caption, fontsize=TITLE_FS)
                ax.axis('off')
            for ax in axes[len(chunk):]:        # blank out the unused cells
                ax.axis('off')

            # Two lines, sized to fit the A4 width (a single line overflows).
            fig.suptitle('%s\n(%d runs)   ($k_0 = %g$, top=%s, bottom=%s)'
                         % (title, n, k0, _tt, _bt),
                         fontsize=9, fontweight='bold',
                         y=1.0 - 0.22 * header_in / PAGE_H)
            fig.text(0.5, 0.012, 'page %d / %d' % (pg + 1, npages),
                     ha='center', va='bottom', fontsize=7)
            # No bbox_inches='tight' -- that would crop the exact A4 page size.
            pdf.savefig(fig, dpi=DPI)
            figs.append(fig)
    return figs, npages, nrows, per_page


# ## 3. Build the sheets
# 
# One sheet per (dataset x figure type). A group with no matching figures is reported and skipped.


written, skipped = [], []

_suffix  = '_normalized' if SOURCE_NORMALIZED else ''
_ext     = SOURCE_EXT.lower().lstrip('.')
_variant = 'normalized' if SOURCE_NORMALIZED else 'raw'
print('source: %s figures, %s  |  A4 %s page %.2f x %.2f in, tile %.0f px @ %d dpi, %d/row'
      % (_ext.upper(), _variant, PAGE_ORIENTATION, PAGE_W, PAGE_H, tile_px(),
         DPI, NCOLS))
if OUT_EXT.lower() != '.pdf':
    print('[warn] multi-page output needs OUT_EXT = ".pdf" (got %r)' % OUT_EXT)

# PDF sources must be rasterized to be tiled; stop early with a clear message if
# that is requested but no backend is installed.
if _ext == 'pdf' and _pdf_backend() is None:
    print('\n[abort] SOURCE_EXT = "pdf" but no PDF raster backend is installed '
          '(pymupdf, pypdfium2 or pdf2image).\n'
          '        Set SOURCE_EXT = "png", or install one of those packages.')
else:
    for base_dir in BASE_DIRS:
        label = os.path.basename(os.path.normpath(base_dir))
        if not os.path.isdir(base_dir):
            print('[skip] %s : not a directory' % label)
            continue
        print('=' * 70)
        print(label)

        for stem, title in GROUPS:
            fname = source_name(stem)
            items = collect(base_dir, fname)
            out_path = os.path.join(base_dir, OUT_PREFIX + stem + _suffix + OUT_EXT)

            if not items:
                print('  [skip] %-30s no "%s" found in any run' % (stem, fname))
                skipped.append((label, stem, 'none: ' + fname))
                continue
            if os.path.isfile(out_path) and not OVERWRITE:
                print('  [skip] %-30s sheet exists (OVERWRITE=False)' % stem)
                skipped.append((label, stem, 'exists'))
                continue

            ttl = '%s  --  %s%s' % (label, title,
                                    '  (normalized)' if SOURCE_NORMALIZED else '')
            figs, npages, nrows, per_page = contact_sheet(items, ttl, out_path)
            src_w = _read(items[0][1]).shape[1]
            print('  [ok]   %-30s %2d tiles | %d page(s) A4 (%dx%d/page) | '
                  'source %4d px -> %3.0f px tile | %s'
                  % (stem, len(items), npages, nrows, NCOLS, src_w, tile_px(),
                     os.path.basename(out_path)))
            written.append(out_path)

            for _f in figs:
                if SHOW_INLINE:
                    plt.show()
                else:
                    plt.close(_f)

print('\n%d sheet(s) written, %d skipped' % (len(written), len(skipped)))
for p in written:
    print('   %.1f MB  %s' % (os.path.getsize(p) / 1e6, p))
if skipped:
    print('skipped:')
    for lab, st, why in skipped:
        print('   %-14s %-30s %s' % (lab, st, why))


# ## 4. Preview
# 
# Re-displays the sheets that were just written, at screen resolution.


# The sheets are multi-page A4 PDFs, so they cannot be read back with imread.
# Set SHOW_INLINE = True above to display every page as it is built, or open the
# files below (each page is exactly A4).
for p in written:
    print('%6.1f MB   %s' % (os.path.getsize(p) / 1e6, p))
