"""Auto-generated .py twin of RES_CURVES.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks



# # RES_CURVES -- resonance curves of energy and velocity
# 
# Builds the resonance curves of one or several datasets from the stored
# per-run spectra (`Velocity.npz` / `KineticEnergy.npz`); **nothing is
# recomputed from the `.mat`**. The user provides the list `BASE_DIRS`.
# 
# Per dataset (figures written INTO each `BASE_DIR`), each with a physical
# LEFT panel and a non-dimensional RIGHT panel (amplitudes / `*_SCALE`,
# frequencies / `f_rot`):
# 
# | figure | y | from |
# |---|---|---|
# | `RES_CURVE_ENERGY` | amplitude of `FFT_EK_ROIaveraged` at `2 f_lib` | `KineticEnergy.npz` |
# | `RES_CURVE_VELOCITY` | amplitude of `FFT_TOTAL_ROIaveraged` at `f_lib` | `Velocity.npz` |
# 
# versus the libration frequency, one point per run. Repeated runs (same
# `frot`, `flib`, `dphi`, different `SSn`) carry **different symbols**; when a
# dataset holds several `dphi` the series are split and labelled.
# 
# With more than one `BASE_DIR`, two overlay figures
# (`RES_CURVE_ENERGY_ALL`, `RES_CURVE_VELOCITY_ALL`) are written into the
# datasets' common parent directory: one curve per dataset, repeated runs
# collapsed to their **mean with a std error bar**.


import glob
import os

import numpy as np
import matplotlib.pyplot as plt


# --- Mute switch -----------------------------------------------------------
# MUTE_PRINT = True silences ALL print() output (this notebook AND the
# library). Off by default: for a viewer the feedback is the point.
import builtins
if not hasattr(builtins, "_piv_real_print"):
    builtins._piv_real_print = builtins.print

MUTE_PRINT = False

builtins.print = (lambda *a, **k: None) if MUTE_PRINT else builtins._piv_real_print


# ## Configuration


# --- edit me --------------------------------------------------------------- #
ROOT_DIR = ("/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/"
            "TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA")
# The datasets to build resonance curves for (each holds the run sub-folders).
BASE_DIRS = [os.path.join(ROOT_DIR, _d) for _d in (
    "FullCylinder", "k20_bottomOnly", "k20_topBottom",
    "k6_TopBottom", "k6_TopBottom_notAligned", "k6_bottomOnly")]

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]   # symbol per run idx (SS1, SS2, ...)
FIG_FORMAT = "png"      # figure format: png or pdf
SAVE = True             # write the figures (into BASE_DIR / the common parent)


# ## Collect one record per run
# 
# Everything comes calibrated from the `.npz` files: the velocity amplitude is
# `FFT_TOTAL_ROIaveraged` interpolated at `f_lib` (x `UCAL`), the energy
# amplitude is `FFT_EK_ROIaveraged` interpolated at `2 f_lib` (x `ECAL`); the
# `*_SCALE` divisors travel along for the non-dimensional panels.


def collect_runs(base_dir):
    """One record per run of `base_dir`: forcing parameters, run idx and the
    two resonance amplitudes (physical), plus their non-dim scales."""
    recs = []
    for d in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(d) or os.path.basename(d).startswith("."):
            continue
        post = os.path.join(d, "PostProcessing")
        rec = {}
        for fname, kind in (("Velocity.npz", "vel"),
                            ("KineticEnergy.npz", "ke")):
            path = os.path.join(post, fname)
            if not os.path.isfile(path):
                continue
            dat = np.load(path, allow_pickle=True)
            _s = lambda k: np.asarray(dat[k]).item()
            if "run" not in rec:
                rec.update(run=str(_s("run")), frot=float(_s("frot_Hz")),
                           flib=float(_s("flib_Hz")),
                           dphi=float(_s("dphi_deg")),
                           idx=int(_s("run_idx")),
                           calibrated=bool(_s("calibrated")))
            f_hz = np.asarray(dat["FREQ"], float) * _s("FCAL")
            if kind == "vel":
                a = np.asarray(dat["FFT_TOTAL_ROIaveraged"], float) * _s("UCAL")
                rec["amp_vel"] = float(np.interp(rec["flib"], f_hz, a))
                rec["U_SCALE"] = float(_s("U_SCALE"))
            else:
                a = np.asarray(dat["FFT_EK_ROIaveraged"], float) * _s("ECAL")
                rec["amp_ke"] = float(np.interp(2.0 * rec["flib"], f_hz, a))
                rec["EK_SCALE"] = float(_s("EK_SCALE"))
        if "run" in rec and np.isfinite(rec["flib"]) and rec["flib"]:
            if not rec["calibrated"]:
                print("  [warn] %s is uncalibrated -- native units" % rec["run"])
            recs.append(rec)
    return recs


# What each of the two curves plots.
CURVES = {
    "energy": dict(
        amp="amp_ke", scale="EK_SCALE", stem="RES_CURVE_ENERGY",
        what=(r"$\langle|\widehat{E_k}|\rangle_{\mathrm{ROI}}$"
              r" at $2f_{\mathrm{lib}}$"),
        ylab=u"amplitude at $2f_{\mathrm{lib}}$  (m\u00b2/s\u00b2)",
        ylab_star=r"amplitude$^{\,*}$ at $2f^*_{\mathrm{lib}}$"),
    "velocity": dict(
        amp="amp_vel", scale="U_SCALE", stem="RES_CURVE_VELOCITY",
        what=(r"$\langle|\widehat{U}|+|\widehat{V}|\rangle_{\mathrm{ROI}}$"
              r" at $f_{\mathrm{lib}}$"),
        ylab=r"amplitude at $f_{\mathrm{lib}}$  (m/s)",
        ylab_star=r"amplitude$^{\,*}$ at $f^*_{\mathrm{lib}}$"),
}


# ## The figures


def _axes_pair(title, spec):
    fig, (axd, axn) = plt.subplots(1, 2, figsize=(12, 5))
    axd.set_xlabel(r"$f_{\mathrm{lib}}$ (Hz)", fontsize=12)
    axn.set_xlabel(r"$f^*_{\mathrm{lib}}$", fontsize=12)
    axd.set_ylabel(spec["ylab"], fontsize=12)
    axn.set_ylabel(spec["ylab_star"], fontsize=12)
    axd.set_title("physical", fontsize=10)
    axn.set_title("non-dimensional", fontsize=10)
    for ax in (axd, axn):
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    return fig, axd, axn


def _finish(fig, axd, out_path):
    """Save the figure and close it -- the curves are never shown."""
    axd.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if SAVE:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print("Figure written to:\n  %s" % out_path)
    plt.close(fig)


def dataset_curve(recs, spec, title, out_path):
    """Figures 1 and 2: one point per run, symbol per run idx (repeated runs
    stand out); a separate series per (dphi, idx) when dphi varies."""
    recs = [r for r in recs if spec["amp"] in r]
    if not recs:
        print("  [skip] nothing stored for %s" % os.path.basename(out_path))
        return
    fig, axd, axn = _axes_pair(title, spec)
    dphis = sorted({r["dphi"] for r in recs})
    for dphi in dphis:
        for i in sorted({r["idx"] for r in recs}):
            sub = sorted((r for r in recs
                          if r["dphi"] == dphi and r["idx"] == i),
                         key=lambda r: r["flib"])
            if not sub:
                continue
            mk = MARKERS[(max(i, 1) - 1) % len(MARKERS)]
            lab = ("SS%d" % i) + (r", $\Delta\phi$=%g$^\circ$" % dphi
                                  if len(dphis) > 1 else "")
            ln, = axd.plot([r["flib"] for r in sub],
                           [r[spec["amp"]] for r in sub],
                           ls="-", marker=mk, ms=6, label=lab)
            axn.plot([r["flib"] / r["frot"] for r in sub],
                     [r[spec["amp"]] / r[spec["scale"]] for r in sub],
                     ls="-", marker=mk, ms=6, color=ln.get_color())
    _finish(fig, axd, out_path)


def overlay_curve(recs_by_ds, spec, title, out_path):
    """Figures 3 and 4: one curve per dataset, repeated runs collapsed to
    their mean with a std error bar."""
    fig, axd, axn = _axes_pair(title, spec)
    drew = False
    for ds, recs in recs_by_ds.items():
        recs = [r for r in recs if spec["amp"] in r]
        dphis = sorted({r["dphi"] for r in recs})
        for dphi in dphis:
            groups = {}
            for r in recs:
                if r["dphi"] != dphi:
                    continue
                groups.setdefault((round(r["frot"], 6),
                                   round(r["flib"], 6)), []).append(r)
            if not groups:
                continue
            pts = []
            for (frot_g, flib_g), g in sorted(groups.items(),
                                              key=lambda kv: kv[0][1]):
                a = np.array([r[spec["amp"]] for r in g], float)
                s = float(np.mean([r[spec["scale"]] for r in g]))
                pts.append((flib_g, frot_g, a.mean(), a.std(), s))
            lab = ds + (r", $\Delta\phi$=%g$^\circ$" % dphi
                        if len(dphis) > 1 else "")
            eb = axd.errorbar([p[0] for p in pts], [p[2] for p in pts],
                              yerr=[p[3] for p in pts], marker="o", ms=4,
                              capsize=3, lw=1.2, label=lab)
            axn.errorbar([p[0] / p[1] for p in pts],
                         [p[2] / p[4] for p in pts],
                         yerr=[p[3] / p[4] for p in pts], marker="o", ms=4,
                         capsize=3, lw=1.2, color=eb.lines[0].get_color())
            drew = True
    if not drew:
        plt.close(fig)
        print("  [skip] nothing stored for %s" % os.path.basename(out_path))
        return
    _finish(fig, axd, out_path)


# --- run: per-dataset curves, then the overlays ----------------------------- #
all_recs = {}
for _base in BASE_DIRS:
    _base = _base.rstrip("/")
    _name = os.path.basename(_base)
    _recs = collect_runs(_base)
    print("dataset %s: %d runs with spectra" % (_name, len(_recs)))
    if not _recs:
        continue
    all_recs[_name] = _recs
    for _kind in ("energy", "velocity"):
        _spec = CURVES[_kind]
        dataset_curve(_recs, _spec,
                      "%s -- %s resonance curve: %s"
                      % (_name, _kind, _spec["what"]),
                      os.path.join(_base, "%s.%s" % (_spec["stem"],
                                                     FIG_FORMAT)))

if len(all_recs) > 1:
    _parent = os.path.commonpath(
        [os.path.dirname(os.path.abspath(b.rstrip("/"))) for b in BASE_DIRS])
    for _kind in ("energy", "velocity"):
        _spec = CURVES[_kind]
        overlay_curve(all_recs, _spec,
                      "%s resonance curves -- mean +/- std over repeated runs"
                      % _kind,
                      os.path.join(_parent, "%s_ALL.%s" % (_spec["stem"],
                                                           FIG_FORMAT)))
