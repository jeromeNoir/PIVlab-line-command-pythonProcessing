#!/usr/bin/env python
"""Generate a runnable .py twin of each notebook in this folder.

The twins SAVE figures but never show them: the matplotlib Agg backend is forced
at the top, so `savefig` works, no windows pop up, and `plt.show()` is a harmless
no-op. Jupyter line magics (`%matplotlib ...`) are stripped; markdown cells become
comments. Everything else is copied verbatim, so the .py runs the same pipeline
as the notebook and writes the same outputs.

Run it from this folder to (re)generate the twins after editing the notebooks:

    python ipynb_to_py.py            # all notebooks
    python ipynb_to_py.py foo.ipynb  # just one
"""
import glob
import io
import json
import os
import re
import sys

MAGIC = re.compile(r"\s*%[A-Za-z]")   # line magics (%matplotlib, ...); NOT '% (' format
SHELL = re.compile(r"\s*![A-Za-z]")   # !shell commands

PREAMBLE = '''"""Auto-generated .py twin of {name}.ipynb -- do not edit by hand.

Figures are SAVED, not shown. Regenerate with `python ipynb_to_py.py` after
editing the notebook.
"""
import matplotlib
matplotlib.use("Agg")   # non-interactive: savefig works, nothing pops up or blocks
'''


def convert(nbpath):
    nb = json.load(io.open(nbpath, encoding="utf-8"))
    name = os.path.splitext(os.path.basename(nbpath))[0]
    chunks = [PREAMBLE.format(name=name)]
    for c in nb["cells"]:
        src = "".join(c["source"]).rstrip("\n")
        if not src.strip():
            continue
        if c["cell_type"] == "markdown":
            chunks.append("# " + "\n# ".join(src.splitlines()))
        elif c["cell_type"] == "code":
            kept = [ln for ln in src.splitlines()
                    if not (MAGIC.match(ln) or SHELL.match(ln))]
            code = "\n".join(kept).strip("\n")
            if code:
                chunks.append(code)
    pypath = os.path.splitext(nbpath)[0] + ".py"
    with io.open(pypath, "w", encoding="utf-8") as fh:
        fh.write("\n\n\n".join(chunks).rstrip("\n") + "\n")
    return pypath


def main(argv):
    targets = argv[1:] or sorted(glob.glob("*.ipynb"))
    for nbpath in targets:
        if not nbpath.endswith(".ipynb"):
            print("[skip] %s (not a notebook)" % nbpath)
            continue
        print("wrote", convert(nbpath))


if __name__ == "__main__":
    main(sys.argv)
