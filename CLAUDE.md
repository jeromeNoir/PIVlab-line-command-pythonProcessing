# Project: CylinderExperimentsGMA — PIV processing

## Overview
This project analyses PIV (Particle Image Velocimetry) images of cylinder
libration/topography experiments using a command-line build of PIVlab.

## Key locations
- **PIVlab command-line code:**
  `/Users/jeromenoir/polybox/CODES/PIV/PIVLab/PIVlab-line-command-pythonProcessing/PIVLAB_commandLine/`
- **Project working dir (results land here):**
  `/Users/jeromenoir/Documents/MyDocuments/TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA`
- **Image source root (example):** `/Volumes/Archives/TOPOLIB_TopBottom/k6_TopBottom`

## Scripts
- **`PIVlab_process_commandline.m`** — original. Processes the images in a
  single `image_folder` (one hardcoded `run_folder` inside `project_root`).
  **Do not modify this file.**
- **`PIV_batchprocessing_commandline.m`** — batch version we created. Loops
  over **every sub-folder** of `project_root` and runs the full PIVlab workflow
  (load pairs → preprocess → PIV → postprocess) on the images inside each.
  Results are written to `local_folder/<run_folder>/`
  (`PIVlab_results_uncalibrated.mat` + first-frame figure).

### Differences in the batch script vs. the original
- Auto-discovers sub-folders of `project_root` (skips `.`, `..`, hidden dirs)
  instead of a hardcoded `run_folder`.
- PIV / preprocessing / postprocessing settings defined once, applied to all folders.
- Per-folder result variables reset each iteration so runs don't mix.
- Skips (with a warning) folders that have no images or an odd image count,
  rather than aborting the whole batch with an `error`.
- Figures saved invisibly (`'Visible','off'` + `close`) to avoid many windows.
- `save` uses an explicit variable list (results + settings used).

## Assumptions / open points
- Images are assumed to sit **directly** inside each sub-folder (one level deep),
  matching how the original treats `image_folder`. Nested layouts would need
  recursive traversal.
- Images are processed as pairs (1+2, 3+4, ...), so each folder needs an even
  image count.
- To point the batch at a different dataset, edit `project_root`, `local_folder`,
  and `file_pattern` at the top of `PIV_batchprocessing_commandline.m`.
