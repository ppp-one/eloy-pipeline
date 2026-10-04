# Photometry Pipeline

Reduce a night of FITS observations into a differential light curve, then turn
that into a self-contained interactive HTML report. Two scripts do the work:
`pipeline.py` (the reduction) and `night_report.py` (the report).

![report screenshot](screenshot.jpg)

## Quick start

```bash
uv sync                                  # install deps (needs Python 3.11.8)

uv run pipeline.py \
  --image_path /path/to/night/fits \
  --target "Sp1746-3214" \
  --query-string "Gaia DR3 4055579464064558592" \
  --output-dir results/ \
  --report                               # also build the HTML report
```

Point `--image_path` at a directory holding all of the night's frames: lights,
darks, flats, and biases. The right calibration frames are matched to the target
automatically by exposure time and filter, so you don't sort them yourself. Add
`--report` to get the interactive report next to the data, or drop it to run the
reduction alone. `ffmpeg` needs to be on your `PATH` for the report's movie.

If the target was observed in more than one filter the pipeline runs once per
filter automatically, producing a separate output folder for each.

## pipeline.py

What it does, per run:

1. Indexes the FITS files by observing night (noon-to-noon, so a run past
   midnight stays on one night), frame type, and target.
2. Builds master bias, dark, and flat frames; estimates read noise and dark
   current.
3. Takes the middle light frame as a reference, detects its stars, solves a WCS
   against Gaia (2MASS for infrared filters), and locates the target via MAST.
4. Processes every light frame in parallel: calibrate, align to the reference,
   refine centroids with the Ballet CNN, then aperture photometry over 40 radii
   (0.5 to 5x FWHM), co-adding a stack as it goes.
5. Runs differential photometry and picks the best aperture from the comparison
   stars' noise.

It saves the two `.npz` bundles below plus two diagnostic PDFs: the light curve,
and a systematics panel (FWHM, sky, centroid drift, airmass).

### Options

| Flag | Default | Description |
|---|---|---|
| `--image_path` | required | Directory of FITS files for the night. |
| `--target` | required | `OBJECT` header value of the science target. |
| `--query-string` | `--target` | Name resolved via MAST. |
| `--fix-bad-pixels` | off | Mask hot/dead pixels from the darks and interpolate over them before photometry. |
| `--output-dir` | `results` | Root folder for outputs; created if missing. Each run gets its own subfolder (see [Output](#output)). |
| `--flat-dir` | — | Directory of flat frames that **override** those found in `--image_path`. Only files whose `IMAGETYP` header marks them as flats and whose `FILTER` matches the target filter are used. |
| `--dark-dir` | — | Directory of dark frames that **override** those found in `--image_path`. Exposure-time matching is still applied. |
| `--bias-dir` | — | Directory of bias frames that **override** those found in `--image_path`. |
| `--report` | off | Build the HTML report (and its assets folder) after reducing. |

### Output

Each run writes to its own folder, `<output-dir>/<target>/<date>_<telescope>_<filter>/`.
The telescope and filter come from the FITS headers. Characters that are not safe in
file names become `-` (so `ETH Hongg` becomes `ETH-Hongg`, and `i'` becomes `i`).
Grouping by target first keeps all nights of one object together. The date comes
first in the run folder, so nights sort in order.

```
results/
└── WASP-33b/
    └── 2026-10-02_ETH-Hongg_i/
        ├── photometry.npz        # per-frame fluxes, backgrounds, centroids, metadata
        ├── night_report.npz      # all of that + stack, master frames, light curves, movie
        ├── lightcurve.pdf        # target and comparison-star light curves
        ├── systematics.pdf       # FWHM, sky, centroid drift, airmass
        ├── night_report.html     # interactive report (--report only)
        └── night_report_assets/  # movie and images the report loads (--report only)
```

A second run of the same target, night, telescope and filter overwrites that folder's files.

## night_report.py

Reads a run folder's `night_report.npz` and writes an HTML page (Plotly and D3 from a CDN)
you can open in any browser, with no server. The movie and stack images go in a
sibling `<report>_assets/` folder that the page references, so keep the two
together when you move or share a report. It gives you:

- a stack viewer (ZScale) with aperture circles and tabs for the flat/dark/bias;
- a night movie whose playhead follows your cursor along the light curve;
- the differential light curve, with aperture and binning controls;
- a systematics panel: FWHM, sky, drift, airmass, ALC, or any single star;
- a light/dark theme toggle.

```bash
uv run night_report.py results/<target>/<date>_<telescope>_<filter>/night_report.npz \
  [-o report.html] \
  [--fps 15] \
  [--crf 28] \      # movie quality: higher number = smaller file
  [--keyint 1]      # 1 = sharpest scrubbing; larger = smaller file
```

The movie is the largest asset. By default each frame is JPEG-like compressed at
`--crf 28`, which keeps scrubbing sharp; raise `--crf` to shrink it. Setting
`--keyint` above 1 also compresses across frames, which helps on real sky where
consecutive frames barely change. The pipeline's `--report` uses these defaults,
so run `night_report.py` yourself when you want to change them.
