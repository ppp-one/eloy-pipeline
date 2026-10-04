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
   against Gaia DR3 (2MASS for infrared filters), and locates the target via MAST.
   The catalogue comes from VizieR (CDS, then its CfA mirror), with the ESA Gaia
   archive as the last fallback. astroquery caches the result, so later runs of
   the same field skip the network.
4. Processes every light frame in parallel: calibrate, align to the reference,
   refine centroids with the Ballet CNN, then aperture photometry over 40 radii
   (0.5 to 5x FWHM), co-adding a stack as it goes.
5. Runs differential photometry against one comparison set for all apertures:
   the 25 stars nearest the target, after dropping stars that are noisier than
   their brightness predicts (likely variables). The set is chosen without
   looking at the target's light curve, so its noise is not underestimated.
   Stars are weighted by inverse variance. The best aperture is the one with
   the lowest short-term (point-to-point and 10-minute) target scatter.

It saves the two `.npz` bundles below and a multi-page PDF summary (see
[pdf_report.py](#pdf_reportpy)).

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
| `--gain` | `EGAIN` header | Native (unbinned) gain in e⁻/ADU, e.g. `0.72`. The camera is assumed to **average** binned pixels, so the effective gain is this × `XBINNING` × `YBINNING`. Read noise and dark current are then also shown in electrons. `GAIN` is not read, because many cameras store the gain *setting* there. |
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
        ├── photometry.npz        # fluxes, backgrounds, light curves, comparison weights, metadata
        ├── images.npz            # co-added stack, master bias/dark/flat (float32), movie frames
        ├── summary.pdf           # observing log, light curve, noise, systematics, comparisons
        ├── night_report.html     # interactive report (--report only)
        └── night_report_assets/  # movie and images the report loads (--report only)
```

A second run of the same target, night, telescope and filter overwrites that folder's files.

## pdf_report.py

Writes `summary.pdf` for a run folder. `pipeline.py` runs it at the end of each run.
You can also run it on its own, without running the pipeline again:

```bash
uv run pdf_report.py results/<target>/<date>_<telescope>_<filter>/ [-o out.pdf]
```

The pages:

1. **Overview**: observing log (coordinates, telescope, camera, airmass, seeing,
   gain, read noise, dark current), photometry and precision (aperture, annulus,
   comparison stars, σ per point and per 10-min bin, red-noise factor β), the
   light curve, and a finder chart with apertures, compass and scale bar.
2. **Noise**: RMS against bin size with the white-noise expectation and β
   (Pont et al. 2006, Winn et al. 2008), and the aperture choice.
3. **Systematics**: light curve, comparison-star flux (extinction and clouds),
   airmass, FWHM, sky, centroid drift and the target's peak counts against the
   saturation limit.
4. **Comparison stars**: each star divided by the others, stacked in the style of
   AstroImageJ, with its weight, brightness, distance and scatter.

Times are mid-exposure BJD_TDB, computed from the site and the target position.
No astrophysical model is fitted: β uses the light curve minus a quadratic in time
(per gap-free segment), so real variability on time scales of tens of minutes
counts as red noise.

## night_report.py

Reads a run folder (`photometry.npz` + `images.npz`) and writes an HTML page (Plotly and D3 from a CDN)
you can open in any browser, with no server. The movie and stack images go in a
sibling `<report>_assets/` folder that the page references, so keep the two
together when you move or share a report. It gives you:

- a stack viewer (ZScale) with aperture circles and tabs for the flat/dark/bias;
- a night movie whose playhead follows your cursor along the light curve;
- the differential light curve, with aperture and binning controls;
- a systematics panel: FWHM, sky, drift, airmass, target peak, ALC, or any single star;
- a light/dark theme toggle.

```bash
uv run night_report.py results/<target>/<date>_<telescope>_<filter>/ \
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
