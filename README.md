# SPECULOOS-South Photometry Pipeline

Two scripts: **`pipeline.py`** reduces a night of FITS observations into a
differential light curve, and **`night_report.py`** turns that output into a
self-contained interactive HTML report.

---

## pipeline.py

### What it does

1. **Indexes** a directory of FITS files by observing night (noon-to-noon,
   handling runs past midnight), frame type, and target using the `IMAGETYP`,
   `OBJECT`, `DATE-OBS`, and `FILTER` header keywords. If the target's frames
   span more than one night, the night with the most frames is reduced.

2. **Builds master calibration frames** — bias, dark (matched to the target's
   exposure time), and flat (matched to the target's filter). Estimates read
   noise from a bias-frame pair and dark current from the master dark.

3. **Sets up the reference frame** using the middle light frame of the night:
   calibrates it, detects stars (threshold + morphological opening, sorted by
   brightness), queries Gaia — or 2MASS for infrared filters — to solve the WCS,
   then resolves the science target's sky coordinates via MAST and identifies
   which detected star it is.

4. **Processes all light frames in parallel** (one chunk per CPU core). Each
   worker, initialised once with the master calibration frames and a Ballet CNN
   centroider:
   - calibrates and trims the frame;
   - solves an affine alignment to the reference to project the reference star
     positions into the current frame (the images themselves are not warped);
   - refines centroids with the Ballet CNN;
   - performs aperture photometry with annulus background subtraction across
     40 aperture sizes (0.5–5× FWHM);
   - co-adds the calibrated frame into a running stack.

5. **Runs differential photometry** (`auto_diff`) across all apertures. Selects
   the optimal aperture by ranking the comparison-star ensemble on two noise
   metrics (point-to-point scatter and within-bin residual scatter); the target
   light curve is used only as a cross-check.

Finally it writes the photometry and night-report bundles (see **Output**) and
two diagnostic PDFs: a light-curve figure (target plus each comparison star) and
a systematics figure (FWHM, sky, centroid drift, airmass).

### Usage

```bash
uv run pipeline.py \
  --image_path /path/to/night/fits \
  --target "Sp0025+5422" \
  [--query-string "SP0025+5422"] \  # name for MAST coordinate resolution if different from --target
  [--fix-bad-pixels] \              # interpolate over hot/dead pixels before photometry
  [--output-dir results/] \        # where to write outputs (default: current directory)
  [--report]                       # also build the interactive HTML report at the end
```

`--image_path` must contain all FITS frames for the night (lights, darks,
flats, biases). Calibration frames are matched automatically by exposure time
and filter.

| Flag | Default | Description |
|---|---|---|
| `--image_path` | — | Directory of FITS files for the night. |
| `--target` | — | `OBJECT` header value of the science target. |
| `--query-string` | same as `--target` | Name resolved via MAST for the target's sky coordinates. Use when the `OBJECT` value isn't resolvable. |
| `--fix-bad-pixels` | off | Build a hot/dead-pixel mask from the dark frames and interpolate flagged pixels (plus negatives and above-full-well pixels) from their valid neighbours before photometry. |
| `--output-dir` | `.` (current dir) | Directory for all outputs (npz bundles and PDFs); created if it doesn't exist. |
| `--report` | off | After processing, build the interactive HTML night report (and its mp4) from the saved bundle, in `--output-dir`. Equivalent to running `night_report.py` on the bundle afterwards. |

### Output

Both files are named with the telescope, filter, target, and date read from
the FITS headers (`TELESCOP`, `FILTER`, `OBJECT`, `DATE-OBS`).

| File | Description |
|---|---|
| `photometry_data_<telescope>_<filter>_<target>_<date>.npz` | Raw per-frame fluxes, backgrounds, centroids, and metadata |
| `night_report_<telescope>_<filter>_<target>_<date>.npz` | Full bundle for the report: all of the above plus the co-added stack, master calibration frames, differential light curves, and per-frame movie thumbnails |

### Requirements

```bash
uv sync   # installs eloy[jax], imageio-ffmpeg, and their dependencies
```

Requires Python 3.11.8 (pinned in `pyproject.toml`). `ffmpeg` must be on
`PATH` for movie encoding (used by `night_report.py`).

---

## night_report.py

### What it does

Reads the `night_report_...npz` bundle and writes a self-contained HTML file
(Plotly + D3, loaded from CDN) with:

- **Stack viewer** — co-added image (ZScale stretch) with aperture circles for
  every detected star; tabs to switch to the master flat, dark, or bias frame;
  zoom and pan.
- **Night movie** — per-frame thumbnails encoded to an embedded MP4; the
  playhead scrubs automatically as you move the cursor across the light curve.
- **Light curve** — raw and binned differential flux; aperture slider and
  binning control update the plot in real time. The best aperture number is
  underlined.
- **Systematics panel** — FWHM, sky background, x/y centroid drift, airmass,
  ALC, or any individual star's differential flux. Hover any star circle on the
  stack image to display its light curve.
- **Light / dark mode toggle** in the header; preference persists in
  `localStorage` and respects the OS `prefers-color-scheme` setting on first
  load.

### Usage

```bash
uv run night_report.py night_report_<telescope>_<filter>_<target>_<date>.npz \
  [-o report.html] \
  [--fps 15] \
  [--crf 28] \      # movie quality/size; higher = smaller file (~18 best, ~28 balanced)
  [--keyint 1] \    # movie keyframe interval; 1 = all-intra, larger (e.g. 15) = smaller
  [--platescale 0.348]
```

The output HTML file is fully self-contained and opens in any modern browser
without a server.

The embedded night movie is the dominant contributor to report size. It is
encoded with x264 at `--crf 28` (each frame JPEG-like intra-coded) by default,
which keeps hover-scrubbing crisp. Raise `--crf` for a smaller file, or set
`--keyint` above 1 to add temporal compression of the near-static frames (this
helps real sky data but not noise-dominated frames). These options also apply
when the pipeline builds the report via `--report` only through its defaults; run
`night_report.py` directly to tune them.
