# SPECULOOS-South Photometry Pipeline

Two scripts: **`pipeline.py`** reduces a night of FITS observations into a
differential light curve, and **`night_report.py`** turns that output into a
self-contained interactive HTML report.

---

## pipeline.py

### What it does

1. **Indexes** a directory of FITS files by observing night (noon-to-noon,
   handling runs past midnight), frame type, and target using the `IMAGETYP`,
   `OBJECT`, `DATE-OBS`, and `FILTER` header keywords.

2. **Builds master calibration frames** — bias, dark (matched to the target's
   exposure time), and flat (matched to the target's filter). Estimates read
   noise from a bias-frame pair and dark current from the master dark.

3. **Sets up the reference frame** using the middle light frame of the night:
   calibrates it, detects stars (DAOStarFinder), queries Gaia to solve the WCS,
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

### Usage

```bash
python pipeline.py \
  --image_path /path/to/night/fits \
  --target "Sp0025+5422" \
  [--query-name "SP0025+5422"]   # name for MAST/Gaia resolution if different from --target
```

`--image_path` must contain all FITS frames for the night (lights, darks,
flats, biases). Calibration frames are matched automatically by exposure time
and filter. `--query-name` defaults to `--target` when omitted.

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

- **Stack viewer** — co-added image with aperture circles for every detected
  star; tabs to switch to the master flat, dark, or bias frame; zoom and pan.
- **Night movie** — per-frame thumbnails encoded to an embedded MP4; the
  playhead scrubs automatically as you move the cursor across the light curve.
- **Light curve** — raw and binned differential flux; aperture slider and
  binning control update the plot in real time. The best aperture number is
  underlined.
- **Systematics panel** — FWHM, sky background, x/y centroid drift, airmass,
  ALC, or any individual star's differential flux. Hover any star circle on the
  stack image to display its light curve.
- **Diagnostic chips** — flags for high airmass, elevated sky background, large
  FWHM, or target saturation.
- **Light / dark mode toggle** in the header; preference persists in
  `localStorage` and respects the OS `prefers-color-scheme` setting on first
  load.

### Usage

```bash
python night_report.py night_report_<telescope>_<filter>_<target>_<date>.npz \
  [-o report.html] \
  [--fps 15] \
  [--platescale 0.348]
```

The output HTML file is fully self-contained and opens in any modern browser
without a server.
