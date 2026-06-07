"""Build an interactive night-report web page from a pipeline report bundle.

Reads the ``night_report_<target>_<date>.npz`` produced by ``pipeline.py`` and
writes a single self-contained HTML file that shows, in the style of the
SPECULOOS portal:

  * the co-added stack with the target/comparison stars overlaid, plus buttons
    to switch to the master flat/dark/bias frames;
  * the night movie (encoded to mp4), whose playhead follows the cursor when you
    hover the light curve;
  * the target differential light curve (raw + binned);
  * a systematics panel with a dropdown (fwhm, sky, dx, dy, airmass, and the
    comparison-star light curves);
  * diagnostic flags (airmass, sky, fwhm, saturation).

The page renders with plotly.js loaded from a CDN, so no Python plotting library
is required.

Usage:
    python night_report.py night_report_<target>_<date>.npz [-o report.html]
"""

import argparse
import base64
import io
import json
import logging
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from astropy.visualization import ZScaleInterval
from PIL import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("night_report")

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
DEFAULT_PLATESCALE = 0.348  # arcsec/pixel, used when not stored in the bundle
_ZSCALE = ZScaleInterval()  # DS9-style display limits, shared with pipeline.py

# Comparison-star marker colour and target colour (mirroring the portal palette).
COLOR_TARGET = "#3F92FF"
COLOR_COMP = "#9340FF"
COLOR_FAINT = "rgba(200,200,200,0.45)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def jsonify(obj):
    """Recursively convert numpy/array data to JSON-safe Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def stretch(image):
    """ZScale (DS9-style) contrast stretch to 0..1 for display."""
    image = np.asarray(image, float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo, hi = _ZSCALE.get_limits(finite)
    norm = np.clip((image - lo) / (hi - lo + 1e-9), 0, 1)
    return np.nan_to_num(norm, nan=0.0)  # non-finite pixels -> black


def png_data_uri(image, max_px=2048, step=None):
    """Downsample + stretch an image to a PNG data URI; return (uri, stride).

    Pass ``step`` to force a specific downsampling factor (useful to make all
    frames share the same coordinate system). Otherwise the step is derived
    from ``max_px``.
    """
    image = np.asarray(image, float)
    if step is None:
        step = max(1, int(np.ceil(max(image.shape) / max_px)))
    arr = (stretch(image[::step, ::step]) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), step


def bin_time(t, y, window_min=10):
    """Bin (t, y) into ``window_min``-minute windows; return centres, means, errors."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    w = window_min / (24 * 60)
    bt, by, be = [], [], []
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] - t[i] < w:
            j += 1
        seg = y[i:j]
        bt.append(float(np.mean(t[i:j])))
        by.append(float(np.nanmean(seg)))
        be.append(float(np.nanstd(seg) / np.sqrt(max(1, len(seg)))))
        i = j
    return np.array(bt), np.array(by), np.array(be)


def encode_movie(frames, mp4_path, fps, crf=28, preset="slow", keyint=1):
    """Encode a (n, h, w) uint8 cube to a browser-friendly mp4; return its bytes.

    Compression knobs (each frame is JPEG-like intra-coded by default):
      * ``crf`` — x264 quality/size trade-off; higher is smaller and lower
        quality (~18 visually lossless, ~28 a good balance for these grayscale
        thumbnails). This is the main lever on file size.
      * ``preset`` — x264 speed/efficiency preset; slower presets compress better
        at the same crf. Encoding is one-off, so "slow" is a reasonable default.
      * ``keyint`` — keyframe interval in frames. ``1`` means all-intra (every
        frame a keyframe), which makes hover-scrubbing seek crisply but is the
        largest. Values >1 add temporal compression of the near-static night
        frames (e.g. ``fps`` for ~1 s GOP), shrinking the file several-fold more.
    """
    f = np.asarray(frames)
    if f.ndim != 3 or f.shape[0] == 0:
        return None
    # ffmpeg/yuv420p needs even dimensions.
    if f.shape[1] % 2:
        f = f[:, :-1, :]
    if f.shape[2] % 2:
        f = f[:, :, :-1]
    rgb = np.repeat(f[..., None], 3, axis=-1)  # grayscale -> RGB
    imageio.mimwrite(
        mp4_path,
        rgb,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv420p",  # browser-compatible
        output_params=["-crf", str(crf), "-preset", preset, "-g", str(max(1, keyint))],
    )
    return Path(mp4_path).read_bytes()


def comparison_indices(weights, diffs, target_index, n_max=8):
    """Pick the comparison stars (highest weight, weight > 0, excluding target).

    ``auto_diff`` may return weights as 1D (per star) or with an extra aperture
    axis; collapse to a per-star vector by averaging any non-star axes.
    """
    n_stars = diffs.shape[1]
    w = np.asarray(weights, float)
    if w.ndim == 1 and w.size == n_stars:
        w1d = w
    elif w.ndim >= 2 and n_stars in w.shape:
        star_axis = list(w.shape).index(n_stars)
        w1d = np.nanmean(np.moveaxis(w, star_axis, 0).reshape(n_stars, -1), axis=1)
    else:
        return []
    order = np.argsort(w1d)[::-1]
    return [int(i) for i in order if i != target_index and w1d[i] > 0][:n_max]


# ---------------------------------------------------------------------------
# Figure specifications (plain dicts -> JSON -> plotly.js)
# ---------------------------------------------------------------------------
def circle_shapes(coords, indices, radius, target_index, selected=None):
    """Aperture circles (data-space) for the given star indices."""
    shapes = []
    for i in indices:
        cx, cy = coords[i]
        shapes.append(
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": cx - radius,
                "x1": cx + radius,
                "y0": cy - radius,
                "y1": cy + radius,
                "line": {
                    "color": COLOR_TARGET if i == target_index else COLOR_COMP,
                    "width": 3 if i == selected else 1.5,
                },
            }
        )
    return shapes


def image_figure(d, n_stars, comps, target_index, init_radius):
    """Stack + master frames with a single hoverable star-overlay trace.

    Returns initial traces/layout for the best aperture; the JS controller
    resizes the aperture circles and toggles the master frames on the client.
    All frames (stack, flat, dark, bias) are cropped to the stack's shape and
    encoded at the same stride so they share an identical coordinate system.
    """
    coords = np.asarray(d["ref_coords"])[:n_stars]
    xs, ys = coords[:, 0].tolist(), coords[:, 1].tolist()
    comp_set = set(comps)

    # Use the stack as the reference shape; calibration frames are full-sensor
    # size (not trimmed), so crop them symmetrically to match.
    stack = np.asarray(d["stack"])
    sh, sw = stack.shape
    # Compute a single stride from the stack so all images share the same scale.
    shared_step = max(1, int(np.ceil(max(sh, sw) / 2048)))

    def _crop_to_stack(img):
        img = np.asarray(img, float)
        if img.shape == (sh, sw):
            return img
        dy = (img.shape[0] - sh) // 2
        dx = (img.shape[1] - sw) // 2
        return img[dy : dy + sh, dx : dx + sw]

    images = []
    for name, key in [
        ("stack", "stack"),
        ("flat", "master_flat"),
        ("dark", "master_dark"),
        ("bias", "master_bias"),
    ]:
        uri, _ = png_data_uri(_crop_to_stack(d[key]), step=shared_step)
        images.append(
            {
                "type": "image",
                "source": uri,
                "x0": 0,
                "y0": 0,
                "dx": shared_step,
                "dy": shared_step,
                "visible": name == "stack",
                "hoverinfo": "skip",
                "name": name,
            }
        )

    # One scatter holding every measured star (curve 4), coloured by role.
    colors, sizes = [], []
    for i in range(n_stars):
        if i == target_index:
            role_color, role_size = COLOR_TARGET, 10
        elif i in comp_set:
            role_color, role_size = COLOR_COMP, 8
        else:
            role_color, role_size = COLOR_FAINT, 5
        colors.append(role_color)
        sizes.append(role_size)
    roles = [
        "target" if i == target_index else ("comparison" if i in comp_set else "star")
        for i in range(n_stars)
    ]
    stars = {
        "type": "scatter",
        "mode": "markers",
        "name": "stars",
        "visible": True,
        "x": xs,
        "y": ys,
        "customdata": list(range(n_stars)),
        "text": roles,
        "marker": {"size": sizes, "color": colors, "line": {"width": 0}},
        "hovertemplate": "%{text} #%{customdata}<extra></extra>",
    }

    shapes = circle_shapes(
        coords.tolist(), [target_index] + comps, init_radius, target_index
    )
    layout = {
        "margin": {"l": 6, "r": 6, "t": 6, "b": 6},
        "height": 460,
        "dragmode": "pan",
        # Fix axis ranges to the stack dimensions so switching images never
        # causes an auto-range jump.
        "xaxis": {
            "visible": False,
            "scaleanchor": "y",
            "constrain": "domain",
            "range": [0, sw],
            "fixedrange": False,
        },
        "yaxis": {
            "visible": False,
            "autorange": False,
            "range": [sh, 0],  # y0=0 at top
            "fixedrange": False,
        },
        "shapes": shapes,
        "showlegend": False,
        "paper_bgcolor": "white",
        "hovermode": "closest",
    }
    return {"data": images + [stars], "layout": layout}


def lightcurve_figure(d, best, target_index, platescale=DEFAULT_PLATESCALE):
    """Target light curve (row 1) + a systematics row (row 2), best aperture."""
    diffs = np.asarray(d["diffs"])
    time = np.asarray(d["time"], float)
    diff = diffs[best, target_index]
    bt, by, be = bin_time(time, diff)

    # Subtract the integer part of the first JD so all x values are small
    # decimals (e.g. 0.432 … 1.187) that plotly renders cleanly as linear floats.
    jd0 = int(np.floor(time.min()))
    t = (time - jd0).tolist()
    btt = (bt - jd0).tolist()

    raw = {
        "type": "scattergl",
        "mode": "markers",
        "name": "raw",
        "x": jsonify(t),
        "y": jsonify(diff),
        "marker": {"size": 4, "color": "rgba(150,150,150,0.6)"},
        "xaxis": "x",
        "yaxis": "y",
        "hoverinfo": "skip",
    }
    binned = {
        "type": "scatter",
        "mode": "markers",
        "name": "binned",
        "x": jsonify(btt),
        "y": jsonify(by),
        "error_y": {
            "type": "data",
            "array": jsonify(be),
            "thickness": 1,
            "color": "#333",
        },
        "marker": {"size": 6, "color": "#222"},
        "xaxis": "x",
        "yaxis": "y",
        "hovertemplate": "JD+%{x:.4f}<br>flux %{y:.4f}<extra></extra>",
    }
    fwhm = np.asarray(d["fwhm"], float) * platescale
    sbt, sby, sbe = bin_time(time, fwhm)
    syst_raw = {
        "type": "scattergl",
        "mode": "markers",
        "name": "fwhm",
        "x": jsonify(t),
        "y": jsonify(fwhm),
        "marker": {"size": 4, "color": "rgba(87,93,100,0.45)"},
        "xaxis": "x2",
        "yaxis": "y2",
        "hoverinfo": "skip",
    }
    syst_binned = {
        "type": "scatter",
        "mode": "markers",
        "name": "fwhm (binned)",
        "x": jsonify((sbt - jd0).tolist()),
        "y": jsonify(sby),
        "error_y": {
            "type": "data",
            "array": jsonify(sbe),
            "thickness": 1,
            "color": "#333",
        },
        "marker": {"size": 6, "color": "#333"},
        "xaxis": "x2",
        "yaxis": "y2",
        "hovertemplate": "JD+%{x:.4f}<br>%{y:.4f}<extra></extra>",
    }

    x_axis = {
        "tickformat": ".3f",
        "hoverformat": ".5f",
    }

    layout = {
        "margin": {"l": 55, "r": 15, "t": 10, "b": 40},
        "showlegend": False,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {
            **x_axis,
            "anchor": "y",
            "domain": [0, 1],
            "matches": "x2",
            "showticklabels": False,
        },
        "yaxis": {
            "anchor": "x",
            "domain": [0.42, 1],
            "title": {"text": "Diff. flux"},
            "range": [0.96, 1.04],
            "autorange": False,
        },
        "xaxis2": {
            **x_axis,
            "anchor": "y2",
            "domain": [0, 1],
            "title": {"text": f"JD \u2212 {jd0}"},
        },
        "yaxis2": {
            "anchor": "x2",
            "domain": [0, 0.32],
            "title": {"text": "FWHM (arcsec)"},
        },
    }
    return (
        {"data": [raw, binned, syst_raw, syst_binned], "layout": layout},
        (float(time.min()), float(time.max()), jd0),
    )


def compute_alc(d, diffs, weights):
    """Per-aperture artificial light curve (weighted mean of normalised comp fluxes).

    Saved by the pipeline as ``alc``; recomputed here from fluxes/bkg/weights if
    an older bundle lacks it.
    """
    if "alc" in d:
        return np.asarray(d["alc"], float)
    fluxes = (
        np.asarray(d["fluxes"], float) - np.asarray(d["bkg"], float)
    ).T  # (ap, star, frame)
    norm = fluxes / np.nanmean(fluxes, axis=-1, keepdims=True)
    wsum = weights.sum(axis=-1, keepdims=True)
    return np.einsum("as,asf->af", weights, norm) / np.where(wsum == 0, np.nan, wsum)


def app_payload(
    d,
    diffs,
    weights,
    alc,
    target_index,
    best,
    ap_radii_per_ap,
    coords,
    t_range,
    jd0=0,
    platescale=DEFAULT_PLATESCALE,
):
    """Client-side data the JS controller needs to react to aperture/star events."""
    return {
        "time": jsonify(np.asarray(d["time"], float)),
        "diffs": jsonify(np.round(diffs, 5)),  # [aperture][star][frame]
        "alc": jsonify(np.round(alc, 5)),  # [aperture][frame]
        "weights": jsonify(np.round(weights, 5)),  # [aperture][star]
        "syst": {
            "fwhm": jsonify(np.asarray(d["fwhm"], float) * platescale),
            "sky": jsonify(
                np.asarray(d["sky"], float)
                / np.asarray(d.get("exptime", np.ones(len(d["sky"]))), float)
            ),
            "dx": jsonify(np.asarray(d["dx"], float)),
            "dy": jsonify(np.asarray(d["dy"], float)),
            "airmass": jsonify(np.asarray(d["airmass"], float)),
        },
        "coords": jsonify(coords),
        "apRadii": jsonify(ap_radii_per_ap),
        "target": int(target_index),
        "best": int(best),
        "nAp": int(diffs.shape[0]),
        "t0": t_range[0],
        "t1": t_range[1],
        "jd0": int(jd0),
        "cTarget": COLOR_TARGET,
        "cComp": COLOR_COMP,
        "cFaint": COLOR_FAINT,
        "platescale": platescale,  # arcsec/pixel, for FWHM labels in JS
    }


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


def render_html(meta, img_fig, lc_fig, app, movie_b64):
    video_html = (
        f'<video id="vid" muted playsinline preload="auto" '
        f'src="data:video/mp4;base64,{movie_b64}"></video>'
        if movie_b64
        else '<div class="novideo">No movie available</div>'
    )
    movie_tab = (
        '<span class="tab-btn" id="movieBtn" data-k="movie" onclick="toggleMovie()">&#9654; Movie</span>'
        if movie_b64
        else ""
    )
    img_tabs = (
        "".join(
            f'<span class="tab-btn{" active" if k == 0 else ""}" data-k="{k}" onclick="showImage({k})">{lbl}</span>'
            for k, lbl in enumerate(["Stack", "Flat", "Dark", "Bias"])
        )
        + movie_tab
    )

    sbtns = "".join(
        f'<span class="hover-syst{" hovered" if key == "fwhm" else ""}" data-k="{key}" '
        f"onmouseenter=\"setSyst('{key}')\">{key}</span>"
        for key in ["fwhm", "sky", "dx", "dy", "airmass", "alc"]
    )

    meta_pills = "".join(
        f'<span class="meta-pill">{label}</span>'
        for label in filter(
            None,
            [
                meta["date"],
                f"filter&nbsp;<b>{meta['band']}</b>",
                f"{meta['n_frames']}&nbsp;frames",
                f"{meta['hours']:.1f}&nbsp;h",
                f"exp&nbsp;<b>{meta['exptime']:.0f}&nbsp;s</b>"
                if meta["exptime"] is not None
                else None,
                f"median&nbsp;fwhm&nbsp;<b>{meta['median_fwhm_as']:.2f}&Prime;&nbsp;({meta['median_fwhm_px']:.1f}&nbsp;px)</b>",
                f"read noise&nbsp;<b>{meta['read_noise']:.2f}&nbsp;ADU</b>"
                if meta["read_noise"] is not None
                else None,
                f"dark current&nbsp;<b>{meta['dark_current']:.4f}&nbsp;ADU/s</b>"
                if meta["dark_current"] is not None
                else None,
            ],
        )
    )

    best = app["best"]
    n_ap = app["nAp"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Night report — {meta["target"]} {meta["date"]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"/>
<script src="{PLOTLY_CDN}"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
  /* Apply saved/OS theme before first paint to avoid flash. */
  (function() {{
    const t = localStorage.getItem('theme') ||
              (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', t);
  }})();
</script>
<style>
/* ── Theme variables ────────────────────────────────────────────── */
:root {{
  --bg:            #f5f6f8;
  --text:          #23292f;
  --text-sec:      #555e66;
  --text-muted:    #aaa;
  --card-bg:       #fff;
  --card-shadow:   0 1px 3px rgba(60,64,67,.15), 0 1px 6px rgba(60,64,67,.08);
  --sep:           #eee;
  --sep-card:      #f0f2f4;
  --tab-idle:      #808891;
  --tab-active:    #23292f;
  --novideo-fg:    #808891;
  --input-border:  #dde1e5;
  --input-bg:      transparent;
  --input-fg:      #23292f;
  --comp-fg:       #bababa;
  --comp-border:   #bababa;
  --comp-badge-bg: rgba(186,186,186,.07);
  --syst-idle:     #808891;
  --syst-active:   #23292f;
}}
[data-theme="dark"] {{
  --bg:            #1c1c1e;
  --text:          #f5f5f7;
  --text-sec:      #8e8e93;
  --text-muted:    #636366;
  --card-bg:       #2c2c2e;
  --card-shadow:   0 1px 3px rgba(0,0,0,.5), 0 1px 8px rgba(0,0,0,.3);
  --sep:           #38383a;
  --sep-card:      #38383a;
  --tab-idle:      #636366;
  --tab-active:    #f5f5f7;
  --novideo-fg:    #636366;
  --input-border:  #38383a;
  --input-bg:      #1c1c1e;
  --input-fg:      #f5f5f7;
  --comp-fg:       #bababa;
  --comp-border:   rgba(186,186,186,.5);
  --comp-badge-bg: rgba(186,186,186,.08);
  --syst-idle:     #636366;
  --syst-active:   #f5f5f7;
}}
/* ── Reset & base ──────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; }}
body {{
  font-family: 'Inter', -apple-system, Segoe UI, sans-serif;
  font-size: 14px;
  color: var(--text);
  background: var(--bg);
  display: flex;
  flex-direction: column;
}}

/* ── Header ────────────────────────────────────────────────────── */
.site-header {{
  background: #23292f;
  color: #fff;
  padding: 0 28px;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-shrink: 0;
  border-bottom: 2px solid #1a1f24;
}}
.site-header .target-name {{
  font-size: 18px;
  font-weight: 600;
  color: #fff;
  letter-spacing: 0.01em;
}}
.site-header .telescope-tag {{
  font-size: 12px;
  font-weight: 500;
  color: #639AD8;
  background: rgba(99,154,216,0.15);
  border: 1px solid rgba(99,154,216,0.35);
  border-radius: 2px;
  padding: 2px 8px;
  letter-spacing: 0.03em;
}}
.meta-pills {{
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-left: auto;
}}
.meta-pill {{
  font-size: 12px;
  color: #C5C7CA;
  background: rgba(255,255,255,0.06);
  border-radius: 0;
  padding: 2px 8px;
}}
.meta-pill b {{ color: #fff; font-weight: 500; }}
.theme-btn {{
  background: none;
  border: none;
  cursor: pointer;
  padding: 6px;
  color: rgba(255,255,255,0.65);
  display: flex;
  align-items: center;
  border-radius: 4px;
  flex-shrink: 0;
  transition: background .15s, color .15s;
}}
.theme-btn:hover {{ background: rgba(255,255,255,0.1); color: #fff; }}
.theme-btn svg {{ width: 16px; height: 16px; display: block; }}

/* ── Main layout ───────────────────────────────────────────────── */
.page-body {{
  display: flex;
  flex-direction: row;
  gap: 20px;
  padding: 20px 24px;
  align-items: flex-start;
  flex-wrap: wrap;
  flex: 1;
  overflow-y: auto;
}}
.col-left  {{ flex: 0 0 min(calc(50vw - 34px), calc(100vh - 136px)); display: flex; flex-direction: column; gap: 12px; }}
.col-right {{ flex: 1 1 480px; max-width: calc(50vw - 34px); min-width: 320px; margin-left: auto; display: flex; flex-direction: column; gap: 10px; }}

/* ── Card ──────────────────────────────────────────────────────── */
.card {{
  background: var(--card-bg);
  border-radius: 4px;
  box-shadow: var(--card-shadow);
  overflow: hidden;
}}
.card-body {{ padding: 14px 16px; }}

/* ── Image box ─────────────────────────────────────────────────── */
.imagebox {{
  position: relative;
  width: 100%;
  padding-top: 100%;   /* overridden in JS once image dimensions are known */
  background: #000;
  border-radius: 4px 4px 0 0;
  overflow: hidden;
}}
.imagebox > #vid {{
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: contain;
  background: #000;
  display: none;
}}
.imagebox > #img-d3 {{
  position: absolute; inset: 0;
}}
.novideo {{
  color: var(--novideo-fg);
  text-align: center;
  padding: 48px 0;
  font-size: 13px;
}}

/* ── Tab buttons (image switcher) ──────────────────────────────── */
.tab-menu {{
  display: flex;
  flex-direction: row;
  border-bottom: 1px solid var(--sep);
  padding: 0 12px;
  gap: 0;
}}
.tab-btn {{
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  color: var(--tab-idle);
  padding: 8px 12px;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  user-select: none;
  transition: color .15s, border-color .15s;
  white-space: nowrap;
}}
.tab-btn:hover {{ color: var(--tab-active); }}
.tab-btn.active {{ color: var(--tab-active); border-bottom-color: #bababa; }}

/* ── Controls row ──────────────────────────────────────────────── */
.controls-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--text-sec);
}}
.controls-label {{ font-weight: 500; color: var(--text); white-space: nowrap; }}
.controls-hint  {{ color: var(--text-muted); font-style: italic; font-size: 11px; }}

/* ── Aperture slider ───────────────────────────────────────────── */
#aplab {{
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  min-width: 130px;
}}
input[type=range] {{
  accent-color: #bababa;
  vertical-align: middle;
  width: 120px;
  cursor: pointer;
}}
input[type=number] {{
  width: 52px;
  border: 1px solid var(--input-border);
  border-radius: 3px;
  padding: 3px 6px;
  font-size: 12px;
  font-family: inherit;
  color: var(--input-fg);
  background: var(--input-bg);
  outline: none;
}}
input[type=number]:focus {{ border-color: #bababa; box-shadow: 0 0 0 2px rgba(186,186,186,.2); }}

/* ── Comparison star list ──────────────────────────────────────── */
#complist {{
  font-size: 12px;
  color: var(--text-sec);
  line-height: 1.6;
  padding: 2px 0;
}}
#complist b {{ color: var(--text); font-weight: 600; }}
.comp-badge {{
  display: inline-block;
  padding: 1px 7px;
  border-radius: 4px;
  margin: 2px 3px 2px 0;
  border: 1px solid var(--comp-border);
  color: var(--comp-fg);
  font-size: 11px;
  font-weight: 500;
  background: var(--comp-badge-bg);
}}

/* ── Light curve plot ──────────────────────────────────────────── */
#lc {{ height: clamp(380px, calc(100vh - 280px), 66.67vh); }}

/* ── Systematics buttons ───────────────────────────────────────── */
.syst-row {{
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--text-sec);
}}
.hover-syst {{
  cursor: pointer;
  padding: 3px 9px;
  font-size: 12px;
  border: 1px solid currentColor;
  border-radius: 3px;
  color: var(--syst-idle);
  opacity: .6;
  user-select: none;
  transition: opacity .12s;
}}
.hover-syst:hover {{ opacity: 1; }}
.hover-syst.hovered {{ opacity: 1; color: var(--syst-active); border-color: var(--syst-active); }}
.hover-syst.comparison {{ color: #bababa; }}
</style>
</head>
<body>

<header class="site-header">
  <span class="target-name">{meta["target"]}</span>
  <span class="telescope-tag">{meta["telescope"]}</span>
  <div class="meta-pills">{meta_pills}</div>
  <button id="themeBtn" class="theme-btn" onclick="toggleTheme()" title="Toggle dark mode"></button>
</header>

<div class="page-body">

  <!-- ── Left column: image + controls ── -->
  <div class="col-left">
    <div class="card">
      <div class="imagebox">
        {video_html}
        <div id="img-d3" style="position:absolute;inset:0;">
          <canvas id="img-canvas" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
          <svg id="img-svg" style="position:absolute;inset:0;width:100%;height:100%;overflow:hidden;"></svg>
        </div>
      </div>
      <div class="tab-menu">{img_tabs}</div>
    </div>
  </div>

  <!-- ── Right column: light curve + controls ── -->
  <div class="col-right">
    <div class="card">
      <div class="card-body" style="border-bottom:1px solid var(--sep-card); padding-bottom:10px;">

        <!-- Aperture selector -->
        <div class="controls-row" style="margin-bottom:8px;">
          <span class="controls-label">Aperture</span>
          <span id="aplab"><u>{best}</u></span>
          <input type="range" min="0" max="{n_ap - 1}" value="{best}" step="1"
                 oninput="setAperture(+this.value)"/>
          <span style="margin-left:8px;" class="controls-label">Bin</span>
          <input type="number" min="1" max="120" value="10" step="1"
                 oninput="setBinning(+this.value)"/>
          <span class="controls-hint">min</span>
        </div>

        <!-- Comparison star ALC summary -->
        <div id="complist"></div>
      </div>

      <!-- Light curve plot -->
      <div id="lc"></div>
    </div>

    <!-- Systematics controls -->
    <div class="card">
      <div class="card-body">
        <div class="syst-row">
          <span class="controls-label">Systematics</span>
          {sbtns}
          <span class="controls-hint">hover a star on the stack to compare it</span>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
  // ── Theme toggle ─────────────────────────────────────────────────────────
  const _MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  const _SUN  = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
  const _THEME_PLOTLY = {{
    light: {{ bg: 'white',    ax: '#e5e7eb', font: '#555e66', bMk: '#222',    bErr: '#333',    sMk: '#333',    sErr: '#333'    }},
    dark:  {{ bg: '#2c2c2e', ax: '#38383a', font: '#8e8e93', bMk: '#e0e4ea', bErr: '#9ca3af', sMk: '#9ca3af', sErr: '#9ca3af' }},
  }};
  function _applyTheme(t) {{
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('theme', t);
    document.getElementById('themeBtn').innerHTML = (t === 'dark') ? _SUN : _MOON;
    const th = _THEME_PLOTLY[t];
    const axUpd = {{ paper_bgcolor: th.bg, plot_bgcolor: th.bg, 'font.color': th.font }};
    ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'].forEach(ax => {{
      axUpd[ax + '.gridcolor']     = th.ax;
      axUpd[ax + '.linecolor']     = th.ax;
      axUpd[ax + '.zerolinecolor'] = th.ax;
    }});
    Plotly.relayout('lc', axUpd);
    Plotly.restyle('lc', {{ 'marker.color': [th.bMk], 'error_y.color': [th.bErr] }}, [TBIN]);
    Plotly.restyle('lc', {{ 'marker.color': [th.sMk], 'error_y.color': [th.sErr] }}, [SBIN]);
  }}
  function toggleTheme() {{
    _applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  }}

  const D = {json.dumps(app)};
  const imgFig = {json.dumps(img_fig)};
  const lcFig = {json.dumps(lc_fig)};
  const cfg = {{responsive: true, displayModeBar: false}};
  const RAW = 0, TBIN = 1, SRAW = 2, SBIN = 3;  // light-curve trace indices
  let A = D.best, sel = {{type: 'syst', key: 'fwhm'}}, binMin = 10, showingStack = true;

  Plotly.newPlot('lc', lcFig.data, lcFig.layout, cfg);

  // ── D3 image viewer ──────────────────────────────────────────────────────
  const _IW = imgFig.layout.xaxis.range[1];
  const _IH = imgFig.layout.yaxis.range[0];

  // Fix the imagebox aspect ratio to match the actual image dimensions.
  document.querySelector('.imagebox').style.paddingTop =
    (_IH / _IW * 100).toFixed(4) + '%';

  const _imgs = {{}};
  imgFig.data.filter(d => d.type === 'image').forEach(d => {{
    const el = new Image();
    el.onload = () => {{ if (_curImg === d.name) _redraw(); }};
    el.src = d.source;
    _imgs[d.name] = el;
  }});
  let _curImg = 'stack', _tr = d3.zoomIdentity;

  const _canvas = document.getElementById('img-canvas');
  const _svg    = d3.select('#img-svg');
  const _gCirc  = _svg.append('g');

  d3.select('#img-svg').call(
    d3.zoom().scaleExtent([1, 30]).on('zoom', ev => {{
      const t = ev.transform;
      const cw = _canvas.clientWidth, ch = _canvas.clientHeight;
      const tx = Math.min(0, Math.max(t.x, cw * (1 - t.k)));
      const ty = Math.min(0, Math.max(t.y, ch * (1 - t.k)));
      _tr = d3.zoomIdentity.translate(tx, ty).scale(t.k);
      _redraw(); _renderOverlay();
    }})
  ).on('dblclick.zoom', null);

  function _toScreen(dx, dy) {{
    const cw = _canvas.clientWidth, ch = _canvas.clientHeight;
    return [_tr.x + _tr.k * (dx / _IW) * cw,
            _tr.y + _tr.k * (dy / _IH) * ch];
  }}
  function _screenR(r) {{
    return _tr.k * (r / _IW) * _canvas.clientWidth;
  }}
  function _redraw() {{
    // Only reallocate the canvas buffer when the display size actually changes;
    // resizing is expensive (clears the buffer and forces a GPU texture upload).
    const w = _canvas.offsetWidth, h = _canvas.offsetHeight;
    if (_canvas.width !== w || _canvas.height !== h) {{
      _canvas.width = w; _canvas.height = h;
    }}
    const ctx = _canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    const img = _imgs[_curImg];
    if (!img || !img.complete || !img.naturalWidth) return;
    ctx.save();
    ctx.imageSmoothingEnabled = false;   // nearest-neighbour → crisp pixels at zoom
    ctx.translate(_tr.x, _tr.y);
    ctx.scale(_tr.k, _tr.k);
    ctx.drawImage(img, 0, 0, w, h);
    ctx.restore();
  }}

  // Cache compsForAp result — it only changes when A changes, not on every hover.
  let _compsCache = null, _compsCacheA = -1;
  function _getComps() {{
    if (A !== _compsCacheA) {{ _compsCache = compsForAp(A); _compsCacheA = A; }}
    return _compsCache;
  }}

  function _renderOverlay() {{
    if (!showingStack) {{ _gCirc.selectAll('circle').remove(); return; }}
    const comps = _getComps();
    const compSet = new Set(comps);
    const s = selectedStar(), r = D.apRadii[A];
    _gCirc.selectAll('circle')
      .data(Array.from({{length: D.coords.length}}, (_, i) => i), i => i)
      .join(
        enter => enter.append('circle')
          .attr('fill', 'none')
          .style('pointer-events', 'all')   // hit-test the interior, not just the ring
          .style('cursor', 'crosshair')
          .on('mouseover', (_, i) => setComp(i))
      )
      .attr('cx', i => _toScreen(D.coords[i][0], D.coords[i][1])[0])
      .attr('cy', i => _toScreen(D.coords[i][0], D.coords[i][1])[1])
      .attr('r', () => Math.max(4, _screenR(r)))
      .attr('stroke', i => i === D.target ? D.cTarget : (compSet.has(i) ? D.cComp : D.cFaint))
      .attr('stroke-width', i => i === s ? 3 : 1.5);
  }}
  // Constrain the left column so the image never overflows the viewport.
  // width  ≤  50 % of viewport   (layout rule)
  // height ≤  viewport − chrome  (avoids scrolling)
  // → max_width = min(50 vw − padding, available_height / aspect_ratio)
  function _setColWidth() {{
    const r    = _IH / _IW;
    const maxW = window.innerWidth  * 0.5 - 44;
    const maxH = window.innerHeight - 160;   // header + body padding + tab menu
    const colW = Math.max(240, Math.min(maxW, maxH / r));
    document.querySelector('.col-left').style.flex = '0 0 ' + colW.toFixed(0) + 'px';
  }}
  _setColWidth();
  window.addEventListener('resize', () => {{ _redraw(); _renderOverlay(); _setColWidth(); }});

  // ── Aperture FWHM helpers ────────────────────────────────────────────────
  (function() {{
    const fwhm = D.syst.fwhm.filter(v => v != null && !isNaN(v)).slice().sort((a,b)=>a-b);
    const mid = fwhm.length >> 1;
    window._medFwhmAs = fwhm.length
      ? (fwhm.length % 2 ? fwhm[mid] : (fwhm[mid-1]+fwhm[mid])/2)
      : 0;
  }})();
  const _medFwhmPx = _medFwhmAs / D.platescale;

  function _apMult(a) {{ return D.apRadii[a] / _medFwhmPx; }}
  function _apLabel(a) {{
    const num = a === D.best ? `<u>${{a}}</u>` : `${{a}}`;
    return `${{num}} — ${{_apMult(a).toFixed(1)}}\xd7 FWHM`;
  }}
  function _apTitle(a) {{
    return `${{_apMult(a).toFixed(2)}}\xd7 FWHM\n`
         + `Median FWHM: ${{_medFwhmPx.toFixed(1)}} px = ${{_medFwhmAs.toFixed(2)}}″`;
  }}
  function _setApLabel(a) {{
    const el = document.getElementById('aplab');
    el.innerHTML = _apLabel(a);
    el.title     = _apTitle(a);
  }}

  function binTime(t, y, win) {{
    const o = [...t.keys()].sort((a, b) => t[a] - t[b]);
    const ts = o.map(i => t[i]), ys = o.map(i => y[i]);
    const bt = [], by = [], be = [];
    let i = 0;
    while (i < ts.length) {{
      let j = i;
      while (j < ts.length && ts[j] - ts[i] < win) j++;
      const segT = ts.slice(i, j);
      const segY = ys.slice(i, j).filter(v => v != null && !isNaN(v));
      const n = Math.max(1, segY.length);
      const m = segY.reduce((a, b) => a + b, 0) / n;
      const sd = Math.sqrt(segY.reduce((a, b) => a + (b - m) * (b - m), 0) / n);
      bt.push(segT.reduce((a, b) => a + b, 0) / segT.length);
      by.push(m); be.push(sd / Math.sqrt(n));
      i = j;
    }}
    return {{ bt, by, be }};
  }}

  // Comparison stars (and their weights) are aperture-dependent: the subset with
  // weight > 0 for aperture A, ordered by weight.
  function compsForAp(a) {{
    const w = D.weights[a];
    return w.map((v, i) => [i, v]).filter(p => p[1] > 0 && p[0] !== D.target)
            .sort((p, q) => q[1] - p[1]).map(p => p[0]);
  }}
  function selectedStar() {{ return sel.type === 'comp' ? sel.i : null; }}

  function refreshShapes() {{ _renderOverlay(); }}
  function updateStars()   {{ _renderOverlay(); }}

  function renderCompList() {{
    const comps = compsForAp(A), w = D.weights[A];
    const tot = comps.reduce((s, i) => s + w[i], 0) || 1;
    const items = comps.map(i =>
      `<span class="comp-badge">#${{i}} ${{(w[i] / tot).toFixed(3)}}</span>`
    ).join('');
    document.getElementById('complist').innerHTML =
      `<b>ALC</b> &mdash; weighted &Sigma; of ${{comps.length}} comparison star`
      + `${{comps.length === 1 ? '' : 's'}}: ${{items || '&mdash;'}}`;
  }}

  function systY() {{
    if (sel.type === 'comp') {{
      const y = D.diffs[A] && D.diffs[A][sel.i];
      // auto_diff sets a comparison star's own differential flux to NaN when
      // it is the sole comparison (circular dependency).  Fall back to the ALC
      // which shows the comparison ensemble quality instead.
      if (!y || y.every(v => v == null)) return D.alc[A];
      return y;
    }}
    if (sel.key === 'alc') return D.alc[A];
    return D.syst[sel.key];
  }}
  function systIsFlux() {{ return sel.type === 'comp' || sel.key === 'alc'; }}

  const _systAxisLabel = {{
    fwhm:    'FWHM (arcsec)',
    sky:     'Sky (ADU·s⁻¹·px⁻¹)',
    dx:      'Δx (px)',
    dy:      'Δy (px)',
    airmass: 'Airmass',
    alc:     'ALC (rel. flux)',
  }};

  function systLabel() {{
    if (sel.type === 'comp') {{
      const w = D.weights[A], tot = compsForAp(A).reduce((s, i) => s + w[i], 0) || 1;
      const y = D.diffs[A] && D.diffs[A][sel.i];
      const isSoleAlc = !y || y.every(v => v == null);
      if (isSoleAlc) return 'comparison #' + sel.i + ' (sole ALC — showing ALC)';
      const wtxt = w[sel.i] > 0 ? ' (weight ' + (w[sel.i] / tot).toFixed(3) + ')' : ' (not a comp here)';
      return 'Star #' + sel.i + ' — diff. flux' + wtxt;
    }}
    return _systAxisLabel[sel.key] || sel.key;
  }}

  function renderSyst() {{
    const y = systY();
    const color = sel.type === 'comp' ? D.cComp
                : (sel.key === 'alc' ? '#8E83E6' : 'rgba(87,93,100,0.45)');
    const b = binTime(D.time, y, binMin / 1440);
    Plotly.restyle('lc', {{ y: [y], 'marker.color': [color] }}, [SRAW]);
    Plotly.restyle('lc', {{ x: [b.bt.map(v => v - D.jd0)], y: [b.by], 'error_y.array': [b.be] }}, [SBIN]);
    const isSkyLog = sel.type === 'syst' && sel.key === 'sky';
    // Comparison star differential flux is constrained to ±4% to keep scale stable;
    // ALC and all other systematics scale freely.
    const constrained = sel.type === 'comp';
    const ax = constrained
      ? {{ 'yaxis2.title.text': systLabel(), 'yaxis2.type': 'linear', 'yaxis2.autorange': false, 'yaxis2.range': [0.96, 1.04] }}
      : {{ 'yaxis2.title.text': systLabel(), 'yaxis2.type': isSkyLog ? 'log' : 'linear', 'yaxis2.autorange': true }};
    Plotly.relayout('lc', ax);
    document.querySelectorAll('.hover-syst').forEach(e =>
      e.classList.toggle('hovered', sel.type === 'syst' && e.dataset.k === sel.key));
  }}

  function updateTarget() {{
    const y = D.diffs[A][D.target];
    const b = binTime(D.time, y, binMin / 1440);
    Plotly.restyle('lc', {{ y: [y] }}, [RAW]);
    Plotly.restyle('lc', {{ x: [b.bt.map(v => v - D.jd0)], y: [b.by], 'error_y.array': [b.be] }}, [TBIN]);
  }}

  function setSyst(key) {{ sel = {{ type: 'syst', key }}; renderSyst(); refreshShapes(); }}

  let _compTimer = null;
  function setComp(i) {{
    if (sel.type === 'comp' && sel.i === i) return;  // same star — nothing to do
    sel = {{ type: 'comp', i }};
    // Immediate lightweight feedback: just flip stroke-width on existing circles.
    // This is O(circles) not O(stars) and involves no Plotly calls.
    _gCirc.selectAll('circle').attr('stroke-width', idx => idx === i ? 3 : 1.5);
    // Debounce the expensive work (3 Plotly calls + full D3 join) so rapid mouse
    // movement over many stars doesn't queue up dozens of heavy updates.
    clearTimeout(_compTimer);
    _compTimer = setTimeout(() => {{ _renderOverlay(); renderSyst(); }}, 60);
  }}
  function setBinning(m) {{ binMin = Math.max(1, m || 1); updateTarget(); renderSyst(); }}
  function setAperture(a) {{
    A = a;
    _setApLabel(a);
    updateTarget();
    renderSyst();      // comp/alc series depend on the aperture
    updateStars();
    refreshShapes();
    renderCompList();
  }}
  function setMovie(on) {{
    const v = document.getElementById('vid');
    if (!v) return;
    document.getElementById('img-d3').style.display = on ? 'none' : 'block';
    v.style.display = on ? 'block' : 'none';
    // Clear all tab highlights first so no image tab stays active when movie plays.
    if (on) document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
    document.getElementById('movieBtn').classList.toggle('active', on);
    if (!on) {{ _redraw(); _renderOverlay(); }}
  }}
  function toggleMovie() {{
    const v = document.getElementById('vid');
    setMovie(v && v.style.display !== 'block');
  }}
  function showImage(k) {{
    setMovie(false);
    showingStack = (k === 0);
    _curImg = ['stack', 'flat', 'dark', 'bias'][k] || 'stack';
    _redraw();
    _renderOverlay();
    document.querySelectorAll('.tab-btn').forEach(e =>
      e.classList.toggle('active', e.dataset.k === String(k)));
  }}

  // Hover crosshair + movie scrub: a thin vertical line follows the cursor
  // across both LC subplots (target and systematics share the x-axis). x values
  // are JD-jd0 (plain floats); converted back to absolute JD to scrub the movie
  // when one is embedded.
  const vid = document.getElementById('vid');
  const lcDiv = document.getElementById('lc');
  lcDiv.style.position = 'relative';
  const _spike = document.createElement('div');
  _spike.style.cssText =
    'position:absolute;top:0;width:1px;pointer-events:none;opacity:0;'
    + 'background:var(--text-muted);z-index:5;transition:opacity .08s;';
  lcDiv.appendChild(_spike);

  lcDiv.addEventListener('mousemove', function(evt) {{
    const fl = lcDiv._fullLayout;
    if (!fl || !fl.xaxis2) return;
    const xax = fl.xaxis2, yTop = fl.yaxis, yBot = fl.yaxis2;
    const bb = lcDiv.getBoundingClientRect();
    const px = evt.clientX - bb.left - xax._offset;
    if (!xax._length || px < 0 || px > xax._length) {{ _spike.style.opacity = 0; return; }}
    // Span the crosshair from the top of the upper subplot to the bottom of the lower one.
    _spike.style.left   = (xax._offset + px) + 'px';
    _spike.style.top    = yTop._offset + 'px';
    _spike.style.height = (yBot._offset + yBot._length - yTop._offset) + 'px';
    _spike.style.opacity = 1;
    if (vid && vid.duration) {{
      const frac = Math.max(0, Math.min(1, px / xax._length));
      const x_val = xax.range[0] + frac * (xax.range[1] - xax.range[0]);  // JD - jd0
      const f = (x_val + D.jd0 - D.t0) / (D.t1 - D.t0);
      vid.currentTime = Math.max(0, Math.min(1, f)) * vid.duration;
    }}
  }});
  lcDiv.addEventListener('mouseleave', function() {{ _spike.style.opacity = 0; }});

  // Initial render of the aperture-dependent overlays.
  _setApLabel(D.best); _redraw(); _renderOverlay(); renderCompList();
  // Apply saved/OS theme (also sets the button icon and syncs Plotly colours).
  _applyTheme(localStorage.getItem('theme') ||
              (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_report(npz_path, out_html=None, fps=15, platescale=None, crf=28, keyint=1):
    npz_path = Path(npz_path)
    d = dict(np.load(npz_path, allow_pickle=True))
    if platescale is None:
        platescale = float(d["platescale"]) if "platescale" in d else DEFAULT_PLATESCALE

    target = str(d["target"])
    date = str(d["date"])
    band = str(d["band"])
    telescope = str(d["telescope"]) if "telescope" in d else "Telescope"
    best = int(d["best_aperture"])
    target_index = int(d["target_index"])
    diffs = np.asarray(d["diffs"])
    n_stars = diffs.shape[1]

    comps = comparison_indices(d["weights"], diffs, target_index)
    logger.info(
        "Best aperture %d, target #%d, %d comparison stars",
        best,
        target_index,
        len(comps),
    )

    # Per-aperture median radius (pixels) so the JS controller can resize circles.
    ap_radii = np.asarray(d["aperture_radii"], float)
    if ap_radii.ndim == 2:
        ap_radii_per_ap = np.nanmedian(ap_radii, axis=0).tolist()
    else:
        ap_radii_per_ap = [6.0] * diffs.shape[0]
    coords = np.asarray(d["ref_coords"])[:n_stars].tolist()

    # Movie -> mp4 (sibling file) + base64 for embedding. Reorder frames into
    # chronological order so the light-curve hover scrub lines up with the video
    # (frames are stored in light-frame order, which need not be time-sorted).
    stem = npz_path.stem.replace("night_report_", "")
    mp4_path = npz_path.with_name(f"night_movie_{stem}.mp4")
    movie = d.get("movie")
    if movie is not None and np.asarray(movie).shape[0] == len(d["time"]):
        movie = np.asarray(movie)[np.argsort(np.asarray(d["time"], float))]
    movie_bytes = encode_movie(movie, mp4_path, fps, crf=crf, keyint=keyint)
    movie_b64 = base64.b64encode(movie_bytes).decode() if movie_bytes else ""
    if movie_bytes:
        logger.info("Wrote movie %s (%.1f MB)", mp4_path.name, len(movie_bytes) / 1e6)

    weights = np.asarray(d["weights"])
    alc = compute_alc(d, diffs, weights)

    img_fig = image_figure(d, n_stars, comps, target_index, ap_radii_per_ap[best])
    lc_fig, t_range = lightcurve_figure(d, best, target_index, platescale=platescale)
    t_min, t_max, jd0 = t_range
    app = app_payload(
        d,
        diffs,
        weights,
        alc,
        target_index,
        best,
        ap_radii_per_ap,
        coords,
        (t_min, t_max),
        jd0,
        platescale=platescale,
    )

    time = np.asarray(d["time"], float)
    _fwhm_px = float(np.nanmedian(d["fwhm"]))
    _exptime = float(np.nanmedian(d["exptime"])) if "exptime" in d else None
    _read_noise = float(d["read_noise"]) if "read_noise" in d else None
    _dark_current = float(d["dark_current"]) if "dark_current" in d else None
    meta = {
        "target": target,
        "date": date,
        "band": band,
        "n_frames": len(time),
        "hours": (np.nanmax(time) - np.nanmin(time)) * 24,
        "median_fwhm_px": _fwhm_px,
        "median_fwhm_as": _fwhm_px * platescale,
        "exptime": _exptime,
        "read_noise": None
        if (_read_noise is None or math.isnan(_read_noise))
        else _read_noise,
        "dark_current": _dark_current,
        "telescope": telescope,
    }

    html = render_html(meta, img_fig, lc_fig, app, movie_b64)
    out_html = Path(out_html) if out_html else npz_path.with_suffix(".html")
    out_html.write_text(html)
    logger.info("Wrote report %s (%.1f MB)", out_html, len(html) / 1e6)
    return out_html


def main():
    ap = argparse.ArgumentParser(
        description="Build an interactive night-report web page."
    )
    ap.add_argument(
        "npz", help="night_report_<target>_<date>.npz produced by pipeline.py"
    )
    ap.add_argument("-o", "--out", help="output HTML path (default: alongside the npz)")
    ap.add_argument("--fps", type=int, default=15, help="night-movie frame rate")
    ap.add_argument(
        "--crf",
        type=int,
        default=28,
        help="movie quality/size (x264 CRF; higher = smaller, ~18 best quality, ~28 balanced)",
    )
    ap.add_argument(
        "--keyint",
        type=int,
        default=1,
        metavar="N",
        help="movie keyframe interval in frames (1 = all-intra/crispest scrub; "
        "larger, e.g. --keyint 15, adds temporal compression for a much smaller file)",
    )
    ap.add_argument(
        "--platescale",
        type=float,
        default=None,
        help=f"plate scale in arcsec/pixel (default: read from npz or {DEFAULT_PLATESCALE})",
    )
    args = ap.parse_args()
    build_report(
        args.npz, args.out, args.fps, args.platescale, crf=args.crf, keyint=args.keyint
    )


if __name__ == "__main__":
    main()
