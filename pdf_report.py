"""Build a multi-page PDF summary of one pipeline run.

Reads a ``pipeline.py`` run folder (``photometry.npz`` + ``images.npz``) and
writes ``summary.pdf`` next to them. The pages follow what AstroImageJ's
multi-plot and time-series photometry papers show:

  1. Overview: observing log, precision summary, the light curve, and a finder
     chart.
  2. Noise: RMS versus bin size with the red-noise factor beta (Winn et al.
     2008), and the aperture choice.
  3. Systematics: light curve, airmass, FWHM, sky, centroid drift, peak counts.
  4. Comparison stars: AstroImageJ-style stacked light curves.

Times are mid-exposure BJD_TDB when the run saved them (``bjd_tdb``), else the
start-of-exposure JD_UTC that older runs stored.

Usage:
    uv run pdf_report.py results/<target>/<date>_<telescope>_<filter>/ [-o out.pdf]
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)

BIN_MINUTES = 10.0  # bin width for binned light curves and per-bin precision
BETA_RANGE_MIN = (10.0, 30.0)  # bin widths (min) averaged for the red-noise factor

C_RAW = "#b5b5b5"
C_BIN = "#1f4e8c"
C_COMP = "#c0504d"
A4_PORTRAIT = (8.27, 11.69)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_run(run_dir):
    """Merge photometry.npz and images.npz (images optional) into one dict."""
    run_dir = Path(run_dir)
    d = {}
    for name in ("photometry.npz", "images.npz"):
        path = run_dir / name
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                d.update({k: z[k] for k in z.files if k != "movie"})
    if "diffs" not in d:
        raise FileNotFoundError(f"No photometry.npz in {run_dir}")
    return d


def _scalar(d, key, default=None):
    """A 0-d entry as a Python value; ``default`` if missing or NaN."""
    if key not in d:
        return default
    v = np.asarray(d[key]).item()
    if isinstance(v, float) and not np.isfinite(v):
        return default
    return v


def bin_lc(t, y, minutes=BIN_MINUTES):
    """Bin into ``minutes`` windows: (t_mean, y_mean, sigma_of_mean, sigma_point, n)."""
    order = np.argsort(t)
    t, y = np.asarray(t)[order], np.asarray(y)[order]
    w = minutes / 1440
    out = []
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] - t[i] < w:
            j += 1
        seg = y[i:j][np.isfinite(y[i:j])]
        if len(seg):
            sd = np.std(seg, ddof=1) if len(seg) > 1 else np.nan
            out.append((t[i:j].mean(), seg.mean(), sd / np.sqrt(len(seg)), sd, len(seg)))
        i = j
    return tuple(np.array(c) for c in zip(*out)) if out else (np.array([]),) * 5


def ptp_sigma(y):
    """Point-to-point scatter: insensitive to slow trends and slow variability."""
    y = np.asarray(y)[np.isfinite(y)]
    return np.std(np.diff(y)) / np.sqrt(2) if len(y) > 2 else np.nan


def segments(t, max_gap_min=30.0):
    """Index arrays of runs of frames without gaps longer than ``max_gap_min``."""
    breaks = np.where(np.diff(t) * 1440 > max_gap_min)[0] + 1
    return np.split(np.arange(len(t)), breaks)


# ---------------------------------------------------------------------------
# Noise analysis
# ---------------------------------------------------------------------------
def detrended_residuals(t, y):
    """Residuals of ``y`` from a quadratic in time, per gap-free segment.

    No astrophysical model is fitted: the quadratic takes out slow trends (from
    airmass or from the source itself) but not shorter correlated noise, so any
    real variability on time scales of tens of minutes counts as red noise here.
    Segments with fewer than 20 frames are dropped.
    """
    res_t, res = [], []
    for seg in segments(t):
        seg = seg[np.isfinite(y[seg])]
        if len(seg) < 20:
            continue
        x = t[seg] - t[seg].mean()
        coef = np.polyfit(x, y[seg], 2)
        res_t.append(t[seg])
        res.append(y[seg] - np.polyval(coef, x))
    if not res:
        return np.array([]), np.array([])
    return np.concatenate(res_t), np.concatenate(res)


def rms_vs_binsize(t, res, cadence_min):
    """Time-averaging test (Pont et al. 2006; Winn et al. 2008).

    Bins the residuals in groups of N consecutive frames (within gap-free
    segments). Returns (bin width in minutes, measured RMS, white-noise
    expectation sigma_1/sqrt(N)*sqrt(M/(M-1))) and beta, the mean ratio of
    measured to expected over ``BETA_RANGE_MIN``.
    """
    sigma1 = np.std(res)
    widths, rms, expected = [], [], []
    n_max = max(1, len(res) // 8)
    for n in np.unique(np.round(np.logspace(0, np.log10(n_max), 25)).astype(int)):
        means = []
        for seg in segments(t):
            r = res[seg]
            k = len(r) // n
            if k:
                means.extend(r[: k * n].reshape(k, n).mean(axis=1))
        m = len(means)
        if m < 4:
            continue
        widths.append(n * cadence_min)
        rms.append(np.std(means))
        expected.append(sigma1 / np.sqrt(n) * np.sqrt(m / (m - 1)))
    widths, rms, expected = map(np.array, (widths, rms, expected))
    sel = (widths >= BETA_RANGE_MIN[0]) & (widths <= BETA_RANGE_MIN[1])
    beta = float(np.mean(rms[sel] / expected[sel])) if sel.any() else np.nan
    return widths, rms, expected, max(beta, 1.0) if np.isfinite(beta) else beta


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7,
            "legend.frameon": False,
        }
    )


def _xlim(ax, t, t0):
    pad = 0.02 * (t.max() - t.min())
    ax.set_xlim(t.min() - t0 - pad, t.max() - t0 + pad)


def _lc_axes(ax, t, y, t0, bins=True, color=C_BIN, ms=2):
    ax.plot(t - t0, y, ".", color=C_RAW, ms=ms, zorder=1, rasterized=True)
    if bins:
        bt, by, be, _, _ = bin_lc(t, y)
        ax.errorbar(bt - t0, by, be, fmt="o", ms=3, color=color, lw=0.8, zorder=3,
                    label=f"{BIN_MINUTES:.0f}-min bins")
    _xlim(ax, t, t0)


def _robust_ylim(ax, y, pad=0.25):
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    span = hi - lo
    ax.set_ylim(lo - pad * span, hi + pad * span)


def _info_block(fig, rect, title, rows):
    """Two-column label/value list drawn as text inside ``rect`` (figure coords)."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.text(0, 1, title, weight="bold", fontsize=9, va="top", transform=ax.transAxes)
    dy = 1 / (len(rows) + 1.5)
    for i, (k, v) in enumerate(rows):
        y = 1 - (i + 1.4) * dy
        ax.text(0, y, k, color="0.35", va="top", transform=ax.transAxes)
        ax.text(0.42, y, v, va="top", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(pdf, d, ctx):
    fig = plt.figure(figsize=A4_PORTRAIT)
    t, y, t0 = ctx["t"], ctx["lc"], ctx["t0"]
    fig.suptitle(ctx["title"], fontsize=12, weight="bold", y=0.975)
    fig.text(0.5, 0.952, ctx["subtitle"], ha="center", fontsize=8, color="0.35")

    _info_block(fig, [0.07, 0.675, 0.42, 0.255], "Observation", ctx["obs_rows"])
    _info_block(fig, [0.53, 0.675, 0.44, 0.255], "Photometry & precision", ctx["phot_rows"])

    ax = fig.add_axes([0.09, 0.375, 0.86, 0.235])
    _lc_axes(ax, t, y, t0)
    _robust_ylim(ax, y)
    ax.axhline(1, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_ylabel("Relative flux")
    ax.set_xlabel(ctx["xlabel"])
    ax.legend(loc="lower left")
    ax.set_title("Target light curve (aperture %d, %d comparison stars)" % (ctx["ap"], len(ctx["comps"])),
                 loc="left")
    # UTC on the top axis: hours since 0h UTC of the first night date (may pass 24)
    midnight = ctx["utc_midnight"] + ctx["utc_offset"] - t0
    sec = ax.secondary_xaxis("top", functions=(lambda x: (x - midnight) * 24,
                                               lambda h: h / 24 + midnight))
    sec.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda h, _: f"{int(h % 24):02d}:{int(round(h % 1 * 60)) % 60:02d}"))
    sec.set_xlabel("UTC", fontsize=7)
    sec.tick_params(labelsize=7)

    ax = fig.add_axes([0.18, 0.015, 0.66, 0.295])
    finder_chart(ax, d, ctx)
    pdf.savefig(fig)
    plt.close(fig)


def finder_chart(ax, d, ctx):
    stack = d.get("stack")
    coords = np.asarray(d["ref_coords"])
    ti, ap = ctx["ti"], ctx["ap"]
    if stack is None:
        ax.text(0.5, 0.5, "no stack image saved", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return
    stack = np.asarray(stack, float)
    vmin, vmax = ZScaleInterval().get_limits(stack[np.isfinite(stack)][::97])
    ax.imshow(stack, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    r = float(np.nanmedian(np.asarray(d["aperture_radii"])[:, ap]))
    ann = np.nanmedian(np.asarray(d["annulus_radii"]), axis=0)
    x, yv = coords[ti]
    ax.add_patch(Circle((x, yv), r, fill=False, color="#1f77b4", lw=1.2))
    for a in ann:
        ax.add_patch(Circle((x, yv), a, fill=False, color="#1f77b4", lw=0.6, ls="--"))
    ax.annotate(ctx["target"], (x, yv), xytext=(8, 8), textcoords="offset points",
                color="#1f77b4", fontsize=7, weight="bold")
    for s in ctx["comps"]:
        cx, cy = coords[s]
        ax.add_patch(Circle((cx, cy), r, fill=False, color=C_COMP, lw=0.8))
        ax.annotate(str(s), (cx, cy), xytext=(5, 4), textcoords="offset points",
                    color=C_COMP, fontsize=5.5)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.7")
    ax.set_title("Finder chart: target (blue, aperture and annulus), comparison stars (red)",
                 loc="left", fontsize=8)
    _compass_and_scale(ax, d, stack.shape)


def _compass_and_scale(ax, d, shape):
    h, w = shape
    scale = _scalar(d, "platescale")
    x0, y0 = 0.05 * w, 0.06 * h
    if scale:
        px = 60 / scale  # 1 arcmin
        ax.plot([x0, x0 + px], [y0, y0], color="k", lw=1.5)
        ax.text(x0 + px / 2, y0 + 0.015 * h, "1′", ha="center", va="bottom", fontsize=7)
    wcs_hdr = _scalar(d, "wcs_header")
    if not wcs_hdr:
        return
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS

    wcs = WCS(fits.Header.fromstring(wcs_hdr, sep="\n"))
    cx, cy = 0.92 * w, 0.10 * h
    c = wcs.pixel_to_world(cx, cy)
    arm = 0.06 * h
    for label, position_angle in (("N", 0), ("E", 90)):
        other = c.directional_offset_by(position_angle * u.deg, 1 * u.arcmin)
        px, py = wcs.world_to_pixel(other)
        v = np.array([px - cx, py - cy])
        v = v / np.hypot(*v) * arm
        ax.annotate("", (cx + v[0], cy + v[1]), (cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
        ax.text(cx + 1.25 * v[0], cy + 1.25 * v[1], label, ha="center", va="center", fontsize=7)


def page_noise(pdf, d, ctx):
    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.suptitle(f"{ctx['title']} — noise", fontsize=11, weight="bold", y=0.975)

    # RMS vs bin size
    ax = fig.add_axes([0.12, 0.55, 0.80, 0.36])
    widths, rms, expected, beta = ctx["rms_curve"]
    if len(widths):
        ax.loglog(widths, rms * 1e3, "o-", ms=3, color=C_BIN, label="measured")
        ax.loglog(widths, expected * 1e3, "--", color="0.4", label="white noise (σ₁/√N)")
        ax.axvspan(*BETA_RANGE_MIN, color="0.9", lw=0, label="β range")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Bin width (min)")
        ax.set_ylabel("RMS of residuals (ppt)")
        ax.legend(loc="lower left")
        ax.set_title(f"Time-averaging test: β = {beta:.2f}" if np.isfinite(beta) else
                     f"Time-averaging test: β not measured (run too short: bins reach "
                     f"{widths.max():.0f} min, β uses {BETA_RANGE_MIN[0]:.0f}–{BETA_RANGE_MIN[1]:.0f} min)",
                     loc="left")
        ax.text(0.98, 0.98, ctx["residual_note"], ha="right", va="top", fontsize=6.5,
                color="0.4", transform=ax.transAxes)
    else:
        ax.axis("off")
        ax.set_title("Time-averaging test", loc="left")
        ax.text(0.0, 0.85, "Not enough frames (needs > 40 in segments of\n"
                "20 or more) to measure red noise.", transform=ax.transAxes, va="top")

    # Aperture choice
    ax = fig.add_axes([0.12, 0.08, 0.80, 0.36])
    r_fwhm = ctx["r_fwhm"]
    ax.plot(r_fwhm, ctx["ap_ptp"] * 1e3, "o-", ms=3, color=C_BIN, label="point-to-point")
    ax.plot(r_fwhm, ctx["ap_bin"] * 1e3, "s-", ms=3, color=C_COMP, label=f"within {BIN_MINUTES:.0f}-min bins")
    ax.axvline(r_fwhm[ctx["ap"]], color="0.4", ls="--", lw=0.8, label=f"chosen ({r_fwhm[ctx['ap']]:.2f}× FWHM)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel("Aperture radius (× median FWHM)")
    ax.set_ylabel("Target scatter per point (ppt)")
    ax.set_title("Aperture choice", loc="left")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def page_systematics(pdf, d, ctx):
    t, t0 = ctx["t"], ctx["t0"]
    scale = _scalar(d, "platescale", 1.0)
    exptime = np.asarray(d["exptime"], float)
    peak = np.asarray(d["peak"])[:, ctx["ti"]]
    alc = np.asarray(d["alc"])[ctx["ap"]] if "alc" in d else None
    rows = [
        ("Relative flux", ctx["lc"], None),
        ("Comparison flux", alc / np.nanmax(alc) if alc is not None else np.full(len(t), np.nan), None),
        ("Airmass", np.asarray(d["airmass"], float), None),
        ("FWHM (″)", np.asarray(d["fwhm"], float) * scale, None),
        ("Sky (ADU s⁻¹ px⁻¹)", np.asarray(d["sky"], float) / exptime, None),
        ("Drift (px)", np.asarray(d["dx"], float), np.asarray(d["dy"], float)),
        ("Target peak (ADU)", peak, None),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=A4_PORTRAIT, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1, 1, 1]})
    fig.suptitle(f"{ctx['title']} — systematics", fontsize=11, weight="bold", y=0.985)
    for ax, (label, y, y2) in zip(axes, rows):
        if label == "Relative flux":
            _lc_axes(ax, t, y, t0)
            _robust_ylim(ax, y)
            ax.legend(loc="lower left", ncol=2)
        else:
            ax.plot(t - t0, y, ".", ms=2, color="#4c4c4c", rasterized=True, label="Δx" if y2 is not None else None)
            if y2 is not None:
                ax.plot(t - t0, y2, ".", ms=2, color="#e08214", rasterized=True, label="Δy")
                ax.legend(loc="upper right", ncol=2, markerscale=3)
            _xlim(ax, t, t0)
        if label == "Comparison flux":
            ax.text(0.01, 0.05, "summed comparison stars, relative to their maximum: "
                    "extinction and transparency (clouds)", transform=ax.transAxes, fontsize=6, color="0.4")
        if label.startswith("Target peak"):
            sat = _scalar(d, "saturation")
            if sat:
                top = np.nanmax(y)
                ax.text(0.01, 0.9, f"max {top:.0f} ADU = {100 * top / sat:.0f}% of the saturation limit "
                        f"({sat:.0f} ADU)", transform=ax.transAxes, fontsize=6.5,
                        color=C_COMP if top > 0.8 * sat else "0.3", va="top")
                if top > 0.5 * sat:
                    ax.axhline(sat, color=C_COMP, lw=0.8, ls="--")
        ax.set_ylabel(label)
    axes[-1].set_xlabel(ctx["xlabel"])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def page_comparisons(pdf, d, ctx):
    t, t0, ap, ti = ctx["t"], ctx["t0"], ctx["ap"], ctx["ti"]
    diffs = np.asarray(d["diffs"])[ap]
    w = np.asarray(d["weights"])[ap]
    coords = np.asarray(d["ref_coords"])
    flux = np.nanmedian((np.asarray(d["fluxes"]) - np.asarray(d["bkg"]))[:, :, ap], axis=0)
    comps = sorted(ctx["comps"], key=lambda s: -w[s])
    lcs = [("target", ctx["lc"], C_BIN, "")]
    wsum = w[comps].sum()
    for s in comps:
        dist = np.hypot(*(coords[s] - coords[ti]))
        lcs.append((f"#{s}", diffs[s], C_COMP,
                    f"w {100 * w[s] / wsum:.0f}%  F/F★ {flux[s] / flux[ti]:.2f}  "
                    f"{dist:.0f} px  σ {ptp_sigma(diffs[s]) * 1e3:.1f} ppt"))
    # Space rows by the typical binned noise (robust MAD), so short features stay
    # visible; large excursions such as twilight ramps may cross into the next row.
    def _mad(y):
        by = bin_lc(t, y)[1]
        return 1.4826 * np.nanmedian(np.abs(by - np.nanmedian(by)))

    spacing = float(np.clip(12 * np.nanmedian([_mad(l[1]) for l in lcs]), 0.01, 0.1))
    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.suptitle(f"{ctx['title']} — comparison stars", fontsize=11, weight="bold", y=0.985)
    ax = fig.add_axes([0.08, 0.05, 0.55, 0.90])
    for i, (name, y, color, note) in enumerate(lcs):
        off = -i * spacing
        raw = np.where(np.abs(y - 1) < 0.6 * spacing, y, np.nan)  # keep raw points in their row
        ax.plot(t - t0, raw + off, ".", ms=1, color=C_RAW, rasterized=True)
        bt, by, be, _, _ = bin_lc(t, y)
        ax.errorbar(bt - t0, by + off, be, fmt="o", ms=1.8, lw=0.5, color=color)
        ax.text(1.01, 1 + off, f"{name}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=6.5, color=color, weight="bold")
        ax.text(1.09, 1 + off, note, transform=ax.get_yaxis_transform(), va="center",
                fontsize=6, color="0.3")
    _xlim(ax, t, t0)
    ax.set_ylim(1 - (len(lcs) - 0.4) * spacing, 1 + 0.8 * spacing)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(ctx["xlabel"])
    ax.set_title(f"Each star divided by the ensemble of the other comparison stars; "
                 f"offset {spacing * 1e3:.0f} ppt", loc="left", fontsize=7.5)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_context(d):
    ti = int(d["target_index"])
    ap = int(d["best_aperture"])
    w = np.asarray(d["weights"])[ap]
    comps = [int(s) for s in np.nonzero(w > 0)[0] if s != ti]
    if "bjd_tdb" in d:
        t, tlabel = np.asarray(d["bjd_tdb"], float), "BJD_TDB (mid-exposure)"
    else:
        t, tlabel = np.asarray(d["time"], float), "JD_UTC (exposure start)"
    order = np.argsort(t)
    t = t[order]
    d = {**d}
    for k in ("time", "bjd_tdb", "airmass", "fwhm", "sky", "dx", "dy", "exptime", "peak",
              "fluxes", "bkg", "aperture_radii", "annulus_radii", "stars_in_exp"):
        if k in d and np.ndim(d[k]) >= 1 and len(d[k]) == len(order):
            d[k] = np.asarray(d[k])[order]
    for k in ("diffs", "alc"):
        if k in d:
            d[k] = np.asarray(d[k])[..., order]
    t0 = float(np.floor(t.min()))
    # offset between the plotted time scale and UTC, for the top axis
    utc_offset = float(np.nanmedian(t - np.asarray(d["time"], float))) if "bjd_tdb" in d else 0.0
    lc = np.asarray(d["diffs"])[ap, ti]
    target = str(d["target"])

    # Precision
    _, _, be, bsd, _ = bin_lc(t, lc)
    sigma_point = ptp_sigma(lc)
    res_t, res = detrended_residuals(t, lc)
    cadence = float(np.median(np.diff(t))) * 1440
    rms_curve = rms_vs_binsize(res_t, res, cadence) if len(res) > 40 else (np.array([]),) * 3 + (np.nan,)
    beta = rms_curve[3]
    residual_note = "residuals: light curve minus a quadratic in time\nper gap-free segment (no model fitted)"

    # Aperture curve
    diffs = np.asarray(d["diffs"])
    r_fwhm = np.nanmedian(np.asarray(d["aperture_radii"]), axis=0) / np.nanmedian(d["fwhm"])
    ap_ptp = np.array([ptp_sigma(diffs[a, ti]) for a in range(diffs.shape[0])])
    ap_bin = np.array([np.nanmedian(bin_lc(t, diffs[a, ti])[3]) for a in range(diffs.shape[0])])

    scale = _scalar(d, "platescale", np.nan)
    gain = _scalar(d, "gain")
    rn, dc = _scalar(d, "read_noise"), _scalar(d, "dark_current")
    r_px = float(np.nanmedian(np.asarray(d["aperture_radii"])[:, ap]))
    ann = np.nanmedian(np.asarray(d["annulus_radii"]), axis=0)
    fwhm = np.asarray(d["fwhm"], float)
    airmass = np.asarray(d["airmass"], float)
    exptime = float(np.nanmedian(d["exptime"]))
    t_utc = Time(np.asarray(d["time"], float)[[0, -1]], format="jd", scale="utc")
    ra, dec = _scalar(d, "target_ra"), _scalar(d, "target_dec")
    n_light = _scalar(d, "n_light_frames")
    coords_str = "—"
    if ra is not None:
        from astropy.coordinates import SkyCoord

        coords_str = SkyCoord(ra, dec, unit="deg").to_string("hmsdms", precision=1, sep=":")

    def e(v, unit):
        return f" ({v * gain:.3g} {unit})" if gain else ""

    obs_rows = [
        ("Target", f"{target}  {coords_str}"),
        ("Telescope / camera", f"{d.get('telescope', '—')}" + (f", D = {_scalar(d, 'aperture_diameter'):.2f} m" if _scalar(d, 'aperture_diameter') else "")
         + (f" / {_scalar(d, 'camera')}" if _scalar(d, 'camera') else "")),
        ("Filter / exposure", f"{d.get('band', '—')} / {exptime:.0f} s"),
        ("Night (UTC)", f"{t_utc[0].iso[:16]} – {t_utc[1].iso[11:16]}"),
        ("Duration", f"{(t.max() - t.min()) * 24:.2f} h"),
        ("Frames used", f"{len(t)}" + (f" of {n_light}" if n_light else "")),
        ("Airmass", f"{np.nanmin(airmass):.2f} – {np.nanmax(airmass):.2f}"),
        ("FWHM (median, range)", f"{np.nanmedian(fwhm) * scale:.2f}″ ({np.nanpercentile(fwhm, 5) * scale:.1f}–{np.nanpercentile(fwhm, 95) * scale:.1f}″)"),
        ("Plate scale", f"{scale:.3f} ″/px"),
        ("Gain / read noise", (f"{gain:.2f} e⁻/ADU / " if gain else "— / ") + f"{rn:.2f} ADU{e(rn, 'e⁻')}" if rn is not None else "—"),
        ("Dark current", f"{dc:.3g} ADU/s{e(dc, 'e⁻/s')}" if dc is not None else "—"),
    ]
    sig_bin = float(np.nanmedian(be))
    phot_rows = [
        ("Aperture", f"#{ap}: r = {r_px:.1f} px = {r_px * scale:.1f}″ = {r_fwhm[ap]:.2f}× FWHM"),
        ("Annulus", f"{ann[0]:.1f} – {ann[1]:.1f} px"),
        ("Comparison stars", f"{len(comps)}" + (f" ({_scalar(d, 'n_rejected_comps')} rejected as noisy/variable)" if _scalar(d, 'n_rejected_comps') is not None else "")),
        ("σ per point (ptp)", f"{sigma_point * 1e3:.2f} ppt"),
        (f"σ per {BIN_MINUTES:.0f}-min bin", f"{sig_bin * 1e3:.2f} ppt (σ/√n, white noise)"),
        ("Red-noise factor β", f"{beta:.2f}" if np.isfinite(beta) else "—"),
        ("σ per bin × β", f"{sig_bin * beta * 1e3:.2f} ppt" if np.isfinite(beta) else "—"),
    ]

    return {
        "t": t, "t0": t0, "lc": lc, "ti": ti, "ap": ap, "comps": comps, "target": target,
        "xlabel": f"{tlabel} − {t0:.0f}", "utc_offset": utc_offset,
        "utc_midnight": float(np.floor(np.asarray(d["time"], float).min() - 0.5) + 0.5),
        "title": f"{target} · {d.get('telescope', '')} · {d.get('band', '')} · {d.get('date', '')}",
        "subtitle": "Differential aperture photometry — eloy pipeline",
        "obs_rows": obs_rows, "phot_rows": phot_rows, "rms_curve": rms_curve,
        "residual_note": residual_note, "r_fwhm": r_fwhm, "ap_ptp": ap_ptp, "ap_bin": ap_bin,
        "sigma_point": sigma_point,
    }, d


def build_pdf(run_dir, out=None):
    """Write ``summary.pdf`` for the run in ``run_dir``; return its path."""
    run_dir = Path(run_dir)
    raw = load_run(run_dir)
    ctx, d = build_context(raw)
    out = Path(out) if out else run_dir / "summary.pdf"
    _style()
    with PdfPages(out) as pdf:
        page_overview(pdf, d, ctx)
        page_noise(pdf, d, ctx)
        page_systematics(pdf, d, ctx)
        page_comparisons(pdf, d, ctx)
        info = pdf.infodict()
        info["Title"] = ctx["title"]
        info["Subject"] = "Differential photometry summary"
    logger.info("Saved summary PDF to %s", out)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Build the PDF summary of a pipeline run.")
    ap.add_argument("run", help="pipeline.py run folder")
    ap.add_argument("-o", "--out", help="output PDF (default: summary.pdf in the run folder)")
    args = ap.parse_args()
    build_pdf(args.run, args.out)


if __name__ == "__main__":
    main()
