"""Differential aperture photometry pipeline for SPECULOOS-South observations.

The pipeline:
  1. Indexes a night's FITS files by date, type, and target (``find_files``).
  2. Builds master bias/dark/flat calibration frames.
  3. Calibrates a reference image, detects stars, and solves for a WCS so the
     science target can be matched to its Gaia source.
  4. Processes the science (light) frames in parallel across worker processes:
     calibrates, aligns to the reference, centroids, and performs aperture
     photometry with annulus background subtraction, while co-adding the aligned
     frames into a stack.
  5. Runs differential photometry and plots the binned light curve.

Run as a script (``python pipeline.py``); the work lives in ``main()`` so the
module can also be imported without triggering a full run, which is required for
the multiprocessing workers (spawned workers re-import this module).
"""

import argparse
import logging
import math
import os
import re
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from glob import glob
from multiprocessing import Manager as _MPManager
from pathlib import Path

import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.mast import Mast
from dateutil import parser
from eloy import (
    alignment,
    calibration,
    centroid,
    detection,
    flux,
    photometry,
    psf,
    utils,
)
from eloy.ballet import Ballet
from skimage.transform import AffineTransform

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from twirl.utils import compute_wcs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quieten noisy third-party loggers so only the pipeline's own logs show: jax
# probes for a (nonexistent) TPU backend and falls back to CPU, and each worker
# logs its Hugging Face fetch of the Ballet model. These are INFO-level chatter,
# not errors. Set at module import so spawned workers inherit it too.
for _noisy in ("jax", "absl", "httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
# Hugging Face warns once per worker about anonymous downloads; silence both its
# logger and the matching UserWarning.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests.*")

# --- Detection / photometry parameters -------------------------------------
N_STARS = 100  # number of stars to track for photometry
CUTOUT_SHAPE = (31, 31)  # cutout size (pixels) used for PSF/centroiding
TRIM = 0  # pixels trimmed from each image edge before processing
SATURATED = 65000 * 0.9  # peak counts considered saturated after calibration (ADU)

N_STARS_ALIGN = 12  # number of brightest stars used to solve frame alignment
RELATIVE_RADII = np.linspace(0.5, 5, 40)  # aperture radii, in units of FWHM
MAX_DRIFT_PX = 20  # skip frames whose median alignment drift exceeds this (pixels)

N_WORKERS = os.cpu_count() or 4  # worker processes for the parallel main loop

MOVIE_MAX_PX = 512  # longest side of the saved night-movie frames (downsampled)
DEFAULT_PLATESCALE = 0.348  # arcsec/pixel, fallback when optics keywords absent

# --- FITS header keywords --------------------------------------------------
KW_DATE_OBS = "DATE-OBS"  # UTC timestamp of the exposure
KW_EXPTIME = "EXPTIME"  # exposure time (s)
KW_IMAGETYP = "IMAGETYP"  # frame type (Light/Dark/Flat/Bias Frame)
KW_OBJECT = "OBJECT"  # target name
KW_FILTER = "FILTER"  # filter name
KW_LONGITUDE = "LONG-OBS"  # site longitude (deg)
KW_RA = "RA"  # pointing right ascension
KW_DEC = "DEC"  # pointing declination
KW_AIRMASS = "AIRMASS"  # airmass at exposure
KW_FOCALLEN = "FOCALLEN"  # focal length (unit read from its header comment)
KW_XPIXSZ = "XPIXSZ"  # pixel size (microns)
KW_TELESCOP = "TELESCOP"  # telescope name
KW_EGAIN = "EGAIN"  # native gain (e-/ADU); not GAIN, which many cameras use for the gain setting
KW_XBINNING = "XBINNING"  # binning factor along x
KW_YBINNING = "YBINNING"  # binning factor along y

# --- FITS frame types (values of the IMAGETYP keyword) ---------------------
TYPE_LIGHT = "Light Frame"
TYPE_DARK = "Dark Frame"
TYPE_FLAT = "Flat Frame"
TYPE_BIAS = "Bias Frame"

# Observation-specific configuration is passed on the command line; see main().


def find_files(glob_pattern: str) -> list[str]:
    """Index FITS files matching ``glob_pattern`` by observing night and type.

    Each file's header is read to determine its observing date, frame type and
    target. Because observations can run past local midnight, the timestamp is
    shifted to the local noon-to-noon "observing night" before the date is taken.

    Returns:
        observations: nested dict ``{date: {frame_type: [sorted file paths]}}``.
        files_meta: dict ``{file path: {date, datetime, type, object, filter,
        exptime}}``.
    """
    files_meta = {}
    observations = defaultdict(lambda: defaultdict(list))

    logger.info("Finding files for pattern '%s'...", glob_pattern)
    files = glob(glob_pattern, recursive=True)
    logger.info("Found %d files", len(files))

    for file in files:
        header = fits.getheader(file)
        file_date = parser.parse(header.get(KW_DATE_OBS, ""))
        exptime = header.get(KW_EXPTIME, np.nan)
        image_type = header.get(KW_IMAGETYP, "unknown")
        object_name = header.get(KW_OBJECT, image_type)
        filter_name = header.get(KW_FILTER, "unknown")
        site_lon = header.get(KW_LONGITUDE, 0)

        # Shift the UTC timestamp to the local noon-to-noon "observing night" so
        # frames taken either side of local midnight share one date. longitude/15
        # converts degrees east to an hour offset; -12 anchors the rollover at noon.
        night_date = (file_date + timedelta(hours=site_lon / 15 - 12)).date()

        files_meta[file] = {
            "date": night_date,
            "datetime": file_date,
            "type": image_type,
            "object": object_name,
            "filter": filter_name,
            "exptime": exptime,
        }
        observations[night_date][image_type].append(file)

    # sort the files by datetime
    for date in observations:
        for obs_type in observations[date]:
            observations[date][obs_type].sort(key=lambda f: files_meta[f]["datetime"])

    # sort files_meta by datetime too, for easier debugging
    files_meta = dict(sorted(files_meta.items(), key=lambda item: item[1]["datetime"]))

    return observations, files_meta


def bad_pixel_map(dark_files, master_bias=None, std_factor_upper=3, std_factor_lower=3):
    """Build a bad-pixel mask from the individual matching dark frames.

    Each dark frame is bias-subtracted and normalised to ADU/s, then the
    per-pixel median across all frames is taken as the master dark. Pixels that
    deviate from the global median by more than N standard deviations are flagged
    as hot (``> median + std_factor_upper * std``) or dead
    (``< median - std_factor_lower * std``). Plain ``np.std`` / ``np.median`` are
    used (no sigma-clipping) to match the reference SPECULOOS implementation.

    Args:
        dark_files: Paths to the individual dark FITS files.
        master_bias: Bias frame subtracted from each dark before normalisation,
            or ``None`` to skip bias subtraction.
        std_factor_upper: Hot-pixel threshold, in standard deviations.
        std_factor_lower: Dead-pixel threshold, in standard deviations.

    Returns:
        Boolean mask the shape of one frame (``True`` = bad pixel).
    """
    logger.info("Building bad-pixel map from %d dark frame(s)", len(dark_files))

    stack = []
    for f in dark_files:
        data = fits.getdata(f).astype(float)
        exptime = fits.getheader(f)[KW_EXPTIME]
        if master_bias is not None:
            data = data - master_bias
        stack.append(data / exptime)

    master_dark = np.median(stack, axis=0)
    logger.info(
        "Master dark (ADU/s): min=%.4f  max=%.4f  median=%.4f  std=%.4f",
        master_dark.min(),
        master_dark.max(),
        np.median(master_dark),
        np.std(master_dark),
    )

    threshold = np.std(master_dark)
    median = np.median(master_dark)

    hot = master_dark > median + std_factor_upper * threshold
    dead = master_dark < median - std_factor_lower * threshold
    mask = hot | dead

    logger.info(
        "Bad-pixel map: %d hot (>%.0f σ above)  %d dead (>%.0f σ below)  |  "
        "%d total (%.2f %%)",
        int(hot.sum()),
        std_factor_upper,
        int(dead.sum()),
        std_factor_lower,
        int(mask.sum()),
        100.0 * mask.mean(),
    )
    return mask


def estimate_dark_current(dark_files, bias_files) -> float:
    """Typical dark current of one pixel, in ADU/s.

    Raw frames are integers, and on a cooled camera the dark signal per exposure
    is far below 1 ADU. A median over frames or pixels then snaps to a
    quantisation step (often exactly 0), so the median master dark cannot be
    used. Instead the bias and dark frames are mean-combined, and a sigma-clipped
    mean is taken over pixels; the clipping removes hot pixels and cosmic rays.
    """
    bias = 0.0
    for f in bias_files:
        bias = bias + fits.getdata(f).astype(float)
    bias = bias / len(bias_files) if bias_files else 0.0

    dark = 0.0
    for f in dark_files:
        data, header = fits.getdata(f, header=True)
        dark = dark + (data.astype(float) - bias) / header[KW_EXPTIME]
    dark = dark / len(dark_files)

    mean, _, _ = sigma_clipped_stats(dark, sigma=3)
    return float(mean)


def interpolate_bad_pixels(image, mask, max_adu=None):
    """Replace bad pixels with the mean of their valid cardinal neighbours.

    Marks bad pixels, unphysical negatives, and pixels above the camera's
    physical maximum (``max_adu``) as NaN, then fills each NaN position with the
    mean of its four N/S/E/W neighbours. The working array is updated in-place as
    each pixel is filled, so pixels on the edge of a cluster can supply values to
    their still-NaN interior neighbours in the same pass.

    Args:
        image: 2-D image to correct.
        mask: Boolean mask the same shape as ``image`` (``True`` = bad pixel).
        max_adu: Camera full-well / physical maximum in ADU, or ``None`` to skip
            the upper-bound check. Any calibrated pixel above this value is
            interpolated over, catching hot pixels, cosmic rays, and flat
            divide-by-near-zero artefacts the dark-based mask misses.

    Returns:
        The corrected image as a 2-D float array.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"interpolate_bad_pixels: image shape {image.shape} != "
            f"mask shape {mask.shape}"
        )

    data = image.astype(float, copy=True)

    # Poison bad pixels, unphysical negatives, and above-maximum pixels.
    data[mask] = np.nan
    data[data < 0] = np.nan
    if max_adu is not None:
        n_above = int(np.sum(data > max_adu))
        if n_above:
            logger.debug(
                "interpolate_bad_pixels: %d pixel(s) above max_adu=%.0f "
                "(hot pixels / cosmic rays / flat artefacts) — interpolating",
                n_above,
                max_adu,
            )
        data[data > max_adu] = np.nan

    nans = np.argwhere(np.isnan(data))
    if len(nans) == 0:
        return data

    # Pad with NaN so boundary pixels need no special-casing.
    padded = np.pad(data, 1, constant_values=np.nan)

    for i, j in nans + 1:  # +1 to account for the padding offset
        mean = np.nanmean(
            [
                padded[i - 1, j],  # north
                padded[i + 1, j],  # south
                padded[i, j - 1],  # west
                padded[i, j + 1],  # east
            ]
        )
        padded[i, j] = mean
        data[i - 1, j - 1] = mean

    n_unfilled = int(np.sum(np.isnan(data)))
    if n_unfilled:
        logger.warning(
            "interpolate_bad_pixels: %d pixel(s) remain NaN after interpolation "
            "(isolated cluster with no valid cardinal neighbours)",
            n_unfilled,
        )

    return data


def calibration_sequence(
    file, master_dark, master_flat, master_bias, bad_pixels=None, max_adu=None
):
    """Calibrate a single frame and measure its stars and PSF width.

    Applies bias/dark/flat calibration, trims the edges, detects stars, removes
    saturated sources, and estimates the seeing (FWHM) from an empirical PSF
    built by median-stacking normalised cutouts of the detected stars.

    Returns a 6-tuple ``(calibrated_data, region_coords_filtered,
    region_coords, fwhm, regions, header)``. The FITS header is returned so
    callers can read keywords (e.g. DATE-OBS, AIRMASS) without re-opening the
    file. If fewer than three stars are detected the same 6-tuple shape is
    returned with empty/``None`` placeholders so callers can unpack it
    unconditionally.
    """
    data = fits.getdata(file)
    header = fits.getheader(file)
    exposure = header[KW_EXPTIME]

    # bias/dark/flat correction, then drop the noisy frame edges
    calibrated_data = calibration.calibrate(
        data, exposure, master_dark, master_flat, master_bias
    )
    if bad_pixels is not None:
        fname = Path(file).name
        _pre = calibrated_data
        n_pre_nan = int(np.sum(~np.isfinite(_pre)))
        n_pre_zero = int(np.sum(_pre == 0))
        logger.debug(
            "%s pre-interpolation:  %d NaN/Inf, %d exact zeros, "
            "min=%.1f  max=%.1f  median=%.1f",
            fname,
            n_pre_nan,
            n_pre_zero,
            float(np.nanmin(_pre)),
            float(np.nanmax(_pre)),
            float(np.nanmedian(_pre)),
        )
        calibrated_data = interpolate_bad_pixels(
            calibrated_data, bad_pixels, max_adu=max_adu
        )
        n_post_nan = int(np.sum(~np.isfinite(calibrated_data)))
        n_post_zero = int(np.sum(calibrated_data == 0))
        logger.debug(
            "%s post-interpolation: %d NaN/Inf, %d exact zeros, "
            "min=%.1f  max=%.1f  median=%.1f",
            fname,
            n_post_nan,
            n_post_zero,
            float(np.nanmin(calibrated_data)),
            float(np.nanmax(calibrated_data)),
            float(np.nanmedian(calibrated_data)),
        )
        if n_post_nan > n_pre_nan:
            logger.warning(
                "%s: bad pixel interpolation increased NaN count %d → %d; "
                "likely caused by NaN/Inf in neighbouring valid pixels",
                fname,
                n_pre_nan,
                n_post_nan,
            )
    if TRIM > 0:
        calibrated_data = calibrated_data[TRIM:-TRIM, TRIM:-TRIM]

    regions = detection.stars_detection(calibrated_data)

    # need at least 3 stars for a usable PSF / alignment solution
    if len(regions) < 3:
        logger.warning("Fewer than 3 stars detected in %s", Path(file).name)
        return None, [], None, None, [], header

    # (x, y) pixel coordinates of every detected star
    region_coords = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
    cutouts = utils.cutout(calibrated_data, region_coords, (50, 50))

    # discard saturated stars so they don't bias the PSF or photometry
    not_saturated = cutouts.max(axis=(1, 2)) < SATURATED
    cutouts = cutouts[not_saturated]
    region_coords_filtered = region_coords[not_saturated]
    regions = [r for r, keep in zip(regions, not_saturated) if keep]

    # empirical PSF: median of peak-normalised cutouts -> Gaussian FWHM
    cutouts_normalized = cutouts / np.nanmax(cutouts, (1, 2))[:, None, None]
    epsf = np.nanmedian(cutouts_normalized, 0)
    psf_params = psf.fit_gaussian(epsf)
    fwhm = psf.gaussian_sigma_to_fwhm * np.mean(
        [psf_params["sigma_x"], psf_params["sigma_y"]]
    )

    # free the large intermediate arrays before returning
    del (
        cutouts_normalized,
        data,
        cutouts,
        epsf,
    )

    return calibrated_data, region_coords_filtered, region_coords, fwhm, regions, header


def _safe_name(value) -> str:
    """Make ``value`` safe as one path component (e.g. "ETH Hongg" -> "ETH-Hongg", "i'" -> "i").

    Letters, digits and ``. _ + -`` are kept; every other run of characters
    becomes a single ``-``.
    """
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", str(value)).strip("-") or "unknown"


def run_output_dir(output_dir, target, day_date, telescope, band) -> Path:
    """Folder for one pipeline run: ``<output_dir>/<target>/<date>_<telescope>_<band>/``.

    Grouping by target first keeps all nights of one object together; the date
    leads the run folder so nights sort chronologically.
    """
    run_dir = (
        Path(output_dir)
        / _safe_name(target)
        / f"{day_date}_{_safe_name(telescope)}_{_safe_name(band)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def effective_gain(header, native_gain=None):
    """Gain of one stored (binned) pixel in e-/ADU, or None if unknown.

    ``native_gain`` (e-/ADU before binning) comes from ``--gain`` or the
    ``EGAIN`` keyword. The camera averages each binning block, so one stored ADU
    stands for XBINNING x YBINNING native ADU.
    """
    if native_gain is None:
        native_gain = header.get(KW_EGAIN)
    if native_gain is None:
        return None
    binning = int(header.get(KW_XBINNING, 1)) * int(header.get(KW_YBINNING, 1))
    return float(native_gain) * binning


def extract_plate_scale(header) -> float:
    """Derive the plate scale (degrees/pixel) from a FITS header's optics keywords.

    Uses the detector pixel size (``XPIXSZ``, microns) and focal length
    (``FOCALLEN``), reading the ``FOCALLEN`` comment to decide whether it is in
    millimetres or metres. Falls back to ``DEFAULT_PLATESCALE`` if the keywords
    are missing or unusable.
    """
    try:
        focallen_comment = header.comments[KW_FOCALLEN].lower()
        # FOCALLEN is usually in mm; treat anything else as metres.
        focallen_m = header[KW_FOCALLEN] * (
            1e-3
            if ("mm" in focallen_comment or "millimeter" in focallen_comment)
            else 1.0
        )
        pixel_m = header[KW_XPIXSZ] * 1e-6  # microns -> metres
        return np.degrees(np.arctan(pixel_m / focallen_m))
    except (KeyError, TypeError, ZeroDivisionError):
        logger.warning(
            "Optics keywords (%s/%s) absent; falling back to %.3f arcsec/pixel",
            KW_XPIXSZ,
            KW_FOCALLEN,
            DEFAULT_PLATESCALE,
        )
        return DEFAULT_PLATESCALE / 3600  # arcsec/pixel -> deg/pixel


# --- Reference-star catalogue for the WCS fit --------------------------------
# VizieR (CDS) answers in ~1 s and sorts by magnitude on the server. The ESA Gaia
# archive is often slow or times out (it is being prepared for Gaia DR4), so it
# is only the last fallback. VizieR results are cached on disk by astroquery, so
# repeat runs of the same field need no network.
VIZIER_SERVERS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")
ESA_TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
CATALOG_TIMEOUT = 30  # seconds without an answer before a service counts as failed
CATALOG_ROUNDS = 2  # times to try the whole list of services
GAIA_DR3_EPOCH = 2016.0  # Julian year of Gaia DR3 positions


def _float_column(table, name):
    """Table column as a float array, with masked entries set to NaN."""
    return np.ma.asarray(table[name].data, dtype=float).filled(np.nan)


def _vizier_stars(server, center, radius_deg, infrared, n):
    """Brightest ``n`` stars in the cone from one VizieR server: (ra, dec, pmra, pmdec)."""
    import astropy.units as u
    from astroquery.vizier import Vizier

    if infrared:  # 2MASS, sorted by J; no proper motions
        catalog, ra_col, dec_col, cols = "II/246/out", "RAJ2000", "DEJ2000", ["+Jmag"]
    else:  # Gaia DR3, sorted by G
        catalog, ra_col, dec_col, cols = "I/355/gaiadr3", "RA_ICRS", "DE_ICRS", ["pmRA", "pmDE", "+Gmag"]
    vizier = Vizier(
        columns=[ra_col, dec_col, *cols],
        row_limit=n,
        timeout=CATALOG_TIMEOUT,
        vizier_server=server,
    )
    tables = vizier.query_region(center, radius=radius_deg * u.deg, catalog=catalog)
    if len(tables) == 0:
        raise RuntimeError(f"no {catalog} sources returned")
    t = tables[0]
    ra, dec = _float_column(t, ra_col), _float_column(t, dec_col)
    if infrared:
        return ra, dec, np.zeros_like(ra), np.zeros_like(ra)
    return ra, dec, _float_column(t, "pmRA"), _float_column(t, "pmDE")


def _esa_stars(center, radius_deg, n):
    """Brightest ``n`` Gaia DR3 stars from the ESA archive: (ra, dec, pmra, pmdec).

    Sorting is done here, not with ORDER BY: on the archive, ORDER BY over a cone
    makes the query far slower. A plain HTTP request is used so the timeout is
    enforced (astroquery.gaia can wait forever).
    """
    import requests
    from astropy.table import Table

    query = (
        "SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {center.ra.deg}, {center.dec.deg}, {radius_deg})) "
        "AND phot_g_mean_mag < 16"
    )
    r = requests.post(
        ESA_TAP_SYNC,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=(10, CATALOG_TIMEOUT),
    )
    r.raise_for_status()
    t = Table.read(r.text, format="ascii.csv")
    t = t[np.argsort(_float_column(t, "phot_g_mean_mag"))][:n]
    cols = ("ra", "dec", "pmra", "pmdec")
    return tuple(_float_column(t, c) for c in cols)


def reference_star_radecs(center, radius_deg, infrared=False, obs_jyear=None, n=200):
    """RA/Dec (deg) of the ``n`` brightest catalogue stars in a cone, brightest first.

    Uses Gaia DR3 (2MASS when ``infrared``, which ranks stars by J as the camera
    sees them). Services are tried in order, VizieR first; the whole list is
    tried ``CATALOG_ROUNDS`` times. When ``obs_jyear`` is given, Gaia positions
    are moved from the DR3 epoch to that date with their proper motions.

    Returns an ``(n, 2)`` array. Raises ``RuntimeError`` if every service fails.
    """
    services = [
        (f"VizieR {s}", lambda s=s: _vizier_stars(s, center, radius_deg, infrared, n))
        for s in VIZIER_SERVERS
    ]
    if not infrared:  # the ESA archive has no fast 2MASS-ordered query
        services.append(("ESA Gaia archive", lambda: _esa_stars(center, radius_deg, n)))

    for attempt in range(1, CATALOG_ROUNDS + 1):
        for name, fetch in services:
            try:
                ra, dec, pmra, pmdec = fetch()
            except Exception as e:
                logger.warning("%s query failed: %s: %s", name, type(e).__name__, e)
                continue
            if len(ra) == 0:
                logger.warning("%s returned no stars", name)
                continue
            logger.info("Got %d reference stars from %s", len(ra), name)
            if obs_jyear is not None and not infrared:
                # pmRA already includes cos(dec); both are in mas/yr.
                years = obs_jyear - GAIA_DR3_EPOCH
                ra = ra + years * np.nan_to_num(pmra) / 3.6e6 / np.cos(np.radians(dec))
                dec = dec + years * np.nan_to_num(pmdec) / 3.6e6
            return np.column_stack([ra, dec])
        if attempt < CATALOG_ROUNDS:
            logger.warning("All catalogue services failed; retrying in %d s", 10 * attempt)
            time.sleep(10 * attempt)
    raise RuntimeError("Could not get reference stars from any catalogue service")


_ZSCALE = ZScaleInterval()  # DS9-style display limits, shared with night_report.py


def _movie_frame(image, max_px=MOVIE_MAX_PX):
    """Downsample and ZScale-stretch a frame into a uint8 night-movie thumbnail.

    Strides the image down so its longest side is at most ``max_px``, then maps
    the ZScale (DS9-style) low/high limits to 0..255 so faint stars stay visible
    without the bright ones washing out. Matches night_report.py's stack stretch.
    """
    step = max(1, int(np.ceil(max(image.shape) / max_px)))
    small = image[::step, ::step]
    finite = small[np.isfinite(small)]
    if finite.size == 0:
        return np.zeros(small.shape, dtype=np.uint8)
    lo, hi = _ZSCALE.get_limits(finite)
    norm = np.clip((small - lo) / (hi - lo + 1e-9), 0, 1)
    # NaN pixels (unfillable bad pixels) survive the clip; map them to black so
    # the uint8 cast is well-defined.
    return (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8)


# ===========================================================================
# Parallel worker: per-frame photometry
# ===========================================================================
# The main loop is embarrassingly parallel across frames. We split the frames
# into one chunk per worker process; each worker holds its own master frames,
# reference positions, and CNN model (set up once via ``_init_worker``), and
# accumulates a partial stack that the parent sums with the others.
_worker = {}  # per-process state, populated by ``_init_worker``


def _init_worker(
    master_dark,
    master_flat,
    master_bias,
    ref_coords,
    ref_coords_all,
    ref_reference,
    stack_shape,
    bp_mask=None,
):
    """Set up a worker process once, before it handles any frames.

    Building the Ballet model is expensive, so it is done once per process here
    rather than once per frame.
    """
    # Keep the CNN single-threaded: parallelism comes from the process pool, so
    # letting each worker's torch spin up many threads would oversubscribe the CPU.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    _worker.update(
        dark=master_dark,
        flat=master_flat,
        bias=master_bias,
        ref_coords=ref_coords,
        ref_coords_all=ref_coords_all,
        ref_reference=ref_reference,
        stack_shape=stack_shape,
        bp_mask=bp_mask,
        cnn=Ballet(),
    )


def _process_frames(files, progress_q=None):
    """Process a chunk of light frames in a worker process.

    Returns ``(results, partial_stack, thumbs)``. After every frame — whether
    it was processed, skipped, or errored — one sentinel is pushed to
    ``progress_q`` so the main process can update tqdm in real time.
    """
    dark, flat, bias = _worker["dark"], _worker["flat"], _worker["bias"]
    ref_coords = _worker["ref_coords"]  # filtered (non-saturated), for photometry
    ref_coords_all = _worker["ref_coords_all"]  # all detected stars, for alignment
    ref_reference = _worker["ref_reference"]
    cnn = _worker["cnn"]

    partial_stack = np.zeros(_worker["stack_shape"], dtype=float)
    results = []
    thumbs = []  # downsampled uint8 frames for the night movie (frame order)

    for file in files:
        filename = Path(file).name
        try:
            # Calibrate this frame and measure its stars and seeing.
            calibrated_data, coords, coords_all, fwhm, regions, header = (
                calibration_sequence(
                    file, dark, flat, bias, _worker["bp_mask"], max_adu=SATURATED
                )
            )

            # Need a handful of stars to solve the alignment reliably.
            if len(coords_all) < 5:
                logger.warning("Skipping %s: only %d stars", filename, len(coords_all))
                continue

            # Solve the rotation/translation that maps this frame to the
            # reference using all detected stars, then project the filtered
            # reference positions into this frame for photometry.
            R = alignment.rotation_matrix(
                coords_all[0:N_STARS_ALIGN],
                ref_coords_all[0:N_STARS_ALIGN],
                ref_reference,
            )
            # rotation_matrix returns R mapping reference -> this frame, so apply
            # it forward (NOT .inverse) to project the reference positions into
            # this frame. Using .inverse placed apertures at ref - drift instead
            # of ref + drift, i.e. off by 2x the drift, which broke photometry on
            # any drifted frame while looking fine on perfectly-aligned ones.
            transform = AffineTransform(R)
            aligned_coords = transform(ref_coords)[0:N_STARS]
            dx, dy = np.median(aligned_coords - ref_coords[0:N_STARS], 0)

            drift = float(np.sqrt(dx**2 + dy**2))
            if drift > MAX_DRIFT_PX:
                logger.warning(
                    "Skipping %s: drift %.1f px (dx=%.1f dy=%.1f) exceeds "
                    "MAX_DRIFT_PX=%d — frame is too misaligned for reliable "
                    "aperture photometry",
                    filename,
                    drift,
                    dx,
                    dy,
                    MAX_DRIFT_PX,
                )
                continue

            # Refine star positions with the CNN centroider.
            centroid_coords = centroid.ballet_centroid(
                calibrated_data, aligned_coords, cnn
            )

            # Aperture photometry over a range of FWHM-scaled radii.
            apertures_radii = RELATIVE_RADII * fwhm
            ap_flux = photometry.aperture_photometry(
                calibrated_data, centroid_coords, apertures_radii
            )

            # Local sky background from a surrounding annulus (sigma-clipped
            # median), scaled to each aperture's area for subtraction.
            annulus_radii = np.max(apertures_radii), 8 * fwhm
            aperture_area = np.pi * apertures_radii**2
            bkg = photometry.annulus_sigma_clip_median(
                calibrated_data, centroid_coords, *annulus_radii
            )
            bkg = bkg[:, None] * aperture_area[None, :]

            # Peak counts per star (saturation / quality diagnostic).
            peaks = np.nanmax(
                utils.cutout(calibrated_data, aligned_coords, (25, 25)), axis=(1, 2)
            )

            # Record this frame's measurements (header already read above).
            results.append(
                {
                    "bkg": bkg,
                    "fluxes": ap_flux,
                    "fwhm": fwhm,
                    "time": Time(parser.parse(header[KW_DATE_OBS])).jd,
                    "dx": dx,
                    "dy": dy,
                    "sky": np.mean(bkg / aperture_area[None, :]),
                    "exptime": header.get(KW_EXPTIME, np.nan),
                    "airmass": header.get(KW_AIRMASS, np.nan),
                    "peak": peaks,
                    "stars_in_exp": len(coords),
                    "aperture_radii": apertures_radii,
                    "annulus_radii": annulus_radii,
                }
            )

            # Co-add into the stack; treat any remaining NaN (unfillable bad
            # pixels) as zero so they don't poison the whole stack at that position.
            partial_stack += np.nan_to_num(calibrated_data, nan=0.0)
            thumbs.append(_movie_frame(calibrated_data))
        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)
        finally:
            if progress_q is not None:
                try:
                    progress_q.put_nowait(1)
                except Exception:
                    pass

    return results, partial_stack, thumbs


def _chunks(seq, n):
    """Split ``seq`` into ``n`` roughly equal contiguous lists."""
    n = max(1, min(n, len(seq)))
    k, m = divmod(len(seq), n)
    return [seq[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _ptp_score(lc: np.ndarray) -> float:
    """Point-to-point scatter: std of adjacent differences / sqrt(2).

    Only sensitive to noise on the timescale of one frame — completely
    blind to astrophysical variability on longer timescales.
    """
    d = np.diff(lc)
    finite = d[np.isfinite(d)]
    return float(np.std(finite) / np.sqrt(2)) if len(finite) >= 2 else np.inf


def _bin_residual_score(lc: np.ndarray, t: np.ndarray, bin_minutes: float) -> float:
    """Within-bin residual scatter over fixed-width time windows.

    Bins the light curve and measures how much each point deviates from its
    bin mean. Insensitive to variability on timescales longer than the bin.
    """
    bin_width = bin_minutes / (24.0 * 60.0)  # JD
    residuals = []
    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] - t[i] < bin_width:
            j += 1
        seg = lc[i:j]
        finite = seg[np.isfinite(seg)]
        if len(finite) >= 2:
            residuals.append(finite - np.mean(finite))
        i = j
    if not residuals:
        return np.inf
    return float(np.std(np.concatenate(residuals)))


N_COMPARISONS = 25  # comparison stars: the N nearest well-behaved stars to the target
COMP_REJECT_MAD = 3.0  # reject stars this many MADs noisier than their brightness predicts
COMP_REF_FWHM = 2.0  # aperture (in FWHM) at which comparison stars are judged


def _ptp_scatter(lcs: np.ndarray) -> np.ndarray:
    """Point-to-point scatter along the last axis (insensitive to slow trends)."""
    return np.nanstd(np.diff(lcs, axis=-1), axis=-1) / np.sqrt(2)


def _ensemble_weights(norm: np.ndarray, comps, n_iter: int = 5) -> np.ndarray:
    """Inverse-variance weights for ``comps`` (Broeg 2005 iteration).

    Each star's variance is the point-to-point scatter of its light curve
    divided by the ensemble of the other comparison stars; weights are 1/var,
    which is the minimum-noise combination (eloy uses 1/sigma, which gives faint
    stars too much weight).
    """
    w = np.zeros(norm.shape[0])
    w[comps] = 1.0
    for _ in range(n_iter):
        scatter = _ptp_scatter(flux.diff(norm, w)[0])
        new = np.zeros_like(w)
        new[comps] = 1.0 / scatter[comps] ** 2
        new[~np.isfinite(new)] = 0.0
        w = new
    return w


def select_comparison_stars(
    fluxes: np.ndarray,
    target_index: int,
    coords: np.ndarray,
    ref_aperture: int,
    n_comps: int = N_COMPARISONS,
    reject_mad: float = COMP_REJECT_MAD,
) -> tuple[list, int]:
    """Choose one comparison set for all apertures, without using the target's light curve.

    Choosing stars by how much they lower the target's noise overfits: the target
    noise looks lower than it is, and the set does not hold up on other parts of
    the night (tested on split halves of two nights). Instead:

    1. Candidates are stars with valid fluxes in every frame and aperture.
    2. Noisy or variable stars are rejected: each star's scatter against the
       ensemble of the others is compared with the scatter expected for its
       brightness (a straight-line fit of log scatter vs log flux), and stars
       more than ``reject_mad`` MADs above it are dropped; repeated until stable.
    3. The ``n_comps`` stars nearest the target are kept. The PSF changes over the
       detector, so nearby stars lose a similar fraction of light outside the
       aperture as the target does when the seeing changes.

    Returns:
        ``(comps, n_rejected)``: comparison star indices, and how many stars were
        rejected as noisy or variable.
    """
    n_stars = fluxes.shape[1]
    norm_all = fluxes / np.nanmean(fluxes, axis=-1, keepdims=True)
    ok = np.isfinite(norm_all).all(axis=(0, 2))
    ok[target_index] = False

    f = fluxes[ref_aperture]
    norm = norm_all[ref_aperture]
    level = np.nanmedian(f, axis=-1)
    ok &= level > 0
    n_candidates = int(ok.sum())
    while ok.sum() > 3:
        comps = np.nonzero(ok)[0]
        scatter = _ptp_scatter(flux.diff(norm, _ensemble_weights(norm, comps))[0])
        x, y = np.log(level[comps]), np.log(scatter[comps])
        slope, offset = np.polyfit(x, y, 1)
        resid = np.log(scatter) - (slope * np.log(level) + offset)
        mad = 1.4826 * np.median(np.abs(resid[comps] - np.median(resid[comps])))
        bad = ok & (resid > reject_mad * mad)
        if not bad.any():
            break
        ok &= ~bad

    dist = np.hypot(*(coords[:n_stars] - coords[target_index]).T)
    good = np.nonzero(ok)[0]
    comps = good[np.argsort(dist[good])][:n_comps].tolist()
    return comps, n_candidates - len(good)


def differential_photometry(fluxes: np.ndarray, comps) -> tuple:
    """Differential light curves of every star against one comparison set.

    Args:
        fluxes: Background-subtracted fluxes, shape ``(n_apertures, n_stars,
            n_frames)``, frames in time order.
        comps: Comparison star indices (the same set at every aperture).

    Returns:
        ``(diffs, weights)`` with shapes ``(n_apertures, n_stars, n_frames)`` and
        ``(n_apertures, n_stars)``; weights are inverse-variance per aperture and
        zero outside ``comps``.
    """
    diffs, weights = [], []
    for f in fluxes:  # one aperture: (n_stars, n_frames)
        norm = f / np.nanmean(f, axis=-1, keepdims=True)
        w = _ensemble_weights(norm, comps)
        diffs.append(flux.diff(norm, w).reshape(f.shape))
        weights.append(w)
    return np.array(diffs), np.array(weights)


def optimal_aperture(
    diffs: np.ndarray,
    target_index: int,
    time: np.ndarray,
    bin_minutes: float = 10.0,
) -> int:
    """Select the aperture that gives the target the least short-term noise.

    Two metrics are computed on the target's light curve at each aperture:
    point-to-point scatter and the scatter within ``bin_minutes`` bins. Both only
    see noise on time scales of a few minutes, so slow astrophysical signals
    (transits lasting hours, pulsations of ~1 h) barely change them. Apertures are
    ranked on each metric and the lowest combined rank wins.

    The comparison stars are not used to score apertures: each aperture has its
    own comparison set, so their median noise mostly reflects which stars are in
    the set, not the aperture.

    Args:
        diffs: Differential light curves, shape ``(n_apertures, n_stars,
            n_frames)``.
        target_index: Column index of the science target in the star axis.
        time: JD timestamps, shape ``(n_frames,)``. Need not be sorted.
        bin_minutes: Width of the time bins used for the within-bin metric.

    Returns:
        Index of the chosen aperture along the aperture axis of ``diffs``.
    """
    order = np.argsort(time)
    t = time[order]
    lcs = diffs[:, target_index][:, order]  # (n_apertures, n_frames)

    ptp = np.array([_ptp_score(lc) for lc in lcs])
    binned = np.array([_bin_residual_score(lc, t, bin_minutes) for lc in lcs])

    def _rank(scores: np.ndarray) -> np.ndarray:
        """Dense rank: inf/NaN values get the worst (highest) rank."""
        return np.argsort(np.argsort(np.nan_to_num(scores, nan=np.inf)))

    best = int(np.argmin(_rank(ptp) + _rank(binned)))
    logger.info(
        "Optimal aperture: %d  (target point-to-point %.4f, within-bin %.4f)",
        best,
        ptp[best],
        binned[best],
    )
    return best


def _bin_lc(t_arr, y_arr, bin_min=10.0):
    """Bin (t, y) into ``bin_min``-minute windows; return (t_bin, y_bin, e_bin)."""
    w = bin_min / (24 * 60)
    order = np.argsort(t_arr)
    ts, ys = t_arr[order], y_arr[order]
    bt, by, be = [], [], []
    i = 0
    while i < len(ts):
        j = i
        while j < len(ts) and ts[j] - ts[i] < w:
            j += 1
        seg = ys[i:j]
        fin = seg[np.isfinite(seg)]
        if len(fin):
            bt.append(float(np.mean(ts[i:j])))
            by.append(float(np.mean(fin)))
            be.append(float(np.std(fin) / np.sqrt(max(1, len(fin)))))
        i = j
    return np.array(bt), np.array(by), np.array(be)


def main():
    ap = argparse.ArgumentParser(
        description="Differential aperture photometry pipeline for SPECULOOS-South."
    )
    ap.add_argument("--image_path", help="Directory containing the night's FITS files.")
    ap.add_argument("--target", help="OBJECT header value of the science target.")
    ap.add_argument(
        "--query-string",
        default=None,
        metavar="NAME",
        help="String to resolve for the target's sky coordinates (default: same as target).",
    )
    ap.add_argument(
        "--fix-bad-pixels",
        action="store_true",
        default=False,
        help=(
            "Interpolate bad pixels before photometry.  Bad pixels are identified "
            "as dark pixels that are exactly zero and flat pixels more than 5 σ "
            "from the median.  The combined mask is filled with the mean of valid "
            "3×3 neighbours."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="results",
        metavar="DIR",
        help=(
            "Root directory for outputs (created if needed; default: results). Each run "
            "writes to its own folder, <DIR>/<target>/<date>_<telescope>_<filter>/."
        ),
    )
    ap.add_argument(
        "--flat-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing flat frames that override those from image_path.  "
            "Only FITS files whose IMAGETYP header identifies them as flat frames and "
            "whose FILTER header matches the target filter are accepted."
        ),
    )
    ap.add_argument(
        "--dark-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing dark frames that override those from image_path.  "
            "Only FITS files whose IMAGETYP header identifies them as dark frames are "
            "accepted; exposure-time matching is still applied."
        ),
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=None,
        metavar="E_PER_ADU",
        help=(
            "Native (unbinned) detector gain in e-/ADU, used to show read noise and "
            "dark current in electrons. The camera is assumed to average binned "
            "pixels, so the effective gain is this times XBINNING x YBINNING. "
            "Default: the EGAIN header keyword, if present."
        ),
    )
    ap.add_argument(
        "--bias-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing bias frames that override those from image_path.  "
            "Only FITS files whose IMAGETYP header identifies them as bias frames are "
            "accepted."
        ),
    )
    ap.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="After processing, build the interactive HTML night report from the saved bundle.",
    )
    args = ap.parse_args()
    image_path = args.image_path
    target = args.target
    query_string = args.query_string if args.query_string is not None else target
    fix_bad_pixels = args.fix_bad_pixels
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index override calibration directories once, if supplied.
    if args.flat_dir:
        _obs, flat_dir_meta = find_files(f"{args.flat_dir}/*.fits")
        override_flats = [f for d in _obs.values() for f in d.get(TYPE_FLAT, [])]
        if not override_flats:
            logger.warning("--flat-dir '%s' contains no flat frames.", args.flat_dir)
    else:
        override_flats = None
        flat_dir_meta = {}

    if args.dark_dir:
        _obs, dark_dir_meta = find_files(f"{args.dark_dir}/*.fits")
        override_darks = [f for d in _obs.values() for f in d.get(TYPE_DARK, [])]
        if not override_darks:
            logger.warning("--dark-dir '%s' contains no dark frames.", args.dark_dir)
    else:
        override_darks = None
        dark_dir_meta = {}

    if args.bias_dir:
        _obs, bias_dir_meta = find_files(f"{args.bias_dir}/*.fits")
        override_bias = [f for d in _obs.values() for f in d.get(TYPE_BIAS, [])]
        if not override_bias:
            logger.warning("--bias-dir '%s' contains no bias frames.", args.bias_dir)
    else:
        override_bias = None

    # =======================================================================
    # Index the night and build master calibration frames
    # =======================================================================
    observations, files_meta = find_files(f"{image_path}/*.fits")

    # Science (light) frames of the target.
    light_frames = [
        f
        for f, meta in files_meta.items()
        if meta["type"] == TYPE_LIGHT and meta["object"] == target
    ]
    if not light_frames:
        raise ValueError(f"No light frames found for target '{target}'")
    logger.info("Target '%s': %d light frames", target, len(light_frames))

    # The pipeline reduces one observing night. If the target's frames span
    # several nights (e.g. a directory holding more than one), keep the night
    # with the most frames so the selection is deterministic.
    frames_per_night = Counter(files_meta[f]["date"] for f in light_frames)
    day_date, _ = frames_per_night.most_common(1)[0]
    if len(frames_per_night) > 1:
        logger.warning(
            "Target light frames span %d nights %s; using %s (most frames) "
            "and ignoring the rest.",
            len(frames_per_night),
            sorted(frames_per_night),
            day_date,
        )
        light_frames = [f for f in light_frames if files_meta[f]["date"] == day_date]
    logger.info("Observing night: %s", day_date)

    # Determine all filters present in the chosen night's light frames.
    target_filters = sorted({files_meta[f]["filter"] for f in light_frames})
    if len(target_filters) > 1:
        logger.info(
            "Found %d filters %s for target '%s'; running the pipeline once per filter.",
            len(target_filters),
            target_filters,
            target,
        )

    # Master bias is filter-independent; build it once for the night.
    bias = (
        override_bias
        if override_bias is not None
        else observations[day_date][TYPE_BIAS]
    )
    if override_bias is not None:
        logger.info("Using %d bias frames from --bias-dir.", len(bias))
    BIAS = calibration.master_bias(files=bias)

    # Read noise: std(B1 - B2) / sqrt(2) in ADU (filter-independent).
    if len(bias) >= 2:
        b1 = fits.getdata(bias[0]).astype(float)
        b2 = fits.getdata(bias[1]).astype(float)
        read_noise = float(np.std(b1 - b2) / np.sqrt(2))
        logger.info("Read noise estimate: %.2f ADU", read_noise)
    else:
        read_noise = float("nan")
        logger.warning("Need at least 2 bias frames to estimate read noise; skipping")

    for target_filter in target_filters:
        if len(target_filters) > 1:
            logger.info("--- Processing filter: %s ---", target_filter)

        # Light frames restricted to this filter.
        filter_light_frames = [
            f for f in light_frames if files_meta[f]["filter"] == target_filter
        ]
        logger.info(
            "Filter '%s': %d light frames", target_filter, len(filter_light_frames)
        )

        # Exposure times for this filter's frames (for dark matching).
        target_exptimes = {files_meta[f]["exptime"] for f in filter_light_frames}

        # Master calibration frames for the chosen night. Flats are restricted to the
        # target's filter; darks/bias are filter-independent. Darks are matched to the
        # target's exposure time(s) where possible, falling back to all of the night's
        # darks otherwise.
        flats = [
            f
            for f in observations[day_date][TYPE_FLAT]
            if files_meta[f]["filter"] == target_filter
        ]
        dark_pool = (
            override_darks
            if override_darks is not None
            else observations[day_date][TYPE_DARK]
        )
        dark_meta = dark_dir_meta if override_darks is not None else files_meta
        matching_darks = [
            f for f in dark_pool if dark_meta[f]["exptime"] in target_exptimes
        ]
        if override_darks is not None:
            logger.info(
                "Using %d dark frames from --dark-dir (%d match target exposure time(s) %s).",
                len(dark_pool),
                len(matching_darks),
                target_exptimes,
            )
        if not matching_darks:
            logger.warning(
                "No dark frames with matching exposure time(s) %s found; "
                "using all available dark frames.",
                target_exptimes,
            )
            darks = dark_pool
        else:
            logger.info(
                "%d dark frames with matching exposure time(s) %s found for target.",
                len(matching_darks),
                target_exptimes,
            )
            darks = matching_darks

        if override_flats is not None:
            flats = [
                f for f in override_flats if flat_dir_meta[f]["filter"] == target_filter
            ]
            logger.info(
                "Using %d '%s' flat frames from --flat-dir.",
                len(flats),
                target_filter,
            )
        if not flats:
            raise ValueError(
                f"No '{target_filter}' flat frames found for {day_date}; cannot build master flat"
            )

        logger.info(
            "Building master frames (%d dark, %d flat in '%s')",
            len(darks),
            len(flats),
            target_filter,
        )
        DARK = calibration.master_dark(bias=BIAS, files=darks)
        FLAT = calibration.master_flat(files=flats, dark=DARK, bias=BIAS)

        # Not np.median(DARK): see estimate_dark_current for why that gives 0.
        dark_current = estimate_dark_current(darks, bias)
        logger.info("Dark current estimate: %.3g ADU/s", dark_current)

        # Bad-pixel mask (computed per filter; passed to every calibration_sequence call).
        bp_mask = bad_pixel_map(darks, master_bias=BIAS) if fix_bad_pixels else None
        if bp_mask is not None:
            logger.info(
                "Bad-pixel correction enabled: %d pixels flagged (%.2f %%)",
                int(bp_mask.sum()),
                100.0 * bp_mask.mean(),
            )

        # =======================================================================
        # Reference frame: detect stars, solve WCS, locate the target
        # =======================================================================
        # Use the middle frame of the night as the alignment/astrometry reference.
        reference_image = filter_light_frames[len(filter_light_frames) // 2]
        logger.info("Using reference image: %s", Path(reference_image))

        ref_data, ref_coords, ref_coords_all, ref_fwhm, _, _ = calibration_sequence(
            reference_image, DARK, FLAT, BIAS, bp_mask, max_adu=SATURATED
        )
        ref_reference = alignment.twirl_reference(ref_coords_all[0:N_STARS_ALIGN])

        # Compute a WCS by matching detected stars to a Gaia query of the field.
        ref_header = fits.getheader(reference_image)
        pixel_scale = extract_plate_scale(ref_header)  # plate scale in degrees/pixel
        fov = ref_data.shape[1] * pixel_scale  # field-of-view width in degrees
        center = SkyCoord(ref_header[KW_RA], ref_header[KW_DEC], unit="deg")
        logger.info("Plate scale: %.4f arcsec/pixel", pixel_scale * 3600)

        # Query a cone of radius 0.75 * FOV (as before) to allow for pointing error.
        logger.info("Querying reference stars and solving WCS...")
        use_tmass = target_filter in ["zYJ", "Y", "J", "H", "Ks"]
        if use_tmass:
            logger.info("Target filter is '%s'; using 2MASS for WCS fit", target_filter)
        all_radecs = reference_star_radecs(
            center,
            0.75 * fov,
            infrared=use_tmass,
            obs_jyear=Time(parser.parse(ref_header[KW_DATE_OBS])).jyear,
        )
        # Match the 20 brightest detected stars to the 20 brightest catalogue stars.
        wcs = compute_wcs(ref_coords_all[0:20], all_radecs[0:20], tolerance=10)
        if wcs is None:
            logger.error(
                "WCS fit failed: no match between the brightest detected and "
                "catalogue stars (clouds, wrong pointing, or too few stars?)"
            )
            exit(1)

        # Check if platescale from WCS is consistent with optics keywords
        # Use the full pixel->sky matrix, not PC1_1 alone: PC1_1 is scale*cos(rotation)
        # and goes to ~0 when the camera is rotated ~90 deg on the sky.
        wcs_platescale = np.mean(proj_plane_pixel_scales(wcs))  # degrees/pixel
        logger.info("WCS plate scale: %.4f arcsec/pixel", wcs_platescale * 3600)
        if abs(wcs_platescale - pixel_scale) / (pixel_scale) > 0.1:
            logger.error(
                "WCS plate scale %.4f arcsec/pixel differs from optics-derived "
                "plate scale %.4f arcsec/pixel by more than 10%%;",
                wcs_platescale * 3600,
                pixel_scale * 3600,
            )
            exit(1)

        # Convert reference-frame star pixel positions to sky coordinates via the
        # WCS, resolve the target's Gaia coordinates, and find which star it is.
        stars_radec = wcs.pixel_to_world(*ref_coords.T)

        mast = Mast()
        target_radec = mast.resolve_object(query_string)
        target_index = int(target_radec.match_to_catalog_sky(stars_radec)[0])
        logger.info("Target matched to star index %d", target_index)

        # =======================================================================
        # Main loop: per-frame photometry, parallelised across worker processes
        # =======================================================================
        stack = np.zeros_like(ref_data, dtype=float)  # co-added, aligned science stack
        data = defaultdict(list)  # per-frame measurements, keyed by quantity
        movie = []  # downsampled uint8 frames (frame order), for the night movie

        chunks = _chunks(filter_light_frames, N_WORKERS)
        logger.info(
            "Processing %d frames across %d workers...",
            len(filter_light_frames),
            len(chunks),
        )
        # Per-frame progress: workers signal this queue once per frame (success,
        # skip, or error). A drain thread reads it and updates tqdm independently
        # of when whole chunks complete, giving smooth and accurate progress.
        # Manager().Queue() produces a proxy object that is picklable across the
        # spawn boundary used by macOS — plain multiprocessing.Queue is not.
        stop_drain = threading.Event()

        with (
            _MPManager() as _manager,
            tqdm(total=len(filter_light_frames), unit="frame") as pbar,
        ):
            progress_q = _manager.Queue()

            def _drain():
                while not stop_drain.is_set():
                    try:
                        progress_q.get(timeout=0.2)
                        pbar.update(1)
                    except Exception:
                        pass

            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()

            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_init_worker,
                initargs=(
                    DARK,
                    FLAT,
                    BIAS,
                    ref_coords,
                    ref_coords_all,
                    ref_reference,
                    stack.shape,
                    bp_mask,
                ),
            ) as pool:
                futures = [
                    pool.submit(_process_frames, chunk, progress_q) for chunk in chunks
                ]
                for future in as_completed(futures):
                    results, partial_stack, thumbs = future.result()
                    stack += partial_stack
                    movie.extend(thumbs)
                    for result in results:
                        for key, value in result.items():
                            data[key].append(value)

            stop_drain.set()
            drain_thread.join(timeout=2)

        # Convert the per-frame lists into stacked arrays for saving/analysis.
        for k, v in data.items():
            data[k] = np.array(v)
        movie = np.array(movie)  # (n_frames, h, w) uint8

        # Put frames in time order. Workers return them in the order they finish,
        # which changes between runs, and the comparison weights and the aperture
        # choice measure noise from consecutive frames, so they depend on this order.
        order = np.argsort(data["time"])
        for k in data:
            data[k] = data[k][order]
        if len(movie) == len(order):
            movie = movie[order]

        telescope_name = ref_header.get(KW_TELESCOP, "unknown")
        gain = effective_gain(ref_header, args.gain)
        if gain is None:
            logger.info("Gain unknown (no --gain or EGAIN); noise stays in ADU")
        else:
            logger.info(
                "Effective gain %.2f e-/ADU: read noise %.1f e-, dark current %.3g e-/s "
                "per stored pixel",
                gain,
                read_noise * gain,
                dark_current * gain,
            )
        run_dir = run_output_dir(
            output_dir, target, day_date, telescope_name, target_filter
        )

        # =======================================================================
        # Differential photometry
        # =======================================================================
        # Background-subtracted fluxes, shaped (apertures, stars, frames).
        fluxes = (data["fluxes"] - data["bkg"]).T

        # One comparison set for all apertures, chosen without the target's light
        # curve, judged at the aperture nearest COMP_REF_FWHM x FWHM.
        radius_fwhm = np.nanmedian(data["aperture_radii"], axis=0) / np.nanmedian(
            data["fwhm"]
        )
        ref_aperture = int(np.argmin(np.abs(radius_fwhm - COMP_REF_FWHM)))
        comps, n_rejected = select_comparison_stars(
            fluxes, target_index, ref_coords, ref_aperture
        )
        if not comps:
            raise RuntimeError("No usable comparison stars found")
        logger.info(
            "Comparison stars: %d nearest well-behaved stars (%d rejected as noisy "
            "or variable): %s",
            len(comps),
            n_rejected,
            comps,
        )
        diffs, weights = differential_photometry(fluxes, comps)

        # Pick the aperture that minimises the target's light-curve scatter.
        best_aperture = optimal_aperture(
            diffs, target_index, data["time"], bin_minutes=10.0
        )
        logger.info("Best aperture index: %d", best_aperture)

        # Artificial (comparison) light curve per aperture: the weighted mean of the
        # normalised comparison fluxes used to detrend the target (Broeg 2005). The
        # comparison weights differ per aperture, so this is shape (apertures, frames).
        norm_fluxes = fluxes / np.nanmean(fluxes, axis=-1, keepdims=True)
        wsum = weights.sum(axis=-1, keepdims=True)
        alc = np.einsum("as,asf->af", weights, norm_fluxes) / np.where(
            wsum == 0, np.nan, wsum
        )

        # =======================================================================
        # Save results: each array goes into exactly one file
        # =======================================================================
        # photometry.npz: everything measured or derived, at full precision (small).
        photometry_file = run_dir / "photometry.npz"
        np.savez_compressed(
            photometry_file,
            ref_coords=ref_coords,
            target_index=target_index,
            diffs=diffs,
            weights=weights,
            alc=alc,
            best_aperture=best_aperture,
            target=target,
            date=str(day_date),
            band=target_filter,
            telescope=telescope_name,
            platescale=pixel_scale * 3600,  # arcsec/pixel
            read_noise=read_noise,  # ADU
            dark_current=dark_current,  # ADU/s
            gain=np.nan if gain is None else gain,  # e-/ADU per stored pixel
            **data,
        )
        logger.info("Saved photometry to %s", photometry_file)

        # images.npz: the pictures the report shows. float32 keeps ~7 significant
        # digits, far finer than the noise, at half the size of float64.
        images_file = run_dir / "images.npz"
        np.savez_compressed(
            images_file,
            stack=stack.astype(np.float32),
            master_bias=np.asarray(BIAS, dtype=np.float32),
            master_dark=np.asarray(DARK, dtype=np.float32),
            master_flat=np.asarray(FLAT, dtype=np.float32),
            movie=movie,
        )
        logger.info(
            'Saved images to %s (build the report with `uv run night_report.py "%s"`)',
            images_file,
            run_dir,
        )

        # =======================================================================
        # Diagnostic figures
        # =======================================================================
        platescale_as = pixel_scale * 3600  # arcsec / pixel

        t_jd = data["time"]
        jd0 = int(np.floor(t_jd.min()))
        # Frames are already time-sorted above; this keeps the plots safe if that
        # ever changes.
        _order = np.argsort(t_jd)
        t_plot = (t_jd - jd0)[_order]

        target_lc = diffs[best_aperture, target_index][_order]
        bt_t, by_t, be_t = _bin_lc(t_plot, target_lc)

        # Comparison stars with non-zero weight at the best aperture.
        comp_idx = [
            i
            for i in range(diffs.shape[1])
            if i != target_index and weights[best_aperture, i] > 0
        ]
        w_total = weights[best_aperture, comp_idx].sum() or 1.0

        # ── Figure 1: target LC + comparison star subplots ───────────────────
        n_comps = len(comp_idx)
        n_cols = min(4, max(1, n_comps))
        n_rows_comp = math.ceil(n_comps / n_cols) if n_comps else 0

        fig1 = plt.figure(figsize=(max(8, 3.5 * n_cols), 3.5 + 2.2 * n_rows_comp))
        gs1 = gridspec.GridSpec(
            1 + n_rows_comp,
            n_cols,
            figure=fig1,
            height_ratios=[3] + [1.8] * n_rows_comp,
            hspace=0.45,
            wspace=0.35,
        )

        ax_top = fig1.add_subplot(gs1[0, :])
        ax_top.scatter(
            t_plot,
            target_lc,
            s=4,
            c="0.65",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        ax_top.errorbar(
            bt_t,
            by_t,
            be_t,
            fmt="o",
            ms=5,
            color="#2166ac",
            elinewidth=1,
            capsize=2,
            label="10-min bins",
            zorder=3,
        )
        ax_top.axhline(1, lw=0.8, ls="--", color="0.45")
        ax_top.set_ylabel("Diff. flux")
        ax_top.set_title(
            f"{target}  ·  {target_filter}  ·  aperture {best_aperture}  ·  {day_date}",
            fontsize=10,
        )
        ax_top.legend(fontsize=8, frameon=False)
        if n_rows_comp:
            ax_top.tick_params(labelbottom=False)
        else:
            ax_top.set_xlabel(f"JD − {jd0}")

        for k, ci in enumerate(comp_idx):
            row = 1 + k // n_cols
            col = k % n_cols
            ax = fig1.add_subplot(gs1[row, col])
            lc_c = diffs[best_aperture, ci][_order]
            bt_c, by_c, be_c = _bin_lc(t_plot, lc_c)
            ax.scatter(
                t_plot, lc_c, s=2, c="0.7", alpha=0.4, linewidths=0, rasterized=True
            )
            ax.errorbar(
                bt_c,
                by_c,
                be_c,
                fmt="o",
                ms=3,
                color="#d6604d",
                elinewidth=0.6,
                capsize=1.5,
                zorder=3,
            )
            w_frac = weights[best_aperture, ci] / w_total
            ax.set_title(f"comp #{ci}  w = {w_frac:.3f}", fontsize=7)
            ax.set_ylim(0.96, 1.04)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("Diff. flux", fontsize=7)
            if row == n_rows_comp:
                ax.set_xlabel(f"JD − {jd0}", fontsize=7)

        lc_path = run_dir / "lightcurve.pdf"
        fig1.savefig(lc_path, bbox_inches="tight")
        logger.info("Saved light-curve figure to %s", lc_path)
        plt.close(fig1)

        # ── Figure 2: target LC + systematics ────────────────────────────────
        fwhm_as = data["fwhm"][_order] * platescale_as
        sky_adu_s = data["sky"][_order] / data["exptime"][_order]

        fig2, axes2 = plt.subplots(
            5,
            1,
            figsize=(10, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5, 1.5], "hspace": 0.06},
        )

        # Target light curve
        axes2[0].scatter(
            t_plot,
            target_lc,
            s=4,
            c="0.65",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        axes2[0].errorbar(
            bt_t,
            by_t,
            be_t,
            fmt="o",
            ms=5,
            color="#2166ac",
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
        axes2[0].axhline(1, lw=0.8, ls="--", color="0.45")
        axes2[0].set_ylabel("Diff. flux")
        axes2[0].set_title(
            f"{target}  ·  {target_filter}  ·  aperture {best_aperture}  ·  {day_date}",
            fontsize=10,
        )

        # FWHM
        fin = np.isfinite(fwhm_as)
        bt_s, by_s, _ = _bin_lc(t_plot[fin], fwhm_as[fin])
        axes2[1].scatter(
            t_plot[fin],
            fwhm_as[fin],
            s=3,
            c="0.7",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        axes2[1].plot(bt_s, by_s, "o-", ms=3, lw=0.9, color="#4393c3")
        axes2[1].set_ylabel('FWHM (")')

        # Sky background
        fin = np.isfinite(sky_adu_s)
        bt_s, by_s, _ = _bin_lc(t_plot[fin], sky_adu_s[fin])
        axes2[2].scatter(
            t_plot[fin],
            sky_adu_s[fin],
            s=3,
            c="0.7",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        axes2[2].plot(bt_s, by_s, "o-", ms=3, lw=0.9, color="#762a83")
        axes2[2].set_ylabel(r"Sky (ADU s$^{-1}$ px$^{-1}$)")

        # Centroid drift dx and dy on the same panel
        for y_arr, color, label in [
            (data["dx"][_order], "#1b7837", "$\Delta x$"),
            (data["dy"][_order], "#e08214", "$\Delta y$"),
        ]:
            fin = np.isfinite(y_arr)
            bt_s, by_s, _ = _bin_lc(t_plot[fin], y_arr[fin])
            axes2[3].scatter(
                t_plot[fin],
                y_arr[fin],
                s=3,
                alpha=0.35,
                linewidths=0,
                color=color,
                rasterized=True,
            )
            axes2[3].plot(bt_s, by_s, "o-", ms=3, lw=0.9, color=color, label=label)
        axes2[3].axhline(0, lw=0.7, ls="--", color="0.5")
        axes2[3].set_ylabel("Centroid drift (px)")
        axes2[3].legend(fontsize=8, frameon=False, ncol=2)

        # Airmass
        airmass = data["airmass"][_order]
        fin = np.isfinite(airmass)
        axes2[4].plot(t_plot[fin], airmass[fin], "o-", ms=3, lw=0.9, color="#636363")
        axes2[4].set_ylabel("Airmass")
        axes2[4].set_xlabel(f"JD − {jd0}")

        for ax in axes2:
            ax.tick_params(labelsize=9)

        syst_path = run_dir / "systematics.pdf"
        fig2.savefig(syst_path, bbox_inches="tight")
        logger.info("Saved systematics figure to %s", syst_path)
        plt.close(fig2)

        # =======================================================================
        # Optional: build the interactive HTML night report from the saved bundle
        # =======================================================================
        if args.report:
            logger.info("Building interactive night report...")
            # Imported lazily so the multiprocessing workers don't pay for
            # night_report's (imageio/PIL) imports on every spawn.
            from night_report import build_report

            build_report(run_dir)

        logger.info("All outputs for filter '%s' are in %s/", target_filter, run_dir)


if __name__ == "__main__":
    main()
