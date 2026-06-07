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
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from glob import glob
from multiprocessing import Manager as _MPManager
from pathlib import Path

import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
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
from photutils.detection import DAOStarFinder
from skimage.transform import AffineTransform

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from twirl.queries import gaia_radecs
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
N_STARS = 400  # number of stars to track for photometry
CUTOUT_SHAPE = (31, 31)  # cutout size (pixels) used for PSF/centroiding
TRIM = 0  # pixels trimmed from each image edge before processing
SATURATED = 10000 * 0.9  # peak counts above which a star is treated as saturated

N_STARS_ALIGN = 12  # number of brightest stars used to solve frame alignment
RELATIVE_RADII = np.linspace(0.5, 5, 40)  # aperture radii, in units of FWHM
MAX_DRIFT_PX = 20  # skip frames whose median alignment drift exceeds this (pixels)

N_WORKERS = os.cpu_count() or 4  # worker processes for the parallel main loop

MOVIE_MAX_PX = 512  # longest side of the saved night-movie frames (downsampled)
DEFAULT_PLATESCALE = 0.348  # arcsec/pixel, fallback when optics keywords absent

USE_TMASS = True  # whether to query 2MASS for WCS-solving reference stars (else Gaia)

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
    files_meta = defaultdict(dict)
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
        site_lat = header.get(KW_LONGITUDE, np.nan)

        # because some observations are taken over midnight
        day_date = file_date + timedelta(hours=site_lat / 15 - 12)

        # for speculoos the type would be in header[KW_IMAGETYP]
        files_meta[file].update(
            {
                "date": day_date.date(),
                "datetime": file_date,
                "type": image_type,
                "object": object_name,
                "filter": filter_name,
                "exptime": exptime,
            }
        )
        observations[day_date.date()][files_meta[file]["type"]].append(file)

    # sort the files by datetime
    for date in observations:
        for obs_type in observations[date]:
            observations[date][obs_type].sort(key=lambda f: files_meta[f]["datetime"])

    return observations, files_meta


def find_stars(
    data: np.ndarray,
    threshold: float = 5.0,
    peak_threshold: float | None = None,
    fwhm: float = 5.0,
    saturation_limit: float | None = None,
) -> np.ndarray:
    """
    Find stars using DAOStarFinder algorithm.

    Uses the photutils DAOStarFinder algorithm to detect point sources in
    astronomical images. The function performs background subtraction and
    returns star coordinates sorted by brightness.

    Parameters:
        data (np.ndarray): The 2D image data array.
        threshold (float, optional): Detection threshold in units of background
            standard deviation. Higher values detect fewer, brighter stars.
            Defaults to 5.0.
        fwhm (float, optional): Expected Full Width at Half Maximum of stars
            in pixels. Should match the typical seeing conditions. Defaults to 5.0.
        peak_threshold (float, optional): Threshold for the peak value of detected stars.
            Stars with peak values below this limit will be excluded. Defaults to None.
        saturation_limit (float, optional): Saturation limit for star detection.
            Stars with flux above this limit will be excluded. Defaults to None.

    Returns:
        np.ndarray: Array of detected star coordinates sorted by brightness.
            Shape is (N, 2) where N is the number of stars, and each row is (x, y).
            Returns an empty array if no stars are found.
    """
    # Calculate background statistics
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    # Use DAOStarFinder for star detection
    dao_find = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold * std,
        exclude_border=True,
        min_separation=2 * fwhm,
    )
    dao_sources = dao_find(data)

    if dao_sources is None or len(dao_sources) == 0:
        return np.array([]).reshape(0, 2)

    # Sort sources by flux (brightness) in descending order
    sorted_indices = np.argsort(dao_sources["flux"])[::-1]
    dao_sources = dao_sources[sorted_indices]

    # Filter sources based on peak value
    if peak_threshold is None:
        peak_threshold = threshold

    dao_sources = dao_sources[dao_sources["peak"] > mean + peak_threshold * std]

    # Filter sources based on saturation limit
    if saturation_limit is not None:
        dao_sources = dao_sources[dao_sources["peak"] < saturation_limit]

    # Convert to (x, y) coordinates
    coordinates = np.column_stack([dao_sources["xcentroid"], dao_sources["ycentroid"]])

    # get to similar output format as the old code, which is a list of regionprops objects with .centroid attribute
    regions = []
    for x, y in coordinates:
        region = type("Region", (), {"centroid": (y, x)})()
        regions.append(region)

    return np.array(coordinates), regions


def bad_pixel_map(dark_files, master_bias=None, std_factor_upper=3, std_factor_lower=3):
    """Build a bad-pixel mask from the individual matching dark frames.

    Replicates the validated SPECULOOS detection logic: each dark frame is
    bias-subtracted and normalised to ADU/s, then the per-pixel median across
    all frames is taken as the master dark.  Pixels that deviate from the
    global median by more than N standard deviations are flagged:

    * **Hot pixels**  — ``master_dark > median + std_factor_upper * std``
    * **Dead pixels** — ``master_dark < median - std_factor_lower * std``

    Plain ``np.std`` / ``np.median`` are used (no sigma-clipping) to stay
    faithful to the reference implementation.

    Parameters
    ----------
    dark_files        : list of str  — paths to the individual dark FITS files
    master_bias       : 2-D ndarray or None  — subtracted from each dark before
                        normalisation; pass ``None`` to skip bias subtraction
    std_factor_upper  : float  — hot-pixel threshold  (default 3)
    std_factor_lower  : float  — dead-pixel threshold (default 3)

    Returns
    -------
    mask : bool ndarray  (True = bad pixel)
    """
    logger.info("Building bad-pixel map from %d dark frame(s)", len(dark_files))

    stack = []
    for f in dark_files:
        data = fits.getdata(f).astype(float)
        exptime = fits.getheader(f)[KW_EXPTIME]
        # if master_bias is not None:
        #     data = data - master_bias
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


def interpolate_bad_pixels(image, mask, max_adu=None):
    """Replace bad pixels with the mean of their valid cardinal neighbours.

    Marks bad pixels, unphysical negatives, and pixels above the camera's
    physical maximum (``max_adu``) as NaN, then fills each NaN position with
    the mean of its four N/S/E/W neighbours.  The working array is updated
    in-place as each pixel is filled, so pixels on the edge of a cluster can
    supply values to their still-NaN interior neighbours in the same pass.

    Parameters
    ----------
    image   : 2-D ndarray
    mask    : 2-D bool ndarray  (True = bad, same shape as image)
    max_adu : float or None
        Camera full-well / physical maximum in ADU.  Any calibrated pixel
        above this value is flagged and interpolated over — this catches hot
        pixels, cosmic rays, and flat divide-by-near-zero artefacts that are
        not detected by the dark-based bad-pixel mask.

    Returns
    -------
    corrected : 2-D float ndarray
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
            logger.info(
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
        logger.info(
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
        logger.info(
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
    # calibrated_data = calibrated_data[TRIM:-TRIM, TRIM:-TRIM]

    regions = detection.stars_detection(calibrated_data)

    # need at least 3 stars for a usable PSF / alignment solution
    if len(regions) < 3:
        logger.warning("Fewer than 3 stars detected in %s", Path(file).name)
        return None, [], None, None, [], header

    # (x, y) pixel coordinates of every detected star
    region_coords = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
    cutouts = utils.cutout(calibrated_data, region_coords, (50, 50))

    # discard saturated stars so they don't bias the PSF or photometry
    not_saturated = np.array([np.max(c) < SATURATED for c in cutouts])
    cutouts = np.array([c for c, keep in zip(cutouts, not_saturated) if keep])
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


def _movie_frame(image, max_px=MOVIE_MAX_PX):
    """Downsample and contrast-stretch a frame into a uint8 night-movie thumbnail.

    Strides the image down so its longest side is at most ``max_px``, clips to a
    robust percentile range, and applies an asinh stretch so faint stars show
    without the bright ones saturating.
    """
    step = max(1, int(np.ceil(max(image.shape) / max_px)))
    small = image[::step, ::step]
    lo, hi = np.percentile(small, [25, 99.5])
    norm = np.clip((small - lo) / (hi - lo + 1e-9), 0, 1)
    stretched = np.arcsinh(norm * 10) / np.arcsinh(10)
    return (stretched * 255).astype(np.uint8)


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
            transform = AffineTransform(R).inverse
            aligned_coords = transform(ref_coords)[0:N_STARS]
            dx, dy = np.median(ref_coords[0:N_STARS] - aligned_coords, 0)

            drift = float(np.sqrt(dx**2 + dy**2))
            if drift > MAX_DRIFT_PX:
                logger.warning(
                    "Skipping %s: drift %.1f px (dx=%.1f dy=%.1f) exceeds "
                    "MAX_DRIFT_PX=%d — frame is too misaligned for reliable "
                    "aperture photometry",
                    filename, drift, dx, dy, MAX_DRIFT_PX,
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


def optimal_aperture(
    diffs: np.ndarray,
    target_index: int,
    time: np.ndarray,
    weights: np.ndarray,
    bin_minutes: float = 10.0,
) -> int:
    """Select the aperture that minimises noise without penalising variability.

    **Primary decision** — comparison-star noise profile
        Comparison stars are not astrophysically variable, so their noise vs.
        aperture gives an uncontaminated view of aperture quality (readnoise-
        dominated at small apertures, sky-dominated at large ones, optimal in
        between).  Two metrics are computed for the comparison ensemble at each
        aperture (PTP scatter and within-bin residual scatter); apertures are
        ranked on each and the one with the lowest combined rank wins.

    **Cross-validation** — target scores
        The same metrics are computed for the target and logged for
        transparency.  If they strongly disagree with the comparison result a
        warning is emitted, but the comparison result is kept because the
        target's own LC may contain real astrophysical signal.

    Parameters
    ----------
    diffs:
        Differential light curves, shape ``(n_apertures, n_stars, n_frames)``.
    target_index:
        Column index of the science target in the star axis.
    time:
        JD timestamps, shape ``(n_frames,)``.  Need not be sorted.
    weights:
        Per-aperture comparison-star weights, shape ``(n_apertures, n_stars)``.
        Stars with ``weight > 0`` are treated as comparisons.
    bin_minutes:
        Width of the time bins used for the within-bin residual metric.
    """
    n_ap, n_stars, _ = diffs.shape

    order = np.argsort(time)
    t = time[order]
    D = diffs[:, :, order]  # (n_ap, n_stars, n_frames)

    comp_ptp = np.full(n_ap, np.inf)
    comp_bin = np.full(n_ap, np.inf)
    tgt_ptp = np.full(n_ap, np.inf)
    tgt_bin = np.full(n_ap, np.inf)

    for ap in range(n_ap):
        tgt_ptp[ap] = _ptp_score(D[ap, target_index])
        tgt_bin[ap] = _bin_residual_score(D[ap, target_index], t, bin_minutes)

        comp_idx = [
            s for s in range(n_stars) if s != target_index and weights[ap, s] > 0
        ]
        if not comp_idx:
            continue

        ptp_vals = [_ptp_score(D[ap, s]) for s in comp_idx]
        bin_vals = [_bin_residual_score(D[ap, s], t, bin_minutes) for s in comp_idx]

        finite_ptp = [v for v in ptp_vals if np.isfinite(v)]
        finite_bin = [v for v in bin_vals if np.isfinite(v)]
        if finite_ptp:
            comp_ptp[ap] = float(np.median(finite_ptp))
        if finite_bin:
            comp_bin[ap] = float(np.median(finite_bin))

    def _rank(scores: np.ndarray) -> np.ndarray:
        """Dense rank: inf values get the worst (highest) rank."""
        return np.argsort(np.argsort(scores))

    # Primary: rank apertures by comparison-ensemble noise.
    comp_best = int(np.argmin(_rank(comp_ptp) + _rank(comp_bin)))

    # Cross-check: what would the target alone prefer?
    tgt_best = int(np.argmin(_rank(tgt_ptp) + _rank(tgt_bin)))

    if abs(comp_best - tgt_best) > 5:
        logger.warning(
            "Aperture selection: comparison ensemble prefers #%d, "
            "target alone prefers #%d. Using comparison result — "
            "target LC may contain astrophysical signal.",
            comp_best,
            tgt_best,
        )

    logger.info(
        "Optimal aperture: %d  "
        "(comp PTP %.4f, comp bin-residual %.4f; "
        "target PTP %.4f, target bin-residual %.4f)",
        comp_best,
        comp_ptp[comp_best],
        comp_bin[comp_best],
        tgt_ptp[comp_best],
        tgt_bin[comp_best],
    )
    return comp_best


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
        "--query-name",
        default=None,
        metavar="NAME",
        help="Name to resolve for the target's sky coordinates (default: same as target).",
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
    args = ap.parse_args()
    image_path = args.image_path
    target = args.target
    query_name = args.query_name if args.query_name is not None else target
    fix_bad_pixels = args.fix_bad_pixels

    # =======================================================================
    # Index the night and build master calibration frames
    # =======================================================================
    observations, files_meta = find_files(f"{image_path}/*.fits")

    # Collect the set of exposure times and observing dates seen per object, so
    # we can match the target to calibrations taken with the same exposure.
    object_info = defaultdict(lambda: {"exptimes": set(), "dates": set()})
    for meta in files_meta.values():
        obj = meta["object"]
        object_info[obj]["exptimes"].add(meta["exptime"])
        object_info[obj]["dates"].add(meta["date"])

    # Darks must match the target's exposure time(s) for dark current to scale.
    target_exptimes = object_info[target]["exptimes"]
    matching_darks = [
        f
        for f, meta in files_meta.items()
        if meta["type"] == TYPE_DARK and meta["exptime"] in target_exptimes
    ]

    # Science (light) frames of the target.
    light_frames = [
        f
        for f, meta in files_meta.items()
        if meta["type"] == TYPE_LIGHT and meta["object"] == target
    ]
    logger.info("Target '%s': %d light frames", target, len(light_frames))

    # Determine the target's filter so flats can be matched to it. A flat-field
    # correction is only valid for the filter it was taken in, so mixing filters
    # would corrupt the master flat.
    target_filters = {files_meta[f]["filter"] for f in light_frames}
    if len(target_filters) != 1:
        raise ValueError(
            f"Expected target light frames in a single filter, found {target_filters}"
        )
    target_filter = target_filters.pop()
    logger.info("Target filter: %s", target_filter)

    # Master calibration frames for the target's observing night. Flats are
    # restricted to the target's filter; darks/bias are filter-independent.
    day_date = list(object_info[target]["dates"])[0]
    flats = [
        f
        for f in observations[day_date][TYPE_FLAT]
        if files_meta[f]["filter"] == target_filter
    ]
    if not matching_darks:
        logger.warning(
            "No dark frames with matching exposure time(s) %s found for target; "
            "using all dark frames for this night.",
            target_exptimes,
        )
        darks = observations[day_date][TYPE_DARK]
    else:
        logger.info(
            "%d dark frames with matching exposure time(s) %s found for target.",
            len(matching_darks),
            target_exptimes,
        )
        darks = matching_darks
    bias = observations[day_date][TYPE_BIAS]

    if not flats:
        raise ValueError(
            f"No '{target_filter}' flat frames found for {day_date}; cannot build master flat"
        )

    logger.info(
        "Building master frames (%d bias, %d dark, %d flat in '%s')",
        len(bias),
        len(darks),
        len(flats),
        target_filter,
    )
    BIAS = calibration.master_bias(files=bias)
    DARK = calibration.master_dark(bias=BIAS, files=darks)
    FLAT = calibration.master_flat(files=flats, dark=DARK, bias=BIAS)

    # Read noise: std(B1 - B2) / sqrt(2) in ADU.
    # Using a difference of two frames cancels fixed-pattern (bias structure) noise.
    if len(bias) >= 2:
        b1 = fits.getdata(bias[0]).astype(float)
        b2 = fits.getdata(bias[1]).astype(float)
        read_noise = float(np.std(b1 - b2) / np.sqrt(2))
        logger.info("Read noise estimate: %.2f ADU", read_noise)
    else:
        read_noise = float("nan")
        logger.warning("Need at least 2 bias frames to estimate read noise; skipping")

    # Dark current: median pixel value of the bias-subtracted master dark divided
    # by the dark exposure time, in ADU/s.
    dark_current = float(np.nanmedian(DARK))
    logger.info("Dark current estimate: %.4f ADU/s", dark_current)

    # Bad-pixel mask (computed once; passed to every calibration_sequence call).
    bp_mask = (
        bad_pixel_map(matching_darks, master_bias=BIAS) if fix_bad_pixels else None
    )
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
    reference_image = light_frames[len(light_frames) // 2]
    logger.info("Using reference image: %s", Path(reference_image).name)

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

    # Query Gaia over a slightly larger area than the FOV to allow for pointing error.
    logger.info("Querying Gaia and solving WCS...")
    all_radecs = gaia_radecs(
        center,
        1.5 * fov,
        tmass=USE_TMASS,
    )
    # Match the 15 brightest detected stars to the 15 brightest Gaia sources.
    wcs = compute_wcs(ref_coords_all[0:15], all_radecs[0:15], tolerance=10)

    # Convert reference-frame star pixel positions to sky coordinates via the
    # WCS, resolve the target's Gaia coordinates, and find which star it is.
    stars_radec = wcs.pixel_to_world(*ref_coords.T)

    mast = Mast()
    target_radec = mast.resolve_object(query_name)
    target_index = int(target_radec.match_to_catalog_sky(stars_radec)[0])
    logger.info("Target matched to star index %d", target_index)

    # =======================================================================
    # Main loop: per-frame photometry, parallelised across worker processes
    # =======================================================================
    stack = np.zeros_like(ref_data, dtype=float)  # co-added, aligned science stack
    data = defaultdict(list)  # per-frame measurements, keyed by quantity
    movie = []  # downsampled uint8 frames (frame order), for the night movie

    chunks = _chunks(light_frames, N_WORKERS)
    logger.info(
        "Processing %d frames across %d workers...", len(light_frames), len(chunks)
    )
    # Per-frame progress: workers signal this queue once per frame (success,
    # skip, or error). A drain thread reads it and updates tqdm independently
    # of when whole chunks complete, giving smooth and accurate progress.
    # Manager().Queue() produces a proxy object that is picklable across the
    # spawn boundary used by macOS — plain multiprocessing.Queue is not.
    stop_drain = threading.Event()

    with _MPManager() as _manager, tqdm(total=len(light_frames), unit="frame") as pbar:
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

    # Persist the raw photometry products for later analysis.
    telescope_name = ref_header.get(KW_TELESCOP, "unknown")
    safe_target_filter = (
        target_filter.replace(" ", "-").replace("/", "-").replace("'", "")
    )
    output_file = (
        f"photometry_data_{telescope_name}_{safe_target_filter}_{target}_{day_date}.npz"
    )
    np.savez(
        output_file,
        platescale=pixel_scale * 3600,
        read_noise=read_noise,
        dark_current=dark_current,
        **data,
    )
    logger.info("Saved photometry to %s", output_file)

    # =======================================================================
    # Differential photometry
    # =======================================================================
    # Background-subtracted fluxes, shaped (apertures, stars, frames).
    fluxes = (data["fluxes"] - data["bkg"]).T

    # Differential photometry against an automatically-chosen comparison set.
    diffs, weights = flux.auto_diff(fluxes, target_index)

    # Pick the aperture that minimises the target's light-curve scatter.
    best_aperture = optimal_aperture(
        diffs, target_index, data["time"], weights, bin_minutes=10.0
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
    # Save the night-report bundle (data + images + movie) for night_report.py
    # =======================================================================
    report_file = (
        f"night_report_{telescope_name}_{safe_target_filter}_{target}_{day_date}.npz"
    )
    np.savez_compressed(
        report_file,
        stack=stack,
        master_bias=BIAS,
        master_dark=DARK,
        master_flat=FLAT,
        ref_coords=ref_coords,
        target_index=target_index,
        diffs=diffs,
        weights=weights,
        alc=alc,
        best_aperture=best_aperture,
        movie=movie,
        target=target,
        date=str(day_date),
        band=target_filter,
        telescope=telescope_name,
        platescale=pixel_scale * 3600,  # arcsec/pixel
        read_noise=read_noise,
        dark_current=dark_current,
        **data,
    )
    logger.info(
        'Saved night-report bundle to %s (visualise with `python night_report.py "%s"`)',
        report_file,
        report_file,
    )

    # =======================================================================
    # Diagnostic figures
    # =======================================================================
    fig_prefix = f"{telescope_name}_{safe_target_filter}_{target}_{day_date}"
    platescale_as = pixel_scale * 3600  # arcsec / pixel

    t_jd = data["time"]
    jd0 = int(np.floor(t_jd.min()))
    # Sort into chronological order once; workers return frames in completion
    # order (as_completed), not observation order.
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
        ax.scatter(t_plot, lc_c, s=2, c="0.7", alpha=0.4, linewidths=0, rasterized=True)
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

    lc_path = f"lc_{fig_prefix}.pdf"
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

    syst_path = f"systematics_{fig_prefix}.pdf"
    fig2.savefig(syst_path, bbox_inches="tight")
    logger.info("Saved systematics figure to %s", syst_path)
    plt.close(fig2)


if __name__ == "__main__":
    main()
