"""
SAR Avalanche Debris Detection via Multi-Temporal RGB Composites and HSV Thresholding

Requirements:
- numpy
- opencv-python
- rasterio

Description:
This script processes a time-series of Sentinel-1 SAR images to detect avalanche debris.
1. Normalizes SAR backscatter (dB scale).
2. Creates a multi-temporal RGB composite:

    Red   = previous SAR acquisition
    Green = current SAR acquisition
    Blue  = previous SAR acquisition

Pixels exhibiting a relative increase in backscatter appear progressively
greener and are subsequently isolated using HSV thresholding.
3. Applies HSV (Hue-Saturation-Value) thresholding to isolate 'Bright Green' pixels.
4. Filters results by connected-component size (cluster size) to reduce noise.
"""

import os
import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from typing import Tuple, Optional

# =========================
# Configuration & Constants
# =========================

VALID_RASTER_EXTENSIONS = (".tif", ".tiff")

# Range used to scale SAR dB values into a [0, 1] interval
CLAMP_MIN = -30.0
CLAMP_MAX = 0.0

# ============================================================
# INPUT DATA TYPE
# ============================================================
# Set to:
#   False -> input SAR images are in linear backscatter (σ⁰, γ⁰, β⁰)
#   True  -> input SAR images are already in decibel (dB) units
#
# Linear SAR images are converted to dB using:
#       dB = 10 * log10(linear)
# ============================================================

INPUT_IS_DB = False

# =========================
# Core Functions
# =========================

def preprocess_raster(band: np.ndarray) -> np.ndarray:
    """
    Converts the input SAR image to dB (if required), clamps the values,
    and normalizes them to the range [0, 1].

    Parameters
    ----------
    band : np.ndarray
        Input SAR image. Can be either linear backscatter or dB,
        depending on the INPUT_IS_DB setting.

    Returns
    -------
    np.ndarray
        Normalized SAR image in the range [0, 1].
    """

    # --------------------------------------------------------
    # Convert to dB if the input is linear SAR
    # --------------------------------------------------------

    if INPUT_IS_DB:

        band_db = band.astype(np.float32)

    else:

        # Prevent log10(0)
        band = np.where(band > 0, band, 1e-10)

        band_db = 10.0 * np.log10(band)

    # --------------------------------------------------------
    # Clamp and normalize
    # --------------------------------------------------------

    band_db = np.clip(
        band_db,
        CLAMP_MIN,
        CLAMP_MAX,
    )

    band_norm = (
        band_db - CLAMP_MIN
    ) / (
        CLAMP_MAX - CLAMP_MIN
    )

    return band_norm.astype(np.float32)

def read_band(path: str) -> Tuple[np.ndarray, dict]:
    """Reads GeoTIFF band and handles NoData."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr = np.nan_to_num(arr, nan=1e-10)
            return arr, src.profile.copy()
    except RasterioIOError as e:
        raise FileNotFoundError(f"Error opening {path}: {e}")

def write_single_band(path: str, array: np.ndarray, profile: dict, dtype: str = "float32", nodata=None) -> None:
    """Writes single-band GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, driver="GTiff", compress="lzw", nodata=nodata)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)

def write_rgb(path: str, rgb: np.ndarray, profile: dict) -> None:
    """Writes 3-band RGB GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(count=3, dtype="float32", driver="GTiff", compress="lzw")
    with rasterio.open(path, "w", **out_profile) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i].astype("float32"), i + 1)

def create_rgb_composite(
    current_norm: np.ndarray,
    previous_norm: Optional[np.ndarray],
) -> np.ndarray:
    """
    Creates a multi-temporal RGB composite.

    Red   : Previous SAR acquisition
    Green : Current SAR acquisition
    Blue  : Previous SAR acquisition

    Parameters
    ----------
    current_norm : np.ndarray
        Normalized current SAR acquisition.

    previous_norm : np.ndarray
        Normalized previous SAR acquisition.

    Returns
    -------
    np.ndarray
        Three-band RGB composite in the range [0, 1].
    """

    if previous_norm is None:
        previous_norm = np.zeros_like(current_norm)

    red = previous_norm
    green = current_norm
    blue = previous_norm

    return np.dstack((red, green, blue)).astype(np.float32)
    
def hsv_threshold(
    hsv_image: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """
    Detects pixels that fall within the specified HSV threshold ranges.

    Parameters
    ----------
    hsv_image : np.ndarray
        HSV image generated from the RGB composite.

    thresholds : dict
        Dictionary containing the HSV threshold limits.

    Returns
    -------
    np.ndarray
        Binary detection mask.
    """

    lower = np.array(
        [
            thresholds["hue"][0],
            thresholds["saturation"][0],
            thresholds["value"][0],
        ],
        dtype=np.uint8,
    )

    upper = np.array(
        [
            thresholds["hue"][1],
            thresholds["saturation"][1],
            thresholds["value"][1],
        ],
        dtype=np.uint8,
    )

    return cv2.inRange(
        hsv_image,
        lower,
        upper,
    )

def filter_by_cluster_size(
    binary_mask: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """
    Removes connected components smaller than the specified area.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary detection mask.

    min_size : int
        Minimum connected-component size (pixels).

    Returns
    -------
    np.ndarray
        Filtered binary mask.
    """

    mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    filtered = np.zeros_like(mask)

    for label in range(1, num_labels):

        if stats[label, cv2.CC_STAT_AREA] >= min_size:

            filtered[labels == label] = 1

    return filtered
    
def process_time_series(
    input_dir: str,
    norm_dir: str,
    output_dir: str,
    thresholds: dict,
    min_cluster: int,
):
    """
    Processes a chronological Sentinel-1 SAR time series and generates:

    - Normalized SAR images
    - RGB composites
    - Binary avalanche detection masks
    """

    files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(VALID_RASTER_EXTENSIONS)
        ]
    )

    if not files:
        return

    mask_dir = os.path.join(output_dir, "Detection_Masks")
    comp_dir = os.path.join(output_dir, "RGB_Composites")

    for directory in (norm_dir, comp_dir, mask_dir):
        os.makedirs(directory, exist_ok=True)

    previous_norm = None

    for fname in files:

        path = os.path.join(input_dir, fname)

        try:

            # --------------------------------------------------------
            # Read SAR image
            # --------------------------------------------------------

            raw, profile = read_band(path)

            current_norm = preprocess_raster(raw)

            # --------------------------------------------------------
            # Save normalized SAR image
            # --------------------------------------------------------

            write_single_band(
                os.path.join(
                    norm_dir,
                    f"{os.path.splitext(fname)[0]}_norm.tif",
                ),
                current_norm,
                profile,
            )

            # --------------------------------------------------------
            # Skip first acquisition (no previous image available)
            # --------------------------------------------------------

            if previous_norm is None:

                previous_norm = current_norm

                print(
                    f"Skipping first acquisition: {fname}"
                )

                continue

            # --------------------------------------------------------
            # Create RGB composite
            # --------------------------------------------------------

            rgb = create_rgb_composite(
                current_norm,
                previous_norm,
            )

            write_rgb(
                os.path.join(
                    comp_dir,
                    f"{os.path.splitext(fname)[0]}_RGB.tif",
                ),
                rgb,
                profile,
            )

            # --------------------------------------------------------
            # RGB → HSV conversion
            # --------------------------------------------------------

            rgb_u8 = np.clip(
                rgb * 255.0,
                0,
                255,
            ).astype(np.uint8)

            hsv = cv2.cvtColor(
                rgb_u8,
                cv2.COLOR_RGB2HSV,
            )

            # --------------------------------------------------------
            # HSV thresholding
            # --------------------------------------------------------

            mask = hsv_threshold(
                hsv,
                thresholds,
            )

            # --------------------------------------------------------
            # Connected-component filtering
            # --------------------------------------------------------

            filtered = filter_by_cluster_size(
                mask,
                min_cluster,
            )

            # --------------------------------------------------------
            # Save detection mask
            # --------------------------------------------------------

            write_single_band(
                os.path.join(
                    mask_dir,
                    f"{os.path.splitext(fname)[0]}_mask.tif",
                ),
                filtered,
                profile,
                dtype="uint8",
            )

            previous_norm = current_norm

            print(f"Processed: {fname}")

        except Exception as e:

           print(f"[ERROR] {fname}: {e}")

if __name__ == "__main__":

    # ======================================================
    # Input / Output Directories
    # ======================================================

    IN_DIR = "./data/raw_sar"
    NORM_DIR = "./data/normalized"
    OUT_DIR = "./results"

    # ======================================================
    # Detection Parameters
    # ======================================================

    HSV_THRESHOLDS = {
        "hue": (45, 75),
        "saturation": (150, 255),
        "value": (50, 255),
    }

    MIN_PIXELS = 40

    # ======================================================
    # Run RGB Avalanche Detection
    # ======================================================

    process_time_series(
        IN_DIR,
        NORM_DIR,
        OUT_DIR,
        HSV_THRESHOLDS,
        MIN_PIXELS,
    )
