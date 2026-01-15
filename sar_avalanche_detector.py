"""
SAR Avalanche Debris Detection via Multi-Temporal RGB Composites and HSV Thresholding

Requirements:
- numpy
- opencv-python
- rasterio

Description:
This script processes a time-series of Sentinel-1 SAR images to detect avalanche debris.
1. Normalizes SAR backscatter (dB scale).
2. Creates an RGB composite where Green represents the current image (highlighting backscatter increase).
3. Applies HSV (Hue-Saturation-Value) thresholding to isolate 'Bright Green' pixels.
4. Filters results by connected-component size (cluster size) to reduce noise.
"""

import os
import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from typing import List, Tuple, Optional

# =========================
# Configuration & Constants
# =========================

VALID_RASTER_EXTENSIONS = (".tif", ".tiff")

# Range used to scale SAR dB values into a [0, 1] interval
CLAMP_MIN = -30.0
CLAMP_MAX = 0.0

# =========================
# Core Functions
# =========================

def preprocess_raster(band: np.ndarray) -> np.ndarray:
    """Converts linear SAR to dB, clamps, and normalizes to [0, 1]."""
    band = np.where(band > 0, band, 1e-10)
    band_db = 10.0 * np.log10(band)
    band_db = np.clip(band_db, CLAMP_MIN, CLAMP_MAX)
    band_norm = (band_db - CLAMP_MIN) / (CLAMP_MAX - CLAMP_MIN)
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

def create_rgb_composite(current_norm: np.ndarray, prev_norm: Optional[np.ndarray]) -> np.ndarray:
    """Creates RGB: Green=Current, Red/Blue=Previous."""
    g = current_norm
    r = prev_norm if prev_norm is not None else np.zeros_like(current_norm)
    b = r.copy()
    return np.dstack([r, g, b]).astype(np.float32)

def hsv_threshold(hsv_image: np.ndarray, thresholds: List[Tuple]) -> List[np.ndarray]:
    """Isolates colors based on HSV ranges."""
    masks = []
    for (h_range, s_range, v_range) in thresholds:
        lower = np.array([h_range[0], s_range[0], v_range[0]])
        upper = np.array([h_range[1], s_range[1], v_range[1]])
        masks.append(cv2.inRange(hsv_image, lower, upper))
    return masks

def filter_by_cluster_size(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """Removes small noise clusters."""
    mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 1
    return filtered

def process_time_series(input_dir: str, norm_dir: str, output_dir: str, thresholds: List, min_cluster: int):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_RASTER_EXTENSIONS)])
    if not files: return
    
    mask_dir = os.path.join(output_dir, "Detection_Masks")
    comp_dir = os.path.join(output_dir, "RGB_Composites")
    for d in [norm_dir, comp_dir, mask_dir]: os.makedirs(d, exist_ok=True)

    prev_norm = None
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            raw, profile = read_band(path)
            current_norm = preprocess_raster(raw)
            
            # Save Normalization
            write_single_band(os.path.join(norm_dir, f"{os.path.splitext(fname)[0]}_norm.tif"), current_norm, profile)

            # RGB Composite
            rgb = create_rgb_composite(current_norm, prev_norm)
            write_rgb(os.path.join(comp_dir, f"{os.path.splitext(fname)[0]}_RGB.tif"), rgb, profile)

            # Thresholding
            rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
            masks = hsv_threshold(hsv, thresholds)

            # Filter & Save
            for idx, m in enumerate(masks):
                filtered = filter_by_cluster_size(m, min_cluster)
                write_single_band(os.path.join(mask_dir, f"{os.path.splitext(fname)[0]}_mask.tif"), filtered, profile, dtype="uint8")

            prev_norm = current_norm
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error {fname}: {e}")

if __name__ == "__main__":
    # --- Configure Paths ---
    IN_DIR = "./data/raw_sar"
    NORM_DIR = "./data/normalized"
    OUT_DIR = "./results"

    MY_THRESHOLDS = [((45, 75), (150, 255), (50, 255))]
    MIN_PIXELS = 49 
    process_time_series(IN_DIR, NORM_DIR, OUT_DIR, MY_THRESHOLDS, MIN_PIXELS)
