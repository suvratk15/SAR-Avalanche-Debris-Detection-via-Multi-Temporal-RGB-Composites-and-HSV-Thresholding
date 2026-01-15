
***

# SAR Avalanche Debris Detection via Multi-Temporal RGB Composites

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rasterio](https://img.shields.io/badge/Dependency-Rasterio-green.svg)](https://rasterio.readthedocs.io/)

This repository contains an automated Python pipeline for detecting and mapping avalanche debris using Sentinel-1 Synthetic Aperture Radar (SAR) time-series data. The method leverages the physical properties of SAR backscatter changes following avalanche events, visualized through RGB composites and isolated via Hue-Saturation-Value (HSV) thresholding.

## 🏔️ Scientific Background

Monitoring snow avalanches in mountainous terrain is often hindered by cloud cover and darkness. Sentinel-1 SAR imagery provides an all-weather, day-and-night solution. 

This tool is designed for scenarios where avalanche debris results in a **backscatter increase** relative to the surrounding undisturbed snowpack. This phenomenon is typically observed due to:
*   **Increased Surface Roughness:** The chaotic nature of debris piles increases surface scattering.
*   **Volume Scattering:** Changes in the internal structure of the snowpack within the debris flow.

## 🛠️ Methodology

The algorithm processes SAR time-series through four distinct stages:

### 1. Pre-processing & dB Normalization
Raw linear-scale SAR intensity is converted to the Decibel (dB) scale. To ensure numerical stability and consistent thresholding, the data is clamped to a standard range (default: -30 to 0 dB) and normalized to a `[0, 1]` interval.

### 2. Multi-Temporal RGB Compositing
The script generates a 3-band GeoTIFF composite for visual and algorithmic analysis:
*   **Green Channel (G):** Current SAR acquisition (Time $i$).
*   **Red (R) & Blue (B) Channels:** Previous SAR acquisition (Time $i-1$).
*   **Result:** Regions where backscatter has increased manifest as **Bright Green** pixels.

### 3. HSV Space Thresholding
The RGB composite is transformed into the **Hue-Saturation-Value (HSV)** color domain. Unlike standard RGB, HSV allows for the isolation of specific color "signatures" (Hue) independent of their brightness (Value) or purity (Saturation). We target the specific "Bright Green" range associated with debris.

### 4. Spatial Noise Filtering
To mitigate the effects of SAR speckle noise, the script applies a connected-component filter using **8-connectivity**. This ensures that only physically plausible debris clusters (user-defined minimum area) are retained in the final detection mask.

## 📦 Requirements

*   **Python 3.8+**
*   **NumPy:** For vectorized array processing.
*   **OpenCV (cv2):** For color space transformations and spatial filtering.
*   **Rasterio:** For georeferenced metadata preservation and I/O.

Install the necessary dependencies:
```shell script
pip install numpy opencv-python rasterio
```


## 🚀 Getting Started

### Data Preparation
1. Ensure your Sentinel-1 images are pre-processed (e.g., via SNAP or S1Tiling) and orthorectified.
2. Place your GeoTIFFs (`.tif`) in a single input directory. Files should be named such that they are chronologically sorted (e.g., `20230101_vv.tif`, `20230112_vv.tif`).

### Configuration
Update the following parameters in `sar_avalanche_detector.py`:
*   `IN_DIR`: Path to your raw SAR images.
*   `NORM_DIR`: Path where normalized dB images will be stored.
*   `OUT_DIR`: Path for detection results and RGB composites.
*   `MY_THRESHOLDS`: The HSV bounds for your specific region (default values provided).
*   `MIN_PIXELS`: Minimum size of an avalanche (in pixels).

### Execution
```shell script
python sar_avalanche_detector.py
```


## 📂 Output Data Structure

```plain text
/results/
├── Detection_Masks/     # Binary (0/1) GeoTIFFs of detected debris
├── RGB_Composites/      # 3-band visualizations for GIS software
└── Normalized/          # Pre-processed dB images [0, 1]
```


## 📝 Citation & License

**License:** Distributed under the MIT License. See `LICENSE` for more information.

If you utilize this code in your research, please cite this repository and the associated methodology for avalanche characterization in SAR imagery.

---
**Disclaimer:** *This tool is intended for research purposes. Avalanche detection accuracy depends heavily on SAR geometry, snow conditions, and terrain correction.*
