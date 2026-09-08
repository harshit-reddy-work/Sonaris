# 🌊 Sonaris: Autonomous Sonar Intelligence Platform

**Sonaris** is an advanced artificial intelligence platform engineered for automated anomaly detection, physical metric estimation, and high-precision segmentation in side-scan sonar (SSS) acoustic imagery.

Designed for hydrographic survey teams, marine archaeology, and autonomous underwater vehicle (AUV) integration, Sonaris bridges rapid real-time acoustic edge screening with deep semantic boundary delineation.

---

## ⚡ Key Capabilities

- **Dual-Engine Architecture**:
  - **Lite Mode (YOLO11n-Seg)**: Ultra-compact, single-stage instance segmentation running at ~9.8 GFLOPs. Optimized for edge compute (NVIDIA Jetson, onboard AUV payloads, field laptops).
  - **Pro Mode (YOLOv8x + SAM)**: Server-grade detection pipeline paired with promptable foundation models for pixel-precise structural boundary segmentation.
- **Geospatial Intelligence**:
  - Direct extraction of spatial coordinates from EXIF metadata, sonar navigation logs, or manual entry.
  - One-click Google Maps location plotting and coordinate-tagged GIS exports.
- **Acoustic Domain Preprocessing**:
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) tailored for sonar shadow/highlight balance.
  - Multi-scale median and Non-Local Means (NLM) acoustic speckle filtering.
- **Morphological & Physical Estimation**:
  - Real-world area ($m^2$), length, and width ($m$) computation via sonar spatial resolution parameters.
  - Automated geometry classification (circular, rectangular, elongated, irregular).
- **Interactive Web Interface & CLI**:
  - Full-featured dashboard with drag-and-drop analysis, batch processing, and dataset exploration.
  - Headless CLI for automated offshore survey batch runs.

---

## 🏗️ Architecture Overview

```text
                        ┌───────────────────────────────┐
                        │    Side-Scan Sonar Imagery    │
                        └───────────────┬───────────────┘
                                        │
                         [Acoustic Denoising & CLAHE]
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
     ┌─────────────────────┐                         ┌─────────────────────┐
     │      Lite Mode      │                         │       Pro Mode      │
     │    (YOLO11n-Seg)    │                         │   (YOLOv8x + SAM)   │
     └──────────┬──────────┘                         └──────────┬──────────┘
                │ Real-time Instance Mask                       │ High-Precision Prompted Mask
                └───────────────────────┬───────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   Physical Size & Geometry    │
                        │    (Area, Dimensions, GPS)    │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │     Interactive Dashboard     │
                        │   & Standardized Reports      │
                        └───────────────────────────────┘
```

---

## 📊 Model Specifications

| Attribute | Lite Model | Pro Model |
|---|---|---|
| **Architecture** | YOLO11n-Seg | YOLOv8x + SAM ViT-B |
| **Parameters** | 2.84M | 68.2M + 91M |
| **Compute Footprint** | 9.8 GFLOPs | Server / GPU Accelerated |
| **Primary Task** | Edge Instance Segmentation | Deep Semantic Detection & Masking |
| **Target Resolution** | 1024×1024 | 1024×1024 / Native |
| **Primary Detectable Classes** | Shipwrecks, Sunken Hulls, Submerged Vessels | Shipwrecks, Aircraft Debris, Seabed Anomalies |
| **Target Hardware** | CPU / Edge AUV Payloads / Mobile Boats | Workstation GPU (CUDA) / Cloud Cluster |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch with CUDA (recommended for Pro mode) or CPU

### Installation
```bash
git clone https://github.com/your-org/sonaris.git
cd sonaris
pip install -r requirements.txt
```

### Launch Interactive Dashboard
```bash
python -m streamlit run app/app.py
```
*Access the interface at `http://localhost:8501`.*

### Headless CLI Batch Run
```bash
# Run Lite mode on test dataset with visualization export
python main.py --mode lite --split test --save-vis

# Run Pro mode with specific geospatial coordinates
python main.py --mode pro --split test --gps-lat 45.0500 --gps-lon -83.4300
```

---

## 📁 Repository Organization

```text
Sonaris/
├── app/
│   └── app.py                  # Streamlit Interactive Web Application
├── data/
│   ├── archive/                # Sonar dataset specifications & configurations
│   └── demo/                   # Benchmark sonar test imagery
├── models/
│   ├── yolo11n_seg_best.pt     # Lite Model trained weights
│   └── README.md               # Model cards and evaluation metrics
├── notebooks/                  # Training and evaluation workflows
├── src/                        # Core modular engine (SOLID architecture)
│   ├── core/                   # System interfaces and configuration objects
│   ├── dataset/                # Dataset loaders and prompt strategies
│   ├── geospatial/             # EXIF parsing & GPS mapping utilities
│   ├── postprocessing/         # Physical size estimation & shape classifiers
│   ├── preprocessing/          # CLAHE & acoustic speckle filtering
│   ├── segmentation/           # Segmentation backends (YOLO11, YOLOv8, SAM)
│   ├── visualization/          # Overlay generators & report writers
│   └── pipeline.py             # Pipeline orchestrator
├── training_results/           # Benchmark curves, confusion matrices, loss curves
├── main.py                     # CLI entry point
├── requirements.txt            # Dependency manifest
└── README.md                   # Platform documentation
```

---

## 📄 License

Proprietary / Dual License for Survey & Research Operations.
