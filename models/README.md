# Sonaris Models Specification

Sonaris implements a 3-tier multi-model architecture engineered specifically for side-scan sonar acoustic characterization across civilian survey, marine ecology, and naval defense domains.

---

## 1. Lite Mode: YOLO11n (Civilian Marine Survey & Debris)

The Lite model delivers low-latency detection designed for edge computing (NVIDIA Jetson, onboard AUV controllers, survey laptops).

### Architecture & Computational Profile
- **Base Architecture**: Ultralytics YOLO11 Nano (`YOLO11n`)
- **Total Parameters**: 2,590,620 (~2.59M)
- **Computational Complexity**: 6.50 GFLOPs
- **Weights File**: `models/yolo11n_seg_best.pt`
- **Trained Classes (4)**:
  - `0: aircraft` (Downed planes, fuselage ruins)
  - `1: fish` (Biological acoustic swarms)
  - `2: other` (Seabed debris, man-made anomalies, containers, frames)
  - `3: shipwreck` (Sunken ships, hulls, maritime architecture)

---

## 2. ⚔️ War Mode: YOLO11n Tactical (Naval Defense & Strategic Infrastructure)

War Mode is specialized for critical maritime infrastructure monitoring, subsea threat detection, and harbor defense.

### Architecture & Computational Profile
- **Base Architecture**: Ultralytics YOLO11 Nano (`YOLO11n`)
- **Total Parameters**: 2,590,815 (~2.59M)
- **Computational Complexity**: 6.50 GFLOPs
- **Training Corpus**: DRISHTI-SSS Dataset (5,198 side-scan sonar image tiles)
- **Weights File**: `models/yolo11n_war_best.pt`
- **Trained Tactical Classes (5)**:
  - `0: crab_pot` (Seafloor cage traps / benthic markers)
  - `1: submarine_pipeline` (Underwater oil, gas, and communications infrastructure)
  - `2: shipwreck` (Sunken naval vessels, hulls, and maritime ruins)
  - `3: ghost_net` (Abandoned fishing nets, propulsion entanglement hazards)
  - `4: mine_cylinder` (Cylindrical naval mines, unexploded ordnance)

---

## 3. 🎯 Pro Mode: YOLOv8x + SAM Hybrid (High Precision)

The Pro pipeline couples high-capacity detection with Meta's foundation model for zero-shot boundary segmentation.

### Components
- **Detector**: YOLOv8 Extra-Large (`YOLOv8x`, 68.2M parameters)
- **Segmentation Engine**: Meta Segment Anything Model (`SAM ViT-B`, 91M parameters)
- **Primary Task**: Deep semantic boundary extraction and acoustic shadow analysis

---

## Deployment & Usage

```python
from ultralytics import YOLO

# Load Lite Mode
lite_model = YOLO("models/yolo11n_seg_best.pt")

# Load War Mode
war_model = YOLO("models/yolo11n_war_best.pt")

# Run Inference
results = war_model.predict("sonar_scan.png", conf=0.4)
```
