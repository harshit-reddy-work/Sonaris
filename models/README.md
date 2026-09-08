# Sonaris Models Specification

Sonaris implements a tiered multi-model architecture engineered specifically for side-scan sonar acoustic characterization.

---

## 1. Lite Model: YOLO11n-Seg (Edge & Real-Time)

The Lite model delivers low-latency instance segmentation designed for compute-constrained platforms (NVIDIA Jetson, onboard AUV controllers, and survey laptops).

### Architecture & Computational Profile
- **Base Architecture**: Ultralytics YOLO11 Nano Instance Segmentation (`YOLO11n-Seg`)
- **Total Parameters**: 2,842,803 (~2.84M)
- **Computational Complexity**: 9.78 GFLOPs
- **Input Resolution**: 1024×1024 (`imgsz=1024`)
- **Weights File**: `models/yolo11n_seg_best.pt`

### Training Dataset & Target Classes
- **Training Corpus**: AI4Shipwrecks Benchmark Dataset (286 high-resolution side-scan sonar image tiles across 28 shipwreck sites collected via AUV).
- **Target Detection Class**: `0: shipwreck` (Sunken vessels, historic shipwrecks, structural hull ruins).
- **Epochs**: 100 epochs (AdamW optimizer with cosine learning rate schedule).
- **Note on Detectable Objects**: The Lite model is specialized strictly for naval and shipwreck structures. Non-vessel objects (e.g., small bicycle frames, hand tools, or fish swarms) are outside its training distribution.

### Validation Benchmark Metrics
| Metric | Value |
|---|---|
| **Box Precision** | 0.580 |
| **Box Recall** | 0.537 |
| **Box mAP@50** | 0.531 |
| **Box mAP@50-95** | 0.317 |
| **Mask Precision** | 0.574 |
| **Mask Recall** | 0.500 |
| **Mask mAP@50** | 0.492 |
| **Mask mAP@50-95** | 0.261 |

---

## 2. Pro Model: YOLOv8x + SAM Hybrid (High Precision)

The Pro pipeline couples high-capacity detection with foundation segmentation for high-fidelity contour extraction.

### Components
- **Detector**: YOLOv8 Extra-Large (`YOLOv8x`, 68.2M parameters).
  - Target Classes: `aircraft`, `shipwreck`, `seabed_anomalies`.
  - Benchmarked Performance: mAP@50 = 0.801 (Precision: 0.936, Recall: 0.695).
- **Segmentation Engine**: Meta Segment Anything Model (`SAM ViT-B`, 91M parameters).
  - Prompt Mechanism: Automated bounding box and point centroid prompting.
  - Zero-shot boundary extraction for acoustic shadow contours.

---

## Deployment & Usage

Models can be loaded via Python or directly within the Sonaris web dashboard:
```python
from src.segmentation.yolo_seg_engine import YOLOSegEngine

engine = YOLOSegEngine(model_path="models/yolo11n_seg_best.pt", confidence=0.5, image_size=1024)
results = engine.segment(sonar_image_array)
```
