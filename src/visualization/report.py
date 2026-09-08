"""JSON/CSV report generation from segmentation results.

Single Responsibility: Only serializes results to report formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.interfaces import SegmentationResult, GeotaggedDetection
from ..geospatial.maps import generate_google_maps_url


class ReportGenerator:
    """Generates structured reports from segmentation results."""

    def to_json(self, result: SegmentationResult, output_path: Path) -> None:
        data = {
            "image_path": result.image_path,
            "inference_time_ms": result.inference_time_ms,
            "num_detections": len(result.detections),
            "model_mode": getattr(result, "model_mode", "Lite"),
            "gps": {
                "latitude": getattr(result, "gps", None).latitude if getattr(result, "gps", None) else None,
                "longitude": getattr(result, "gps", None).longitude if getattr(result, "gps", None) else None,
                "google_maps_url": generate_google_maps_url(result.gps.latitude, result.gps.longitude) if getattr(result, "gps", None) else None,
                "source": getattr(result, "gps", None).source if getattr(result, "gps", None) else None
            },
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": {
                        "x1": d.bbox.x1, "y1": d.bbox.y1,
                        "x2": d.bbox.x2, "y2": d.bbox.y2,
                    },
                    "area_m2": d.area_m2,
                    "width_m": d.width_m,
                    "height_m": d.height_m,
                    "shape_type": d.shape_type,
                }
                for d in result.detections
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def to_csv_row(self, result: SegmentationResult) -> list[dict]:
        rows = []
        for d in result.detections:
            rows.append({
                "image_path": result.image_path,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_x1": d.bbox.x1,
                "bbox_y1": d.bbox.y1,
                "bbox_x2": d.bbox.x2,
                "bbox_y2": d.bbox.y2,
                "area_m2": d.area_m2,
                "width_m": d.width_m,
                "height_m": d.height_m,
                "shape_type": d.shape_type,
                "inference_ms": result.inference_time_ms,
                "latitude": getattr(result, "gps", None).latitude if getattr(result, "gps", None) else None,
                "longitude": getattr(result, "gps", None).longitude if getattr(result, "gps", None) else None,
                "model_mode": getattr(result, "model_mode", "Lite"),
            })
        return rows

    def write_csv(
        self,
        results: list[SegmentationResult],
        output_path: Path,
    ) -> None:
        all_rows = []
        for r in results:
            all_rows.extend(self.to_csv_row(r))

        if not all_rows:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
