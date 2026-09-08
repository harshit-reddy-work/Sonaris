"""Geometric shape classification from segmentation masks.

Single Responsibility: Only classifies shape type based on geometry.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core.config import SonarConfig
from ..core.interfaces import BBox, PostProcessor


class ShapeClassifier(PostProcessor):
    """Classifies detected objects by geometric shape."""

    def __init__(self, config: SonarConfig) -> None:
        self._circ_thresh = config.circularity_threshold
        self._rect_thresh = config.rectangularity_threshold
        self._elong_thresh = config.elongation_threshold

    def analyze(self, mask: np.ndarray, bbox: BBox) -> dict[str, Any]:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return {"shape_type": "unknown", "circularity": 0.0, "rectangularity": 0.0}

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

        # Rectangularity: area / bounding_rect_area
        _, (rw, rh), _ = cv2.minAreaRect(largest)
        rectangularity = area / (rw * rh + 1e-6)

        # Aspect ratio (elongation)
        aspect_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)

        if circularity > self._circ_thresh:
            shape_type = "circular"
        elif aspect_ratio > self._elong_thresh:
            shape_type = "elongated"
        elif rectangularity > self._rect_thresh:
            shape_type = "rectangular"
        else:
            shape_type = "irregular"

        return {
            "shape_type": shape_type,
            "circularity": float(circularity),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
        }
