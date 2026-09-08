"""Physical size estimation from segmentation masks.

Single Responsibility: Only estimates real-world dimensions.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core.config import SonarConfig
from ..core.interfaces import BBox, PostProcessor


class SizeEstimator(PostProcessor):
    """Converts pixel-space mask area to real-world metric estimates."""

    def __init__(self, config: SonarConfig) -> None:
        self._resolution = config.resolution_m_per_px

    def analyze(self, mask: np.ndarray, bbox: BBox) -> dict[str, Any]:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return {"area_m2": 0.0, "width_m": 0.0, "height_m": 0.0}

        largest = max(contours, key=cv2.contourArea)
        pixel_area = cv2.contourArea(largest)
        rect = cv2.minAreaRect(largest)
        rect_w, rect_h = rect[1]

        return {
            "area_m2": pixel_area * (self._resolution ** 2),
            "width_m": rect_w * self._resolution,
            "height_m": rect_h * self._resolution,
        }
