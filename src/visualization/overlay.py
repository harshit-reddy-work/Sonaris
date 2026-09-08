"""Mask overlay visualization on sonar images.

Single Responsibility: Only handles visual rendering of detections.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.interfaces import Detection, Visualizer

# Per-class colors (BGR)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),     # aircraft  -> green
    1: (255, 255, 0),   # fish      -> cyan
    2: (0, 165, 255),   # other     -> orange
    3: (0, 0, 255),     # shipwreck -> red
}

DEFAULT_COLOR = (200, 200, 200)


class MaskOverlayVisualizer(Visualizer):
    """Renders segmentation masks and bounding boxes on the image."""

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = alpha

    def render(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        # Ensure uint8 for drawing
        if image.dtype != np.uint8:
            canvas = (image * 255).astype(np.uint8).copy()
        else:
            canvas = image.copy()

        overlay = canvas.copy()

        for det in detections:
            color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)

            # Resize mask to match image dimensions if needed
            mask = det.mask
            h, w = canvas.shape[:2]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = color

            # Draw bounding box
            pt1 = (int(det.bbox.x1), int(det.bbox.y1))
            pt2 = (int(det.bbox.x2), int(det.bbox.y2))
            cv2.rectangle(canvas, pt1, pt2, color, 2)

            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                canvas, label,
                (pt1[0], pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Blend mask overlay
        result = cv2.addWeighted(overlay, self._alpha, canvas, 1 - self._alpha, 0)
        return result
