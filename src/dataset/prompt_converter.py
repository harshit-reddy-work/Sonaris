"""SAM prompt generation strategies.

Open/Closed: Each strategy is a separate class implementing PromptStrategy.
New strategies can be added without modifying existing ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.interfaces import BBox, PromptStrategy


class BoxPrompt(PromptStrategy):
    """Uses the bounding box directly as a SAM box prompt."""

    def generate(self, bbox: BBox) -> dict[str, Any]:
        return {"box": bbox.as_array()}


class PointPrompt(PromptStrategy):
    """Uses the bbox center as a foreground point prompt."""

    def generate(self, bbox: BBox) -> dict[str, Any]:
        cx, cy = bbox.center
        return {
            "point_coords": np.array([[cx, cy]]),
            "point_labels": np.array([1]),  # 1 = foreground
        }


class BoxPointPrompt(PromptStrategy):
    """Combines box prompt with center point for higher accuracy."""

    def generate(self, bbox: BBox) -> dict[str, Any]:
        cx, cy = bbox.center
        return {
            "box": bbox.as_array(),
            "point_coords": np.array([[cx, cy]]),
            "point_labels": np.array([1]),
        }
