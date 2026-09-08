"""SAM2 segmentation engine.

Implements the Segmenter interface (Dependency Inversion).
If SAM3 is released, a new SAM3Segmenter can be created implementing
the same interface — no changes needed in Pipeline (Open/Closed).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from ..core.config import SAMConfig
from ..core.interfaces import Segmenter

logger = logging.getLogger(__name__)


class SAMSegmenter(Segmenter):
    """Wraps Meta's SAM2 model for sonar image segmentation."""

    def __init__(self, config: SAMConfig) -> None:
        self._config = config
        self._predictor = None

    def load_model(self) -> None:
        """Load SAM2 model checkpoint.

        Attempts sam2 import first; falls back to segment_anything.
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_cfg = f"{self._config.model_type}.yaml"
            checkpoint = self._config.checkpoint
            model = build_sam2(model_cfg, checkpoint)
            self._predictor = SAM2ImagePredictor(model)
            logger.info("SAM2 model loaded: %s", self._config.model_type)

        except ImportError:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_b"](checkpoint=self._config.checkpoint)
            sam.to(self._config.device)
            self._predictor = SamPredictor(sam)
            logger.info("SAM (v1) model loaded as fallback")

    def segment(
        self,
        image: np.ndarray,
        prompts: list[dict[str, Any]],
    ) -> list[tuple[np.ndarray, float]]:
        """Run segmentation for each prompt on the given image.

        Returns list of (best_mask, best_score) tuples.
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert float [0,1] back to uint8 if needed
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image

        self._predictor.set_image(img_uint8)

        results: list[tuple[np.ndarray, float]] = []
        for prompt in prompts:
            masks, scores, _ = self._predictor.predict(
                box=prompt.get("box"),
                point_coords=prompt.get("point_coords"),
                point_labels=prompt.get("point_labels"),
                multimask_output=self._config.multimask_output,
            )
            # Pick the mask with the highest confidence
            best_idx = int(np.argmax(scores))
            results.append((masks[best_idx], float(scores[best_idx])))

        return results

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None
