"""Evaluation metrics for segmentation quality.

Interface Segregation: MetricCalculator is separate from PostProcessor.
Each metric is its own class (Single Responsibility).
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import MetricCalculator


class IoUCalculator(MetricCalculator):
    """Intersection over Union (Jaccard Index)."""

    def compute(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, float]:
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = float(intersection / (union + 1e-6))
        return {"iou": iou}


class DiceCalculator(MetricCalculator):
    """Dice Coefficient (F1 Score for segmentation)."""

    def compute(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, float]:
        pred = pred_mask.astype(bool)
        gt = gt_mask.astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        dice = float(2 * intersection / (pred.sum() + gt.sum() + 1e-6))
        return {"dice": dice}
