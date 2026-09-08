"""Abstract base classes defining the contracts for each system component.

Dependency Inversion: High-level modules (Pipeline) depend on these abstractions,
not on concrete implementations. New strategies can be added without modifying
existing code (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Value Objects ────────────────────────────────────────────────────────────


@dataclass
class BBox:
    """Bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])


@dataclass(frozen=True)
class GPSCoordinate:
    """GPS coordinate with source tracking."""
    latitude: float
    longitude: float
    source: str  # "manual", "exif", "metadata"


@dataclass
class Detection:
    """A single detected object with its mask and metadata."""

    class_id: int
    class_name: str
    bbox: BBox
    mask: np.ndarray          # binary mask (H, W)
    confidence: float
    area_m2: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    shape_type: str | None = None  # "circular", "rectangular", "elongated", "irregular"
    gps: GPSCoordinate | None = None


@dataclass
class GeotaggedDetection(Detection):
    """Detection with GPS coordinate attached."""
    pass


@dataclass
class SegmentationResult:
    """Complete result for a single image."""

    image_path: str
    detections: list[Detection]
    inference_time_ms: float
    preprocessed_image: np.ndarray | None = None
    model_mode: str = "LITE"
    gps: GPSCoordinate | None = None


# ── Interfaces (Single Responsibility + Interface Segregation) ───────────────


class Preprocessor(ABC):
    """Interface for a single preprocessing step.

    Each concrete preprocessor does ONE thing (Single Responsibility).
    They are composed in a pipeline (see PreprocessingPipeline).
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply the preprocessing step and return the transformed image."""
        ...


class PromptStrategy(ABC):
    """Converts a bounding box into SAM-compatible prompts.

    Different strategies (box-only, point-only, box+point) implement this
    interface. New strategies can be added without changing existing code.
    """

    @abstractmethod
    def generate(self, bbox: BBox) -> dict[str, Any]:
        """Return a dict with keys expected by SAM predictor.predict().

        Possible keys: 'box', 'point_coords', 'point_labels'.
        """
        ...


class Segmenter(ABC):
    """Interface for segmentation models.

    Abstracts away the model implementation (SAM2, SAM3, or any future model).
    Pipeline depends on this interface, not on a specific model.
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        prompts: list[dict[str, Any]],
    ) -> list[tuple[np.ndarray, float]]:
        """Run inference on a single image with given prompts.

        Returns list of (mask, score) tuples, one per prompt.
        """
        ...


class SegmentationEngine(ABC):
    """Interface for unified segmentation engines (like YOLO11n-Seg)."""

    @abstractmethod
    def segment(self, image: np.ndarray, detections: list[Detection] | None = None) -> SegmentationResult:
        """Run end-to-end segmentation and return results."""
        ...


class PostProcessor(ABC):
    """Interface for post-processing operations on segmentation masks."""

    @abstractmethod
    def analyze(self, mask: np.ndarray, bbox: BBox) -> dict[str, Any]:
        """Analyze a binary mask and return computed properties."""
        ...


class MetricCalculator(ABC):
    """Interface for evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics between predicted and ground-truth masks."""
        ...


class Visualizer(ABC):
    """Interface for rendering segmentation results."""

    @abstractmethod
    def render(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Overlay detections on the original image and return the result."""
        ...
