"""Main pipeline orchestrator.

Dependency Inversion: Depends only on abstractions (Segmenter, Preprocessor, etc.),
not on concrete implementations. All concrete classes are injected from outside.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .core.interfaces import (
    GPSCoordinate,
    GeotaggedDetection,
    BBox,
    Detection,
    PostProcessor,
    PromptStrategy,
    Segmenter,
    SegmentationResult,
    Visualizer,
)
from .preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class SonarAnalysisPipeline:
    """Orchestrates the full sonar analysis workflow.

    All dependencies are injected via constructor (Dependency Inversion).
    Supports dual-mode inference: LITE (YOLO11n-Seg) and PRO (YOLOv8x+SAM).
    """

    def __init__(
        self,
        preprocessor: PreprocessingPipeline,
        segmenter: Segmenter,
        prompt_strategy: PromptStrategy,
        post_processors: list[PostProcessor],
        visualizer: Visualizer | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._segmenter = segmenter
        self._prompt_strategy = prompt_strategy
        self._post_processors = post_processors
        self._visualizer = visualizer
        self._class_names = class_names or {}

    def process_single(
        self,
        image: np.ndarray,
        model_mode: str = "LITE",
        use_preprocessing: bool = True,
        gps_coord: GPSCoordinate | None = None,
    ) -> SegmentationResult:
        """Run inference on a single image (used by Streamlit dashboard).

        Args:
            image: Raw sonar image (BGR, uint8).
            model_mode: "LITE" or "PRO".
            use_preprocessing: Whether to apply acoustic filters.
            gps_coord: Optional GPS coordinate to attach to results.

        Returns:
            SegmentationResult with all detections and metadata.
        """
        processed = self._preprocessor.run(image) if use_preprocessing else image

        # Use the segmentation engine directly
        result = self._segmenter.segment(processed, [])

        # Attach metadata
        if gps_coord:
            result.gps = gps_coord
            for d in result.detections:
                d.gps = gps_coord
        result.model_mode = model_mode

        # Post-process each detection
        for det in result.detections:
            extra = {}
            for pp in self._post_processors:
                extra.update(pp.analyze(det.mask, det.bbox))
            det.area_m2 = extra.get("area_m2")
            det.width_m = extra.get("width_m")
            det.height_m = extra.get("height_m")
            det.shape_type = extra.get("shape_type")

        return result

    def process_image(
        self,
        image: np.ndarray,
        annotations: list[tuple[int, BBox]],
        image_path: str = "",
    ) -> SegmentationResult:
        """Run the full pipeline on a single image with ground-truth annotations.

        Args:
            image: Raw sonar image (BGR, uint8).
            annotations: List of (class_id, bbox) from labels.
            image_path: Path for reporting.

        Returns:
            SegmentationResult with all detections.
        """
        # 1. Preprocess
        processed = self._preprocessor.run(image)

        # 2. Generate prompts from annotations
        prompts = [self._prompt_strategy.generate(bbox) for _, bbox in annotations]

        # 3. Run segmentation
        t0 = time.perf_counter()
        mask_results = self._segmenter.segment(processed, prompts)
        inference_ms = (time.perf_counter() - t0) * 1000

        # 4. Build detections with post-processing
        detections: list[Detection] = []
        for (class_id, bbox), (mask, score) in zip(annotations, mask_results):
            # Merge all post-processor outputs
            extra: dict = {}
            for pp in self._post_processors:
                extra.update(pp.analyze(mask, bbox))

            det = Detection(
                class_id=class_id,
                class_name=self._class_names.get(class_id, f"class_{class_id}"),
                bbox=bbox,
                mask=mask,
                confidence=score,
                area_m2=extra.get("area_m2"),
                width_m=extra.get("width_m"),
                height_m=extra.get("height_m"),
                shape_type=extra.get("shape_type"),
            )
            detections.append(det)

        return SegmentationResult(
            image_path=image_path,
            detections=detections,
            inference_time_ms=inference_ms,
            preprocessed_image=processed,
        )

    def visualize(
        self,
        image: np.ndarray,
        result: SegmentationResult,
    ) -> np.ndarray | None:
        """Render detections on the image if a visualizer is available."""
        if self._visualizer is None:
            return None
        return self._visualizer.render(image, result.detections)
