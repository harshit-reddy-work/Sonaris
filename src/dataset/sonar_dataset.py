"""PyTorch-compatible dataset for sonar images.

Single Responsibility: Loads images and labels, delegates parsing to YOLOLabelParser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.interfaces import BBox
from ..dataset.label_parser import YOLOLabelParser


class SonarDataset:
    """Loads sonar images and their YOLO labels from a split directory."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self._images_dir = Path(images_dir)
        self._labels_dir = Path(labels_dir)
        self._parser = YOLOLabelParser()
        self._class_names = class_names or {}

        self._image_paths = sorted(
            p for p in self._images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self._image_paths[idx]
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        label_path = self._labels_dir / f"{img_path.stem}.txt"
        annotations: list[tuple[int, BBox]] = []
        if label_path.exists():
            annotations = self._parser.parse(label_path, w, h)

        return {
            "image": image,
            "image_path": str(img_path),
            "annotations": annotations,
            "image_size": (w, h),
        }

    def get_class_name(self, class_id: int) -> str:
        return self._class_names.get(class_id, f"class_{class_id}")
