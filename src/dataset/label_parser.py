"""YOLO label file parser.

Single Responsibility: Only parses YOLO-format label files into BBox objects.
"""

from __future__ import annotations

from pathlib import Path

from ..core.interfaces import BBox


class YOLOLabelParser:
    """Parses YOLOv8 label files (class_id cx cy w h) into BBox objects."""

    def parse(
        self,
        label_path: Path,
        img_width: int,
        img_height: int,
    ) -> list[tuple[int, BBox]]:
        """Return list of (class_id, BBox) from a YOLO label file.

        Converts normalized YOLO coordinates to absolute pixel coordinates.
        """
        results: list[tuple[int, BBox]] = []
        text = label_path.read_text().strip()
        if not text:
            return results

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height

            bbox = BBox(
                x1=cx - w / 2,
                y1=cy - h / 2,
                x2=cx + w / 2,
                y2=cy + h / 2,
            )
            results.append((class_id, bbox))

        return results
