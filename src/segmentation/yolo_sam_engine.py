"""YOLO + SAM birleşik pipeline.

YOLO: Nesne tespiti + sınıflandırma (ne ve nerede?)
SAM: Piksel bazında segmentasyon (kesin sınırlar)

Bu yaklaşım her modelin güçlü yanını kullanır:
- YOLO sonar verisine özel eğitilmiştir → doğru sınıf tahmini
- SAM genel segmentasyon ustasıdır → piksel-hassas maske
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.interfaces import BBox, Detection, Segmenter

logger = logging.getLogger(__name__)


class YOLOSAMSegmenter(Segmenter):
    """YOLO ile tespit, SAM ile segmentasyon yapan birleşik motor.

    Akış:
    1. YOLO görüntüyü tarar → bbox + sınıf + güven
    2. Her tespit için SAM'a bbox prompt verilir → piksel maskesi
    """

    def __init__(
        self,
        yolo_weights: str = "models/yolo_sonar/weights/best.pt",
        sam_checkpoint: str = "models/sam_vit_b_01ec64.pth",
        device: str = "mps",
        confidence_threshold: float = 0.25,
    ) -> None:
        self._yolo_weights = yolo_weights
        self._sam_checkpoint = sam_checkpoint
        self._device = device
        self._conf_threshold = confidence_threshold
        self._yolo = None
        self._sam_predictor = None

    def load_model(self) -> None:
        from ultralytics import YOLO
        from segment_anything import sam_model_registry, SamPredictor

        # YOLO
        self._yolo = YOLO(self._yolo_weights)
        logger.info("YOLO loaded: %s", self._yolo_weights)

        # SAM
        sam = sam_model_registry["vit_b"](checkpoint=self._sam_checkpoint)
        sam.to(self._device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("SAM loaded on %s", self._device)

    def detect_and_segment(
        self,
        image: np.ndarray,
    ) -> list[Detection]:
        """Tam pipeline: YOLO tespit + SAM segmentasyon.

        Args:
            image: BGR uint8 görüntü (orijinal boyut)

        Returns:
            Detection listesi (sınıf, bbox, maske, güven)
        """
        if self._yolo is None or self._sam_predictor is None:
            raise RuntimeError("Model yüklenmedi. load_model() çağırın.")

        # 1. YOLO tespiti
        results = self._yolo.predict(
            image, conf=self._conf_threshold, verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        class_names = results[0].names

        # 2. SAM için görüntüyü hazırla
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(img_rgb)

        # 3. Her YOLO tespiti için SAM maskesi üret
        detections: list[Detection] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            cls_name = class_names[cls_id]

            bbox = BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])

            # SAM box prompt
            sam_box = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
            masks, scores, _ = self._sam_predictor.predict(
                box=sam_box,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx]

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                bbox=bbox,
                mask=mask.astype(np.uint8),
                confidence=conf,
            ))

        return detections

    # Segmenter interface uyumluluğu
    def segment(
        self,
        image: np.ndarray,
        prompts: list[dict[str, Any]],
    ) -> list[tuple[np.ndarray, float]]:
        """Segmenter interface - geriye uyumluluk."""
        detections = self.detect_and_segment(image)
        return [(d.mask, d.confidence) for d in detections]

    @property
    def is_loaded(self) -> bool:
        return self._yolo is not None and self._sam_predictor is not None
