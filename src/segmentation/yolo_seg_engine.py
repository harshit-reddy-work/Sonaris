from __future__ import annotations
import numpy as np
import time
from ultralytics import YOLO
from ..core.interfaces import BBox, Detection, SegmentationResult, SegmentationEngine

class YOLOSegEngine(SegmentationEngine):
    """Wraps YOLO11n-Seg inference."""
    def __init__(self, model_path: str, confidence: float = 0.5, image_size: int = 1024):
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.model = None

    def load_model(self) -> None:
        if self.model is None:
            self.model = YOLO(self.model_path)

    def segment(self, image: np.ndarray, detections: list[Detection] | None = None) -> SegmentationResult:
        self.load_model()
        t0 = time.perf_counter()
        
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )
        
        inference_time_ms = (time.perf_counter() - t0) * 1000
        
        out_detections = []
        if len(results) > 0:
            result = results[0]
            names = result.names
            if result.boxes is not None and result.masks is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                masks = result.masks.data.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cls_id = int(clss[i])
                    conf = float(confs[i])
                    class_name = names.get(cls_id, str(cls_id))
                    
                    import cv2
                    mask_img = masks[i]
                    if mask_img.shape != image.shape[:2]:
                        mask_img = cv2.resize(mask_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_img > 0.5).astype(np.uint8) * 255
                    
                    det = Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        bbox=BBox(x1, y1, x2, y2),
                        mask=mask_binary,
                        confidence=conf
                    )
                    out_detections.append(det)
        
        return SegmentationResult(
            image_path="",
            detections=out_detections,
            inference_time_ms=inference_time_ms,
            preprocessed_image=image
        )
