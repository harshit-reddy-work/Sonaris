from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class ModelMode(Enum):
    LITE = "LITE"
    WAR = "WAR"
    PRO = "PRO"


@dataclass(frozen=True)
class GPSConfig:
    mode: str = "manual"
    manual_lat: float = 0.0
    manual_lon: float = 0.0
    metadata_path: Path | None = None


@dataclass(frozen=True)
class PathConfig:
    root: Path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir: Path = Path("data/archive")
    demo_dir: Path = Path("data/demo")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    yolo11n_seg_weights: Path = Path("models/yolo11n_seg_best.pt")
    yolo11n_war_weights: Path = Path("models/yolo11n_war_best.pt")
    yolov8x_weights: Path = Path("models/yolov8x_best.pt")
    sam_weights: Path = Path("models/sam_vit_b_01ec64.pth")

    @property
    def train_images(self) -> Path:
        return self.data_dir / "train" / "images"

    @property
    def train_labels(self) -> Path:
        return self.data_dir / "train" / "labels"

    @property
    def valid_images(self) -> Path:
        return self.data_dir / "valid" / "images"

    @property
    def valid_labels(self) -> Path:
        return self.data_dir / "valid" / "labels"

    @property
    def test_images(self) -> Path:
        return self.data_dir / "test" / "images"

    @property
    def test_labels(self) -> Path:
        return self.data_dir / "test" / "labels"


@dataclass(frozen=True)
class PreprocessConfig:
    median_kernel: int = 5
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    nlm_h: int = 10
    target_size: tuple[int, int] = (1024, 1024)


@dataclass(frozen=True)
class SAMConfig:
    model_type: str = "vit_b"
    checkpoint: str = "models/sam_vit_b_01ec64.pth"
    device: str = "cpu"
    multimask_output: bool = True
    score_threshold: float = 0.5


@dataclass(frozen=True)
class SonarConfig:
    resolution_m_per_px: float = 1.0
    circularity_threshold: float = 0.85
    rectangularity_threshold: float = 0.80
    elongation_threshold: float = 3.0


CLASS_NAMES: dict[int, str] = {
    0: "aircraft",
    1: "fish",
    2: "other",
    3: "shipwreck",
}


@dataclass(frozen=True)
class SonarisMasterConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    sonar: SonarConfig = field(default_factory=SonarConfig)
    class_names: dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    model_mode: ModelMode = ModelMode.LITE
    gps: GPSConfig = field(default_factory=GPSConfig)


# Alias for backward compatibility
Config = SonarisMasterConfig
