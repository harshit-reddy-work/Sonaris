"""Entry point: wires up all components and runs the Sonaris pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from src.core.config import Config  # alias for SonarisMasterConfig
from src.dataset import SonarDataset, BoxPrompt, PointPrompt, BoxPointPrompt
from src.preprocessing import (
    PreprocessingPipeline,
    MedianDenoiser,
    CLAHEEnhancer,
    GrayscaleToRGB,
    ImageResizer,
)
from src.segmentation import SAMSegmenter
from src.postprocessing import SizeEstimator, ShapeClassifier
from src.visualization import MaskOverlayVisualizer, ReportGenerator
from src.pipeline import SonarAnalysisPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_STRATEGIES = {
    "box": BoxPrompt,
    "point": PointPrompt,
    "box_point": BoxPointPrompt,
}

BANNER = r"""
  ____                        _     
 / ___|  ___  _ __   __ _ _ __(_)___ 
 \___ \ / _ \| '_ \ / _` | '__| / __|
  ___) | (_) | | | | (_| | |  | \__ \
 |____/ \___/|_| |_|\__,_|_|  |_|___/
                                     
 Unified Sonar Intelligence Platform
"""

def build_pipeline(config: Config, prompt_type: str, mode: str) -> SonarAnalysisPipeline:
    """Composition Root: assemble all components."""

    # Preprocessing chain
    preprocessor = PreprocessingPipeline([
        MedianDenoiser(config.preprocess.median_kernel),
        CLAHEEnhancer(
            config.preprocess.clahe_clip_limit,
            config.preprocess.clahe_tile_grid,
        ),
        GrayscaleToRGB(),
        ImageResizer(config.preprocess.target_size),
    ])

    # Segmentation engine
    # In Sonaris we support mode selection (lite vs pro)
    # The segmenter logic would adapt based on the mode
    segmenter = SAMSegmenter(config.sam)
    segmenter.load_model()

    # Prompt strategy
    strategy_cls = PROMPT_STRATEGIES.get(prompt_type, BoxPrompt)
    prompt_strategy = strategy_cls()

    # Post-processors
    post_processors = [
        SizeEstimator(config.sonar),
        ShapeClassifier(config.sonar),
    ]

    # Visualizer
    visualizer = MaskOverlayVisualizer(alpha=0.4)

    return SonarAnalysisPipeline(
        preprocessor=preprocessor,
        segmenter=segmenter,
        prompt_strategy=prompt_strategy,
        post_processors=post_processors,
        visualizer=visualizer,
        class_names=config.class_names,
    )


def main() -> None:
    print(BANNER)
    parser = argparse.ArgumentParser(description="Sonaris CLI - Unified Sonar Intelligence Platform")
    parser.add_argument(
        "--mode", choices=["lite", "pro"], default="lite",
        help="Inference mode: lite (YOLO11n-Seg) or pro (YOLOv8x)",
    )
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], default="test",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--prompt", choices=list(PROMPT_STRATEGIES.keys()), default="point",
        help="SAM prompt strategy",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max images to process (0 = all)",
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="Save visualization images",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Inference device",
    )
    parser.add_argument(
        "--gps-lat", type=float, default=None,
        help="Optional GPS Latitude context",
    )
    parser.add_argument(
        "--gps-lon", type=float, default=None,
        help="Optional GPS Longitude context",
    )
    args = parser.parse_args()

    config = Config()

    # Override device from CLI
    sam_cfg = config.sam
    if args.device != sam_cfg.device:
        from dataclasses import replace
        config = replace(config, sam=replace(sam_cfg, device=args.device))

    # Select split paths
    split_map = {
        "train": (config.paths.train_images, config.paths.train_labels),
        "valid": (config.paths.valid_images, config.paths.valid_labels),
        "test": (config.paths.test_images, config.paths.test_labels),
    }
    img_dir, lbl_dir = split_map[args.split]

    # Load dataset
    dataset = SonarDataset(img_dir, lbl_dir, config.class_names)
    logger.info("Loaded %d images from '%s' split", len(dataset), args.split)

    # Build pipeline
    pipeline = build_pipeline(config, args.prompt, args.mode)
    report_gen = ReportGenerator()

    # Process
    results = []
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))

    for i in tqdm(range(n), desc="Processing"):
        sample = dataset[i]
        
        # Add GPS info if provided
        metadata = {}
        if args.gps_lat is not None and args.gps_lon is not None:
            metadata["gps"] = (args.gps_lat, args.gps_lon)
            
        result = pipeline.process_image(
            image=sample["image"],
            annotations=sample["annotations"],
            image_path=sample["image_path"],
            # metadata=metadata # Pass metadata here if pipeline is updated to accept it
        )
        results.append(result)

        # Save per-image JSON report
        img_name = Path(sample["image_path"]).stem
        report_gen.to_json(
            result,
            Path(config.paths.outputs_dir) / "reports" / f"{img_name}.json",
        )

        # Save visualization
        if args.save_vis:
            vis = pipeline.visualize(sample["image"], result)
            if vis is not None:
                vis_path = Path(config.paths.outputs_dir) / "vis" / f"{img_name}.jpg"
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(vis_path), vis)

    # Summary CSV
    report_gen.write_csv(
        results,
        Path(config.paths.outputs_dir) / "results.csv",
    )

    # Print summary
    total_dets = sum(len(r.detections) for r in results)
    avg_time = sum(r.inference_time_ms for r in results) / max(len(results), 1)
    logger.info("Done: %d images, %d detections, avg %.1f ms/image", n, total_dets, avg_time)


if __name__ == "__main__":
    main()
