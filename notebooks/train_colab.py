"""Google Colab YOLOv8x Training Script.

This script was used to train the final YOLOv8x model on Google Colab (A100 GPU).
Run this on Colab after uploading the dataset — see COLAB_EGITIM.md for full instructions.

Results: mAP50=0.801 | Aircraft AP50=0.778 | Shipwreck AP50=0.824
"""

from ultralytics import YOLO

# 1. Load YOLOv8x (Extra Large) — highest capacity model
model = YOLO('yolov8x.pt')

# 2. Train with class filtering and optimized strategy
results = model.train(
    data='/content/sonar.yaml',
    epochs=200,
    imgsz=800,
    batch=8,
    device=0,
    project='/content/sonar_yolo_runs',
    name='v7_colab_filtered',
    exist_ok=True,
    patience=50,

    # Only train on 'aircraft' (0) and 'shipwreck' (3) classes
    classes=[0, 3],

    # Optimizer
    optimizer='AdamW',
    lr0=0.0005,
    weight_decay=0.001,
    cos_lr=True,

    # Augmentation
    augment=True,
    mosaic=0.5,
    flipud=0.5,
    fliplr=0.5,
    scale=0.3,
    translate=0.1,
    degrees=5.0,
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,

    verbose=True,
)

print('Training complete!')
print(f'Best mAP50: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
