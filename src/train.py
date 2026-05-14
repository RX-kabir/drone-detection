import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO
from src.config import *

def main():
    print("=" * 60)
    print("  DRONE DETECTION — YOLOv8 TRAINING")
    print("=" * 60)
    print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset : {DATASET_YAML}")
    print(f"Model   : yolov8s.pt")
    print(f"Epochs  : {EPOCHS}")
    print(f"Batch   : 8")
    print("=" * 60)

    model = YOLO("yolov8s.pt")

    results = model.train(
        data             = str(DATASET_YAML),
        epochs           = EPOCHS,
        imgsz            = IMG_SIZE,
        batch            = 8,
        device           = 0 if torch.cuda.is_available() else "cpu",
        project          = str(MODELS_DIR / "runs"),
        name             = "visdrone_yolov8",
        exist_ok         = True,
        pretrained       = True,
        optimizer        = "AdamW",
        lr0              = 0.001,
        lrf              = 0.01,
        momentum         = 0.937,
        weight_decay     = 0.0005,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        box              = 7.5,
        cls              = 0.5,
        hsv_h            = 0.015,
        hsv_s            = 0.7,
        hsv_v            = 0.4,
        degrees          = 0.0,
        translate        = 0.1,
        scale            = 0.5,
        fliplr           = 0.5,
        mosaic           = 1.0,
        mixup            = 0.1,
        copy_paste       = 0.1,
        cache            = False,
        val              = True,
        save             = True,
        save_period      = 10,
        plots            = True,
        verbose          = True,
        workers          = 4,       # reduced workers for Windows stability
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best weights : {MODELS_DIR}/runs/visdrone_yolov8/weights/best.pt")


if __name__ == '__main__':
    main()