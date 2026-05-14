import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
from src.config import *


def load_model(weights_path=None):
    path = weights_path or str(WEIGHTS_DIR / "best.pt")
    model = YOLO(path)
    print(f"✅ Model loaded: {path}")
    return model


def detect_and_count(model, image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
    """
    Run detection on a single image (numpy BGR array).
    Returns: annotated image, human_count, car_count, raw results
    """
    results  = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    annotated = image.copy()

    human_count = 0
    car_count   = 0
    detections  = []

    for box in results.boxes:
        cid        = int(box.cls)
        confidence = float(box.conf)
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        label      = CLASS_NAMES.get(cid, f"class_{cid}")
        color      = CLASS_COLORS.get(cid, (255, 255, 255))

        # Count
        if cid in HUMAN_CLASS_IDS:
            human_count += 1
        elif cid in CAR_CLASS_IDS:
            car_count += 1

        detections.append({
            "class_id"  : cid,
            "label"     : label,
            "confidence": confidence,
            "bbox"      : [x1, y1, x2, y2],
        })

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text       = f"{label} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text     = max(y1 - 4, th + 4)
        cv2.rectangle(annotated,
                      (x1, y_text - th - 4),
                      (x1 + tw + 2, y_text),
                      color, -1)
        cv2.putText(annotated, text,
                    (x1 + 1, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)

    # Draw counting overlay
    annotated = draw_count_overlay(annotated, human_count, car_count)

    return annotated, human_count, car_count, detections


def draw_count_overlay(image, human_count, car_count):
    """Draw a clean count panel on top-left of image."""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Panel background
    panel_w, panel_h = 220, 90
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Title
    cv2.putText(image, "DETECTION COUNT",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Human count (green)
    cv2.putText(image, f"Humans : {human_count}",
                (18, 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 100), 2, cv2.LINE_AA)

    # Car count (blue)
    cv2.putText(image, f"Cars   : {car_count}",
                (18, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 120, 255), 2, cv2.LINE_AA)

    return image


def process_image(image_path, output_path=None, model=None, conf=CONF_THRESHOLD):
    """Process a single image file."""
    if model is None:
        model = load_model()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated, humans, cars, detections = detect_and_count(model, img, conf=conf)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"✅ Saved → {output_path}")

    return annotated, humans, cars, detections


def process_video(video_path, output_path=None, model=None, conf=CONF_THRESHOLD):
    """Process a video file frame by frame."""
    if model is None:
        model = load_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_id    = 0
    total_humans = 0
    total_cars   = 0

    print(f"Processing video: {Path(video_path).name}")
    print(f"  Resolution : {width}×{height} @ {fps}fps")
    print(f"  Frames     : {total}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, humans, cars, _ = detect_and_count(model, frame, conf=conf)
        total_humans = max(total_humans, humans)
        total_cars   = max(total_cars, cars)

        if writer:
            writer.write(annotated)

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  Frame {frame_id}/{total} — humans: {humans}, cars: {cars}")

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Video saved → {output_path}")

    return total_humans, total_cars


if __name__ == "__main__":
    import random

    model   = load_model()
    val_dir = DATA_RAW_DIR / "VisDrone2019-DET-val" / IMAGES_FOLDER
    img_path = random.choice(list(val_dir.glob("*.jpg")))
    out_path = OUTPUT_IMAGES_DIR / f"detected_{img_path.name}"

    annotated, humans, cars, detections = process_image(
        img_path, out_path, model
    )

    print(f"\nImage    : {img_path.name}")
    print(f"Humans   : {humans}")
    print(f"Cars     : {cars}")
    print(f"Total    : {len(detections)} objects detected")
    print(f"Output   : {out_path}")