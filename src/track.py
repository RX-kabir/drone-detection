import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import random
from collections import defaultdict
from ultralytics import YOLO
from src.config import *


# Store trajectory history per track ID
trajectory_history = defaultdict(list)
TRAJECTORY_LENGTH  = 30   # max trail length in frames


def draw_tracking_overlay(frame, track_data, human_count, car_count):
    """Draw count panel + track IDs + trajectories."""
    h, w = frame.shape[:2]

    # ── Count panel ──────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (240, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "TRACKING COUNT",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Humans : {human_count}",
                (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Cars   : {car_count}",
                (18, 86), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 120, 255), 2, cv2.LINE_AA)

    # ── Per-track boxes + IDs + trails ───────────────────────
    for tid, cid, bbox, color in track_data:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: class + track ID
        label = f"{CLASS_NAMES.get(cid, '?')} #{tid}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(y1 - 4, th + 4)
        cv2.rectangle(frame,
                      (x1, y_text - th - 4),
                      (x1 + tw + 2, y_text),
                      color, -1)
        cv2.putText(frame, label,
                    (x1 + 1, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory trail
        trajectory_history[tid].append((cx, cy))
        if len(trajectory_history[tid]) > TRAJECTORY_LENGTH:
            trajectory_history[tid].pop(0)

        pts = trajectory_history[tid]
        for i in range(1, len(pts)):
            alpha = int(255 * i / len(pts))
            trail_color = tuple(int(c * alpha / 255) for c in color)
            cv2.line(frame, pts[i-1], pts[i], trail_color, 2)

    return frame


def track_video(video_path, output_path=None, model=None,
                conf=CONF_THRESHOLD, max_frames=None):
    """Run ByteTrack on a video file."""

    if model is None:
        model = YOLO(str(WEIGHTS_DIR / "best.pt"))
        print(f"✅ Model loaded")

    trajectory_history.clear()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {Path(video_path).name}")
    print(f"  {width}×{height} @ {fps}fps | {total} frames")

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height))

    frame_id   = 0
    all_humans = set()
    all_cars   = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_id >= max_frames:
            break

        # Run tracking (ByteTrack built-in)
        results = model.track(
            frame,
            conf      = conf,
            iou       = IOU_THRESHOLD,
            tracker   = "bytetrack.yaml",
            persist   = True,      # keep track IDs across frames
            verbose   = False,
        )[0]

        track_data    = []
        human_count   = 0
        car_count     = 0

        if results.boxes.id is not None:
            for box in results.boxes:
                cid   = int(box.cls)
                tid   = int(box.id)
                bbox  = list(map(int, box.xyxy[0].tolist()))
                color = CLASS_COLORS.get(cid, (255, 255, 255))

                track_data.append((tid, cid, bbox, color))

                if cid in HUMAN_CLASS_IDS:
                    human_count += 1
                    all_humans.add(tid)
                elif cid in CAR_CLASS_IDS:
                    car_count += 1
                    all_cars.add(tid)

        frame = draw_tracking_overlay(
            frame, track_data, human_count, car_count)

        if writer:
            writer.write(frame)

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"  Frame {frame_id}/{total} | "
                  f"humans: {human_count}, cars: {car_count} | "
                  f"unique human IDs: {len(all_humans)}, "
                  f"unique car IDs: {len(all_cars)}")

    cap.release()
    if writer:
        writer.release()
        print(f"\n✅ Tracking video saved → {output_path}")

    print(f"\n── Tracking Summary ──────────────────")
    print(f"  Frames processed      : {frame_id}")
    print(f"  Unique human track IDs: {len(all_humans)}")
    print(f"  Unique car track IDs  : {len(all_cars)}")

    return frame_id, len(all_humans), len(all_cars)


def track_image_sequence(image_paths, output_dir, model=None, conf=CONF_THRESHOLD):
    """Run tracking on a sequence of images (simulates video)."""

    if model is None:
        model = YOLO(str(WEIGHTS_DIR / "best.pt"))

    trajectory_history.clear()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_humans = set()
    all_cars   = set()

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.track(
            frame,
            conf    = conf,
            iou     = IOU_THRESHOLD,
            tracker = "bytetrack.yaml",
            persist = True,
            verbose = False,
        )[0]

        track_data  = []
        human_count = 0
        car_count   = 0

        if results.boxes.id is not None:
            for box in results.boxes:
                cid   = int(box.cls)
                tid   = int(box.id)
                bbox  = list(map(int, box.xyxy[0].tolist()))
                color = CLASS_COLORS.get(cid, (255, 255, 255))
                track_data.append((tid, cid, bbox, color))

                if cid in HUMAN_CLASS_IDS:
                    human_count += 1
                    all_humans.add(tid)
                elif cid in CAR_CLASS_IDS:
                    car_count += 1
                    all_cars.add(tid)

        frame = draw_tracking_overlay(
            frame, track_data, human_count, car_count)

        out_file = output_dir / f"tracked_{i:04d}_{Path(img_path).name}"
        cv2.imwrite(str(out_file), frame)

    print(f"✅ Tracked {len(image_paths)} images → {output_dir}")
    print(f"   Unique human IDs : {len(all_humans)}")
    print(f"   Unique car IDs   : {len(all_cars)}")
    return len(all_humans), len(all_cars)


if __name__ == "__main__":
    # Demo: track a sequence of 20 val images
    model   = YOLO(str(WEIGHTS_DIR / "best.pt"))
    val_dir = DATA_RAW_DIR / "VisDrone2019-DET-val" / IMAGES_FOLDER
    imgs    = sorted(val_dir.glob("*.jpg"))[:20]

    humans, cars = track_image_sequence(
        imgs,
        output_dir = OUTPUT_TRACKING_DIR,
        model      = model,
    )
    print(f"\nDone! Unique humans tracked: {humans}, cars: {cars}")