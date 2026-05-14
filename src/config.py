import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Data Paths ──────────────────────────────────────────────
DATA_RAW_DIR        = ROOT_DIR / "data" / "raw" / "VisDrone"
DATA_PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
DATASET_YAML        = ROOT_DIR / "dataset.yaml"

# ─── Model Paths ─────────────────────────────────────────────
MODELS_DIR          = ROOT_DIR / "models"
WEIGHTS_DIR         = MODELS_DIR / "weights"
RUNS_DIR            = MODELS_DIR / "runs"

# ─── Output Paths ────────────────────────────────────────────
OUTPUTS_DIR         = ROOT_DIR / "outputs"
OUTPUT_IMAGES_DIR   = OUTPUTS_DIR / "images"
OUTPUT_VIDEOS_DIR   = OUTPUTS_DIR / "videos"
OUTPUT_TRACKING_DIR = OUTPUTS_DIR / "tracking"

# ─── Dataset Folder Names ────────────────────────────────────
LABELS_FOLDER       = "labels"
IMAGES_FOLDER       = "images"

# ─── Model Config ────────────────────────────────────────────
MODEL_BASE          = "yolov8n.pt"
IMG_SIZE            = 640
CONF_THRESHOLD      = 0.25
IOU_THRESHOLD       = 0.45
EPOCHS              = 50
BATCH_SIZE          = 16
DEVICE              = "0"

# ─── All Class Names (10 classes) ────────────────────────────
CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# ─── Counting Logic Class IDs ────────────────────────────────
HUMAN_CLASS_IDS     = [0, 1]   # pedestrian + people → humans
CAR_CLASS_IDS       = [3, 4]   # car + van           → cars

# ─── Display Colors (BGR) ────────────────────────────────────
CLASS_COLORS = {
    0: (0,   255, 100),   # pedestrian → green
    1: (0,   200,  80),   # people     → green shade
    2: (255, 165,   0),   # bicycle    → orange
    3: (0,   120, 255),   # car        → blue
    4: (0,    80, 200),   # van        → blue shade
    5: (220,  20,  60),   # truck      → red
    6: (148,   0, 211),   # tricycle   → purple
    7: (255,  20, 147),   # awning     → pink
    8: (139,  69,  19),   # bus        → brown
    9: (0,   206, 209),   # motor      → cyan
}

# ─── Web App ─────────────────────────────────────────────────
UPLOAD_FOLDER       = ROOT_DIR / "webapp" / "static" / "uploads"
RESULTS_FOLDER      = ROOT_DIR / "webapp" / "static" / "results"
ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}
MAX_CONTENT_MB      = 100