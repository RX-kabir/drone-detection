import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import uuid
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from src.config import *
from src.detect import detect_and_count
from src.track  import trajectory_history

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = YOLO(str(WEIGHTS_DIR / "best.pt"))
print("✅ Model ready!")


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    ext      = filename.rsplit(".", 1)[1].lower()
    in_name  = f"{uid}_input.{ext}"
    out_name = f"{uid}_result.jpg"
    in_path  = UPLOAD_FOLDER  / in_name
    out_path = RESULTS_FOLDER / out_name
    file.save(str(in_path))

    conf = float(request.form.get("conf", CONF_THRESHOLD))

    try:
        if ext in {"mp4", "avi", "mov"}:
            cap    = cv2.VideoCapture(str(in_path))
            fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_name = f"{uid}_result.mp4"
            out_path = RESULTS_FOLDER / out_name
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(
                str(out_path), fourcc, fps, (width, height))

            max_humans, max_cars, frame_count = 0, 0, 0
            trajectory_history.clear()

            while frame_count < min(total, fps * 30):
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, h, c, _ = detect_and_count(
                    model, frame, conf=conf)
                max_humans = max(max_humans, h)
                max_cars   = max(max_cars, c)
                writer.write(annotated)
                frame_count += 1

            cap.release()
            writer.release()

            return jsonify({
                "type"      : "video",
                "result_url": f"/results/{out_name}",
                "humans"    : max_humans,
                "cars"      : max_cars,
                "frames"    : frame_count,
            })

        else:
            start    = time.time()
            img      = cv2.imread(str(in_path))
            if img is None:
                return jsonify({"error": "Could not read image"}), 400

            h_img, w_img = img.shape[:2]
            annotated, humans, cars, detections = detect_and_count(
                model, img, conf=conf)
            cv2.imwrite(str(out_path), annotated)
            elapsed = round((time.time() - start) * 1000, 1)

            breakdown = {}
            for d in detections:
                lbl = d["label"]
                breakdown[lbl] = breakdown.get(lbl, 0) + 1

            return jsonify({
                "type"      : "image",
                "result_url": f"/results/{out_name}",
                "humans"    : humans,
                "cars"      : cars,
                "total"     : len(detections),
                "breakdown" : breakdown,
                "size"      : f"{w_img}×{h_img}",
                "time_ms"   : elapsed,
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(str(RESULTS_FOLDER), filename)


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)