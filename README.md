# 🚁 Drone Human & Car Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![Flask](https://img.shields.io/badge/Flask-3.0-green)

A drone-based computer vision system for detecting and counting **humans** and **cars** from aerial images and videos using a fine-tuned YOLOv8 model.

---

## 📌 Features

- Human and car detection from drone images
- Object counting
- Video detection and tracking
- Flask-based web interface
- YOLOv8 training and evaluation notebooks
- Fine-tuned on the VisDrone dataset

---

## 📸 Demo

### Detection Result
![Detection Result](assets/sample_images/inference_results.png)

### Training Curves
![Training Curves](assets/sample_images/training_curves.png)

### Sample Annotations
![Sample Annotations](assets/sample_images/sample_annotations_grid.png)

---

## 📊 Results

| Metric | Value |
|---|---:|
| Overall mAP@50 | 29.6% |
| Human mAP@50 | 27.2% |
| Car mAP@50 | 52.8% |
| Precision | 58.6% |
| Recall | 35.4% |
| FPS | 39.1 |

---

## 📁 Project Structure

```text
drone-detection/
├── assets/
│   └── sample_images/
├── models/
│   └── weights/
│       └── best.pt
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_inference.ipynb
├── outputs/
├── src/
│   ├── config.py
│   ├── detect.py
│   ├── train.py
│   └── track.py
├── webapp/
│   ├── app.py
│   ├── static/
│   └── templates/
├── dataset.yaml
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

This project uses the **VisDrone2019-DET** dataset.

Download from Kaggle:

[VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

Place the dataset inside:

```text
data/raw/VisDrone/
```

---

## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/RX-kabir/drone-detection.git
cd drone-detection
```

Create and activate virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Web App

```bash
python webapp/app.py
```

Open in browser:

```text
http://localhost:5000
```

### Train Model

```bash
python src/train.py
```

### Run Detection

```python
from src.detect import load_model, process_image

model = load_model()

annotated, humans, cars, detections = process_image(
    "path/to/image.jpg",
    "outputs/result.jpg",
    model
)

print("Humans:", humans)
print("Cars:", cars)
```

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| Model | YOLOv8s |
| Epochs | 50 |
| Batch Size | 8 |
| Image Size | 640 |
| Optimizer | AdamW |
| Dataset | VisDrone2019-DET |
| GPU | NVIDIA RTX 4050 |

---

## 🔍 Challenges

- Small objects in aerial images
- Dense crowds and traffic
- Occlusion between objects
- Imbalanced dataset classes
- Aerial-view domain specificity

---

## 🛠️ Tech Stack

- Python
- YOLOv8
- PyTorch
- OpenCV
- Flask
- Pandas
- Matplotlib

---

## 👤 Author

**Mohammad Adnan Kabir**

Antlings Internship Program 2026  
AI/ML Technical Assessment
