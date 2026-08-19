# 🚗 Traffic Violation Detection System Using YOLOv8

An end-to-end Computer Vision system built with YOLOv8 to detect vehicles and traffic safety violations (seatbelt / no-seatbelt compliance) in real-world camera feeds. Developed as part of the AIRI Team PITB AI Internship Task 1.

## 🎯 Problem Selection

This project addresses the problem of automated seatbelt-compliance detection in vehicles, with the system trained to detect three object classes, vehicle, seatbelt, and no_seatbelt, from traffic camera imagery. This problem is useful because seatbelt non-compliance is a major contributor to road traffic injuries and fatalities, and manual monitoring for compliance is not scalable across large road networks; an automated system makes safety enforcement more consistent and efficient. It can be applied in traffic surveillance setups at intersections and highways, automated violation-monitoring systems for law enforcement, fleet and ride-hailing driver compliance checks, and broader smart-city road-safety infrastructure. The final output of the system is a set of annotated images or video frames, where each detected vehicle and occupant is marked with a bounding box, a predicted class label (seatbelt or no_seatbelt), and a confidence score.

## 🛠️ Project Structure

Traffic-Violation-Detector/
├── dataset/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── models/
│   ├── best/
│   └── best.pt
├── notebooks/
│   └── Traffic_Violation_Detector.ipynb
├── outputs/
│   ├── predictions/
│   └── training_results/
│       └── yolov8n_violation/
├── report/
│   └── _Computer_Vision_Object_Detection_Report.pdf
├── README.md
└── requirements.txt


## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/azkatahir10/Traffic-Violation-Detector.git
cd Traffic-Violation-Detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 How to Run Inference

```python
from ultralytics import YOLO

# Load the fine-tuned best model
model = YOLO('models/best.pt')

# Run inference on test images
results = model.predict(
    source='dataset/images/test',
    conf=0.35,
    save=True,
    project='outputs/predictions',
    name='test_results'
)
```

## 📄 Project Report

### 1. Project Overview
I built an end-to-end Computer Vision object detection system using YOLOv8 to identify vehicles and seatbelt-compliance violations in image feeds. The goal was to execute a complete AI development workflow — problem definition, dataset curation, manual annotation verification, YOLO dataset formatting, model training, evaluation, and inference on unseen test data. The final system detects target classes and renders bounding boxes with confidence scores on unseen images.

### 2. Tools and Technologies Used
- **Language & Environment:** Python 3, Google Colab (Tesla T4 GPU)
- **CV & ML Libraries:** Ultralytics YOLOv8, PyTorch, OpenCV
- **Annotation & Dataset Tools:** Roboflow Annotate
- **Data Handling & Visualization:** Matplotlib, OS, Shutil, Pandas

### 3. Dataset Preparation
- **Source:** [Roboflow — `violation-atfhu-agqyn` project](https://app.roboflow.com/azka-tahir/violation-atfhu-agqyn/1/export)
- **Volume:** 200+ images
- **Classes:** `vehicle`, `seatbelt`, `no_seatbelt`
- **Annotations:** Manually reviewed and adjusted bounding boxes; exported in normalized YOLO format (`<class_id> <x_center> <y_center> <width> <height>`)
- **Split:** 70% train / 20% val / 10% test

### 4. Model Training
- **Architecture:** YOLOv8 Nano (`yolov8n.pt` pretrained weights)
- **Epochs:** 30
- **Image size:** 640x640
- **Batch size:** 8
- **Environment:** Google Colab, T4 GPU
- **Checkpoint:** Best weights saved to `models/best.pt`

### 5. Evaluation Results
- **Metrics:** Precision, Recall, mAP@0.5, mAP@0.5:0.95
- **Summary:** The model converged steadily over 30 epochs, with low bounding box and classification loss on validation data. Precision and recall showed accurate detection with tight bounding box localization.

### 6. Inference Results
Inference was run on unseen test images using `best.pt` at a confidence threshold of 0.35. The model successfully detected `vehicle`, `seatbelt`, and `no_seatbelt` instances, generating annotated outputs with bounding boxes, predicted labels, and confidence scores.

### 7. Error Analysis

| Image Name | Actual Object | Model Prediction | Error Type | Possible Reason |
|---|---|---|---|---|
| `camera1_...17_39_30.jpg` | `no_seatbelt` | `no_seatbelt (0.47)` | Low Confidence | Dark clothing contrast and windshield glare |
| `camera1_...17_43_22.jpg` | `seatbelt` | `seatbelt (0.46)` | Low Confidence | Small object scale relative to full 640x640 frame |
| `camera1_...17_43_22.jpg` | `no_seatbelt` | `no_seatbelt (0.46)` | Low Confidence | Interior vehicle shadowing and low contrast |
| `camera1_...17_43_22.jpg` | `seatbelt` | `seatbelt (0.37)` | Low Confidence | Partial occlusion by steering wheel/driver arm |
| `camera1_...17_53_05.jpg` | `no_seatbelt` | `no_seatbelt (0.41)` | Low Confidence | Distance of camera from vehicle interior |
| `camera1_...18_01_22.jpg` | `no_seatbelt` | `no_seatbelt (0.40)` | Low Confidence | Blurry contours due to motion/camera resolution |
| `camera2_...17_43_22.jpg` | `vehicle` | `vehicle (0.36)` | Low Confidence | Partial cropping at image boundary |
| `camera2_...17_59_42.jpg` | `no_seatbelt` | `no_seatbelt (0.49)` | Low Confidence | Low ambient light conditions during capture |
| `camera2_...17_59_42.jpg` | `seatbelt` | `seatbelt (0.47)` | Low Confidence | Visual similarity between strap and seat fabric |
| `camera2_...17_59_42.jpg` | `seatbelt` | `seatbelt (0.38)` | Low Confidence | Overlapping bounding boxes in passenger cabin |

**Key Findings:** Occluded objects and small instances contributed to false negatives. Adding more training samples captured under varied lighting conditions should resolve most misclassifications.

### 8. What I Learned
- Executed the complete end-to-end lifecycle of a real-world Computer Vision project
- Learned manual annotation techniques and normalized YOLO format structures
- Fine-tuned pretrained YOLOv8 models on GPU hardware using Ultralytics
- Evaluated models using Precision, Recall, mAP scores, and confusion matrices
- Analyzed failure modes to outline dataset quality improvements

### 9. Future Improvements
- Expand the dataset to 500+ images for greater class balance
- Experiment with larger YOLO variants (YOLOv8s or YOLOv8m)
- Deploy the model as an interactive web demo using Streamlit or FastAPI
- Add low-light and occluded examples to strengthen weak-confidence classes

## Author

Azka Tahir
