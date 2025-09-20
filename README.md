# Multihead Mask CNN with YOLOv8-Seg

This project combines **YOLOv8 instance segmentation** with **PCA-based orientation analysis** to detect, segment, and analyze objects in real-time from a webcam feed.  
Each detected object is segmented, overlaid with a semi-transparent mask, and annotated with its bounding box, label, confidence, and principal component axes.

---

## Features
- Uses **YOLOv8-Seg** (from [Ultralytics](https://github.com/ultralytics/ultralytics)) for real-time object segmentation.
- Extracts and visualizes **object orientation** using **Principal Component Analysis (PCA)** on contours.
- Overlays **semi-transparent masks** on detected objects.
- Annotates bounding boxes with labels and confidence scores.
- Runs on **GPU (CUDA)** if available, otherwise falls back to CPU.

---

## Requirements

Install the required dependencies:

```bash
pip install ultralytics opencv-python torch numpy
