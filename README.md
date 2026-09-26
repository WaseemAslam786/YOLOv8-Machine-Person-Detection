# 🏗️ Machine & Person Detection using YOLOv8

An end-to-end Computer Vision pipeline built with **Ultralytics YOLOv8** and **OpenCV** to detect, process, and crop objects (*Machines* and *Persons*) in construction and industrial environments.

This project handles custom annotation conversion (CVAT/Darknet YOLO format to Ultralytics YOLOv8 structure), model training, batch processing across image directories, and automated object cropping.

---

## 📌 Features

- **Automated Dataset Structuring:** Converts CVAT/YOLO custom annotations (`obj_train_data`, `obj.names`) into standard YOLOv8 directory structures (`images/train`, `labels/train`).
- **YOLOv8 Model Training:** Custom training pipeline for detecting `Machines` and `Persons`.
- **Batch Processing:** Runs batch inference on entire image directories and outputs annotated images with confidence scores.
- **OpenCV Object Cropping:** Extracts bounding box coordinates ($x_1, y_1, x_2, y_2$) and saves detected objects as standalone cropped images.

---

## 📂 Project Structure

```text
├── obj_train_data/         # Raw annotations (.txt) and source images
├── images/
│   └── train/              # Processed training images
├── labels/
│   └── train/              # Processed annotation files
├── cropped_detections/     # Output directory for OpenCV cropped objects
├── runs/                   # YOLOv8 training & inference outputs
├── obj.names               # Class label definitions (Machines, Persons)
├── obj.data                # Dataset metadata
├── data.yaml               # Generated YOLOv8 configuration file
├── prepare_and_train.py    # Dataset restructuring & model training script
├── batch_detection.py      # Batch inference script for image folders
└── crop_machines.py        # OpenCV bounding box cropping script
