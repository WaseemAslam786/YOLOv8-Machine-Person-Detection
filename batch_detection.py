import os
from ultralytics import YOLO

# 1. Paths Configuration
base_dir = r"D:\HCCDA-AI\Week-3\job_4493434_annotations_2026_09_20_11_26_47_yolo 1.1"

# Jis folder mein kafi sari images hain, uska path yahan dein:
input_images_folder = os.path.join(base_dir, "images", "train")  # ya koi bhi images wala folder

# Detections save hone ka target folder name
output_folder_name = "batch_detection_results"

# 2. Load Model (Trained Weights ya Base YOLOv8)
custom_weights = os.path.join(base_dir, "runs", "train", "custom_yolo_model", "weights", "best.pt")

if os.path.exists(custom_weights):
    print("Loading Trained Custom Model...")
    model = YOLO(custom_weights)
else:
    print("Custom weights not found. Loading Pre-trained YOLOv8 Model...")
    model = YOLO("yolov8n.pt")

# 3. Batch Detection across all images in the folder
print(f"\nStarting batch detection for images in: '{input_images_folder}'...")

results = model.predict(
    source=input_images_folder,  # Single image ki jagah poore folder ka path
    conf=0.25,                   # Confidence threshold (25%)
    save=True,                   # Detections wali images auto-save hongi
    project=base_dir,            # Output location
    name=output_folder_name      # Sub-folder name
)

print(f"\nBatch Detection Complete!")
print(f"All annotated images with bounding boxes are saved in:")
print(f"-> {os.path.join(base_dir, output_folder_name)}")