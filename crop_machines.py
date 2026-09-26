import os
import cv2
from ultralytics import YOLO

# 1. Paths Setup
base_dir = r"D:\HCCDA-AI\Week-3\job_4493434_annotations_2026_09_20_11_26_47_yolo 1.1"
image_path = os.path.join(base_dir, "construction-675-_jpg.rf.b84f55bdd16f9e64e8933d093eedd012.jpg")
output_folder = os.path.join(base_dir, "cropped_detections")

# Folder for cropped images
os.makedirs(output_folder, exist_ok=True)

# 2. Load Model
custom_weights = os.path.join(base_dir, "runs", "train", "custom_yolo_model", "weights", "best.pt")
if os.path.exists(custom_weights):
    print("Using Custom Model...")
    model = YOLO(custom_weights)
else:
    print("Using Base YOLOv8 Model...")
    model = YOLO("yolov8n.pt")

# 3. Read Image & Run Detection
img = cv2.imread(image_path)
results = model.predict(source=image_path, conf=0.25)

# 4. Crop & Save Detected Objects
crop_count = 0
for r in results:
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # Extract Bounding Box Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Crop using OpenCV / Numpy Slicing
        cropped_img = img[y1:y2, x1:x2]
        
        # Save Cropped Portion
        save_name = f"{class_name}_{i+1}.jpg"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, cropped_img)
        
        crop_count += 1
        print(f"Saved Crop: {save_name}")

print(f"\nDone! All {crop_count} crops saved inside: '{output_folder}'")