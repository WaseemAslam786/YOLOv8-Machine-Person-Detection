import os
import shutil
from ultralytics import YOLO

# 1. Directory Paths Setup
base_dir = r"D:\HCCDA-AI\Week-3\job_4493434_annotations_2026_09_20_11_26_47_yolo 1.1"
obj_train_dir = os.path.join(base_dir, "obj_train_data")
obj_names_path = os.path.join(base_dir, "obj.names")

images_train_dir = os.path.join(base_dir, "images", "train")
labels_train_dir = os.path.join(base_dir, "labels", "train")

# Ensure target directories exist
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)

# 2. Extract Class Names
classes = []
if os.path.exists(obj_names_path):
    with open(obj_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
else:
    print(f"Error: {obj_names_path} not found!")
    exit()

print(f"Detected Classes ({len(classes)}): {classes}")

# 3. Robust Search & Copy Mechanism (Walk through obj_train_data)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
moved_images = 0
moved_labels = 0

for root, _, files in os.walk(obj_train_dir):
    for file in files:
        src_path = os.path.join(root, file)
        
        if file.lower().endswith(image_extensions):
            dst_path = os.path.join(images_train_dir, file)
            shutil.copy2(src_path, dst_path)
            moved_images += 1
        elif file.lower().endswith('.txt') and file != 'train.txt':
            dst_path = os.path.join(labels_train_dir, file)
            shutil.copy2(src_path, dst_path)
            moved_labels += 1

print(f"Dataset Successfully Formatted:")
print(f" - Images copied to 'images/train': {moved_images}")
print(f" - Labels copied to 'labels/train': {moved_labels}")

if moved_images == 0:
    print("\n[CRITICAL ERROR] No images were found in 'obj_train_data'!")
    print("Please verify that your images exist inside 'obj_train_data' directory.")
    exit()

# 4. Generate data.yaml with Absolute Paths
yaml_path = os.path.join(base_dir, "data.yaml")
clean_base_path = base_dir.replace('\\', '/')

yaml_content = f"""path: {clean_base_path}
train: images/train
val: images/train

names:
"""
for idx, name in enumerate(classes):
    yaml_content += f"  {idx}: {name}\n"

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Configuration file written to: '{yaml_path}'")

# 5. Remove Stale Caches
for cache in [os.path.join(images_train_dir, "train.cache"), os.path.join(base_dir, "data.cache")]:
    if os.path.exists(cache):
        try:
            os.remove(cache)
        except Exception:
            pass

# 6. Run Training
print("\n--- Initiating YOLOv8 Training ---")
model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,  # Avoid multi-threading file locks on Windows
    name="custom_yolo_model",
    project="runs/train"
)

print("\nTraining complete!")