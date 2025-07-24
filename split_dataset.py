import os
import shutil

DATA_ROOT = "data/food-101"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
DEST_DIR = os.path.join(DATA_ROOT, "images_split")
os.makedirs(DEST_DIR, exist_ok=True)

for split in ["train", "test"]:
    with open(os.path.join(DATA_ROOT, "meta", f"{split}.txt")) as f:
        lines = f.read().splitlines()
    for line in lines:
        class_name, image_name = line.split("/")[0], line.split("/")[1]
        src = os.path.join(IMAGE_DIR, class_name, image_name + ".jpg")
        dst_dir = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, image_name + ".jpg"))

print("✅ Dataset split completed.")
