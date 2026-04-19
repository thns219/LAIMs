import os
import shutil
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "diffusion_data")

random.seed(42)

classes = ["0_real", "1_fake"]

for cls in classes:
    src_folder = os.path.join(DATA_DIR, cls)

    files = [f for f in os.listdir(src_folder)
             if f.lower().endswith((".jpg",".jpeg",".png"))]

    random.shuffle(files)

    total = len(files)

    train_end = int(total * 0.7)
    val_end   = int(total * 0.85)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        dst_folder = os.path.join(DATA_DIR, split_name, cls)
        os.makedirs(dst_folder, exist_ok=True)

        for file in split_files:
            shutil.copy(
                os.path.join(src_folder, file),
                os.path.join(dst_folder, file)
            )

print("DONE SPLIT DATASET")