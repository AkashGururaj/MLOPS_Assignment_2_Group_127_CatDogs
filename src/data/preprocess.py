import os
import shutil
import random
import subprocess
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===============================
# Config
# ===============================
RAW_DIR = "data/raw/PetImages"
PROCESSED_DIR = "data/processed"
CLASSES = ["Cat", "Dog"]
SAMPLES_PER_CLASS = 250   # 250 Cat + 250 Dog = 500 total

# ===============================
# DVC Tracking
# ===============================
def track_raw_with_dvc():
    try:
        if not os.path.exists(f"{RAW_DIR}.dvc"):
            subprocess.run(["dvc", "add", RAW_DIR], check=True)
            subprocess.run(["git", "add", f"{RAW_DIR}.dvc"], check=True)
            subprocess.run(["git", "commit", "-m", "Track raw data with DVC"], check=True)
            print("[INFO] Raw data tracked with DVC")
        else:
            print("[INFO] Raw data already tracked with DVC")
    except Exception:
        print("[WARNING] DVC/Git not available — skipping tracking")

# ===============================
# Simple Preprocessing (500 imgs)
# ===============================
def preprocess_data():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        cls_path = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(".jpg")]
        random.shuffle(images)
        images = images[:SAMPLES_PER_CLASS]

        n = len(images)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"

            dst = os.path.join(PROCESSED_DIR, split, cls, img)
            shutil.copy2(src, dst)

    print("[INFO] Dataset prepared with 500 samples")

# ===============================
# DataLoaders
# ===============================
def get_loaders(batch_size=32):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(f"{PROCESSED_DIR}/train", transform=tfm)
    val_ds   = datasets.ImageFolder(f"{PROCESSED_DIR}/val", transform=tfm)
    test_ds  = datasets.ImageFolder(f"{PROCESSED_DIR}/test", transform=tfm)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    track_raw_with_dvc()
    preprocess_data()

    train_loader, val_loader, test_loader = get_loaders()

    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val:   {len(val_loader.dataset)}")
    print(f"Test:  {len(test_loader.dataset)}")
