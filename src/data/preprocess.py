import os
import shutil
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import subprocess

# ===============================
# Config
# ===============================
RAW_DIR = "data/raw/PetImages"
PROCESSED_DIR = "data/processed"
CLASSES = ["Cat", "Dog"]

# ===============================
# Preprocessing Functions
# ===============================
def track_raw_with_dvc(raw_dir=RAW_DIR):
    """Track raw data with DVC if not already tracked."""
    try:
        dvc_file = os.path.join(raw_dir, "PetImages.dvc")
        if not os.path.exists(dvc_file):
            subprocess.run(["dvc", "add", raw_dir], check=True)
            subprocess.run(["git", "add", f"{raw_dir}.dvc"], check=True)
            subprocess.run(["git", "commit", "-m", f"Track raw data {raw_dir} with DVC"], check=True)
            print(f"[INFO] Raw data {raw_dir} is now tracked by DVC.")
        else:
            print(f"[INFO] Raw data {raw_dir} is already tracked by DVC.")
    except FileNotFoundError:
        print("[WARNING] DVC or Git not found. Skipping raw data tracking.")
    except subprocess.CalledProcessError:
        print("[WARNING] DVC stage already exists or git commit failed. Skipping.")

def preprocess_data(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, train_ratio=0.8, val_ratio=0.1):
    """
    Split dataset into train/val/test folders with exact ratios: 80:10:10.
    Won't delete processed folder if it exists.
    """
    if os.path.exists(processed_dir):
        print(f"[INFO] Processed folder {processed_dir} already exists. Skipping preprocessing.")
        return

    os.makedirs(processed_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    for cls in CLASSES:
        cls_path = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)  # remaining goes to test automatically

        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            if i < train_end:
                dst = os.path.join(processed_dir, "train", cls, img)
            elif i < val_end:
                dst = os.path.join(processed_dir, "val", cls, img)
            else:
                dst = os.path.join(processed_dir, "test", cls, img)
            shutil.copy2(src, dst)

    print(f"[INFO] Preprocessing done! Train/Val/Test split created at: {processed_dir}")
    print(f"Split ratios: Train {train_ratio*100}%, Val {val_ratio*100}%, Test {100 - int(train_ratio*100 + val_ratio*100)}%")

# ===============================
# DataLoader Function
# ===============================
def get_loaders(processed_dir=PROCESSED_DIR, batch_size=32, augment=True):
    """Return PyTorch DataLoaders for train/val/test."""
    if augment:
        train_tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.8,1.2)),
            transforms.ToTensor()
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(processed_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(processed_dir, "val"), transform=test_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(processed_dir, "test"), transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# ===============================
# Combined Helper
# ===============================
def prepare_data(batch_size=32, augment=True):
    """Track raw data, preprocess (if needed), and return DataLoaders."""
    track_raw_with_dvc()
    preprocess_data()  # will skip if folder exists
    return get_loaders(batch_size=batch_size, augment=augment)

# ===============================
# Main (for testing)
# ===============================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_data(batch_size=32, augment=True)
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader.dataset)}")
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")
