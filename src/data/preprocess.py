import os
import random
import shutil
import subprocess
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===============================
# Configurable Paths & Parameters
# ===============================
RAW_DIR = "data/raw/PetImages"
PROCESSED_DIR = "data/processed"
SMALL_DATASET = True           # True to use small subset
SMALL_SAMPLE_SIZE = 500        # Total images if using small dataset
CLASSES = ["Cat", "Dog"]

# ===============================
# Helper Functions
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
        print("[WARNING] Raw data tracking already exists or git commit failed. Skipping.")

def create_small_dataset(raw_dir=RAW_DIR, sample_size=SMALL_SAMPLE_SIZE):
    """Create a small dataset subset (~sample_size images total)."""
    small_raw_dir = raw_dir + "_small"
    if os.path.exists(small_raw_dir):
        shutil.rmtree(small_raw_dir)  # remove if exists
    os.makedirs(small_raw_dir, exist_ok=True)

    for cls in CLASSES:
        cls_dir = os.path.join(small_raw_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        images = [f for f in os.listdir(os.path.join(raw_dir, cls)) if f.lower().endswith((".jpg", ".png"))]
        selected = random.sample(images, min(len(images), sample_size // len(CLASSES)))
        for img in selected:
            shutil.copy2(os.path.join(raw_dir, cls, img), os.path.join(cls_dir, img))

    print(f"[INFO] Small dataset created at {small_raw_dir} ({sample_size} images total).")
    return small_raw_dir

def preprocess_data(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, train_ratio=0.8, val_ratio=0.1, small_dataset=SMALL_DATASET):
    """Preprocess dataset: optionally reduce size and split into train/val/test."""
    
    # Clear processed folder if exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    
    if small_dataset:
        raw_dir = create_small_dataset(raw_dir)

    # Create processed folders
    os.makedirs(processed_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    # Split images
    for cls in CLASSES:
        cls_path = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            if i < train_end:
                dst = os.path.join(processed_dir, "train", cls, img)
            elif i < val_end:
                dst = os.path.join(processed_dir, "val", cls, img)
            else:
                dst = os.path.join(processed_dir, "test", cls, img)
            shutil.copy2(src, dst)

    print(f"[INFO] Data preprocessing complete! Train/Val/Test folders created at: {processed_dir}")

# ===============================
# DataLoaders
# ===============================
def get_loaders(processed_dir=PROCESSED_DIR, batch_size=32, augment=True):
    """Return PyTorch DataLoaders with optional augmentation."""
    
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
    val_ds = datasets.ImageFolder(os.path.join(processed_dir, "val"), transform=test_tfms)
    test_ds = datasets.ImageFolder(os.path.join(processed_dir, "test"), transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    track_raw_with_dvc()
    preprocess_data(small_dataset=SMALL_DATASET)
