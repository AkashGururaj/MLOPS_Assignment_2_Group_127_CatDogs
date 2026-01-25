import os
import shutil
import kagglehub

RAW_DIR = "data/raw"
DATASET_ID = "bhavikjikadara/dog-and-cat-classification-dataset"

def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Downloading dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download(DATASET_ID)

    print("Copying dataset to data/raw...")
    for item in os.listdir(dataset_path):
        src = os.path.join(dataset_path, item)
        dst = os.path.join(RAW_DIR, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print("Dataset successfully saved to data/raw")

if __name__ == "__main__":
    download_dataset()
