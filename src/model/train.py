import os
import subprocess
import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.model import SimpleCNN
from src.data.preprocess import get_loaders, preprocess_data, RAW_DIR, SMALL_DATASET

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# DVC Preprocessing
# ------------------------------
def run_dvc_preprocessing():
    try:
        raw_dvc_file = os.path.join(RAW_DIR, "PetImages.dvc")
        if not os.path.exists(raw_dvc_file):
            subprocess.run(["dvc", "add", RAW_DIR], check=True)
            subprocess.run(["git", "add", f"{RAW_DIR}.dvc"], check=True)
            subprocess.run(["git", "commit", "-m", "Track raw data with DVC"], check=True)
        subprocess.run(["dvc", "repro"], check=True)
        print("[INFO] DVC preprocessing complete.")
    except FileNotFoundError:
        print("[WARNING] DVC or Git not found. Ensure data is preprocessed manually.")
    except subprocess.CalledProcessError:
        print("[WARNING] DVC stage already up-to-date or failed.")


# ------------------------------
# Training function
# ------------------------------
def train_model(epochs=5, batch_size=32, learning_rate=1e-3, augment=True):
    run_dvc_preprocessing()
    preprocess_data(small_dataset=SMALL_DATASET)

    train_loader, val_loader, _ = get_loaders(batch_size=batch_size, augment=augment)

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("cats_vs_dogs_simple_cnn")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_fn": "BCELoss",
            "augmentation": augment
        })

        for epoch in range(epochs):
            # Training
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            model.eval()
            val_loss, correct_val, total_val = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)
            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/simple_cnn.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Loss & Accuracy plots
        plt.figure()
        plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        plt.figure()
        plt.plot(range(1, epochs+1), train_accs, label="Train Acc")
        plt.plot(range(1, epochs+1), val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("accuracy_curve.png")
        mlflow.log_artifact("accuracy_curve.png")

        # Confusion matrix
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print("[INFO] Training complete. Model and artifacts logged to MLflow.")
    return model

if __name__ == "__main__":
    train_model(epochs=5, batch_size=32, learning_rate=1e-3)
