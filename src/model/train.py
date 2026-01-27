import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import the existing functions
from src.data.preprocess import get_loaders, preprocess_data
from src.model.model import SimpleCNN


# Device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")


# Train Function with MLflow

def train_model(epochs=5, batch_size=32, lr=1e-3):
    # Preprocess data (prepare train/val/test splits)
    preprocess_data()

    # Get data loaders
    train_loader, val_loader, _ = get_loaders(batch_size=batch_size)

    # Initialize model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # MLflow experiment
    mlflow.set_experiment("cats_vs_dogs_simple_cnn")
    train_losses, train_accs, val_accs = [], [], []

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss_fn": "BCELoss",
            "augmentation": False
        })

        for epoch in range(epochs):
            # -------- Train --------
            model.train()
            running_loss, correct, total = 0, 0, 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # -------- Validation --------
            model.eval()
            correct, total = 0, 0
            all_labels, all_preds = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    outputs = model(images)
                    preds = (outputs > 0.5).float()

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            val_acc = correct / total
            val_accs.append(val_acc)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, step=epoch)

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # -------- Save Model --------
        os.makedirs("models", exist_ok=True)
        model_path = "models/simple_cnn.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"[INFO] Model saved: {model_path}")

        # -------- Plot Loss & Accuracy --------
        plt.figure()
        plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
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

        # -------- Confusion Matrix --------
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print("[INFO] Training complete. Model, metrics, and artifacts logged to MLflow.")

    return model


# Main

if __name__ == "__main__":
    train_model(epochs=5, batch_size=32, lr=1e-3)
