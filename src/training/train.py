"""
Training script for the Hybrid Quantum-Classical Brain Tumor model.
Trains the model on brain MRI images and tracks everything with MLflow.
"""

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.data_loader import get_dataloaders
from src.models.hybrid_model import HybridModel

# ----- Settings (hyperparameters) -----
EPOCHS = 10           # how many times we go through all the data
BATCH_SIZE = 8        # images per batch
LEARNING_RATE = 0.001  # smaller steps = more careful learning
N_CLASSES = 2         # tumor / no tumor


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train the model for one full pass over the training data."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # reset gradients
        outputs = model(images)        # forward pass (predict)
        loss = loss_fn(outputs, labels)  # how wrong were we?
        loss.backward()                # backward pass (learn)
        optimizer.step()               # update weights

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, loss_fn, device):
    """Check how well the model does on validation data (no learning here)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # don't compute gradients (faster, we're just testing)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"📊 Classes: {class_names} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Build model
    model = HybridModel(n_classes=N_CLASSES).to(device)

    # ----- Loss function with class weights (fixes imbalance: 155 tumor vs 98 healthy) -----
    # Giving the smaller class (healthy) more weight so the model can't just guess "tumor"
    class_weights = torch.tensor([155.0 / 98.0, 1.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (only train parameters that need it)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=LEARNING_RATE)

    # ----- MLflow: track this experiment -----
    mlflow.set_experiment("brain-tumor-quantum")

    with mlflow.start_run():
        # Log our settings
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model": "HybridQuantumResNet18",
        })

        print("\n🚀 Starting training...\n")
        best_val_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            print(
                f"Epoch {epoch}/{EPOCHS}  "
                f"| Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2%}  "
                f"| Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2%}"
            )

            # Save the best model so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "brain_tumor_model.pth")

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_artifact("brain_tumor_model.pth")
        print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2%}")
        print("Model saved as brain_tumor_model.pth")


if __name__ == "__main__":
    main()