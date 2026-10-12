from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[Union[str, torch.Tensor, list, tuple]] = None,
    label_smoothing: float = 0.0,
    scheduler_type: Optional[str] = "cosine",
    device: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Train GCP Classifier model (binary presence or multi-class shape classifier).

    Args:
        model: GCPClassifier or GCPShapeClassifier PyTorch model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        num_epochs: Number of epochs to train.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for regularisation.
        class_weights: 'balanced' or tensor/list of class weights for cross-entropy.
        label_smoothing: Label smoothing factor (e.g. 0.05).
        scheduler_type: 'cosine', 'step', or None.
        device: 'cuda' or 'cpu'. Automatically detected if None.
        save_path: Path to save the best model weights based on validation F1/accuracy.

    Returns:
        Dictionary with training history (losses, accuracies, precisions, recalls, f1s, confusion matrix).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Class weight calculation
    weights_tensor = None
    if class_weights == "balanced":
        all_labels = []
        ds = train_loader.dataset
        if hasattr(ds, "samples") and len(ds.samples) > 0:
            all_labels = [ds.shape_to_idx[s["shape"]] for s in ds.samples]
        elif hasattr(ds, "files") and len(ds.files) > 0:
            shape_map = getattr(ds, "SHAPE_TO_IDX", None)
            for f in ds.files:
                d = np.load(f, allow_pickle=True)
                if shape_map is not None and "shape" in d.files:
                    s = str(d["shape"])
                    if s in shape_map:
                        all_labels.append(shape_map[s])
                elif "label" in d.files:
                    all_labels.append(int(d["label"]))
        elif hasattr(ds, "tensors") and len(ds.tensors) > 1:
            all_labels = ds.tensors[1].cpu().numpy().tolist()

        if all_labels:
            num_classes = getattr(model, "num_classes", 3)
            counts = np.bincount(all_labels, minlength=num_classes)
            total = len(all_labels)
            weights_arr = np.zeros(num_classes, dtype=np.float32)
            for c in range(num_classes):
                weights_arr[c] = total / (num_classes * max(1, counts[c]))
            weights_tensor = torch.tensor(weights_arr, dtype=torch.float32).to(device)
            print(f"[train_classifier] Applied balanced class weights: {weights_arr.round(3).tolist()}")
    elif isinstance(class_weights, torch.Tensor):
        weights_tensor = class_weights.to(device)
    elif isinstance(class_weights, (list, tuple)):
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.5)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    history: Dict[str, Any] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
        "best_val_acc": 0.0,
        "best_val_f1": 0.0,
        "final_confusion_matrix": None,
    }

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = preds.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d} [Val]"):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)

                val_loss += loss.item()
                predicted = preds.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                all_val_preds.extend(predicted.cpu().numpy().tolist())
                all_val_labels.extend(labels.cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        # Compute Classification Metrics
        if len(all_val_labels) > 0:
            val_prec = float(precision_score(all_val_labels, all_val_preds, average="macro", zero_division=0))
            val_rec = float(recall_score(all_val_labels, all_val_preds, average="macro", zero_division=0))
            val_f1 = float(f1_score(all_val_labels, all_val_preds, average="macro", zero_division=0))
            num_cls = getattr(model, "num_classes", 3)
            cm = confusion_matrix(all_val_labels, all_val_preds, labels=list(range(num_cls)))
        else:
            val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0
            cm = np.zeros((1, 1), dtype=int)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["final_confusion_matrix"] = cm.tolist()

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}"
        )
        print("Validation Confusion Matrix:\n", cm)

        if scheduler is not None:
            scheduler.step()

        # Save Best Checkpoint (prioritizing macro F1 across balanced shapes)
        is_best = False
        if epoch == 0:
            is_best = True
            best_val_f1 = val_f1
            best_val_acc = val_acc
        elif val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            is_best = True
            best_val_f1 = val_f1
            best_val_acc = val_acc

        if is_best and save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_file)
            print(f"Saved best classifier checkpoint with Val F1: {val_f1:.4f}, Acc: {val_acc:.4f} -> {save_file}")

    history["best_val_acc"] = best_val_acc
    history["best_val_f1"] = best_val_f1
    return history

