from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_localizer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    device: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Train GCP Localizer model.

    Args:
        model: GCPLocalizer PyTorch model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        num_epochs: Number of epochs to train.
        learning_rate: Optimizer learning rate.
        device: 'cuda' or 'cpu'. Automatically detected if None.
        save_path: Path to save the best model weights based on validation loss.

    Returns:
        Dictionary with training history (train_loss, val_loss, best_val_loss).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss (MAE): {train_loss:.4f} | Val Loss (MAE): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                save_file = Path(save_path)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_file)
                print(f"Saved best localizer checkpoint with Val Loss: {val_loss:.4f}")

    history["best_val_loss"] = best_val_loss
    return history
