import random
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gcp_localizer.data.datasets import IMAGENET_MEAN, IMAGENET_STD
from gcp_localizer.evaluation.metrics import compute_localization_error


def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) normalized ImageNet tensor back to (H, W, 3) RGB numpy array in [0, 1]."""
    img_vis = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_vis = img_vis * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img_vis, 0.0, 1.0)


def plot_predictions(
    localizer_model: nn.Module,
    val_dataset: Dataset,
    save_path: Optional[Union[str, Path]] = None,
    num_samples: int = 9,
    device: str = "cpu"
) -> None:
    """Plot ground truth vs predicted GCP coordinates on sample tiles.

    Lime green dot = Ground Truth
    Red dot = Model Prediction

    Args:
        localizer_model: Trained localizer model.
        val_dataset: Validation PyTorch dataset.
        save_path: Path to save figure (optional).
        num_samples: Number of tiles to plot (default 9 in 3x3 grid).
        device: 'cuda' or 'cpu'.
    """
    localizer_model.eval()
    localizer_model.to(device)

    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if num_samples == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flat

    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

    for i, idx in enumerate(indices):
        ax = axes_list[i]
        img, target = val_dataset[idx]

        with torch.no_grad():
            pred = localizer_model(img.unsqueeze(0).to(device)).cpu()[0]

        pred_x, pred_y = pred[0].item(), pred[1].item()
        gt_x, gt_y = target[0].item(), target[1].item()

        err = compute_localization_error((pred_x, pred_y), (gt_x, gt_y))

        img_vis = denormalize_image(img)
        ax.imshow(img_vis)
        ax.scatter(gt_x, gt_y, c="lime", s=30, label="GT")
        ax.scatter(pred_x, pred_y, c="red", s=30, label="Pred")
        ax.set_title(f"Error: {err:.1f}px")
        ax.axis("off")

    # Hide extra subplots
    for j in range(len(indices), len(axes_list)):
        axes_list[j].axis("off")

    plt.tight_layout()
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_shape_confusion(
    classifier_model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    save_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> None:
    """Plot sample false positive and false negative classification cases.

    Args:
        classifier_model: Trained binary classifier.
        val_loader: Validation dataloader.
        save_path: Path to save plot artifact.
        device: 'cuda' or 'cpu'.
    """
    classifier_model.eval()
    classifier_model.to(device)

    false_positives = []
    false_negatives = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = classifier_model(images)
            preds = logits.argmax(dim=1)

            for i in range(len(labels)):
                pred = preds[i].item()
                gt = labels[i].item()

                if pred == 1 and gt == 0 and len(false_positives) < 5:
                    false_positives.append(images[i].cpu())
                elif pred == 0 and gt == 1 and len(false_negatives) < 5:
                    false_negatives.append(images[i].cpu())

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for idx, img in enumerate(false_positives):
        ax = axes[0, idx]
        ax.imshow(denormalize_image(img))
        ax.set_title("FP (Pred: 1, GT: 0)")
        ax.axis("off")

    for idx, img in enumerate(false_negatives):
        ax = axes[1, idx]
        ax.imshow(denormalize_image(img))
        ax.set_title("FN (Pred: 0, GT: 1)")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_dataset_distribution(
    processed_dir: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot dataset split counts (Positive vs Negative) and shape distribution.

    Args:
        processed_dir: Path to processed directory containing train/ and val/ splits.
        save_path: Path to save plot artifact.
    """
    proc_dir = Path(processed_dir)
    from glob import glob
    from collections import Counter

    splits = ["train", "val"]
    counts = {"train": {"positive": 0, "negative": 0}, "val": {"positive": 0, "negative": 0}}
    shapes_counter = Counter()

    for split in splits:
        pos_files = glob(str(proc_dir / split / "positive" / "*.npz"))
        neg_files = glob(str(proc_dir / split / "negative" / "*.npz"))
        counts[split]["positive"] = len(pos_files)
        counts[split]["negative"] = len(neg_files)

        for pf in pos_files:
            try:
                data = np.load(pf, allow_pickle=True)
                shape = str(data["shape"]) if "shape" in data.files else "Unknown"
                shapes_counter[shape] += 1
            except Exception:
                pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Pos vs Neg bar chart
    x = np.arange(len(splits))
    width = 0.35
    pos_vals = [counts[s]["positive"] for s in splits]
    neg_vals = [counts[s]["negative"] for s in splits]

    rects1 = ax1.bar(x - width / 2, pos_vals, width, label="Positive Tiles", color="#2ecc71")
    rects2 = ax1.bar(x + width / 2, neg_vals, width, label="Negative Tiles", color="#e74c3c")
    ax1.set_ylabel("Tile Count")
    ax1.set_title("Dataset Split Class Balance (Pos vs Neg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Train Split", "Validation Split"])
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f"{height}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    # Panel 2: Shape Distribution
    if shapes_counter:
        shapes = list(shapes_counter.keys())
        shape_counts = list(shapes_counter.values())
        colors = ["#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#1abc9c"][:len(shapes)]
        ax2.pie(shape_counts, labels=shapes, autopct="%1.1f%%", startangle=140, colors=colors)
        ax2.set_title("GCP Marker Shape Distribution")
    else:
        ax2.text(0.5, 0.5, "No shape metadata found", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_end_to_end_pipeline(
    sample_npz_path: Union[str, Path],
    classifier_model: nn.Module,
    localizer_model: nn.Module,
    save_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> None:
    """Visualize end-to-end GCP detection pipeline from candidate crop tile to localizer coordinate output.

    Steps visualized:
    1. Input candidate tile.
    2. Classifier stage (Marker presence prediction & confidence score).
    3. Localizer stage (GCP coordinate regression on tile).
    4. Combined pipeline output with GT vs Prediction overlay and error metric.

    Args:
        sample_npz_path: Path to sample preprocessed .npz tile.
        classifier_model: Trained GCP classifier model.
        localizer_model: Trained GCP localizer model.
        save_path: Output save path for plot figure.
        device: 'cuda' or 'cpu'.
    """
    from gcp_localizer.data.datasets import preprocess_image_tensor

    classifier_model.eval().to(device)
    localizer_model.eval().to(device)

    data = np.load(sample_npz_path, allow_pickle=True)
    raw_img = data["image"]  # (224, 224, 3)
    gt_target = data["target"]  # (rel_x, rel_y)
    gt_shape = str(data["shape"]) if "shape" in data.files else "Unknown"
    gt_label = int(data["label"]) if "label" in data.files else 1

    img_tensor = preprocess_image_tensor(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits = classifier_model(img_tensor)
        probs = torch.softmax(cls_logits, dim=1)[0]
        pos_prob = probs[1].item() * 100.0
        cls_pred = cls_logits.argmax(dim=1).item()

        loc_pred = localizer_model(img_tensor).cpu().numpy()[0]

    pred_x, pred_y = loc_pred[0], loc_pred[1]
    gt_x, gt_y = gt_target[0], gt_target[1]
    err = compute_localization_error((pred_x, pred_y), (gt_x, gt_y))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Step 1: Input Candidate Tile
    axes[0].imshow(raw_img)
    axes[0].set_title(f"1. Input Tile (224x224)\nGT Label: {'Marker' if gt_label==1 else 'Background'}")
    axes[0].axis("off")

    # Step 2: Classifier Stage
    cls_color = "green" if cls_pred == gt_label else "red"
    cls_text = f"Stage 1: Classifier\nPred: {'Marker' if cls_pred==1 else 'Background'}\nConf: {pos_prob:.1f}%"
    axes[1].imshow(raw_img)
    axes[1].text(10, 30, cls_text, color="white", fontsize=10, bbox=dict(boxstyle="round", facecolor=cls_color, alpha=0.8))
    axes[1].set_title(f"2. Marker Classifier Filter")
    axes[1].axis("off")

    # Step 3: Localizer Stage
    axes[2].imshow(raw_img)
    axes[2].scatter(pred_x, pred_y, c="red", s=60, marker="x", label="Pred Coordinate")
    axes[2].set_title(f"3. Localizer Regression\nPred: ({pred_x:.1f}, {pred_y:.1f})")
    axes[2].legend(loc="lower right")
    axes[2].axis("off")

    # Step 4: Final Overlay & Error Metric
    axes[3].imshow(raw_img)
    axes[3].scatter(gt_x, gt_y, c="lime", s=70, label="GT Location")
    axes[3].scatter(pred_x, pred_y, c="red", s=70, marker="x", label="Pred Location")
    axes[3].plot([gt_x, pred_x], [gt_y, pred_y], "yellow", linestyle="--", linewidth=1.5, label=f"Error: {err:.1f}px")
    axes[3].set_title(f"4. End-to-End Output\nShape: {gt_shape} | Error: {err:.1f}px")
    axes[3].legend(loc="lower right")
    axes[3].axis("off")

    plt.suptitle("GCP Two-Stage Detection & Localization Pipeline Flow", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

