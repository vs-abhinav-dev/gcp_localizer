from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_tensor(image_np: np.ndarray) -> torch.Tensor:
    """Normalize image numpy array [0, 255] RGB to ImageNet normalized PyTorch float tensor (C, H, W)."""
    img = image_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)


class FastGCPDataset(Dataset):
    """PyTorch Dataset for GCP center coordinate localization (regression)."""

    def __init__(self, files: List[Union[str, Path]]):
        self.files = [str(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        image = data["image"]
        target = data["target"]

        image_tensor = preprocess_image_tensor(image)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return image_tensor, target_tensor


class MarkerClassifierDataset(Dataset):
    """PyTorch Dataset for binary GCP marker presence classification (0 = negative, 1 = positive)."""

    def __init__(self, files: List[Union[str, Path]]):
        self.files = [str(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        image = data["image"]
        label = int(data["label"])

        image_tensor = preprocess_image_tensor(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class ShapeDataset(Dataset):
    """PyTorch Dataset for 3-class GCP shape classification (L-Shape, Square, Cross)."""

    SHAPE_TO_IDX: Dict[str, int] = {
        "L-Shape": 0,
        "Square": 1,
        "Cross": 2
    }
    IDX_TO_SHAPE: Dict[int, str] = {v: k for k, v in SHAPE_TO_IDX.items()}

    def __init__(self, files: List[Union[str, Path]]):
        self.files = []
        for f in files:
            data = np.load(f, allow_pickle=True)
            shape = str(data["shape"]) if "shape" in data.files else "Unknown"
            if shape in self.SHAPE_TO_IDX:
                self.files.append(str(f))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)
        image = data["image"]
        shape = str(data["shape"])

        image_tensor = preprocess_image_tensor(image)
        label_tensor = torch.tensor(self.SHAPE_TO_IDX[shape], dtype=torch.long)

        return image_tensor, label_tensor


class RawShapeDataset(Dataset):
    """PyTorch Dataset extracting GCP shape crops directly from raw images + annotations on-the-fly."""

    def __init__(
        self,
        annotations: Dict[str, Any],
        raw_images_dir: Union[str, Path],
        crop_size: int = 512,
        output_size: int = 224,
        is_train: bool = True,
        max_offset: int = 64,
        shape_to_idx: Optional[Dict[str, int]] = None
    ):
        import cv2
        from gcp_localizer.data.preprocessing import crop_positive_sample

        self.raw_images_dir = Path(raw_images_dir)
        self.crop_size = crop_size
        self.output_size = output_size
        self.is_train = is_train
        self.max_offset = max_offset
        self.shape_to_idx = shape_to_idx or ShapeDataset.SHAPE_TO_IDX
        self._crop_fn = crop_positive_sample

        self.samples = []
        for rel_path, info in annotations.items():
            if not isinstance(info, dict):
                continue
            mark = info.get("mark")
            if mark is None or not isinstance(mark, dict):
                continue
            shape = info.get("verified_shape")
            if shape not in self.shape_to_idx:
                continue
            img_path = self.raw_images_dir / rel_path
            if img_path.exists():
                self.samples.append({
                    "img_path": img_path,
                    "cx": float(mark["x"]),
                    "cy": float(mark["y"]),
                    "shape": shape
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2
        sample = self.samples[idx]
        image = cv2.imread(str(sample["img_path"]))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {sample['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        offset = self.max_offset if self.is_train else 0
        crop_res = self._crop_fn(
            image=image,
            cx=sample["cx"],
            cy=sample["cy"],
            crop_size=self.crop_size,
            output_size=self.output_size,
            max_offset=offset
        )

        crop_img = crop_res["image"]
        img_tensor = preprocess_image_tensor(crop_img)
        label_tensor = torch.tensor(self.shape_to_idx[sample["shape"]], dtype=torch.long)
        return img_tensor, label_tensor

