from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    from PIL import Image

    bilinear = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), bilinear)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

class SyntheticImageDataset(Dataset):
    """
    A deterministic "real enough" dataset: each index generates the same pseudo-random sample.
    No downloads, no internet, no excuses.

    For real data: replace this with a dataset that reads from DATA_ROOT (mounted from /localscratch).
    """
    def __init__(self, size: int, image_size: int, num_classes: int, seed: int = 123):
        self.size = int(size)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        # Make each sample deterministic but "random-looking"
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))
        x = torch.randn(3, self.image_size, self.image_size, generator=g)
        y = torch.randint(low=0, high=self.num_classes, size=(1,), generator=g).item()
        return x, y


class ImageFolderDataset(Dataset):
    """
    Minimal image-folder dataset:
      root/class_a/*.jpg
      root/class_b/*.png
    """

    def __init__(self, root: Union[str, Path], image_size: int):
        self.root = Path(root)
        self.image_size = int(image_size)

        if not self.root.is_dir():
            raise ValueError(f"imagefolder root does not exist: {self.root}")

        class_dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        if not class_dirs:
            raise ValueError(f"expected class subdirectories under: {self.root}")

        self.class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
        self.samples = []
        for class_dir, target in self.class_to_idx.items():
            full_dir = self.root / class_dir
            for path in sorted(full_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, target))

        if not self.samples:
            raise ValueError(f"no image files found under: {self.root}")

        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        x = _load_image_tensor(path, self.image_size)
        return x, target


def build_dataset(dataset_type: str, image_size: int, num_classes: int, dataset_size: int, seed: int, data_root: str = ""):
    if dataset_type == "synthetic":
        dataset = SyntheticImageDataset(size=dataset_size, image_size=image_size, num_classes=num_classes, seed=seed)
        return dataset, int(num_classes)

    if dataset_type == "imagefolder":
        if not data_root:
            raise ValueError("--data_root is required when --dataset_type imagefolder")
        dataset = ImageFolderDataset(root=data_root, image_size=image_size)
        return dataset, dataset.num_classes

    raise ValueError(f"unsupported dataset_type: {dataset_type}")
