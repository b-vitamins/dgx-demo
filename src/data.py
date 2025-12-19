import math
import torch
from torch.utils.data import Dataset

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
