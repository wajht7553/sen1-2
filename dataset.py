# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms


class SEN12Dataset(Dataset):
    def __init__(self, split_file, split="train", transform=None):
        self.splits = np.load(split_file, allow_pickle=True).item()
        self.image_pairs = self.splits[split]
        self.transform = transform or self._get_default_transform(split)

    def __len__(self):
        return len(self.image_pairs)

    def _get_default_transform(self, split):
        if split == "train":
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        # Load images
        s1_img = Image.open(pair["s1"])
        s2_img = Image.open(pair["s2"])

        # Apply transforms
        if self.transform:
            s1_img = self.transform(s1_img)
            s2_img = self.transform(s2_img)

        return {
            "s1": s1_img,
            "s2": s2_img,
            "season": pair["season"],
            "roi": pair["roi"],
        }
