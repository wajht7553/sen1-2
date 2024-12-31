import os
import random
import shutil
from pathlib import Path
import numpy as np


class DatasetOrganizer:
    def __init__(self, root_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio

    def _get_paired_images(self):
        """Get all paired S1-S2 images across seasons."""
        paired_images = []

        # Walk through all seasons
        for season_folder in self.root_dir.glob("Sentinel_*"):
            for roi_folder in season_folder.glob("ROIs*"):
                # Get all s1 images
                s1_folders = list(roi_folder.glob("s1_*"))

                for s1_folder in s1_folders:
                    scene_num = s1_folder.name.split("_")[1]
                    s2_folder = roi_folder / f"s2_{scene_num}"

                    if s2_folder.exists():
                        # Match corresponding images
                        s1_images = list(s1_folder.glob("*.png"))
                        for s1_img in s1_images:
                            img_num = s1_img.name.split("_")[-1]
                            s2_img = s2_folder / f"s2_{scene_num}_{img_num}"

                            if s2_img.exists():
                                paired_images.append(
                                    {
                                        "s1": str(s1_img),
                                        "s2": str(s2_img),
                                        "season": season_folder.name,
                                        "roi": roi_folder.name,
                                    }
                                )

        return paired_images

    def create_splits(self):
        """Create train/val/test splits."""
        paired_images = self._get_paired_images()
        random.shuffle(paired_images)

        # Calculate split sizes
        n_samples = len(paired_images)
        n_train = int(n_samples * self.split_ratio[0])
        n_val = int(n_samples * self.split_ratio[1])

        # Split the data
        train_pairs = paired_images[:n_train]
        val_pairs = paired_images[n_train : n_train + n_val]
        test_pairs = paired_images[n_train + n_val :]

        # Save splits
        splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        # Save split information
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(self.output_dir / "dataset_splits.npy", splits)

        return splits
