# data_loading.py
from dataset import SEN12Dataset
from torch.utils.data import DataLoader


def get_dataloaders(split_file, batch_size=32, num_workers=4):
    # Create datasets for each split
    train_dataset = SEN12Dataset(split_file, split="train")
    val_dataset = SEN12Dataset(split_file, split="val")
    test_dataset = SEN12Dataset(split_file, split="test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
