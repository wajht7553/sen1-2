# main.py
from dataset_organization import DatasetOrganizer
from data_loading import get_dataloaders


def main():
    # Set up paths
    root_dir = "path/to/tum-sentinel-1-2"
    output_dir = "path/to/processed_dataset"

    # Organize dataset
    organizer = DatasetOrganizer(root_dir, output_dir)
    splits = organizer.create_splits()

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        split_file=f"{output_dir}/dataset_splits.npy", batch_size=32
    )

    # Example of accessing data
    for batch in train_loader:
        s1_imgs = batch["s1"]
        s2_imgs = batch["s2"]
        seasons = batch["season"]
        rois = batch["roi"]

        # Your training code here
        break


if __name__ == "__main__":
    main()
