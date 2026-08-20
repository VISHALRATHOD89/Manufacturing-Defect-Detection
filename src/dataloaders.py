from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.dataset import SteelDefectDataset
from src.transforms import train_transforms, val_transforms


def get_dataloaders(labels_csv="data/severstal/image_labels.csv",
                     images_dir="data/severstal/train_images",
                     batch_size=32):

    full_df = pd.read_csv(labels_csv)

    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_df.to_csv("data/severstal/train_split.csv", index=False)
    val_df.to_csv("data/severstal/val_split.csv", index=False)

    train_dataset = SteelDefectDataset("data/severstal/train_split.csv", images_dir, transform=train_transforms)
    val_dataset = SteelDefectDataset("data/severstal/val_split.csv", images_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    images, labels = next(iter(train_loader))
    print("\nOne batch of images shape:", images.shape)
    print("One batch of labels shape:", labels.shape)