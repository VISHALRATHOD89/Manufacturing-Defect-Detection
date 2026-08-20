import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SteelDefectDataset(Dataset):
    def __init__(self, labels_csv, images_dir, transform=None):
        self.labels_df = pd.read_csv(labels_csv)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row["ImageId"]

        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row[["defect_1", "defect_2", "defect_3", "defect_4"]].values.astype("float32")

        return image, label
