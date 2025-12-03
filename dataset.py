import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = float(row['label'])
        return img, label
