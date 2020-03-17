import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import os


class ImageDataset(Dataset):
    def __init__(self, mode, data_path, csv_path, transformer=None, ext='.jpg'):
        self.mode = mode
        self.image_ext = ext
        self.data_frame = pd.read_csv(csv_path)
        self.data_dir = data_path
        self.transformer = transformer

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data_frame.iloc[idx, 0] + self.image_ext
        image_path = os.path.join(self.data_dir, image_name)
        image = io.imread(image_path)

        return self.transformer.get_augmented(image)