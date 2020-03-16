import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers.transformer_factory import Transformer
from skimage import io
import os


class TrainDataset(Dataset):
    def __init__(self, data_dir='data/images', train_filename='data/train.csv', image_ext='.jpg'):
        self.image_ext = image_ext
        self.train_frame = pd.read_csv(train_filename)
        self.data_dir = data_dir
        self.transformer = Transformer()

    def __len__(self):
        return len(self.train_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        image_name = self.train_frame.iloc[idx, 0] + self.image_ext
        image_path = os.path.join(self.data_dir, image_name)
        image = io.imread(image_path)

        print( image_name )
        print( image_path )
        return self.transformer.get_augmented(image)
