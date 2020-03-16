from torch.utils.data import (DataLoader)
from dataloader.train_dataset import TrainDataset


class CustomDataLoader:
    def __init__(self):
        self.train_data_loader = DataLoader(TrainDataset())
        # TODO: Add a val data loader
        # TODO: Add a test data loader

    def get_training_dataloader(self):
        # TODO: Implement k fold validation here
        return self.train_data_loader
