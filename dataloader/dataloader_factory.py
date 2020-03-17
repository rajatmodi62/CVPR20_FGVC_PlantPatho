from os import (path, mkdir)
import pandas as pd
import numpy as np
from shutil import copy
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (DataLoader)

# list of datasets
from dataloader.image_dataset import ImageDataset


class CustomDataLoader:
    def __init__(self, root_data_dir):
        self.FOLDS = 5
        self.root_data_dir = root_data_dir

    def generate_folds(self, train_csv_path, fold_dir, train_data_path):
        # assign fold to each frame item
        data_frame = pd.read_csv(train_csv_path)
        X = data_frame.values
        one_hot = data_frame.iloc[:, -4:].values
        Y = [np.where(r == 1)[0][0] for r in one_hot]

        skf = StratifiedKFold(n_splits=self.FOLDS)

        # generate and copy k folds
        fold_idx = 0
        for train_index, val_index in skf.split(X, Y):
            # Copy train csv and images to fold
            for idx in train_index:
                src_image_path = path.join(
                    train_data_path, data_frame.iloc[idx, 0] + '.jpg')
                dst_image_path = path.join(
                    fold_dir, str(fold_idx), "train", data_frame.iloc[idx, 0] + '.jpg')
                copy(src_image_path, dst_image_path)
            data_frame.loc[train_index].to_csv(path.join(
                fold_dir, str(fold_idx), "train.csv"))

            # Copy val csv and images to fold
            for idx in val_index:
                src_image_path = path.join(
                    train_data_path, data_frame.iloc[idx, 0] + '.jpg')
                dst_image_path = path.join(
                    fold_dir, str(fold_idx), "val", data_frame.iloc[idx, 0] + '.jpg')
                copy(src_image_path, dst_image_path)
            data_frame.loc[val_index].to_csv(path.join(
                fold_dir, str(fold_idx), "val.csv"))

            fold_idx += 1

    def create_fold_directory(self, dataset_name):
        if not path.exists('folds'):
            mkdir('folds')

        if not path.exists(path.join('folds', dataset_name)):
            print("[Creating Folds directory]")
            mkdir(path.join('folds', dataset_name))

            for i in range(self.FOLDS):
                mkdir(path.join('folds', dataset_name, str(i)))
                mkdir(path.join('folds', dataset_name, str(i), "train"))
                mkdir(path.join('folds', dataset_name, str(i), "val"))

        return path.join('folds', dataset_name)

    def get_dataset(self, dataset_name, transformer):
        fold_dir = self.create_fold_directory(dataset_name)

        # path for creating test dataset
        test_csv_path = path.join(self.root_data_dir, "test.csv")
        test_data_path = path.join(self.root_data_dir, "images")

        train_csv_path = path.join(self.root_data_dir, "train.csv")
        train_data_path = path.join(self.root_data_dir, "images")

        # copy data to folds
        self.generate_folds(train_csv_path, fold_dir, train_data_path)

        if dataset_name == 'fgvc7':
            test_dataloader = DataLoader(ImageDataset(
                'test', test_data_path, test_csv_path, transformer))

            train_dataloader_list = []
            val_dataloader_list = []
            for i in range(self.FOLDS):
                # pushing fold_i train to train dataset list
                fold_train_csv_path = path.join(fold_dir, str(i), 'train.csv')
                fold_train_data_path = path.join(fold_dir, str(i), 'train')
                fold_train_dataloader = DataLoader(ImageDataset('train', fold_train_data_path,
                                                                fold_train_csv_path, transformer))
                train_dataloader_list.append(
                    fold_train_dataloader
                )

                # pushing fold_i val to val dataset list
                fold_val_csv_path = path.join(fold_dir, str(i), 'val.csv')
                fold_val_data_path = path.join(fold_dir, str(i), 'val')
                fold_val_dataloader = DataLoader(ImageDataset('val', fold_val_data_path,
                                                              fold_val_csv_path, transformer))
                val_dataloader_list.append(
                    fold_val_dataloader
                )
            return (train_dataloader_list, val_dataloader_list, test_dataloader)
