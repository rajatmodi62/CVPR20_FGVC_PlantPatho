from os import path

# list of datasets
from dataset.fgvc7_dataset import FGVC7_Dataset


class Dataset:
    def __init__(self, org_data_dir):
        self.FOLDS = 5
        self.org_data_dir = org_data_dir

    def get_dataset(self, mode, dataset_name, transformer=None, fold_number=None):

        if mode not in ["train", "test", "val"]:
            print("[Mode should either be train/test/val]")
        else:
            dataset_dir = path.join(self.org_data_dir, dataset_name)

            if dataset_name == "fgvc7":
                return FGVC7_Dataset(mode, dataset_dir, transformer, fold_number)
            else:
                print("[Dataset not found]")
