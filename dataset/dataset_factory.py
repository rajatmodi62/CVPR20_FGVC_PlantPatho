from os import path

# list of datasets
from dataset.fgvc7_dataset import FGVC7_Dataset


class DatasetFactory:
    def __init__(self, org_data_dir="data",):
        self.FOLDS = 5
        self.org_data_dir = org_data_dir

    def get_dataset(self, mode, dataset_name, transformer=None, fold_number=None):

        if mode not in ["train", "test", "val"]:
            print("[ Dataset Mode should either be train/test/val ]")
            exit()
        else:
            dataset_dir = path.join(self.org_data_dir, dataset_name)
            dataset = None

            if dataset_name == "fgvc7":
                print("[ Dataset : fgvc7 <", mode, "/",
                      "raw" if fold_number is None else fold_number, "> ]")
                dataset = FGVC7_Dataset(
                    mode, dataset_dir, transformer, fold_number)
            else:
                print("[ Dataset not found ]")
                exit()

        print("[ Transformer : ", str(transformer), " ]")

        return dataset
