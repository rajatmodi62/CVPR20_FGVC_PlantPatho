from tqdm import (trange, tqdm)
from math import ceil


class CustomBar:
    def __init__(self, epoch, len_training_data_set, batch_size):
        self.bar = tqdm(
            total=epoch * ceil(len_training_data_set / batch_size),
            desc="Progress",
            postfix=[
                dict(batch_idx=0),
                ceil(len_training_data_set / batch_size),
                dict(epoch_idx=0),
                epoch
            ],
            bar_format='{desc}: {percentage:3.0f}%|{bar}| [ETA:{remaining}] [Batch:{postfix[0][batch_idx]}/{postfix[1]} Epoch:{postfix[2][epoch_idx]}/{postfix[3]}]'
        )

    def step(self):
        self.bar.update()

    def update_epoch_info(self,i):
        self.bar.postfix[2]["epoch_idx"] = i + 1
    
    def update_batch_info(self, batch_ndx):
        self.bar.postfix[0]["batch_idx"] = batch_ndx + 1

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.bar.close()
