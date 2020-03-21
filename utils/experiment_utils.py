import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd


def model_checkpoint(experiment_name, validation_loss, training_loss, validation_acc, epochs, state_dict):
    df = pd.DataFrame(
        [[epochs, validation_loss, training_loss, validation_acc]])
    result_path = path.join('results', experiment_name, 'result.csv')
    if not path.isfile(result_path):
        df.to_csv(result_path, header=[
                  "epoch", "Loss ( Val )", "Loss ( Train )", "Accuracy ( Val )"], index=False)
    else:  # else it exists so append without writing the header
        df.to_csv(result_path, mode='a', header=False, index=False)
    torch.save(
        state_dict,
        path.join('results', experiment_name, 'weights.pth')
    )


def experiment_dir_creator(experiment_name, rewrite=False):
    if path.exists(path.join('results', experiment_name)) == False:
        makedirs(path.join('results', experiment_name))
    else:
        if rewrite:
            print("[ Experiment output already exists - Overwriting! ]")
            rmtree(path.join('results', experiment_name))
            makedirs(path.join('results', experiment_name))
        else:
            print("[ Experiment output already exists - Manual deletion needed ]")
            exit()
