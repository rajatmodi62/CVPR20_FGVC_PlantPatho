import torch
from os import (makedirs, path)
import pandas as pd

def model_checkpoint(experiment_name, training_loss, validation_loss, epochs, state_dict):
    df = pd.DataFrame([[ epochs, validation_loss, training_loss ]])
    result_path = path.join('results', experiment_name, 'result.csv')
    if not path.isfile(result_path):
        df.to_csv(result_path, header=["epoch", "validation loss", "training loss"])
    else: # else it exists so append without writing the header
        df.to_csv(result_path, mode='a', header=False)
    torch.save( 
        state_dict,  
        path.join('results', experiment_name, 'weights.pth')
    )

def experiment_dir_creator( experiment_name ):
    if path.exists(path.join('results', experiment_name)) == False:
        makedirs(path.join('results', experiment_name))
    else:
        print("[ Experiment output already exists - Manual deletion needed ]")
        exit()
