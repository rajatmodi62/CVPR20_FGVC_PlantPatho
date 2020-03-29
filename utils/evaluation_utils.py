import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import numpy as np
from utils.kaggle_metric import post_process_output


class EvaluationHelper:
    def __init__(self, experiment_name, rewrite=False):
        self.experiment_name = experiment_name

        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if rewrite:
                print("[ <", experiment_name, "> results exists - Overwriting! ]")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                print("[ <", experiment_name,
                      "> results exists - Manual deletion needed ]")
                exit()

    def evaluate(self, test_csv_path, test_output, model_name):
        test_output = post_process_output(test_output)

        result_path = path.join(
            'results', self.experiment_name, model_name + '.csv')
        test_df = pd.read_csv(test_csv_path)

        with torch.no_grad():
            # saving results to csv
            df = pd.DataFrame(
                np.hstack(
                    (
                        test_df.to_numpy(),
                        test_output.cpu().numpy()
                    )
                )
            )

            df.to_csv(
                result_path,
                header=["image_id", "healthy",
                        "multiple_diseases", "rust", "scab"],
                index=False
            )
