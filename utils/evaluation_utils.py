import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import numpy as np
from utils.kaggle_metric import (post_process_output, kaggle_output_header)
from utils.regression_utils import covert_to_classification


class EvaluationHelper:
    def __init__(self, experiment_name, overwrite=False):
        self.experiment_name = experiment_name

        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if overwrite:
                print("[ <", experiment_name, "> results exists - Overwriting! ]")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                print("[ <", experiment_name,
                      "> results exists - Manual deletion needed ]")
                exit()

    def evaluate(self, pred_type, num_classes, model_path, test_csv_path, test_output):
        if pred_type == 'classification':
            test_output = post_process_output(test_output)
        elif pred_type == 'regression':
            test_output = covert_to_classification(
                test_output,
                num_classes,
            )

        result_path = path.join(
            'results', self.experiment_name, model_path + '.csv')
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
                header=kaggle_output_header,
                index=False
            )
