import torch
from torch.nn.functional import softmax
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import numpy as np
from utils.kaggle_metric import (post_process_output, kaggle_output_header)
from utils.regression_utils import covert_to_classification
from utils.print_util import cprint


class EvaluationHelper:
    def __init__(self, experiment_name, overwrite=False, ensemble=False):
        self.experiment_name = experiment_name

        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if overwrite:
                cprint("[ <", experiment_name,
                       "> results exists - Overwriting! ]", type="warn")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                cprint("[ <", experiment_name,
                       "> results exists - Manual deletion needed ]", type="warn")
                exit()

        self.is_ensemble = ensemble
        self.ensemble_list = []

    def evaluate(self, pred_type, num_classes, experiment_path, test_csv_path, test_output, tta=False):
        test_output = torch.mean(test_output, dim=2)
        
        if pred_type == 'classification':
            test_output = post_process_output(test_output)
        elif pred_type == 'regression' or pred_type == 'mixed':
            test_output = covert_to_classification(
                test_output,
                num_classes,
            )

        if self.is_ensemble:
            self.ensemble_list.append(test_output)

        result_path = path.join(
            'results', self.experiment_name, experiment_path + '.csv')
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

    def ensemble(self, test_csv_path, type = "softmax"):
        cprint("[ Ensembling results ]", type="success")

        self.ensemble_list = torch.stack(self.ensemble_list, dim=2)

        if self.ensemble_list.size()[2] < 3:
            cprint("[ Too few experiments for ensembling ]", type="warn")
            exit()
        elif self.ensemble_list.size()[2] % 2 == 0:
            cprint("[ Ensemble logic needs odd majority < ",
                   str(self.ensemble_list.size()[2]), " > ]", type="warn")
            exit()

        # Voting logic
        if type == "softmax":
            self.results = torch.sum(self.ensemble_list, dim=2)
            self.results = torch.softmax(self.results, dim=1)
        elif type == "thresholding":
            self.results = torch.mean(self.ensemble_list, dim=2)
            OHV_target = torch.zeros(self.results.size()).to(
                self.results.get_device())
            OHV_target[range(self.results.size()[0]),
                       torch.argmax(self.results, dim=1)] = 1
            self.results = OHV_target
        elif type == "mean":
            self.results = torch.mean(self.ensemble_list, dim=2)
        

        result_path = path.join(
            'results', self.experiment_name, 'ensembled.csv')
        ensemble_df = pd.read_csv(test_csv_path)

        with torch.no_grad():
            # saving results to csv
            df = pd.DataFrame(
                np.hstack(
                    (
                        ensemble_df.to_numpy(),
                        self.results.cpu().numpy()
                    )
                )
            )

            df.to_csv(
                result_path,
                header=kaggle_output_header,
                index=False
            )
