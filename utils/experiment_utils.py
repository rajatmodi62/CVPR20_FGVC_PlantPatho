import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import math
from utils.regression_utils import covert_to_classification
from utils.kaggle_metric import (
    kaggle_metric_generator, accuracy_generator, confusion_matrix_generator)
from utils.print_util import cprint
from utils.telegram_update import publish_msg
from utils.wandb_update import (wandb_init, publish_intermediate)


class ExperimentHelper:
    def __init__(self, experiment_name, freq=None, tb_writer=None, overwrite=False, publish=False, config=None):
        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if overwrite:
                cprint("[ <", experiment_name,
                       "> output exists - Overwriting! ]", type="warn")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                cprint("[ <", experiment_name,
                       "> output exists - Manual deletion needed ]", type="warn")
                exit()

        self.publish = publish
        self.publish and wandb_init(config)

        self.experiment_name = experiment_name
        self.best_val_loss = float('inf')
        self.best_val_kaggle_metric = 0
        self.tb_writer = tb_writer
        self.freq = freq
        self.progress_loss = False
        self.progress_kaggle_metric = False

    def should_trigger(self, i):
        if self.freq:
            return (i + 1) % self.freq == 0
        return True

    def save_checkpoint(self, state_dict):
        if self.progress_loss:
            torch.save(
                state_dict,
                path.join('results', self.experiment_name, 'weights_loss.pth')
            )
        if self.progress_kaggle_metric:
            torch.save(
                state_dict,
                path.join('results', self.experiment_name,
                          'weights_kaggle_metric.pth')
            )
        torch.save(
            state_dict,
            path.join('results', self.experiment_name, 'weights.pth')
        )

    def validate(self, pred_type, num_classes, val_loss, train_loss, val_output_list, val_target_list, train_output_list, train_target_list, epoch):
        if pred_type == 'regression' or pred_type == 'mixed':
            train_output_list = covert_to_classification(
                train_output_list,
                num_classes,
            )
            val_output_list = covert_to_classification(
                val_output_list,
                num_classes,
            )

        # generating accuracy measures
        val_acc = accuracy_generator(val_output_list, val_target_list)
        train_acc = accuracy_generator(
            train_output_list, train_target_list)

        # generating kaggle metric measures
        val_kaggle_metric = kaggle_metric_generator(
            val_output_list, val_target_list)
        train_kaggle_metric = kaggle_metric_generator(
            train_output_list, train_target_list)

        # generate confusion matrix (validation only)
        confusion_matrix_generator(
            val_output_list, val_target_list, self.experiment_name)

        # saving results to csv
        df = pd.DataFrame(
            [[epoch + 1, val_loss, train_loss, val_acc, train_acc, val_kaggle_metric, train_kaggle_metric]])
        result_path = path.join(
            'results', self.experiment_name, 'result.csv')

        if not path.isfile(result_path):
            df.to_csv(
                result_path,
                header=[
                    "epoch",
                    "Loss ( Val )",
                    "Loss ( Train )",
                    "Accuracy ( Val )",
                    "Accuracy ( Train )",
                    "Kaggle Metric ( Val )",
                    "Kaggle Metric ( Train )"
                ],
                index=False
            )
        else:
            df.to_csv(result_path, mode='a', header=False, index=False)

        # creating tensorboard events
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
            self.tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.tb_writer.add_scalar(
                'Accuracy/Validation', val_acc, epoch)
            self.tb_writer.add_scalar(
                'Kaggle Metric/Train', train_kaggle_metric, epoch)
            self.tb_writer.add_scalar(
                'Kaggle Metric/Validation', val_kaggle_metric, epoch)

        # storing loss for check
        if self.best_val_loss >= val_loss:
            self.best_val_loss = val_loss
            self.progress_loss = True
        else:
            self.progress_loss = False

        # storing Kaggle Metric for check
        if self.best_val_kaggle_metric <= val_kaggle_metric:
            self.best_val_kaggle_metric = val_kaggle_metric
            self.progress_kaggle_metric = True
        else:
            self.progress_kaggle_metric = False

        # publish intermediate results
        self.publish and self.publish_intermediate({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_kaggle_metric": val_kaggle_metric,
            "train_kaggle_metric": train_kaggle_metric,
            "epoch": epoch + 1
        },
            val_output_list,
            val_target_list
        )

    def publish_final(self, config):
        # telegram
        # publish_msg(self.result)
        pass

    def publish_intermediate(self, results, val_output_list, val_target_list):
        # wandb
        publish_intermediate(
            results, 
            self.best_val_loss,
            self.best_val_kaggle_metric, 
            val_output_list, 
            val_target_list
        )
