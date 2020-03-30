import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import math
from utils.regression_utils import covert_to_classification
from utils.kaggle_metric import (roc_auc_score_generator, accuracy_generator)


class ExperimentHelper:
    def __init__(self, experiment_name, num_classes, device=None, rewrite=False, freq=None, tb_writer=None, pred_type='classification'):
        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if rewrite:
                print("[ <", experiment_name, "> output exists - Overwriting! ]")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                print("[ <", experiment_name,
                      "> output exists - Manual deletion needed ]")
                exit()

        self.pred_type = pred_type
        self.experiment_name = experiment_name
        self.best_val_loss = float('inf')
        self.tb_writer = tb_writer
        self.freq = freq
        self.progress = False
        self.num_classes = num_classes
        self.device = device

    def should_trigger(self, i):
        if self.freq is None:
            return True
        else:
            return i % self.freq == 0

    def is_progress(self):
        return self.progress

    def save_checkpoint(self, state_dict):
        torch.save(
            state_dict,
            path.join('results', self.experiment_name, 'weights.pth')
        )

    def validate(self, loss_fn, val_output_list, val_target_list, train_output_list, train_target_list, epoch):
        with torch.no_grad():
            # loss calculation
            val_loss = loss_fn(
                val_output_list, val_target_list).item()
            train_loss = loss_fn(
                train_output_list, train_target_list).item()

            if self.pred_type == 'regression':
                train_output_list = covert_to_classification(
                    train_output_list,
                    self.num_classes,
                    self.device
                )
                val_output_list = covert_to_classification(
                    val_output_list,
                    self.num_classes,
                    self.device
                )

            val_acc = accuracy_generator(val_output_list, val_target_list)
            train_acc = accuracy_generator(
                train_output_list, train_target_list)

            val_roc = roc_auc_score_generator(val_output_list, val_target_list)
            train_roc = roc_auc_score_generator(
                train_output_list, train_target_list)

            # saving results to csv
            df = pd.DataFrame(
                [[epoch, val_loss, train_loss, val_acc, train_acc, val_roc, train_roc]])
            result_path = path.join(
                'results', self.experiment_name, 'result.csv')

            if not path.isfile(result_path):
                df.to_csv(
                    result_path,
                    header=[
                        "epoch", "Loss ( Val )", "Loss ( Train )", "Accuracy ( Val )", "Accuracy ( Train )", "ROC ( Val )", "ROC ( Train )"
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
                self.tb_writer.add_scalar('ROC/Train', train_roc, epoch)
                self.tb_writer.add_scalar('ROC/Validation', val_roc, epoch)

            # storing loss for check
            if self.best_val_loss >= val_loss:
                self.best_val_loss = val_loss
                self.progress = True
            else:
                self.progress = False
