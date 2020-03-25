import torch
from os import (makedirs, path)
from shutil import rmtree
import pandas as pd
import math
from sklearn.metrics import roc_auc_score


class ExperimentHelper:
    def __init__(self, experiment_name, rewrite=False, freq=None, tb_writer=None):
        if path.exists(path.join('results', experiment_name)) == False:
            makedirs(path.join('results', experiment_name))
        else:
            if rewrite:
                print("[ <", experiment_name, "> output exists - Overwriting! ]")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                print("[ <", experiment_name, "> output exists - Manual deletion needed ]")
                exit()

        self.experiment_name = experiment_name
        self.best_val_loss = float('inf')
        self.tb_writer = tb_writer
        self.freq = freq
        self.progress = False

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
                val_output_list, torch.argmax(val_target_list, dim=1)).item()
            train_loss = loss_fn(
                train_output_list, torch.argmax(train_target_list, dim=1)).item()

            val_acc = torch.argmax(val_target_list, dim=1).eq(
                torch.argmax(val_output_list, dim=1))
            val_acc = 1.0 * torch.sum(val_acc.int()).item() / \
                val_output_list.size()[0]

            train_acc = torch.argmax(train_target_list, dim=1).eq(
                torch.argmax(train_output_list, dim=1))
            train_acc = 1.0 * torch.sum(train_acc.int()
                                        ).item() / train_output_list.size()[0]

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


def roc_auc_score_generator(output_list, target_list):
    return roc_auc_score(
        target_list.cpu().numpy(), 
        output_list.cpu().numpy(), 
        average="macro"
    )
