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
                print("[ Experiment output already exists - Overwriting! ]")
                rmtree(path.join('results', experiment_name))
                makedirs(path.join('results', experiment_name))
            else:
                print("[ Experiment output already exists - Manual deletion needed ]")
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

        # saving results to csv
        df = pd.DataFrame(
            [[epoch, val_loss, train_loss, val_acc, train_acc]])
        result_path = path.join('results', self.experiment_name, 'result.csv')

        if not path.isfile(result_path):
            df.to_csv(result_path, header=[
                "epoch", "Loss ( Val )", "Loss ( Train )", "Accuracy ( Val )", "Accuracy ( Train )"], index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(result_path, mode='a', header=False, index=False)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
            self.tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.tb_writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # storing loss for check
        if self.best_val_loss >= val_loss:
            self.best_val_loss = val_loss
            self.progress = True
        else:
            self.progress = False


def roc_auc_score_generator(output_list, target_list):
    roc_auc_score(target_list, output_list, average="macro")
