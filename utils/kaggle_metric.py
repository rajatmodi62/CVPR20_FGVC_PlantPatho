import torch
from torch.nn.functional import softmax
from sklearn.metrics import (roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sn


def accuracy_generator(output_list, target_list):
    acc = torch.argmax(target_list, dim=1).eq(
        torch.argmax(output_list, dim=1))
    return 1.0 * torch.sum(acc.int()).item() / output_list.size()[0]


def confusion_matrix_generator(output_list, target_list, experiment_name):
    output_list = post_process_output(output_list)
    matrix = confusion_matrix(
        torch.argmax(target_list, dim=1).numpy(),
        torch.argmax(output_list, dim=1).numpy()
    )
    
    labels = ['H','MD','R','S']
    plt.figure()
    figure = sn.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True)
    plt.savefig('./results/' + experiment_name + '/confusion_matrix.jpg')


def post_process_output(output):
    # implementation based on problem statement
    return softmax(output, dim=1)


def kaggle_metric_generator(output_list, target_list):
    # implementation based on problem statement
    output_list = post_process_output(output_list)

    return roc_auc_score(
        target_list.numpy(),
        output_list.numpy(),
        average="macro"
    )


# implementation based on problem statement
kaggle_output_header = ["image_id", "healthy",
                        "multiple_diseases", "rust", "scab"]
