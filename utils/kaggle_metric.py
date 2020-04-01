import torch
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score


def post_process_output(output):
    # implementation based on problem statement
    return softmax(output, dim=1)


def accuracy_generator(output_list, target_list):
    # implementation based on problem statement
    acc = torch.argmax(target_list, dim=1).eq(
        torch.argmax(output_list, dim=1))
    return 1.0 * torch.sum(acc.int()).item() / output_list.size()[0]


def roc_auc_score_generator(output_list, target_list):
    # implementation based on problem statement
    output_list = post_process_output(output_list)

    return roc_auc_score(
        target_list.cpu().numpy(),
        output_list.cpu().numpy(),
        average="macro"
    )


# implementation based on problem statement
kaggle_output_header = ["image_id", "healthy",
                        "multiple_diseases", "rust", "scab"]

# implementation based on problem statement
custom_threshhold = [0.7, 1.5, 2.7]
