import torch


def covert_to_classification(output, num_classes, device = None):
    # takes a value and number of classes, returns a onehot vector
    batch_size = output.size()[0]

    one_hot = None
    if device:
        one_hot = torch.zeros(batch_size, num_classes).to( device )
    else:
        one_hot = torch.zeros(batch_size, num_classes)

    for class_idx in range(num_classes):
        one_hot[:, class_idx] = (
            (class_idx - 0.5) < output[:, 0]) & (output[:, 0] < (class_idx + 0.5))

    return one_hot
