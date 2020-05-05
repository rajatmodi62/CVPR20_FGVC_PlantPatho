import torch

# implementation based on problem statement
custom_threshhold = [0.8, 1.9, 2.2]


def covert_to_classification(output, num_classes, threshholding_type='custom'):
    output_size = output.size()[0]
    tta_size = output.size()[2]

    # consider only first column ( for mixed )
    output = output[:, 0, :].view(output_size, 1, tta_size)

    one_hot = None
    if output.is_cuda:
        one_hot = torch.zeros(output_size, num_classes,
                              tta_size).to(output.get_device())
    else:
        one_hot = torch.zeros(output_size, num_classes, tta_size)

    if threshholding_type == 'even':
        for class_idx in range(num_classes):
            one_hot[:, class_idx, :] = (
                (class_idx - 0.5) < output[:, 0, :]) & (output[:, 0, :] < (class_idx + 0.5))
    elif threshholding_type == 'custom':
        if (num_classes != (len(custom_threshhold) + 1)):
            print("[ Custom Threshold list length incorrect ]")
            exit()

        for class_idx in range(num_classes):
            if class_idx == 0:
                one_hot[:, class_idx, :] = output[:,
                                                  0, :] < custom_threshhold[class_idx]
            elif class_idx == len(custom_threshhold):
                one_hot[:, class_idx, :] = custom_threshhold[class_idx -
                                                             1] < output[:, 0, :]
            else:
                one_hot[:, class_idx, :] = (
                    custom_threshhold[class_idx - 1] < output[:, 0, :]) & (output[:, 0, :] < custom_threshhold[class_idx])

    return one_hot
