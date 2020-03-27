from torch.nn import (CrossEntropyLoss, NLLLoss)
from losses.focal_loss import FocalLoss


class LossFactory:
    def __init__(self):
        pass

    def get_loss_function(self, function_name, hyper_params=None):
        loss_function = None

        if function_name == 'focal-loss':
            print("[ Loss : Focal Loss ]")
            if hyper_params is not None:
                loss_function = FocalLoss(
                    size_average=hyper_params['size_average']
                )
            else:
                loss_function = FocalLoss()
        if function_name == 'cross-entropy-loss':
            print("[ Loss : Cross Entropy Loss ]")
            loss_function = CrossEntropyLoss()

        if function_name == 'negative-log-likelihood-loss':
            print("[ Loss : Negative Log Likelihood Loss ]")
            loss_function = NLLLoss()

        return loss_function
