from torch.nn import (CrossEntropyLoss, NLLLoss, MSELoss)
from losses.focal_loss import FocalLoss
from losses.utils import (ClassificationLossWrapper, RegressionLossWrapper)


class LossFactory:
    def __init__(self):
        pass

    def modified(self, ):
        CrossEntropyLoss()

    def get_loss_function(self, function_name, hyper_params=None):
        loss_function = None

        if function_name == 'focal-loss':
            print("[ Loss : Focal Loss ]")
            if hyper_params:
                loss_function = FocalLoss(
                    size_average=hyper_params['size_average']
                )
            else:
                loss_function = FocalLoss()
        if function_name == 'cross-entropy-loss':
            print("[ Loss : Cross Entropy Loss ]")
            loss_function = ClassificationLossWrapper(CrossEntropyLoss())

        if function_name == 'negative-log-likelihood-loss':
            print("[ Loss : Negative Log Likelihood Loss ]")
            loss_function = ClassificationLossWrapper(NLLLoss())

        if function_name == 'mean-squared-error-loss':
            print("[ Loss : Mean Squared Error Loss ]")
            loss_function = RegressionLossWrapper(MSELoss())

        return loss_function
