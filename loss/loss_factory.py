from torch.nn import (CrossEntropyLoss, NLLLoss, MSELoss)
from loss.focal_loss import FocalLoss
from loss.arcface_loss import ArcfaceLoss
from loss.utils import (ClassificationLossWrapper,
                          RegressionLossWrapper, MixedLossWrapper)


class LossFactory:
    def __init__(self):
        pass

    def get_pure(self, function_name, hyper_params=None):
        loss_function = None

        if function_name == 'focal-loss':
            print("[ Loss : Focal Loss ]")
            if hyper_params:
                loss_function = FocalLoss(
                    size_average=hyper_params['size_average']
                )
            else:
                loss_function = FocalLoss()

        elif function_name == 'cross-entropy-loss':
            print("[ Loss : Cross Entropy Loss ]")
            loss_function = CrossEntropyLoss()

        elif function_name == 'negative-log-likelihood-loss':
            print("[ Loss : Negative Log Likelihood Loss ]")
            loss_function = NLLLoss()

        elif function_name == 'mean-squared-error-loss':
            print("[ Loss : Mean Squared Error Loss ]")
            loss_function = MSELoss()

        elif function_name == 'arcface-loss':
            print("[ Loss: Arcface Loss ]")
            loss_function = ArcfaceLoss()

        return loss_function

    def get_loss_function(self, function_name, pred_type, hyper_params=None):
        wrapped_loss_function = None

        if pred_type == 'regression':
            wrapped_loss_function = RegressionLossWrapper(
                self.get_pure(function_name, hyper_params))
        elif pred_type == 'classification':
            wrapped_loss_function = ClassificationLossWrapper(
                self.get_pure(function_name, hyper_params))
        elif pred_type == 'mixed':
            wrapped_loss_function = MixedLossWrapper(
                self.get_pure(
                    function_name,
                    hyper_params
                ),
                self.get_pure(
                    hyper_params['classification_loss'],
                    hyper_params
                ),
                hyper_params['classification_coefficient']
            )

        return wrapped_loss_function
