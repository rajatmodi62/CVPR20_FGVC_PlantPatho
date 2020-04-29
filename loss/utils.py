import torch


class ClassificationLossWrapper:
    def __init__(self, loss_obj):
        self.loss = loss_obj

    def to(self, device):
        self.loss.to(device)

    def __call__(self, output, target):
        return self.loss(output, torch.argmax(target, dim=1))


class RegressionLossWrapper:
    def __init__(self, loss_obj):
        self.loss = loss_obj

    def to(self, device):
        self.loss.to(device)

    def __call__(self, output, target):
        return self.loss(output, torch.argmax(target, dim=1).view(-1, 1).float())


class MixedLossWrapper:
    def __init__(self, reg_loss_obj, class_loss_obj, classification_coefficient):
        self.reg_loss = reg_loss_obj
        self.class_loss = class_loss_obj
        self.class_coeff = classification_coefficient
        self.reg_coeff = 1 - classification_coefficient

    def to(self, device):
        self.reg_loss.to(device)
        self.class_loss.to(device)

    def __call__(self, output, target):
        mixed_loss = self.reg_coeff * \
            self.reg_loss(output[:, 0].view(-1, 1), torch.argmax(
                target, dim=1).view(-1, 1).float())
        mixed_loss += self.class_coeff * \
            self.class_loss(output[:, 1:], torch.argmax(target, dim=1))
        return mixed_loss
