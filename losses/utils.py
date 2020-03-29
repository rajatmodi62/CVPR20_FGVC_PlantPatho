import torch


class LossWrapper:
    def __init__(self, loss_obj):
        self.loss = loss_obj
        pass

    def to(self, device):
        self.loss.to(device)

    def __call__(self, output, target):
        return self.loss(output, torch.argmax(target, dim=1))
