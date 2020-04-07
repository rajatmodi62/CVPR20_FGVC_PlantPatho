import torch
from torch.nn import (CrossEntropyLoss)
from math import (cos, sin)


class ArcfaceLoss:
    def __init__(self, s=30.0, m=0.5, reduction="mean"):
        self.s = s
        self.m = m
        self.loss_fn = CrossEntropyLoss(reduction=reduction)

        self.cos_m = cos(m)
        self.sin_m = sin(m)

    def __call__(self, output, target):
        cos_th = output
        sin_th = torch.sqrt(1 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        output = target * cos_th_m + (1 - target) * cos_th
        return self.loss_fn(output, torch.argmax(target, dim=1))
