import torch
from torch.nn import (CrossEntropyLoss)
from math import (cos, sin, pi)


class ArcfaceLoss:
    def __init__(self, s=30.0, m=0.5, reduction="mean"):
        self.s = s
        self.m = m
        self.loss_fn = CrossEntropyLoss(reduction=reduction)

        self.cos_m = cos(m)
        self.sin_m = sin(m)

        self.th = cos(pi - m)
        self.mm = sin(pi - m) * m

    def __call__(self, output, target):
        # converting labels to One hot labels
        OHV_target = torch.zeros(output.size()).to(target.get_device())
        OHV_target[range(output.size()[0]), target] = 1

        cos_th = output
        sin_th = torch.sqrt(torch.clamp(1.0 - torch.pow(cos_th, 2), 1e-9, 1))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m

        # not sure why this is here??
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        output = OHV_target * cos_th_m + (1 - OHV_target) * cos_th
        output *= self.s

        return self.loss_fn(output, target) / 2
