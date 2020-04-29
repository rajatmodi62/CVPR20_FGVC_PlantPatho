import torch
import torch.nn.functional as F


class FocalLoss:
    def __init__(self, alpha=None, gamma=None, size_average=None):
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, output, target):
        if output.dim() > 2:
            # N,C,H,W => N,C,H*W
            output = output.view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)    # N,C,H*W => N,H*W,C
            output = output.contiguous().view(-1, output.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != output.data.type():
                self.alpha = self.alpha.type_as(output.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def __call__(self, output, target):
        return self.forward(output, target)
