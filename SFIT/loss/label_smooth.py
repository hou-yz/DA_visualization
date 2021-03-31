import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, e=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.e = e

    def forward(self, x, target):
        target = torch.zeros_like(x).scatter_(1, target.unsqueeze(1), 1)
        smoothed_target = (1 - self.e) * target + self.e / x.size(1)
        loss = (- F.log_softmax(x, dim=1) * smoothed_target).sum(dim=1)
        return loss.mean()


if __name__ == '__main__':
    loss = LabelSmoothLoss()
    output = torch.randn(64, 10)
    label = torch.randint(0, 10, [64])
    loss(output, label)
