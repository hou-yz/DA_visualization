import torch
from torch import nn
import torch.nn.functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = -F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1).mean()
        return b
