import torch
from torch import nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C, H * W]), featmap_tgt_S.view([B, C, H * W])
        # calculate Gram matrices
        A_src, A_tgt = torch.bmm(f_src, f_src.transpose(1, 2)), torch.bmm(f_tgt, f_tgt.transpose(1, 2))
        A_src, A_tgt = A_src / (H * W), A_tgt / (H * W)
        loss = F.mse_loss(A_src, A_tgt)
        return loss


if __name__ == '__main__':
    feat1, feat2 = torch.ones([16, 2048, 7, 7]), torch.zeros([16, 2048, 7, 7])
    style_loss = StyleLoss()
    l1 = style_loss(feat1, feat2)
    pass
