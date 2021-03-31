import torch
from torch import nn
import torch.nn.functional as F


class ChannelLoss(nn.Module):
    def __init__(self):
        super(ChannelLoss, self).__init__()

    def forward(self, feat_src_T, feat_tgt_S):
        B, C = feat_src_T.shape
        loss = torch.zeros([]).cuda()
        for b in range(B):
            f_src, f_tgt = feat_src_T[b].view([C, 1]), feat_tgt_S[b].view([C, 1])
            A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss += torch.norm(A_src - A_tgt) ** 2 / C
            # loss += torch.norm(A_src - A_tgt, p=1)
        loss /= B
        return loss


class ChannelLoss2D(nn.Module):
    def __init__(self):
        super(ChannelLoss2D, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss += torch.norm(A_src - A_tgt) ** 2 / C
            # loss += torch.norm(A_src - A_tgt, p=1)
        loss /= B
        return loss


if __name__ == '__main__':
    from SFIT.loss.style_loss import StyleLoss

    feat1, feat2 = torch.ones([16, 2048]), torch.zeros([16, 2048])
    channel_loss = ChannelLoss()
    l1 = channel_loss(feat1, feat2)
    pass
    feat1, feat2 = torch.ones([16, 2048, 7, 7]), torch.zeros([16, 2048, 7, 7])
    channel_loss = ChannelLoss2D()
    style_loss = StyleLoss()
    l1 = channel_loss(feat1, feat2)
    l2 = style_loss(feat1, feat2)
    pass
