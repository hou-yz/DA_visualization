import torch
from torch import nn
import torch.nn.functional as F


class BatchSimLoss(nn.Module):
    def __init__(self):
        super(BatchSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C * H * W]), featmap_tgt_S.view([B, C * H * W])
        A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_batch = torch.norm(A_src - A_tgt) ** 2 / B
        return loss_batch


class PixelSimLoss(nn.Module):
    def __init__(self):
        super(PixelSimLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss_pixel = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src.T @ f_src, f_tgt.T @ f_tgt
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_pixel += torch.norm(A_src - A_tgt) ** 2 / (H * W)
        loss_pixel /= B
        return loss_pixel


class ChannelSimLoss(nn.Module):
    def __init__(self):
        super(ChannelSimLoss, self).__init__()

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


class ChannelSimLoss1D(nn.Module):
    def __init__(self):
        super(ChannelSimLoss1D, self).__init__()

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


if __name__ == '__main__':
    from SFIT.loss.style_loss import StyleLoss

    feat1, feat2 = torch.ones([16, 2048, 7, 7]), torch.zeros([16, 2048, 7, 7])
    batch_loss = BatchSimLoss()
    l1 = batch_loss(feat1, feat2)
    pixel_loss = PixelSimLoss()
    l2 = pixel_loss(feat1, feat2)
    channel_loss = ChannelSimLoss()
    l3 = channel_loss(feat1, feat2)
    style_loss = StyleLoss()
    l4 = style_loss(feat1, feat2)

    feat1, feat2 = torch.ones([16, 2048]), torch.zeros([16, 2048])
    channel_loss = ChannelSimLoss1D()
    l1 = channel_loss(feat1, feat2)
    pass
