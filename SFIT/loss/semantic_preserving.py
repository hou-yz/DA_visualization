import torch
from torch import nn
import torch.nn.functional as F


class BatchSimilarityLoss(nn.Module):
    def __init__(self):
        super(BatchSimilarityLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        f_src, f_tgt = featmap_src_T.view([B, C * H * W]), featmap_tgt_S.view([B, C * H * W])
        A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
        A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
        loss_sim = torch.norm(A_src - A_tgt) ** 2 / B
        return loss_sim


class ImageSemanticLoss(nn.Module):
    def __init__(self):
        super(ImageSemanticLoss, self).__init__()

    def forward(self, featmap_src_T, featmap_tgt_S):
        B, C, H, W = featmap_src_T.shape
        loss_semantic = 0
        for b in range(B):
            f_src, f_tgt = featmap_src_T[b].view([C, H * W]), featmap_tgt_S[b].view([C, H * W])
            A_src, A_tgt = f_src.T @ f_src, f_tgt.T @ f_tgt
            A_src, A_tgt = F.normalize(A_src, p=2, dim=1), F.normalize(A_tgt, p=2, dim=1)
            loss_semantic += torch.norm(A_src - A_tgt) ** 2 / (H * W)
        loss_semantic /= B
        return loss_semantic


if __name__ == '__main__':
    feat1, feat2 = torch.ones([16, 2048, 7, 7]), torch.zeros([16, 2048, 7, 7])
    sim_loss = BatchSimilarityLoss()
    l1 = sim_loss(feat1, feat2)
    semantic_loss = ImageSemanticLoss()
    l2 = semantic_loss(feat1, feat2)
    pass
