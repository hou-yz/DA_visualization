import torch.nn as nn
import torch.nn.functional as F


class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()

    def forward(self, p_output, q_output):
        # 1/2*KL(p,m) + 1/2*KL(q,m)
        p = F.softmax(p_output, dim=1)
        q = F.softmax(q_output, dim=1)
        log_m = (0.5 * (p + q)).log()
        # F.kl_div(x, y) -> F.kl_div(log_q, p)
        l_js = 0.5 * (F.kl_div(log_m, p) + F.kl_div(log_m, q))
        return l_js
