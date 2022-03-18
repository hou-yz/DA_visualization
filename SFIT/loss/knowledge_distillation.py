import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, temperature=1):
        super(KDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        """
        NOTE: the KL Divergence for PyTorch comparing the prob of teacher and log prob of student,
        mimicking the prob of ground truth (one-hot) and log prob of network in CE loss
        """
        # x -> input -> log(q)
        log_q = F.log_softmax(student_output / self.temperature, dim=1)
        # y -> target -> p
        p = F.softmax(teacher_output / self.temperature, dim=1)
        # F.kl_div(x, y) -> F.kl_div(log_q, p)
        # l_n = y_n \cdot \left( \log y_n - x_n \right) = p * log(p/q)
        l_kl = F.kl_div(log_q, p, reduction='batchmean')  # forward KL
        return l_kl
