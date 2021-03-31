from torch import nn


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        # COMPUTE total variation regularization loss
        loss_var_l2 = ((image[:, :, :, 1:] - image[:, :, :, :-1]) ** 2).mean() + \
                      ((image[:, :, 1:, :] - image[:, :, :-1, :]) ** 2).mean()

        loss_var_l1 = ((image[:, :, :, 1:] - image[:, :, :, :-1]).abs()).mean() + \
                      ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs()).mean()
        return loss_var_l1, loss_var_l2
