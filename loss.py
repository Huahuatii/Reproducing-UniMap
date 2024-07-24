import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calc_coeff(iter_num, upper_bound=1.0, lower_bound=0.0, alpha=10.0, max_iter=10000.0):
    coeff = 2.0 * (upper_bound - lower_bound) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (upper_bound - lower_bound) + lower_bound
    return coeff


def Entropy(input_):
    entropy = torch.sum(-input_ * torch.log(input_ + 1e-7), dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def DANN_multi(st_z, ad_nets, celltype_weights, st_cell_w, st_cell_prob, st_cell_entropy=None, coeff=None, cell_weights=None, batch_size=128):
    ad_outs = [ad_net(st_z) for ad_net in ad_nets]
    ad_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(st_z.device)

    st_cell_w.register_hook(grl_hook(coeff))

    loss_all = 0
    for j in range(len(ad_outs)):

        ct_w_j = celltype_weights[j]
        st_cell_w = st_cell_w.view(-1, 1)
        st_p_j = st_cell_prob[:, j:j + 1]
        bce_loss = nn.BCELoss(reduction='none')(ad_outs[j], ad_target)

        loss_all += torch.sum(ct_w_j * st_cell_w * st_p_j * bce_loss) / torch.sum(st_cell_w).detach().item()

    return loss_all


def marginloss(yHat, y, classes=32, alpha=1, weight=None, device='cuda'):
    batch_size = len(y)
    yint = torch.nonzero(y)[:, 1]
    yHat = F.softmax(yHat, dim=1)

    Yg = torch.gather(yHat, 1, yint.view(-1, 1))
    Yg_ = (1 - Yg) + 1e-7
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)
    y_zerohot = torch.ones(batch_size, classes).scatter_(1, yint.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.to(device)
    loss = torch.sum(output, dim=1) / np.log(classes - 1)
    Yg_ = Yg_ ** alpha

    weight = (weight * (Yg_.view(len(yHat), ) / Yg_.sum())).detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)
    return loss


# def mmd_loss(source_features, target_features, device):

#     def pairwise_distance(x, y):
#         if not len(x.shape) == len(y.shape) == 2:
#             raise ValueError('Both inputs should be matrices.')

#         if x.shape[1] != y.shape[1]:
#             raise ValueError('The number of features should be the same.')

#         x = x.view(x.shape[0], x.shape[1], 1)
#         y = torch.transpose(y, 0, 1)
#         output = torch.sum((x - y) ** 2, 1)
#         output = torch.transpose(output, 0, 1)

#         return output

#     def gaussian_kernel_matrix(x, y, sigmas):
#         sigmas = sigmas.view(sigmas.shape[0], 1)
#         beta = 1. / (2. * sigmas)
#         dist = pairwise_distance(x, y).contiguous()
#         dist_ = dist.view(1, -1)
#         s = torch.matmul(beta, dist_)

#         return torch.sum(torch.exp(-s), 0).view_as(dist)

#     def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
#         cost = torch.mean(kernel(x, x))
#         cost += torch.mean(kernel(y, y))
#         cost -= 2 * torch.mean(kernel(x, y))

#         return cost
#     sigmas = [
#         1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#         1e3, 1e4, 1e5, 1e6
#     ]
#     gaussian_kernel = partial(
#         gaussian_kernel_matrix, sigmas=Variable(torch.Tensor(sigmas), requires_grad=False).to(device)
#     )

#     loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
#     loss_value = loss_value

#     return loss_value
