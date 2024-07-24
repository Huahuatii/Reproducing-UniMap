import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from typing import List
from torch.autograd import Variable
from functools import partial

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class RevGrad(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


class SharedEncoder(nn.Module):
    # This is used for CODE_adv
    def __init__(self, in_feature=1126, latent_feature=128, drop=0.1, gr_flag=False):
        super(SharedEncoder, self).__init__()

        self.shared_encoder = nn.Sequential()
        if gr_flag:
            self.shared_encoder.add_module('f_revgrad', RevGrad())
        self.shared_encoder.add_module('se_fc1', nn.Linear(in_feature, 512, bias=True))
        self.shared_encoder.add_module('se_af1', nn.SELU())
        self.shared_encoder.add_module('se_dp1', nn.Dropout(drop))
        self.shared_encoder.add_module('se_fc2', nn.Linear(512, 256, bias=True))
        self.shared_encoder.add_module('se_af2', nn.SELU())
        self.shared_encoder.add_module('se_dp2', nn.Dropout(drop))
        self.shared_encoder.add_module('se_fc3', nn.Linear(256, latent_feature, bias=True))
        self.apply(init_weights)

    def forward(self, x):
        feature = self.shared_encoder(x)
        return feature


class SharedDecoder(nn.Module):
    # This is used for CODE_adv
    def __init__(self, in_feature=256, latent_feature=1126, drop=0.1, gr_flag=False):
        super(SharedDecoder, self).__init__()
        self.shared_decoder = nn.Sequential()
        if gr_flag:
            self.shared_decoder.add_module('f_revgrad', RevGrad())
        self.shared_decoder.add_module('se_fc1', nn.Linear(in_feature, 256, bias=True))
        self.shared_decoder.add_module('se_af1', nn.SELU())
        self.shared_decoder.add_module('se_dp1', nn.Dropout(drop))
        self.shared_decoder.add_module('se_fc2', nn.Linear(256, 512, bias=True))
        self.shared_decoder.add_module('se_af2', nn.SELU())
        self.shared_decoder.add_module('se_dp2', nn.Dropout(drop))
        self.shared_decoder.add_module('se_fc3', nn.Linear(512, latent_feature, bias=True))
        self.apply(init_weights)

    def forward(self, x):
        feature = self.shared_decoder(x)
        return feature


class DSNAE(nn.Module):
    def __init__(self, shared_encoder, shared_decoder, in_feature=1126, alpha=1.0, drop=0.1):
        super(DSNAE, self).__init__()
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.alpha = alpha

        self.private_encoder = nn.Sequential()
        self.private_encoder.add_module('pe_fc1', nn.Linear(in_feature, 512, bias=True))
        self.private_encoder.add_module('pe_af1', nn.SELU())
        self.private_encoder.add_module('pe_dp1', nn.Dropout(drop))
        self.private_encoder.add_module('pe_fc2', nn.Linear(512, 256, bias=True))
        self.private_encoder.add_module('pe_af2', nn.SELU())
        self.private_encoder.add_module('pe_dp2', nn.Dropout(drop))
        self.private_encoder.add_module('pe_fc3', nn.Linear(256, 128, bias=True))
        self.apply(init_weights)

    def p_encode(self, x):
        return self.private_encoder(x)

    def s_encode(self, x):
        return self.shared_encoder(x)

    def encode(self, x):
        p_z = self.p_encode(x)
        s_z = self.s_encode(x)
        z = torch.cat((p_z, s_z), dim=1)
        return z

    def decode(self, x):
        z = self.encode(x)
        recons = self.shared_decoder(z)
        return recons

    def forward(self, x):
        z = self.encode(x)
        recons = self.shared_decoder(z)
        return [x, recons, z]

    def loss_function(self, *args) -> dict:
        x = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]

        recons_loss = F.mse_loss(x, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

        loss = recons_loss + self.alpha * ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss}


class CODE_AE_MLP(nn.Module):

    def __init__(self, input_dim=256, output_dim=1, hidden_dims: List = [64, 32], dop: float = 0.1, act_fn=nn.SELU, out_fn=None, gr_flag=False, **kwargs) -> None:
        super(CODE_AE_MLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        if gr_flag:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                # nn.BatchNorm1d(hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn()
            )
        self.apply(init_weights)

    def forward(self, input):
        embed = self.module(input)
        output = self.output_layer(embed)

        return output


def mmd_loss(source_features, target_features, device):

    def pairwise_distance(x, y):
        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)

        return output

    def gaussian_kernel_matrix(x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)

        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))

        return cost
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=Variable(torch.Tensor(sigmas), requires_grad=False).to(device)
    )

    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value
