import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from typing import List


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# def grl_hook(coeff):
#     def fun1(grad):
#         return -coeff * grad.clone()
#         # return (coeff - 1) * grad.clone()
#     return fun1


def calc_coeff(iter_num, upper_bound=1.0, lower_bound=0.0, alpha=10.0, max_iter=10000.0):
    coeff = 2.0 * (upper_bound - lower_bound) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (upper_bound - lower_bound) + lower_bound
    return coeff


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


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd)
    return GradReverse.apply(x, lam)


class DSBN(nn.Module):
    def __init__(self, in_feature, n_domain, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_feature = in_feature
        self.n_domain = n_domain
        self.bns = nn.ModuleList([nn.BatchNorm1d(in_feature, eps=eps, momentum=momentum) for _ in range(n_domain)])
        self.apply(init_weights)

    def forward(self, x, domain):
        out = torch.zeros(x.size(0), self.in_feature, device=x.device)
        for i in range(self.n_domain):
            indices = np.where(domain.cpu().numpy() == i)[0]
            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        return out


class MLP(nn.Module):
    def __init__(self, in_feature=1126, latent_feature=128, num_classes=32, drop=0.1, num_batches=2, batch_emb_dims=10, *args, **kwargs):
        super(MLP, self).__init__()

        # feature extractor
        self.batch_embedding = nn.Embedding(num_batches, batch_emb_dims)
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(in_feature, 512, bias=True))
        self.feature.add_module('f_af1', nn.ReLU())
        self.feature.add_module('f_dp1', nn.Dropout(drop))

        self.feature.add_module('f_fc2', nn.Linear(512, 512, bias=True))
        self.feature.add_module('f_af2', nn.ReLU())
        self.feature.add_module('f_dp2', nn.Dropout(drop))

        self.feature.add_module('f_fc3', nn.Linear(512, latent_feature, bias=True))
        self.feature.add_module('f_af3', nn.ReLU())
        self.feature.add_module('f_dp3', nn.Dropout(drop))

        # classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(latent_feature, latent_feature, bias=True))
        self.classifier.add_module('c_af1', nn.ReLU())
        self.classifier.add_module('c_dp1', nn.Dropout(drop))

        self.classifier.add_module('c_fc2', nn.Linear(latent_feature, num_classes, bias=True))
        self.classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # initialization
        self.apply(init_weights)

    def forward(self, x):
        feature = self.feature_forward(x)
        y = self.classifier_forward(feature)
        return feature, y

    def feature_forward(self, x):
        x = self.feature.f_fc1(x)

        x = self.feature.f_af1(x)
        x = self.feature.f_dp1(x)

        x = self.feature.f_fc2(x)

        x = self.feature.f_af2(x)
        x = self.feature.f_dp2(x)

        x = self.feature.f_fc3(x)

        # x = self.feature.f_af3(x)
        # x = self.feature.f_dp3(x)
        return x

    def classifier_forward(self, x):
        x = self.classifier.c_fc1(x)
        x = self.classifier.c_af1(x)
        x = self.classifier.c_dp1(x)
        x = self.classifier.c_fc2(x)
        # x = self.classifier.c_softmax(x)
        return x


class CMLP(nn.Module):
    def __init__(self, in_feature=1126, latent_feature=128, num_classes=32, drop=0.1, num_batches=2, batch_emb_dims=10, *args, **kwargs):
        super(CMLP, self).__init__()

        self.batch_embedding = nn.Embedding(num_batches, batch_emb_dims)
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(in_feature + batch_emb_dims, 512, bias=True))
        self.feature.add_module('f_af1', nn.ReLU())
        self.feature.add_module('f_dp1', nn.Dropout(drop))

        self.feature.add_module('f_fc2', nn.Linear(512, 512, bias=True))
        self.feature.add_module('f_af2', nn.ReLU())
        self.feature.add_module('f_dp2', nn.Dropout(drop))

        self.feature.add_module('f_fc3', nn.Linear(512, latent_feature, bias=True))
        self.feature.add_module('f_af3', nn.ReLU())
        self.feature.add_module('f_dp3', nn.Dropout(drop))

        # classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(latent_feature, latent_feature, bias=True))
        self.classifier.add_module('c_af1', nn.ReLU())
        self.classifier.add_module('c_dp1', nn.Dropout(drop))

        self.classifier.add_module('c_fc2', nn.Linear(latent_feature, num_classes, bias=True))
        self.classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # initialization
        self.apply(init_weights)

    def forward(self, x, conditions=None):
        feature = self.feature_forward(x, conditions)
        y = self.classifier_forward(feature)
        return feature, y

    def feature_forward(self, x, conditions):

        c = self.batch_embedding(conditions)
        x = torch.cat((x, c), dim=1)
        x = self.feature.f_fc1(x)

        x = self.feature.f_af1(x)
        x = self.feature.f_dp1(x)

        x = self.feature.f_fc2(x)

        x = self.feature.f_af2(x)
        x = self.feature.f_dp2(x)

        x = self.feature.f_fc3(x)

        # x = self.feature.f_af3(x)
        # x = self.feature.f_dp3(x)
        return x

    def classifier_forward(self, x):
        x = self.classifier.c_fc1(x)
        x = self.classifier.c_af1(x)
        x = self.classifier.c_dp1(x)
        x = self.classifier.c_fc2(x)
        # x = self.classifier.c_softmax(x)
        return x


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, dp=0.2, max_iter=25000):
        super(AdversarialNetwork, self).__init__()
        self.revgrad = grad_reverse
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x = self.revgrad(x, coeff)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y


class AE(nn.Module):
    def __init__(self, in_feature=1126, hidden_feature=128, drop=0.1, noise_flag=False):
        super(AE, self).__init__()

        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('e_fc1', nn.Linear(in_feature, 512, bias=True))
        self.encoder.add_module('e_af1', nn.ReLU())
        self.encoder.add_module('e_dp1', nn.Dropout(drop))
        self.encoder.add_module('e_fc2', nn.Linear(512, hidden_feature, bias=True))
        self.encoder.add_module('e_af2', nn.ReLU())
        self.encoder.add_module('e_dp2', nn.Dropout(drop))

        # decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('d_fc1', nn.Linear(hidden_feature, hidden_feature, bias=True))
        self.decoder.add_module('d_af1', nn.ReLU())
        self.decoder.add_module('d_dp1', nn.Dropout(drop))
        self.decoder.add_module('d_fc2', nn.Linear(hidden_feature, 512, bias=True))
        self.decoder.add_module('d_af2', nn.ReLU())
        self.decoder.add_module('d_dp2', nn.Dropout(drop))
        self.decoder.add_module('d_fc3', nn.Linear(512, in_feature, bias=True))

        self.apply(init_weights)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recons = self.decode(z)
        return [x, recons, z]

    def loss_function(self, *args) -> dict:
        input = args[0]
        recons = args[1]

        recons_loss = F.mse_loss(input, recons)
        loss = recons_loss

        return {'loss': loss, 'recons_loss': recons_loss}


class SharedEncoder(nn.Module):
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


class Critic(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(Critic, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

        self.iter_num = 0

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    # def output_num(self):
    #     return 1

    # def get_parameters(self):
    #     return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
