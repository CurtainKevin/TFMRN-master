import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.fft as fft


def select_freq(x, k):
    x_fft = fft.rfft(x, dim=1)
    eps = 1e-6
    amp = torch.sqrt((x_fft.real + eps).pow(2) + (x_fft.imag + eps).pow(2))
    topk = torch.topk(amp, k, 1)
    idx = topk.indices
    fft_selected = torch.gather(x_fft, 1, idx)
    return idx, fft_selected


def decomp_k(x, k):
    B, L, N = x.size()
    out = torch.zeros((B, L // 2 + 1, N), dtype=torch.cdouble).to(x.device)
    x_fft = fft.rfft(x, dim=1)
    eps = 1e-6
    amp = torch.sqrt((x_fft.real + eps).pow(2) + (x_fft.imag + eps).pow(2))
    topk = torch.topk(amp, k, 1)
    idx = topk.indices
    fft_selected = torch.gather(x_fft, 1, idx)
    out = out.scatter(1, idx, fft_selected)
    out = fft.irfft(out, n=L, dim=1)
    noisy = x - out
    return out, noisy


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


def complex_softmax(input, dim=-1):
    return F.softmax(input.real, dim=dim).type(torch.complex128) + 1j * F.softmax(input.imag, dim=dim).type(
        torch.complex128)


def complex_dropout(input, dropout):
    mask = dropout(torch.ones_like(input.real))
    return input * mask


def complex_activate(input, activation):
    return activation(input.real).type(torch.complex128) + 1j * activation(input.imag).type(torch.complex128)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
           against Distribution Shift, ICLR2021.
    Parameters
    ----------
    num_features: int, the number of features or channels.
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Attention_MLP(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, in_dim, out_dim, hidden, channel):
        super(Attention_MLP, self).__init__()
        self.hidden = hidden

        # self.gmlp = torch.nn.Linear(in_dim, out_dim, bias=False)
        # self.gmlp = nn.Sequential(nn.Tanh(), nn.Linear(in_dim, self.hidden, bias=False), nn.Tanh(),
        #                            nn.Linear(self.hidden, out_dim, bias=False), nn.Softmax(-1), nn.LayerNorm(out_dim))
        self.gmlp1 = nn.Sequential(nn.Tanh(), nn.Linear(in_dim, self.hidden, bias=False), nn.ReLU())
        self.gmlp2 = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden, out_dim, bias=False), nn.Softmax(-1),
                                   nn.LayerNorm(out_dim))
        self.mem = nn.Parameter(torch.zeros(self.hidden, self.hidden), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.mem, gain=1.0)

    def forward(self, x_att, x, freq=False):
        # gate = self.gmlp(x_att.permute(0, 2, 1))
        #
        gate1 = self.gmlp1(x_att.permute(0, 2, 1))
        gate2 = torch.einsum("bdh,he->bde", gate1, self.mem)
        # gate2 = F.dropout(gate2, 0.1)

        gate = self.gmlp2(gate2)
        # gate = F.dropout(gate, 0.1)
        out = torch.einsum("bls,bsd->bds", x, gate)
        # if freq:
        #     out = complex_activate(out, F.relu)
        # else:
        #     out = F.relu(out)

        return out
