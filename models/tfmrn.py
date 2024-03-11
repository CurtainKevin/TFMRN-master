import math
import statistics

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Layers import RevIN, Attention_MLP
import torch.fft as fft


class FourierBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_node, topk=None, padding=1.0, hid_dim=128):
        super(FourierBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.hid_dim = hid_dim
        self.tk = topk
        self.padding = padding
        g_in = (padding + 1) * in_channel // 2 + 1 + in_channel // 2 + 1

        self.attn = Attention_MLP(g_in, in_channel // 2 + 1, hidden=self.hid_dim, channel=self.num_node)

    def forward(self, x):
        B, L, N = x.size()
        zero = torch.zeros(B, self.padding * L, N, device=x.device)
        x_fft = fft.rfft(x, dim=1)

        #   fence
        x_pad = torch.cat([x, zero], dim=1)
        x_p_fft = fft.rfft(x_pad, dim=1)

        # top_amp
        eps = 1e-6
        amp_p = torch.sqrt((x_p_fft.real + eps).pow(2) + (x_p_fft.imag + eps).pow(2))
        amp = torch.sqrt((x_fft.real + eps).pow(2) + (x_fft.imag + eps).pow(2))
        topk = torch.topk(amp, self.tk, 1)
        idx = topk.indices
        values = topk.values

        amp_topk = torch.zeros_like(amp, device=amp.device)
        amp_topk = amp_topk.scatter(1, idx, values)

        x_gate = torch.cat([amp_topk, amp_p], dim=1)

        out_fft = self.attn(x_gate, x_fft, True)

        out = fft.irfft(out_fft, n=self.in_channel, dim=1)

        return out


class TimeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_node, padding, chunk_size, hid_dim=128, ratio=0.3):
        super(TimeBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_node = num_node
        self.chunk_size = chunk_size
        self.hid_dim = hid_dim
        assert (in_channel % chunk_size == 0)
        self.num_chunks = in_channel // chunk_size
        self.padding = padding
        self.ratio = ratio
        num_sel = math.floor(ratio * in_channel * padding)
        g_in = num_sel + chunk_size + in_channel
        self.attn = Attention_MLP(g_in, self.in_channel, hidden=self.hid_dim, channel=self.num_node)

    def forward(self, x):
        B, L, N = x.size()

        x1 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x1 = x1[:, :, 0, :]
        x1 = x1.permute(0, 2, 1)

        x2 = F.interpolate(x.permute(0, 2, 1), scale_factor=self.padding + 1, mode='linear')
        x2_int = x2[:, :, 1::2]

        shape = x2_int.shape
        num_sel = math.floor(self.ratio * shape[-1])
        tensor = torch.zeros(shape, device=x2.device)
        indices_to_set = torch.randperm(shape[-1])[:num_sel]
        tensor[..., indices_to_set] = 1
        # shuffle
        shuffled_tensor = tensor[..., torch.randperm(shape[-1])]
        result_matrix = x2_int.masked_select(shuffled_tensor.bool()).reshape(B, N, -1)

        x_gate = torch.cat([x1, x.permute(0, 2, 1), result_matrix], dim=-1)

        out = self.attn(x_gate.permute(0, 2, 1), x)

        return out


class MResBlock(nn.Module):
    def __init__(self, lookback, lookahead, hid_dim, num_node, dropout=0.1, chunk_size=40, c_dim=40):
        super(MResBlock, self).__init__()
        self.lookback = lookback
        self.lookahead = lookahead
        self.chunk_size = chunk_size
        self.num_chunks = lookback // chunk_size
        self.hid_dim = int(hid_dim)
        self.num_node = int(num_node)
        self.c_dim = int(c_dim)
        self.tk = 8
        # self.padding = 0
        self.padding_t = 1
        self.padding_f = 1
        self.dropout = dropout
        self._build()

    def _build(self):
        self.fb = FourierBlock(self.lookback, self.lookahead, self.num_node, self.tk, self.padding_f, self.hid_dim)
        self.tb = TimeBlock(self.lookback, self.lookahead, self.num_node, self.padding_t, self.chunk_size, self.hid_dim)

    def forward(self, x):
        fout = self.fb(x)  # (B,L,N)
        tout = self.tb(x)  # (B,L,N)
        return fout, tout


class TFMRN(nn.Module):
    def __init__(self, lookback, lookahead, hid_dim, num_node, dropout=0, chunk_size=40, c_dim=40):
        super(TFMRN, self).__init__()
        self.lookback = int(lookback)
        self.lookahead = int(lookahead)
        self.chunk_size = int(chunk_size)
        self.num_chunks = lookback // chunk_size

        self.hid_dim = int(hid_dim)
        self.num_node = int(num_node)
        self.c_dim = int(c_dim)
        self.dropout = dropout

        self._build()

    def _build(self):
        self.revinlayer = RevIN(num_features=self.num_node)

        self.layer = MResBlock(
            lookback=self.lookback,
            lookahead=self.lookahead,
            hid_dim=self.hid_dim,
            num_node=self.num_node,
            chunk_size=self.chunk_size)

        self.out_proj = nn.Linear(self.lookback * 2, self.lookahead)

    def forward(self, x):
        x = self.revinlayer(x, mode='norm')

        fout, tout = self.layer(x)
        out = torch.cat([fout, tout], dim=1)
        out = self.out_proj(out.permute(0, 2, 1))
        # out = fout + tout
        out = self.revinlayer(out.permute(0, 2, 1), mode='denorm')

        return out
