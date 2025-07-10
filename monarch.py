import torch
import torch.nn as nn
from typing import Tuple


class SquaredMonarchMatrix(nn.Module):

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(torch.randn(2, size, size, size))

    def forward(self, x: torch.Tensor):
        # x (..., size^2), output has same shape as input
        x = x.view(*x.shape[:-1], self.size, self.size).transpose(-1, -2)
        x = torch.einsum("nij,...nj->...in", self.param[0], x)  # n is number of blocks, transpose is included in einsum
        x = torch.einsum("nij,...nj->...in", self.param[1], x)  # n is number of blocks, transpose is included in einsum
        return x.flatten(-2)


class GeneralizedMonarchMatrix(nn.Module):

    def __init__(self, in_dim: Tuple[int, int], out_dim: Tuple[int, int]):
        super().__init__()
        """ monarch models a linear mapping from R^(in_dim[0]*in_dim[1]) -> R^(out_dim[0]*out_dim[1]) """
        """ e.g. to apply it to a Tucker layer with K units just set in_dims=(K, K) and out_dims=(sqrt(K),sqrt(K)) """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_features = in_dim[0] * in_dim[1]
        self.out_features = out_dim[0] * out_dim[1]
        self.R = nn.Parameter(torch.randn(in_dim[1], out_dim[0], in_dim[0]))
        self.L = nn.Parameter(torch.randn(out_dim[0], out_dim[1], in_dim[1]))

    def forward(self, x: torch.Tensor):
        x = x.view(*x.shape[:-1], self.in_dim[1], self.in_dim[0])
        x = torch.einsum("nij,...nj->...in", self.R, x)
        x = torch.einsum("nij,...nj->...in", self.L, x).flatten(-2)
        return x

    def get_monarch_matrix(self):
        idx = torch.arange(self.out_dim[0] * self.in_dim[1])
        PR = torch.eye(len(idx))[idx.view(self.in_dim[1], self.out_dim[0]).t().flatten()]
        idx = torch.arange(self.out_features)
        PL = torch.eye(len(idx))[idx.view(self.out_dim[0], self.out_dim[1]).t().flatten()]
        return PL @ torch.block_diag(*self.L) @ PR @ torch.block_diag(*self.R)


def __main__():
    in_dim = (8, 32)
    out_dim = (64, 128)
    mm = GeneralizedMonarchMatrix(in_dim=in_dim, out_dim=out_dim)

    batch_size = 1024
    x = torch.randn(batch_size, in_dim[0] * in_dim[1])

    out = mm(x)
    mm_materialized = mm.get_monarch_matrix()
    out1 = x @ mm_materialized.T
    print(torch.allclose(out, out1, atol=1e-4))
