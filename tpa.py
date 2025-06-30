import torch
import torch.nn as nn
from rope import RoPE


class CPLinear(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, rank_kv: int = 1, rank_q: int = 12):
        super(CPLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank_kv = rank_kv
        self.rank_q = rank_q

        self.W_A_q = nn.Linear(hidden_dim, num_heads * rank_q, bias=False)
        self.W_B_q = nn.Linear(hidden_dim, rank_q * head_dim,  bias=False)

        self.W_A_k = nn.Linear(hidden_dim, num_heads * rank_kv, bias=False)
        self.W_B_k = nn.Linear(hidden_dim, rank_kv * head_dim,  bias=False)

        self.W_A_v = nn.Linear(hidden_dim, num_heads * rank_kv, bias=False)
        self.W_B_v = nn.Linear(hidden_dim, rank_kv * head_dim,  bias=False)

        self.rotary = RoPE(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for name, rank in [('q', self.rank_q), ('k', self.rank_kv), ('v', self.rank_kv)]:
            weight = getattr(self, f'W_A_{name}').weight
            tensor = weight.view(self.hidden_dim, self.num_heads, rank)
            nn.init.xavier_uniform_(tensor)
            weight.data.copy_(tensor.view_as(weight))
        for name, rank in [('q', self.rank_q), ('k', self.rank_kv), ('v', self.rank_kv)]:
            weight = getattr(self, f'W_B_{name}').weight
            tensor = weight.view(self.hidden_dim, rank, self.head_dim)
            nn.init.xavier_uniform_(tensor)
            weight.data.copy_(tensor.view_as(weight))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compute intermediate variables A for Q, K, and V
        A_q = self.W_A_q(x).view(batch_size, seq_len, self.num_heads, self.rank_q)
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.num_heads, self.rank_kv)
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.num_heads, self.rank_kv)

        # Compute intermediate variables B for Q, K, and V
        B_q = self.W_B_q(x).view(batch_size, seq_len, self.rank_q, self.head_dim)
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank_kv, self.head_dim)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank_kv, self.head_dim)

        B_q_rope = self.rotary(B_q)
        B_k_rope = self.rotary(B_k)

        q = (A_q @ B_q_rope).div_(self.rank_q)
        k = (A_k @ B_k_rope).div_(self.rank_kv)
        v = (A_v @ B_v).div_(self.rank_kv)

        return q, k, v


def __main__():
    batch, seq_len, hidden_dim, num_heads, head_dim = 2, 128, 256, 8, 6
    x = torch.randn(batch, seq_len, hidden_dim)
    self = CPLinear(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, rank_kv=1, rank_q=12)
    q, k, v = self(x)
    print(q.shape, k.shape, v.shape)
