import torch


class RoPE(torch.nn.Module):

    def __init__(self, dim, base=10_000):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        assert x.size(-1) == self.dim
        self._build_sin_cos(x)
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        y1 = x1 * self.cos_cached + x2 * self.sin_cached
        y2 = x2 * self.cos_cached - x1 * self.sin_cached
        return torch.cat([y1, y2], dim=-1)

    def _build_sin_cos(self, x):
        seq_len = x.size(1)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            seq_idx = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            angles = torch.outer(seq_idx, self.inv_freq).to(x.device)
            self.cos_cached = angles.cos()[None, :, None, :]
            self.sin_cached = angles.sin()[None, :, None, :]


def __main__():
    batch_size, seq_len, num_heads, head_dim = 32, 128, 12, 6
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    rope = RoPE(head_dim)
    x_rope = rope(x)
    print(x.shape, x_rope.shape)

    # example 1: different positions i, j → ⟨RoPE(x, i), RoPE(y, j)⟩ depends only on (j - i)
    i, j, shift = 10, 25, 100
    assert i + shift < seq_len and j + shift < seq_len
    v_i = torch.randn(batch_size, num_heads, head_dim)
    x[:, i] = x[:, i + shift] = v_i
    v_j = torch.randn(batch_size, num_heads, head_dim)
    x[:, j] = x[:, j + shift] = v_j
    v_i_rope = rope(x)[:, i]
    v_j_rope = rope(x)[:, j]
    v_i_rope_shift = rope(x)[:, i + shift]
    v_j_rope_shift = rope(x)[:, j + shift]
    dot_rope = (v_i_rope * v_j_rope).sum(dim=-1)
    dot_rope_shift = (v_i_rope_shift * v_j_rope_shift).sum(dim=-1)
    print(torch.allclose(dot_rope, dot_rope_shift, atol=1e-5)) # True

    # example 2: same position t for two different sequences a, b → rotation cancels out
    x1 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    x2 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    x1_rope = rope(x1)
    x2_rope = rope(x2)
    dot = (x1 * x2).sum(dim=-1)
    dot_rope = (x1_rope * x2_rope).sum(dim=-1)
    print(torch.allclose(dot, dot_rope, atol=1e-5)) # True
