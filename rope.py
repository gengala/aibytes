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
        return torch.cat([y1, y2], 3)

    def _build_sin_cos(self, x):
        seq_len = x.size(1)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            seq_idx = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            angles = torch.outer(seq_idx, self.inv_freq).to(x.device)
            self.cos_cached = angles.cos()[None, :, None, :]
            self.sin_cached = angles.sin()[None, :, None, :]

def __main__():
    batch, seq_len, num_heads, head_dim = 2, 128, 12, 6
    x = torch.randn(batch, seq_len, num_heads, head_dim)
    rope = RoPE(head_dim)
    x_rope = rope(x)
    print(x.shape, x_rope.shape)
