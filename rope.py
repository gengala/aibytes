import torch

def apply_rope(x: torch.Tensor, base: float = 10_000.0):
    """ applies eq. 34 in https://arxiv.org/abs/2104.09864 to every head """

    batch, seq_len, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=x.device).float()
    angles = seq_idx[:, None] * inv_freq[None, :]
    sin = angles.sin().repeat_interleave(2, dim=-1).view(1, seq_len, 1, head_dim)  # can be precomputed and cached
    cos = angles.cos().repeat_interleave(2, dim=-1).view(1, seq_len, 1, head_dim)  # can be precomputed and cached

    def rotate_every_two(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    return x * cos + rotate_every_two(x) * sin


batch, seq_len, num_heads, head_dim = 2, 128, 12, 64
x = torch.randn(batch, seq_len, num_heads, head_dim)
x_rope = apply_rope(x)
print(x.shape, x_rope.shape)  # both should be (2, 128, 12, 64)
