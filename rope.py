import torch


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """ eq. 31 in https://arxiv.org/abs/2104.09864 """

    batch_size, seq_len, hidden_dim = x.shape
    assert hidden_dim % 2 == 0, "hidden dimension must be even for RoPE"

    theta = 1.0 / (base ** (torch.arange(0, hidden_dim, 2, device=x.device).float() / hidden_dim))  # (d//2, )
    seq_idx = torch.arange(seq_len, device=x.device).float()    # (t, ))
    angle = torch.einsum("t,d->td", seq_idx, theta)     # (t, d//2)
    sin = angle.sin().repeat_interleave(2, dim=-1).unsqueeze(0)  # (1, t, d), can be precomputed and cached
    cos = angle.cos().repeat_interleave(2, dim=-1).unsqueeze(0)  # (1, t, d), can be precomputed and cached

    def rotate_every_two(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    x_rope = x * cos + rotate_every_two(x) * sin
    return x_rope


batch_size, seq_len, hidden_dim = 2, 128, 768
x = torch.randn(batch_size, seq_len, hidden_dim)
x_rope = apply_rope(x)
print(x.shape, x_rope.shape)
