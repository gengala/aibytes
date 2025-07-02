import torch


def gqa(X, W_q, W_k, W_v, W_o):
    """
        X: (b, t, d)
        W_q: (num_heads, d, k)
        W_k: (num_heads_kv, d, k)
        W_v: (num_heads_kv, d, k)
        W_o: (h, k, d)
    """
    num_heads = W_q.size(0)
    num_heads_kv = W_k.size(0)
    assert num_heads_kv == W_v.size(0)
    assert num_heads % num_heads_kv == 0
    seq_len = X.size(1)
    head_dim = W_q.size(-1)
    num_groups = num_heads // num_heads_kv
    Q = torch.einsum("btd,hdk->bthk", X, W_q).view(-1, seq_len, num_heads_kv, num_groups, head_dim)
    K = torch.einsum("btd,hdk->bthk", X, W_k)
    V = torch.einsum("btd,hdk->bthk", X, W_v)
    A = torch.softmax(torch.einsum("bthgk,bshk->bhgts", Q, K) / (head_dim ** 0.5), dim=-1)
    AV = torch.einsum("bhgts,bshk->bthgk", A, V).flatten(2, 3)
    O = torch.einsum("bthk,hkd->btd", AV, W_o)
    return A, O


def gqa_repeat(X, W_q, W_k, W_v, W_o):
    """
        X: (b, t, d)
        W_q: (num_heads, d, k)
        W_k: (num_heads_kv, d, k)
        W_v: (num_heads_kv, d, k)
        W_o: (h, k, d)
    """
    num_heads = W_q.size(0)
    num_heads_kv = W_k.size(0)
    assert num_heads_kv == W_v.size(0)
    assert num_heads % num_heads_kv == 0
    head_dim = W_q.size(-1)
    num_groups = num_heads // num_heads_kv
    Q = torch.einsum("btd,hdk->bthk", X, W_q)
    K = torch.einsum("btd,hdk->bthk", X, W_k).repeat_interleave(num_groups, dim=2)
    V = torch.einsum("btd,hdk->bthk", X, W_v).repeat_interleave(num_groups, dim=2)
    A = torch.softmax(torch.einsum("bthk,bshk->bhts", Q, K) / (head_dim ** 0.5), dim=-1)
    AV = torch.einsum("bhts,bshk->bthk", A, V)
    O = torch.einsum("bthk,hkd->btd", AV, W_o)
    return A, O


def __main__():
    seq_len = 128           # t
    hidden_dim = 256        # d
    num_heads = 32          # h
    num_heads_kv = 16       # h
    head_dim = 8            # k     # head_dim = hidden_dim // num_heads

    W_q = torch.randn(num_heads,    hidden_dim, head_dim) * 0.2
    W_v = torch.randn(num_heads_kv, hidden_dim, head_dim) * 0.2
    W_k = torch.randn(num_heads_kv, hidden_dim, head_dim) * 0.2
    W_o = torch.randn(num_heads,    head_dim, hidden_dim) * 0.2

    batch_size = 96         # b
    X = torch.randn(batch_size, seq_len, hidden_dim) * 0.2

    O = gqa(X, W_q, W_k, W_v, W_o)[1]
    O_repeat = gqa_repeat(X, W_q, W_k, W_v, W_o)[1]
    print(torch.allclose(O, O_repeat, atol=1e-5))
