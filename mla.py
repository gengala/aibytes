from rope import RoPE
import torch


def mla_norope_train(X, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o):
    L_q = torch.einsum("btd,dl->btl", X, W_dq)
    L_kv = torch.einsum("btd,dl->btl", X, W_dkv)
    Q = torch.einsum("btl,hlk->bthk", L_q, W_uq)
    K = torch.einsum("btl,hlk->bthk", L_kv, W_uk)
    V = torch.einsum("btl,hlk->bthk", L_kv, W_uv)
    A = torch.softmax(torch.einsum("bthk, bshk -> bhts", Q, K) / (W_uq.size(-1) ** 0.5), dim=-1)
    AV = torch.einsum("bhts,bshk->bthk", A, V)
    O = torch.einsum("bthk,hkd->btd", AV, W_o)
    return A, O


def mla_norope_test(X, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o):
    W_q = torch.einsum("dl,hlk->dhk", W_dq, W_uq)           # only needed in the next einsum
    W_q_absorbed = torch.einsum("dhk,hlk->dhl", W_q, W_uk)  # can be precomputed and cached
    W_o_absorbed = torch.einsum("hkd,hlk->dhl", W_o, W_uv)  # can be precomputed and cached
    L_kv = torch.einsum("btd,dl->btl", X, W_dkv)
    Q_absorbed = torch.einsum("btd,dhl->bthl", X, W_q_absorbed)
    A = torch.softmax(torch.einsum("bthl, bsl -> bhts", Q_absorbed, L_kv) / (W_uq.size(-1) ** 0.5), dim=-1)
    AL_kv = torch.einsum("bhts,bsl->bthl", A, L_kv)
    O = torch.einsum("bthl,dhl->btd", AL_kv, W_o_absorbed)
    return A, O


def mla_norope_train_test_comparison():
    seq_len = 128           # t
    hidden_dim = 256        # d
    num_heads = 32          # h
    head_dim = 12           # k     # head_dim = hidden_dim // num_heads
    latent_dim_q = 8        # l
    latent_dim_kv = 16      # l
    W_dq = torch.randn(hidden_dim, latent_dim_q) * 0.2
    W_uq = torch.randn(num_heads, latent_dim_q, head_dim) * 0.2
    W_dkv = torch.randn(hidden_dim, latent_dim_kv) * 0.2
    W_uk = torch.randn(num_heads, latent_dim_kv, head_dim) * 0.2
    W_uv = torch.randn(num_heads, latent_dim_kv, head_dim) * 0.2
    W_o = torch.randn(num_heads, head_dim, hidden_dim) * 0.2

    batch_size = 96  # b
    X = torch.randn(batch_size, seq_len, hidden_dim) * 0.2

    A_train, O_train = mla_norope_train(X, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o)
    A_test, O_test = mla_norope_test(X, W_dq, W_uq, W_dkv, W_uk, W_uv, W_o)
    print(torch.allclose(A_train, A_test, atol=1e-4))   # True
    print(torch.allclose(O_train, O_test, atol=1e-4))   # True


def mla_rope_train(X, W_dq, W_uq, W_rq, W_dkv, W_uk, W_rk, W_uv, W_o):
    nope_dim = W_uq.size(-1)
    rope_dim = W_rq.size(-1)
    num_heads = W_uq.size(0)
    rope = RoPE(rope_dim)
    L_q = torch.einsum("btd,dl->btl", X, W_dq)
    L_kv = torch.einsum("btd,dl->btl", X, W_dkv)
    Q_u = torch.einsum("btl,hlk->bthk", L_q, W_uq)
    Q_r = rope(torch.einsum("btl,hlr->bthr", L_q, W_rq))
    Q = torch.cat([Q_u, Q_r], dim=-1)
    K_u = torch.einsum("btl,hlk->bthk", L_kv, W_uk)
    K_r = rope(torch.einsum("btd,dr->btr", X, W_rk).unsqueeze(2)).expand(-1, -1, num_heads, -1)
    K = torch.cat([K_u, K_r], dim=-1)
    V = torch.einsum("btl,hlk->bthk", L_kv, W_uv)
    A = torch.softmax(torch.einsum("bthk, bshk -> bhts", Q, K) / ((nope_dim + rope_dim) ** 0.5), dim=-1)
    AV = torch.einsum("bhts,bshk->bthk", A, V)
    O = torch.einsum("bthk,hkd->btd", AV, W_o)
    return A, O


def mla_rope_test(X, W_dq, W_uq, W_rq, W_dkv, W_uk, W_rk, W_uv, W_o):
    W_drq = torch.einsum("dl,hlr->dhr", W_dq, W_rq)         # can be precomputed and cached
    W_q = torch.einsum("dl,hlk->dhk", W_dq, W_uq)           # only needed in the next einsum
    W_q_absorbed = torch.einsum("dhk,hlk->dhl", W_q, W_uk)  # can be precomputed and cached
    W_o_absorbed = torch.einsum("hkd,hlk->dhl", W_o, W_uv)  # can be precomputed and cached
    nope_dim = W_uq.size(-1)
    rope_dim = W_rq.size(-1)
    rope = RoPE(rope_dim)
    L_kv = torch.einsum("btd,dl->btl", X, W_dkv)
    Q_u_absorbed = torch.einsum("btd,dhl->bthl", X, W_q_absorbed)
    Q_r_absorbed = rope(torch.einsum("btd,dhr->bthr", X, W_drq))
    Q_absorbed = torch.cat([Q_u_absorbed, Q_r_absorbed], dim=-1)
    K_r = rope(torch.einsum("btd,dr->btr", X, W_rk).unsqueeze(2))[:, :, 0]
    K_latent = torch.cat([L_kv, K_r], dim=-1)
    A = torch.softmax(torch.einsum("bthk,bsk->bhts", Q_absorbed, K_latent) / ((nope_dim + rope_dim) ** 0.5), dim=-1)
    AL_kv = torch.einsum("bhts,bsl->bthl", A, L_kv)
    O = torch.einsum("bthl,dhl->btd", AL_kv, W_o_absorbed)
    return A, O


def __main__():
    seq_len = 128           # t
    hidden_dim = 256        # d
    num_heads = 32          # h
    nope_dim = 12           # k     # head_dim = hidden_dim // num_heads
    rope_dim = 8            # r
    latent_dim_q = 8        # l
    latent_dim_kv = 16      # l

    W_dq = torch.randn(hidden_dim, latent_dim_q) * 0.2
    W_uq = torch.randn(num_heads, latent_dim_q, nope_dim) * 0.2
    W_rq = torch.randn(num_heads, latent_dim_q, rope_dim) * 0.2
    W_dkv = torch.randn(hidden_dim, latent_dim_kv) * 0.2
    W_uk = torch.randn(num_heads, latent_dim_kv, nope_dim) * 0.2
    W_rk = torch.randn(hidden_dim, rope_dim) * 0.2
    W_uv = torch.randn(num_heads, latent_dim_kv, nope_dim) * 0.2
    W_o = torch.randn(num_heads, nope_dim, hidden_dim) * 0.2

    batch_size = 96  # b
    X = torch.randn(batch_size, seq_len, hidden_dim) * 0.2

    A_train, O_train = mla_rope_train(X, W_dq, W_uq, W_rq, W_dkv, W_uk, W_rk, W_uv, W_o)
    A_test, O_test = mla_rope_test(X, W_dq, W_uq, W_rq, W_dkv, W_uk, W_rk, W_uv, W_o)
    print(torch.allclose(A_train, A_test, atol=1e-4))   # True
    print(torch.allclose(O_train, O_test, atol=1e-4))   # True
