import torch
torch.set_default_dtype(torch.float64)


seq_len = 128       # t
hidden_dim = 256    # d
num_heads = 32      # h
head_dim = 8        # k     # head_dim = hidden_dim // num_heads
latent_dim = 8      # l


W_dkv = torch.randn(hidden_dim, latent_dim) * 0.2
W_dq = torch.randn(hidden_dim, latent_dim) * 0.2
W_uv = torch.randn(num_heads, latent_dim, head_dim) * 0.2
W_uk = torch.randn(num_heads, latent_dim, head_dim) * 0.2
W_uq = torch.randn(num_heads, latent_dim, head_dim) * 0.2
W_o = torch.randn(num_heads, head_dim, hidden_dim) * 0.2

batch_size = 96    # b
H = torch.randn(batch_size, seq_len, hidden_dim) * 0.2


def mla_einsum_train():
    L_kv = torch.einsum("btd,dl->btl", H, W_dkv)
    L_q = torch.einsum("btd,dl->btl", H, W_dq)
    K = torch.einsum("btl,hlk->bthk", L_kv, W_uk)
    V = torch.einsum("btl,hlk->bthk", L_kv, W_uv)
    Q = torch.einsum("btl,hlk->bthk", L_q, W_uq)
    A = torch.softmax(torch.einsum("bthk, bshk -> bhts", Q, K) / (head_dim ** 0.5), dim=-1)
    AV = torch.einsum("bhts,bshk->bthk", A, V)
    O = torch.einsum("bthk,hkd->btd", AV, W_o)
    return A, O


def mla_einsum_test():
    W_q = torch.einsum("dl,hlk->dhk", W_dq, W_uq)           # only needed in the next einsum
    W_q_absorbed = torch.einsum("dhk,hlk->dhl", W_q, W_uk)  # can be precomputed and cached
    W_o_absorbed = torch.einsum("hkd,hlk->dhl", W_o, W_uv)  # can be precomputed and cached
    L_kv = torch.einsum("btd,dl->btl", H, W_dkv)
    Q_absorbed = torch.einsum("btd,dhl->bthl", H, W_q_absorbed)
    A = torch.softmax(torch.einsum("bthl, bsl -> bhts", Q_absorbed, L_kv) / (head_dim ** 0.5), dim=-1)
    AL_kv = torch.einsum("bhts,bsl->bthl", A, L_kv)
    O = torch.einsum("bthl,dhl->btd", AL_kv, W_o_absorbed)
    return A, O


A_train, O_train = mla_einsum_train()
A_test, O_test = mla_einsum_test()
print(torch.allclose(A_train, A_test, atol=1e-4))   # True
print(torch.allclose(O_train, O_test, atol=1e-4))   # True
