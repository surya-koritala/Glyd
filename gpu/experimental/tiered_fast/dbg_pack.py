import numpy as np, torch
import pack_v2 as P
O, K = 64, 64
RB, KS = O // 64, K // 16
w = (torch.randn(O, K) * 0.02).to(torch.bfloat16)
u = w.contiguous().view(torch.int16)
v = u.view(RB, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten()
out = v.numpy().astype(np.uint16).reshape(RB * KS, 1024)
o = out.reshape(RB, KS, 8, 4, 8, 2, 2)
W = o.transpose(0, 4, 2, 1, 5, 3, 6).reshape(O, K)
print("layout inverse ok:", np.array_equal(W, u.numpy().astype(np.uint16)))
data, blocks, bb, tiers = P.pack(w, 1)
steps = O * K // 1024
d = data.view(steps, 1280).numpy()
print("tiers", [hex(t) for t in tiers], "bb", bb[:5].tolist(), "blocks", blocks.numel())
# digits check
pd = d[0, :256].view("<u4").reshape(2, 32)
dig = np.zeros((32, 32), dtype=np.int64)
for wd in range(2):
    for j in range(16):
        dig[:, wd * 16 + j] = (pd[wd] >> (2 * j)) & 3
e = ((out[0].astype(np.int64) >> 7) & 0xFF).reshape(32, 32)
sym = [(tiers[k] >> (8 * j)) & 0xFF for k in range(3) for j in range(4)]
print("sym", sym)
s1 = sym[0:3]
ok = True
for l in range(32):
    for i in range(32):
        if dig[l, i] < 3 and s1[dig[l, i]] != e[l, i]:
            ok = False
print("tier-1 digits give the exponents:", ok)
