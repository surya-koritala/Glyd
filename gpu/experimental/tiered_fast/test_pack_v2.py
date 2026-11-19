import numpy as np, torch
import pack_v2 as P
torch.manual_seed(1)
for O, K, wild in [(64, 64, 0.0), (128, 256, 0.02), (192, 128, 0.2), (64, 32, 0.5)]:
    w = torch.randn(O, K) * 0.02
    m = torch.rand(O, K) < wild
    w[m] = torch.randn(int(m.sum())) * torch.exp2(torch.randint(-30, 20, (int(m.sum()),)).float())
    w = w.to(torch.bfloat16)
    ref = w.contiguous().view(torch.int16).numpy().astype(np.uint16)
    for v in (1, 2):
        data, blocks, bb, tiers = P.pack(w, v)
        W = P.unpack((O, K), data, blocks, bb, tiers, v)
        ok = np.array_equal(W.astype(np.uint16), ref)
        steps = O * K // 1024
        print(O, K, wild, "v%d" % v, "ok" if ok else "DIFFERENT", f"{8 * (data.numel() + blocks.numel() + 4 * bb.numel()) / (O * K):.3f} bits")
        assert ok
