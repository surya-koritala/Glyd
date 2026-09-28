"""The new prompt kernels' outputs against main's kernels of the same tiling (variants 8: 256 by one row block, 9:
128 by two), bit for bit, with and without bias, on odd shapes and exceptions."""
import os, sys, torch
sys.path.insert(0, os.environ.get("GLYD_GPU_DIR", "."))
import glyd_gpu as g
torch.manual_seed(0)
n = 0
for O, K, wild in [(64, 64, 0), (192, 128, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (2560, 9728, 0)]:
    w = torch.randn(O, K, device="cuda") * 0.02
    m = torch.rand(O, K, device="cuda") < wild
    w[m] = torch.randn(int(m.sum()), device="cuda") * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device="cuda").float())
    w = w.to(torch.bfloat16)
    bias = torch.randn(O, device="cuda").to(torch.bfloat16)
    for q in (g.pack_mma(w), g.pack_mma12(w)):
        for M in (65, 100, 128, 129, 200, 256, 257, 300, 384, 600, 1000, 1024, 1500, 2100):
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            for b in (None, bias):
                for new, old in ((2, 8), (1, 9), (0, 8 if M > 128 and (M % 256 == 0 or M % 256 > 128 or M > (4224 if isinstance(q, g.Mma12) else 1024)) else 9)):
                    y, z = g.mma_gemm_big(q, x, b, new), g.mma_gemm_big(q, x, b, old)
                    assert torch.equal(y, z), (type(q).__name__, O, K, M, new, old)
                    n += 1
print(f"bits: {n} products the same as main's kernels of the same tiling")
