"""The prompt kernel's products through the library in GLYD_GPU_LIB (the tree's Python in GLYD_GPU_DIR): each one's
sha256 saved to OUT.json, or compared with OLD.json's. Main's library, then the branch's:

    GLYD_GPU_DIR=main/gpu GLYD_GPU_LIB=main.so python bits.py main.json
    GLYD_GPU_DIR=branch/gpu GLYD_GPU_LIB=branch.so python bits.py branch.json main.json

Both layouts, the choice of blocks (variant 0) and each tiling asked for (1: 128 tokens by two row blocks, 2: 256 by
one), 65-2100 tokens, with and without bias, odd row blocks, exceptions few and many."""
import hashlib, json, os, sys, torch
sys.path.insert(0, os.environ.get("GLYD_GPU_DIR", "."))
import glyd_gpu as g
torch.manual_seed(0)
out = {}
for O, K, wild in [(64, 64, 0), (192, 128, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (2560, 9728, 0)]:
    w = torch.randn(O, K, device="cuda") * 0.02
    m = torch.rand(O, K, device="cuda") < wild
    w[m] = torch.randn(int(m.sum()), device="cuda") * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device="cuda").float())
    w = w.to(torch.bfloat16)
    bias = torch.randn(O, device="cuda").to(torch.bfloat16)
    for q in (g.pack_mma(w), g.pack_mma12(w)):
        for M in (65, 100, 128, 129, 200, 256, 257, 300, 384, 400, 600, 1000, 1024, 1500, 1920, 2047, 2100):
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            for b in (None, bias):
                for v in (0, 1, 2):
                    y = g.mma_gemm_big(q, x, b, v)
                    out[f"{type(q).__name__} {O}x{K} {M} bias {b is not None} variant {v}"] = hashlib.sha256(y.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
if len(sys.argv) > 2:
    old = json.load(open(sys.argv[2]))
    bad = [k for k in out if out[k] != old[k]]
    print(f"{len(out)} products compared bit for bit with {sys.argv[2]}: {len(out) - len(bad)} identical" + (f"; differ: {bad}" if bad else ""))
else:
    json.dump(out, open(sys.argv[1], "w"), indent=0)
    print(f"{len(out)} products saved to {sys.argv[1]}")
