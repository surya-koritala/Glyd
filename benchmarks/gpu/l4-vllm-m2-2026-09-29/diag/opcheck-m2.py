"""The library's routed linear (glyd_gpu_mma[12]_linear, route -1: the L4's GEMM, MID and BIG) on random packs, for
compute-sanitizer (initcheck: reads of memory never written; memcheck: reads and writes out of bounds): Qwen3-8B's
shapes, both layouts, 1-1024 tokens, workspaces per call as the plugin takes them."""
import torch
from glyd.gpu import _lib, kernels as g

g.lib()  # the library loaded, as the plugin loads it before its first product
torch.manual_seed(0)
nob = torch.empty(0, dtype=torch.bfloat16, device="cuda")
for O, K in ((6144, 4096), (4096, 12288)):
    w = (torch.randn(O, K, device="cuda") * 0.02).to(torch.bfloat16)
    for layout in ("mma", "mma12"):
        p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
        t = (p.data, p.exc, p.exc_base) if layout == "mma12" else (p.data, p.blocks, p.block_base)
        words = list(p.sym if layout == "mma12" else p.tiers)
        f = _lib.mma12_linear if layout == "mma12" else _lib.mma_linear
        for M in (1, 8, 17, 64, 65, 513, 1024):
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            y = torch.empty(M, O, dtype=torch.bfloat16, device="cuda")
            _lib.local.fresh = True
            f(*t, words, O, K, x, nob, y, -1)
            torch.cuda.synchronize()
            ref = x.float() @ g.mma_unpack(p).float().T
            print(O, K, layout, M, f"{((y.float() - ref).abs().max() / ref.abs().max()).item():.2e}", flush=True)
print("opcheck: done")
