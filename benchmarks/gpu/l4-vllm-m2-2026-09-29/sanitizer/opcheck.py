"""The library's routed linear (glyd_gpu_mma[12]_linear, route -1: this GPU's routes for M) and its whole-matrix decode
on random packs, for compute-sanitizer: initcheck (reads of memory never written), memcheck (accesses out of bounds),
racecheck (shared-memory hazards between threads) and synccheck (barriers misused). Both layouts, each shape
(OPCHECK_SHAPES, "O,K;O,K": default Qwen3-8B's qkv and down), each M (OPCHECK_MS), workspaces per call as the vLLM
plugin takes them; each product checked against the decoded matrix's."""
import os

import torch
from glyd.gpu import _lib, kernels as g

g.lib()  # the library loaded, as the plugin loads it before its first product
torch.manual_seed(0)
nob = torch.empty(0, dtype=torch.bfloat16, device="cuda")
shapes = [tuple(int(v) for v in s.split(",")) for s in os.environ.get("OPCHECK_SHAPES", "6144,4096;4096,12288").split(";")]
ms = [int(v) for v in os.environ.get("OPCHECK_MS", "1,8,17,64,65,513,1024").split(",")]
for O, K in shapes:
    w = (torch.randn(O, K, device="cuda") * 0.02).to(torch.bfloat16)
    for layout in ("mma", "mma12"):
        p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
        t = (p.data, p.exc, p.exc_base) if layout == "mma12" else (p.data, p.blocks, p.block_base)
        words = list(p.sym if layout == "mma12" else p.tiers)
        f = _lib.mma12_linear if layout == "mma12" else _lib.mma_linear
        ref_w = g.mma_unpack(p).float()
        assert torch.equal(ref_w.to(torch.bfloat16).view(torch.int16), w.view(torch.int16)), "decoded to other bits"
        for M in ms:
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            y = torch.empty(M, O, dtype=torch.bfloat16, device="cuda")
            _lib.local.fresh = True
            f(*t, words, O, K, x, nob, y, -1)
            torch.cuda.synchronize()
            ref = x.float() @ ref_w.T
            print(O, K, layout, M, f"{((y.float() - ref).abs().max() / ref.abs().max()).item():.2e}", flush=True)
print("opcheck: done")
