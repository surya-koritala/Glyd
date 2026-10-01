"""The 12-bit layout's MID route (mma12_mid_kernel: 17-64 tokens on Ampere and Ada) called again and again on the same
inputs, against its first output and the decoded matrix's product: whether racecheck's hazards there ever show in
the bits. The tiered layout's step kernel (GEMM) beside it. argv: REPS (default 2000)."""
import sys

import torch
from glyd.gpu import _lib, kernels as g

g.lib()
torch.manual_seed(0)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
nob = torch.empty(0, dtype=torch.bfloat16, device="cuda")
for O, K in ((2048, 2048), (6144, 2048), (2048, 6144), (12288, 2048)):
    w = (torch.randn(O, K, device="cuda") * 0.02).to(torch.bfloat16)
    for layout in ("mma12", "mma"):
        p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
        t = (p.data, p.exc, p.exc_base) if layout == "mma12" else (p.data, p.blocks, p.block_base)
        words = list(p.sym if layout == "mma12" else p.tiers)
        f = _lib.mma12_linear if layout == "mma12" else _lib.mma_linear
        wd = g.mma_unpack(p).float()
        for M in (17, 33, 64):
            route = (_lib.mma12_route if layout == "mma12" else _lib.mma_route)(_lib.gpu(), O, K, M)[0]
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            y0 = torch.empty(M, O, dtype=torch.bfloat16, device="cuda")
            _lib.local.fresh = True
            f(*t, words, O, K, x, nob, y0, -1)
            ys = torch.empty(reps, M, O, dtype=torch.bfloat16, device="cuda")
            for i in range(reps):
                f(*t, words, O, K, x, nob, ys[i], -1)
            torch.cuda.synchronize()
            diff = (ys.view(torch.int16) != y0.view(torch.int16)).flatten(1).any(1).sum().item()
            err = ((y0.float() - x.float() @ wd.T).abs().max() / (x.float() @ wd.T).abs().max()).item()
            print(f"{layout} O {O} K {K} M {M} route {route}: {reps} calls, {diff} with other bits than the first; first against the decoded product {err:.2e}", flush=True)
print("midstress: done")
