"""The dense products of a layer (a Linear's weights W [O, K]) on this GPU by tokens a step: the library's one-call product (mma_linear: the
kernel this GPU takes for M tokens), each variant of the step loop's settings, against bf16's cuBLAS, under CUDA graphs (kernel GPU time; a ring of
matrices larger than the L2 cache is walked so that the weights come from memory as in a step through the layers).

    python dense_layer.py --shapes "5120,2048;2048,4096" --tokens "1 2 4 8 16 32 64" --variants "new:;pfd4:GLYD_GPU_PFD=4" [--layout mma12|mma] [--check] [--json OUT]
    python dense_layer.py --shapes "5120,2048;2048,4096;128,2048" --tokens "16 64 128 232 512 1024" --routes "default gemm wg mid decode"
        each route of the library (kernels.GEMM, MID, WG: its kernel; decode: the matrix unpacked, then cuBLAS) beside the one this GPU takes for M, where it takes that M

shapes: O,K of each matrix (Qwen3-30B-A3B's q, k, v as one and its o: 5120,2048 and 2048,4096)."""
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

import glyd_gpu_dev as g
from glyd_gpu import _lib
from moe_layer import graph_us, parse_variants, weights

BF = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="5120,2048;2048,4096")
    ap.add_argument("--tokens", default="1 2 4 8 16 32 64")
    ap.add_argument("--layout", default="auto", choices=["auto", "mma", "mma12"])
    ap.add_argument("--variants", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--routes", default="", help="routes timed in place of the settings: default (the library's for M), gemm (M up to 64), mid, wg, decode (unpack, then cuBLAS)")
    ap.add_argument("--ring-mb", type=int, default=256, help="the matrices walked, MB of bf16 at least (a few times the L2 cache)")
    a = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes.split(";") if s.strip()]
    variants = parse_variants(a.variants)
    routes = a.routes.split()
    if routes:
        variants = [(r, {}) for r in routes]
    out = {"gpu": torch.cuda.get_device_name(), "rows": [], "env": {x: v for x, v in os.environ.items() if x.startswith("GLYD_")}}
    bad = set()
    for O, K in shapes:
        n = max(3, -(-(a.ring_mb << 20) // (O * K * 2)))
        ws = [weights((1, O, K), gen=gen)[0] for _ in range(n)]
        layout = a.layout if a.layout != "auto" else g.best_layout(O * K * 2 * 8, 0, 1, 0)[0]
        pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
        packs = [pack(w) for w in ws]
        print(f"{O}x{K}: {n} matrices walked, layout {layout}, {8 * packs[0].nbytes() / (O * K):.2f} bits a weight", flush=True)
        for M in (int(t) for t in a.tokens.split()):
            xs = [torch.randn(M, K, dtype=BF, device="cuda", generator=gen) * 0.1 for _ in range(n)]
            bf = [lambda i=i: F.linear(xs[i], ws[i]) for i in range(n)]
            bf16_us = graph_us(bf, 5)
            print(f"{O}x{K} M={M:4d}        bf16: cuBLAS {bf16_us:7.1f} us ({O * K * 2 / (bf16_us * 1e3):6.0f} GB/s)", flush=True)
            for vname, env in variants:
                saved = {x: os.environ.get(x) for x in env}
                os.environ.update(env)
                _lib._need.cache_clear()
                try:
                    if a.check and not routes:
                        y = g.mma_linear(packs[0], xs[0])
                        ref = F.linear(xs[0], ws[0]).float()
                        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                        assert err < 1.5e-2, (O, K, M, vname, err)
                        assert torch.equal(y.view(torch.int16), g.mma_linear(packs[0], xs[0]).view(torch.int16)), "the same bits twice"
                    if routes:
                        if vname == "decode":
                            buf = torch.empty(O * K, dtype=BF, device="cuda")
                            fns = [lambda i=i: F.linear(xs[i], g.mma_unpack(packs[i], out=buf)) for i in range(n)]
                        else:
                            rt = -1 if vname == "default" else {"gemm": g.GEMM, "mid": g.X2, "wg": g.X3}[vname]
                            if rt == g.GEMM and M > 64:
                                continue
                            fns = [lambda i=i: g.mma_linear(packs[i], xs[i], way=rt) for i in range(n)]
                    else:
                        fns = [lambda i=i: g.mma_linear(packs[i], xs[i]) for i in range(n)]
                    us = graph_us(fns, 5)
                    out["rows"].append({"O": O, "K": K, "T": M, "variant": vname, "glyd_us": us, "bf16_us": bf16_us})
                    print(f"{O}x{K} M={M:4d} {vname:>10s}: {us:7.1f} us ({O * K * 1.5 / (us * 1e3):6.0f} GB/s) = {us / bf16_us:4.2f}x bf16", flush=True)
                except (AssertionError, RuntimeError) as ex:
                    bad.add(vname)
                    print(f"{O}x{K} M={M:4d} {vname:>10s}: FAILED {type(ex).__name__}: {ex}", flush=True)
                finally:
                    for x, v in saved.items():
                        if v is None:
                            os.environ.pop(x, None)
                        else:
                            os.environ[x] = v
                    _lib._need.cache_clear()
            if a.json:
                json.dump(out, open(a.json, "w"), indent=1)
    if bad:
        print(f"FAILED: {', '.join(sorted(bad))}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
