"""The tiered layout (mma) against the 12-bit one (mma12) and bf16's cuBLAS on a model's own matrices, by tokens a step: each layout's
product as the vLLM plugin takes it on this GPU (default: the library's route for M, the matrix decoded for cuBLAS where that route
says so), and its other routes where they take M and are not the default (gemm to 64 tokens; past 64, big, the prompt kernel, but on
Hopper, and decode, the matrix unpacked and then cuBLAS), under CUDA graphs over a ring of copies of each matrix four times the L2 cache (the weights from memory, as a step through
the layers reads them); with --hot also over one copy where its pack fits half the L2 cache (to 64 tokens: a kernel as fast from the
L2 cache as from memory is bound by its instructions, not by memory). Each pack unpacked and compared bit for bit as it is made;
each product against fp32 (within 1.5e-2 of the largest output) and the same bits twice. A row's clk a step: its GPU time by the
SMs and the SM clock (--mhz: the GPU's under load, from nvidia-smi's log of the run), by the matrix's steps of 1,024 weights.

    python layouts.py FILE.safetensors [--tokens "1 8 16 32 64 128 512 1024 2048"] [--routes "gemm big decode"] [--hot] [--json OUT]
        [--variants "mid1:GLYD_GPU_TMID=1,GLYD_GPU_TMID_V=1;..."]   the tiered layout's product again under each variant's variables
        (a development library reads them at each call), its default route at every M, checked as the rest

FILE: the matrices, each an [O, K] bf16 tensor by its name (fetch_tensors.py)."""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import glyd_gpu_dev as g
from glyd_gpu import _lib
from moe_layer import graph_us, parse_variants

BF = torch.bfloat16


def time_route(a, name, w0, ps, xs, buf, layout, vname, route, M, code, props, l2, n):
    """One product's row (None where the route does not take M): checked against fp32 and twice, then timed over the ring."""
    O, K = w0.shape
    way = g.way(ps[0], code, M)[0]
    dec = way in (g.DECODE, g.EARLY) or (way == g.X1 and K % 64)  # (the plugin's _decoded)
    if route == "gemm" and (M > 64 or way == g.GEMM) or route in ("big", "decode") and M <= 64 or route == "big" and (K % 64 or props.major == 9):
        return None  # (gemm where it is not the default; the prompt routes past 64 tokens; no prompt kernel on Hopper, where the routes never take it)
    if route == "decode" or route == "default" and dec:
        f = lambda i: F.linear(xs[i], g.mma_unpack(ps[i], out=buf))
    else:
        rt = {"default": -1, "gemm": g.GEMM, "big": g.X1}[route]
        f = lambda i: g.mma_linear(ps[i], xs[i], way=rt)
    ref = F.linear(xs[0].float(), w0.float())
    y = f(0)
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 1.5e-2 and torch.equal(y.view(torch.int16), f(0).view(torch.int16)), (name, layout, vname, route, M, err)
    row = {"name": name, "O": O, "K": K, "M": M, "layout": layout, "variant": vname, "route": route, "way": way, "err": err, "us": graph_us([lambda i=i: f(i) for i in range(n)], 5)}
    if a.hot and M <= 64 and route == "default" and ps[0].nbytes() * 2 <= l2:
        row["hot_us"] = graph_us([lambda: f(0)] * n, 5)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--tokens", default="1 2 4 8 16 24 32 48 64 128 256 512 1024 2048")
    ap.add_argument("--routes", default="gemm big decode", help="routes timed beside the default where they take M")
    ap.add_argument("--layouts", default="mma mma12")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--mhz", type=float, default=0, help="the SM clock for the rows' clk a step (0: the GPU's top clock)")
    ap.add_argument("--json", default="")
    ap.add_argument("--variants", default="")
    a = ap.parse_args()
    variants = [("base", {})] + (parse_variants(a.variants) if a.variants else [])
    props = torch.cuda.get_device_properties(0)
    sms, l2 = props.multi_processor_count, props.L2_cache_size
    mhz = a.mhz or getattr(props, "clock_rate", 0) / 1e3
    code = _lib.gpu()
    gen = torch.Generator(device="cuda").manual_seed(0)
    out = {"gpu": props.name, "cc": f"{props.major}.{props.minor}", "sms": sms, "l2": l2, "code": code, "rows": []}
    tokens = [int(t) for t in a.tokens.split()]
    bad = []
    print(f"{props.name} (compute {props.major}.{props.minor}, {sms} SMs, L2 {l2 >> 20} MiB, route code {code}, clk a step at {mhz:.0f} MHz); {a.file}", flush=True)
    for name, w0 in load_file(a.file, device="cuda").items():
        O, K = w0.shape
        steps = O * K // 1024
        n = max(2, math.ceil(4 * l2 / (O * K * 2 * 0.6)))  # copies: the ring's packs four times the L2 cache
        ws = [w0] + [w0.clone() for _ in range(n - 1)]
        xs = {M: [torch.randn(M, K, dtype=BF, device="cuda", generator=gen) * 0.1 for _ in range(n)] for M in tokens}
        bf16 = {}
        for M in tokens:
            bf16[M] = graph_us([lambda i=i: F.linear(xs[M][i], ws[i]) for i in range(n)], 5)
        print(f"{name} {O}x{K}: {n} copies; bf16 " + " ".join(f"{M}:{bf16[M]:.1f}" for M in tokens) + " us", flush=True)
        for layout in a.layouts.split():
            pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
            ps = [pack(w) for w in ws]
            assert torch.equal(g.mma_unpack(ps[0]).view(torch.int16), w0.view(torch.int16)), f"{name} {layout}: the pack is not the matrix"
            bits = 8 * ps[0].nbytes() / (O * K)
            flagged = ps[0].flagged() / steps if layout == "mma" else 0.0  # (pack_smallest takes the 12-bit layout past 25%)
            buf = torch.empty(O * K, dtype=BF, device="cuda")
            for vname, env in variants if layout == "mma" else variants[:1]:
                saved = {x: os.environ.get(x) for x in env}
                os.environ.update(env)
                _lib._need.cache_clear()  # (a workspace's size can depend on a variable)
                try:
                    for M in tokens:
                        for route in ["default"] + (a.routes.split() if vname == "base" else []):
                            try:
                                row = time_route(a, name, w0, ps, xs[M], buf, layout, vname, route, M, code, props, l2, n)
                            except (AssertionError, RuntimeError) as ex:  # (reported; the rest go on, and the run ends with an error)
                                bad.append((name, layout, vname, route, M))
                                print(f"{name} {O}x{K} M={M:5d} {layout:>6s} {vname:>6s} {route:>8s}: FAILED {type(ex).__name__}: {ex}", flush=True)
                                continue
                            if row:
                                row.update(bf16_us=bf16[M], bits=bits, flagged=flagged, mhz=mhz, clk_step=row["us"] * 1e-6 * sms * mhz * 1e6 / steps)
                                out["rows"].append(row)
                                print(f"{name} {O}x{K} M={M:5d} {layout:>6s} {vname:>6s} {route:>8s} (way {row['way']}): {row['us']:8.1f} us = {row['us'] / bf16[M]:5.2f}x bf16, {row['clk_step']:6.0f} clk/step"
                                      + (f", hot {row['hot_us']:8.1f}" if "hot_us" in row else ""), flush=True)
                finally:
                    for x, v in saved.items():
                        if v is None:
                            os.environ.pop(x, None)
                        else:
                            os.environ[x] = v
                    _lib._need.cache_clear()
            del ps, buf
            torch.cuda.empty_cache()
        if a.json:
            json.dump(out, open(a.json, "w"), indent=1)
        del ws, xs
        torch.cuda.empty_cache()
    if bad:
        print(f"FAILED: {len(bad)} products: {bad[:20]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
