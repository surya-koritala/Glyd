"""Every route and plan the library offers for a product, timed on this GPU on 2026 models' own matrices, both layouts, by tokens a step:
the route the library takes today (as the vLLM plugin calls it: the matrix decoded for cuBLAS where its route says so; an LM head's
never decoded) and each alternative where it takes M: gemm (to 64 tokens; the tiered layout's 8-warp kernel to 16, the 12-bit
layout's prefetch and batched exception loads), mid and wg (the 12-bit layout's staged products), big (the prompt kernel's plans 1,
2 and 3), decode (the matrix unpacked, then cuBLAS). Each config checked once before it is timed: every pack unpacked bit for bit,
the product against fp32 (within 1.5e-2 of the largest output) and the same bits twice, the decode route bit for bit against the
bf16 product of the matrix itself. Timed as layouts.py times (CUDA graphs over a ring of copies four times the L2 cache, the median
of 5 replays), twice in opposite orders, the faster kept.

    python sweep.py FILE.safetensors [FILE ...] --json OUT [--tokens "1 2 4 8 16 24 32 48 64 128 512 2048"] [--head NAME]

FILE: [O, K] bf16 matrices by name (gpu/fetch_tensors.py); --head: the names that are an LM head (the plugin never decodes one)."""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu"))
import glyd_gpu_dev as g  # noqa: E402
from glyd_gpu import _lib  # noqa: E402
from moe_layer import graph_us  # noqa: E402

BF = torch.bfloat16
ROUTE = {"gemm": g.GEMM, "mid": g.X2, "wg": g.X3}


def configs(twelve, M, K, hopper, head):
    """(route, variant, env) of every alternative that takes M tokens of a matrix of K columns."""
    k64 = K % 64 == 0
    out = []
    if M <= 64:
        out.append(("gemm", 0, {}))
        if not twelve and M <= 16:
            out.append(("gemm", 0, {"GLYD_GPU_STEP": "0"}))
        if twelve:
            out += [("gemm", 0, {"GLYD_GPU_PFD": "2"}), ("gemm", 0, {"GLYD_GPU_PFD": "4"}), ("gemm", 0, {"GLYD_GPU_PATCH_B": "1"})]
    if twelve and k64:
        out.append(("mid", 0, {}))
        if hopper:
            out.append(("wg", 0, {}))
    if k64:
        out += [("big", v, {}) for v in ((1, 2, 3) if twelve else (1, 2))]
    if not head:
        out.append(("decode", 0, {}))
    return out


def caller(route, variant, p, xs, buf):
    if route == "decode":
        return lambda i: F.linear(xs[i], g.mma_unpack(p[i], out=buf))
    if route == "big":
        return lambda i: g.mma_gemm_x1(p[i], xs[i], variant=variant)
    if route == "default":
        return lambda i: g.mma_linear(p[i], xs[i], way=variant)
    return lambda i: g.mma_linear(p[i], xs[i], way=ROUTE[route])


def with_env(env, fn):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _lib._need.cache_clear()  # (a workspace's size can depend on a variable)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _lib._need.cache_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", required=True)
    ap.add_argument("--tokens", default="1 2 4 8 16 24 32 48 64 128 512 2048")
    ap.add_argument("--head", default="lm_head")
    ap.add_argument("--layouts", default="mma mma12")
    a = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    hopper = props.major == 9
    code, l2 = _lib.gpu(), props.L2_cache_size
    tokens = [int(t) for t in a.tokens.split()]
    heads = set(a.head.split())
    out = {"gpu": props.name, "cc": f"{props.major}.{props.minor}", "sms": props.multi_processor_count, "l2": l2, "code": code, "tokens": tokens, "rows": [], "failed": []}
    print(f"{props.name} (compute {props.major}.{props.minor}, {props.multi_processor_count} SMs, L2 {l2 >> 20} MiB, code {code})", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    t0 = time.time()
    for path in a.files:
        model = os.path.basename(path).rsplit(".", 1)[0]
        for name, w0 in load_file(path, device="cuda").items():
            O, K = w0.shape
            head = name in heads
            n = max(2, math.ceil(4 * l2 / (O * K * 2 * 0.6)))  # the ring: its packs four times the L2 cache
            ws = [w0] + [w0.clone() for _ in range(n - 1)]
            for layout in a.layouts.split():
                twelve = layout == "mma12"
                ps = [(g.pack_mma12 if twelve else g.pack_mma)(w) for w in ws]
                assert torch.equal(g.mma_unpack(ps[0]).view(torch.int16), w0.view(torch.int16)), f"{model} {name} {layout}: the pack is not the matrix"
                buf = None if head else torch.empty(O * K, dtype=BF, device="cuda")
                for M in tokens:
                    if head and M > 128:  # (an LM head multiplies a step's sequences, a prompt's last token each)
                        continue
                    xs = [torch.randn(M, K, dtype=BF, device="cuda", generator=gen) * 0.1 for _ in range(n)]
                    ref = F.linear(xs[0].float(), w0.float())
                    exact = None if head else F.linear(xs[0], w0)  # (the decode route's bits)
                    way = g.way(ps[0], code, M)[0]
                    if head:
                        dflt = ("default", g.X1 if way in (g.DECODE, g.EARLY) else -1, {})
                    else:
                        dflt = ("decode", 0, {}) if way in (g.DECODE, g.EARLY) or (way == g.X1 and K % 64) else ("default", -1, {})
                    key = lambda c: c[:2] + (tuple(sorted(c[2].items())),)  # noqa: E731
                    cands = [dflt] + [c for c in configs(twelve, M, K, hopper, head) if key(c) != key(dflt)]
                    fns = {}
                    for c in cands:  # each config checked once
                        route, variant, env = c
                        f = caller(route, variant, ps, xs, buf)

                        def check(f=f, route=route):
                            y = f(0)
                            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                            assert err < 1.5e-2, f"error {err}"
                            assert torch.equal(y.view(torch.int16), f(0).view(torch.int16)), "not the same bits twice"
                            if route == "decode":
                                assert torch.equal(y.view(torch.int16), exact.view(torch.int16)), "decode not bit for bit"
                            return err

                        try:
                            fns[key(c)] = (c, f, with_env(env, check))
                        except (AssertionError, RuntimeError) as ex:
                            out["failed"].append({"model": model, "name": name, "O": O, "K": K, "M": M, "layout": layout, "route": route, "variant": variant, "env": env, "why": f"{type(ex).__name__}: {ex}"})
                            print(f"FAILED {model} {name} {O}x{K} M={M} {layout} {route}/{variant} {env}: {ex}", flush=True)
                    best = {}
                    for order in (list(fns), list(fns)[::-1]):  # two passes, opposite orders: the faster of each config's two
                        for k in order:
                            c, f, _ = fns[k]
                            us = with_env(c[2], lambda f=f: graph_us([lambda i=i: f(i) for i in range(n)], 5))
                            best[k] = min(best.get(k, us), us)
                    for k, (c, f, err) in fns.items():
                        out["rows"].append({"model": model, "name": name, "O": O, "K": K, "M": M, "layout": layout, "head": head, "way": way, "route": c[0], "variant": c[1],
                                            "env": c[2], "default": k == key(dflt), "us": best[k], "err": err})
                    if key(dflt) in best:
                        d, top = best[key(dflt)], min(best, key=best.get)
                        print(f"{model} {name} {O}x{K} M={M:5d} {layout:>5s} way {way}: today {d:9.1f} us; best {top[0]}/{top[1]} {dict(top[2]) or ''} {best[top]:9.1f} us ({d / best[top]:.3f}x)", flush=True)
                    del xs
                del ps, buf
                torch.cuda.empty_cache()
            del ws
            torch.cuda.empty_cache()
            json.dump(out, open(a.json, "w"))
    print(f"{len(out['rows'])} timings, {len(out['failed'])} failed, {time.time() - t0:.0f} s", flush=True)
    sys.exit(1 if out["failed"] else 0)


if __name__ == "__main__":
    main()
