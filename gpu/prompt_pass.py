"""A prompt's pass through a 2026 model's layers' products (its Linears in a forward pass's order, every layer), timed whole on the GPU
at each prompt length, by way of:
  bf16      cuBLAS on the bf16 matrices (F.linear)
  glyd      the vLLM plugin's op (glyd::vllm_linear): the library's route for the tokens, the matrix decoded first where it says so
  decode    each matrix decoded first, then cuBLAS (the decoded route at every length)
  ahead:N   each matrix decoded ahead on a second stream beside the product before it, then cuBLAS (the plugin's development path,
            GLYD_AHEAD_PROMPT; N: the decode's warps an SM, 0 the library's own launch), at every length
  glinear   glyd.gpu's own Linear (model.py's GLinear: its routes, the decode ahead of GeForce Ada, the route SPLIT of an H100 SXM)
Weights: random bf16 of a trained matrix's spread (N(0, 0.02)), two copies of each kind of matrix taken in turn (no matrix read from
the L2 cache); one input a K. Checks, on a pass of each way before it is timed (no wait between its products): every pack decodes to
its matrix bit for bit; each product's last token's row and last output's column (the whole matrix and every token) the same bits as
bf16's for decode and ahead, within 1.5e-2 of the largest fp32 output for glyd and glinear. A pass: 2 untimed, then the median of --reps.

    python prompt_pass.py MODEL [--tokens "128 512 1024 2048 4096 8192"] [--layouts "mma12 mma"] [--ways "glyd decode ahead glinear"]
                          [--warps "0 2 4"] [--reps 3] [--json OUT]

MODEL: Qwen3.5-9B, Qwen3.5-27B or gemma-4-31B-it (their configs' shapes and layer types)."""
import argparse
import json
import statistics
import sys

import torch
import torch.nn.functional as F

import glyd_gpu_dev  # noqa: F401 (the package from this checkout, the library from GLYD_GPU_LIB)
from glyd_gpu import _lib, kernels as g, model as gm, vllm_plugin as vp

BF = torch.bfloat16


def qwen35(layers, H, qkv, o, gdn_in, gdn_out, inter):  # full attention every 4th layer, Gated DeltaNet the others
    return {"layers": [("full" if i % 4 == 3 else "gdn") for i in range(layers)],
            "full": [("qkv", qkv, H), ("o", H, o)], "gdn": [("gdn_in", gdn_in, H), ("gdn_out", H, gdn_out)],
            "mlp": [("gate_up", 2 * inter, H), ("down", H, inter)]}


MODELS = {
    "Qwen3.5-9B": qwen35(32, 4096, 10240, 4096, 12288, 4096, 12288),
    "Qwen3.5-27B": qwen35(64, 5120, 14336, 6144, 16384, 6144, 17408),
    "gemma-4-31B-it": {"layers": [("full" if i % 6 == 5 else "sliding") for i in range(60)],  # sliding window 5 of 6, global the 6th
                       "sliding": [("qkv", 16384, 5376), ("o", 5376, 8192)], "full": [("qk_global", 18432, 5376), ("o_global", 5376, 16384)],
                       "mlp": [("gate_up", 43008, 5376), ("down", 5376, 21504)]},
}


def calls(spec):
    """The pass: (kind, O, K, copy) a product, in order."""
    seen, out = {}, []
    for t in spec["layers"]:
        for kind, O, K in spec[t] + spec["mlp"]:
            c = seen.get(kind, 0)
            seen[kind] = c + 1
            out.append((kind, O, K, c % 2))
    return out


def sample(y):
    return torch.cat([y[-1], y[:, -1]])  # the last token's outputs and the last output's tokens


def run(pas, prods, xs, keep):
    """One pass: each product (a function of x), its samples kept where asked (no wait between products)."""
    out = []
    for (kind, O, K, c), f in zip(pas, prods):
        y = f(xs[K])
        if keep:
            out.append(sample(y))
    return out


def timed(pas, prods, xs, reps):
    for _ in range(2):
        run(pas, prods, xs, False)
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        run(pas, prods, xs, False)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--tokens", default="128 512 1024 2048 4096 8192")
    ap.add_argument("--layouts", default="mma12 mma")
    ap.add_argument("--ways", default="glyd decode ahead glinear")
    ap.add_argument("--warps", default="0 2 4", help="ahead's decode: warps an SM (0: the library's own launch)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    spec, dev = MODELS[a.model], torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    pas = calls(spec)
    kinds = {(kind, O, K) for kind, O, K, _ in pas}
    gen = torch.Generator(device=dev).manual_seed(0)
    W = {(kind, c): (torch.randn(O, K, device=dev, generator=gen) * 0.02).to(BF) for kind, O, K in kinds for c in (0, 1)}
    nW = sum(O * K for _, O, K, _ in pas)
    vp._define()
    code = _lib.gpu()
    out = {"gpu": props.name, "sms": props.multi_processor_count, "code": code, "model": a.model, "products": len(pas), "weights": nW, "rows": []}
    print(f"{props.name} ({props.multi_processor_count} SMs, route code {code}); {a.model}: {len(pas)} products, {nW / 1e9:.2f} G weights a pass", flush=True)
    bad = []

    def row(layout, way, M, ms, ts, err=None, same=None):
        r = {"layout": layout, "way": way, "M": M, "ms": ms, "runs": ts, "err": err, "same": same}
        out["rows"].append(r)
        b = next((x["ms"] for x in reversed(out["rows"]) if x["way"] == "bf16" and x["M"] == M), None)  # (the bf16 pass timed just before)
        print(f"M={M:5d} {layout:>5s} {way:>9s}: {ms:9.2f} ms" + (f" = {ms / b:5.3f}x bf16" if b and way != "bf16" else "") + (f", err {err:.1e}" if err is not None else "")
              + ("" if same is None else ", bf16's bits" if same else ", NOT bf16's bits"), flush=True)

    tokens = [int(t) for t in a.tokens.split()]
    Ks = sorted({K for _, _, K, _ in pas})
    xs = {M: {K: torch.randn(M, K, device=dev, generator=gen).to(BF) * 0.5 for K in Ks} for M in tokens}
    bprods = [lambda x, w=W[(kind, c)]: F.linear(x, w) for kind, O, K, c in pas]
    ref = {M: run(pas, bprods, xs[M], True) for M in tokens}  # bf16's samples
    f32 = {M: [torch.cat([xs[M][K][-1].float() @ W[(kind, c)].float().t(), xs[M][K].float() @ W[(kind, c)][-1].float()]) for kind, O, K, c in pas] for M in tokens}
    ways = []
    for w in a.ways.split():
        ways += [f"ahead:{n}" for n in a.warps.split()] if w == "ahead" else [w]
    vp._BIGGEST[dev] = max(O * K for _, O, K in kinds)
    for layout in a.layouts.split():
        P = {}
        for k, w in W.items():
            p = P[k] = g.pack_mma12(w) if layout == "mma12" else g.pack_mma(w)
            assert torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)), f"{k} {layout}: the pack is not the matrix"
        bits = 8 * sum(P[(kind, c)].nbytes() for kind, O, K, c in pas) / nW
        print(f"{layout}: {bits:.2f} bits a weight over the pass", flush=True)
        t = {k: ((p.data, p.exc, p.exc_base, list(p.sym)) if layout == "mma12" else (p.data, p.blocks, p.block_base, list(p.tiers))) for k, p in P.items()}
        for M in tokens:
            row(layout, "bf16", M, *timed(pas, bprods, xs[M], a.reps))
            for way in ways:
                lins = prods = None
                try:
                    if way == "glyd":
                        prods = [lambda x, o=O, tt=t[(kind, c)]: torch.ops.glyd.vllm_linear(x, *tt[:3], tt[3], None, o, False) for kind, O, K, c in pas]
                    elif way == "decode":
                        prods = [lambda x, o=O, k=K, tt=t[(kind, c)]: F.linear(x, vp._unpack(*tt, o, k)) for kind, O, K, c in pas]
                    elif way.startswith("ahead:"):
                        vp.AHEAD_WARPS = int(way.split(":")[1])
                        vp._Ahead.of.clear()
                        ah = vp._Ahead.get(dev)

                        def one(x, o, k, tt, i, ah=ah):
                            y = ah.product(x, *tt[:3], tt[3], None, o, k, key=i)  # (a key a place: the copies come again in a pass)
                            return F.linear(x, vp._unpack(*tt, o, k)) if y is None else y  # (the order being recorded: decoded first)

                        prods = [lambda x, o=O, k=K, tt=t[(kind, c)], i=i: one(x, o, k, tt, i) for i, (kind, O, K, c) in enumerate(pas)]
                    elif way == "glinear":
                        lins = [gm.GLinear(P[(kind, c)], None) for kind, O, K, c in pas]
                        gm.set_scratch(torch.nn.ModuleList(lins), False)
                        prods = [lambda x, lin=lin: lin(x) for lin in lins]
                    else:
                        raise SystemExit(f"no way {way!r}")
                    run(pas, prods, xs[M], False)  # (an order recorded: ahead's, GLinear's decode ahead and SPLIT)
                    got = run(pas, prods, xs[M], True)
                    if way in ("glyd", "glinear"):
                        err, same = max(((s.float() - r).abs().max() / r.abs().max()).item() for s, r in zip(got, f32[M])), None
                        ok = err < 1.5e-2
                    else:
                        err, same = None, all(torch.equal(s.view(torch.int16), r.view(torch.int16)) for s, r in zip(got, ref[M]))
                        ok = same
                    if not ok:
                        bad.append((layout, way, M))
                    row(layout, way, M, *timed(pas, prods, xs[M], a.reps), err=err, same=same)
                except Exception as ex:  # (reported; the rest go on, and the run ends with an error)
                    bad.append((layout, way, M))
                    print(f"M={M:5d} {layout:>5s} {way:>9s}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
                finally:
                    if way.startswith("ahead:") and dev in vp._Ahead.of:
                        vp._Ahead.of[dev].drop()
                    if way == "glinear":
                        gm.Early.of.clear()
                    del lins, prods
                    torch.cuda.synchronize()
            if a.json:
                json.dump(out, open(a.json, "w"), indent=1)
        del P, t
        torch.cuda.empty_cache()
    if bad:
        print(f"FAILED: {len(bad)}: {bad}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
