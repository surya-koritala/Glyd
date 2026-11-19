#!/usr/bin/env python3
"""The mixture-of-experts layer against PyTorch. DUMP_DIR is what `glyd-engine moecheck --dump DIR` wrote (or `--cpu-dump DIR`, the CPU reference in the GPU's place, which is how this script is tried where there
is no GPU): a directory a case, with the case's inputs, its experts' weights as the Int5 or Int6 packs decode them (the bf16 values the GPU multiplies by) and the engine's outputs (the experts, their weights, the layer's
output). Each case is run through the layer as transformers' eager code runs it, in torch's own bf16: the router in fp32 (GLM-5.x, Nemotron 3) or as a bf16 Linear (Qwen3.5/3.6, Gemma 4), softmax or sigmoid
scores (the bias to pick, DeepSeek's noaux_tc), the k largest, the weights renormalized, scaled; the experts one at a time on the tokens that chose them (gate and up product in bf16, the activation in bf16 with
its clamp, the down product in bf16, times the pair's weight, added into a bf16 sum); the shared expert; and compared with the engine's output: the largest difference relative to the largest value, bounded at
2e-2 (the library's own checks bound a bf16 layer at 1.5e-2; transformers rounds after every operation, the engine after the activation and once at the end: on the CPU reference's dump the layers differ by 4e-3 to 1e-2).

    python3 tools/moe_ref.py DUMP_DIR [--bound 2e-2] [CASE-LAYOUT ...]

The router is compared on its own too: torch's experts against the engine's (a token whose experts differ is fine where the scores of the two sets are equal to 1e-6: a tie that falls the other way), and the
weights torch makes of the engine's experts against the engine's. The experts' stage runs on the engine's experts and weights, so that a tie does not move it. Exit status 1 where a bound is not kept."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BF = torch.bfloat16


def read(d, name, dtype):
    p = os.path.join(d, name)
    return np.fromfile(p, dtype=dtype) if os.path.exists(p) else None


def bf16(a, *shape):
    """bf16 words (a uint16 array) as a bf16 tensor of `shape`."""
    if a is None:
        return None
    t = torch.from_numpy(a.view(np.int16).copy()).view(BF)
    return t.reshape(*shape) if shape else t


def rel(a, b):
    """The largest difference of a from b relative to the largest value of b."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-30)).item()


def activation(cfg, gu, inter):
    """The activation of the first product's output [n, 2 inter] (gate then up) or [n, inter] (relu squared), in bf16 as eager code runs it."""
    if cfg["act"] == "relu2":
        return F.relu(gu).pow(2)
    gate, up = gu[..., :inter], gu[..., inter:]
    if cfg["limit"] is not None:  # GLM-5.x's swiglu_limit: gate at most it, up within it
        gate = gate.clamp(max=cfg["limit"])
        up = up.clamp(min=-cfg["limit"], max=cfg["limit"])
    act = F.silu(gate) if cfg["act"] == "swiglu" else F.gelu(gate, approximate="tanh")
    return act * up


def scores_of(cfg, x, router, bias):
    """The router's logits, scores and the values it ranks them by, as the model's code computes them."""
    if cfg["bf16_logits"]:
        logits = F.linear(x, router).float()  # a bf16 gate: bf16 logits
    else:
        logits = F.linear(x.float(), router.float())  # a router computed in fp32
    if cfg["scoring"] == "softmax":
        s = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return s, s
    s = torch.sigmoid(logits)
    return s, (s + bias[None, :] if bias is not None else s)


def weights_of(cfg, s, idx, escale):
    """The weights of chosen experts `idx` [T, k] from the scores: renormalized, times the per-expert scale, times routed_scaling_factor."""
    w = s.gather(1, idx.clamp(min=0))
    if cfg["renorm"]:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    if escale is not None:
        w = w * escale[idx.clamp(min=0)]
    return w * cfg["scale"]


def run_case(d, name, bound):
    cfg = json.load(open(os.path.join(d, "cfg.json")))
    T, E, k, hr, H, I = cfg["tokens"], cfg["experts"], cfg["top_k"], cfg["router_hidden"], cfg["hidden"], cfg["inter"]
    rows = I if cfg["act"] == "relu2" else 2 * I
    x_route, x_exp = bf16(read(d, "x_route.bf16", np.uint16), T, hr), bf16(read(d, "x_exp.bf16", np.uint16), T, H)
    router = bf16(read(d, "router.bf16", np.uint16), E, hr)
    bias, escale = read(d, "bias.f32", np.float32), read(d, "escale.f32", np.float32)
    bias, escale = (torch.from_numpy(bias) if bias is not None else None), (torch.from_numpy(escale) if escale is not None else None)
    up, down = bf16(read(d, "up.bf16", np.uint16), E, rows, H), bf16(read(d, "down.bf16", np.uint16), E, H, I)
    ids = torch.from_numpy(read(d, "ids.i64", np.int64)).view(T, k)
    wts = torch.from_numpy(read(d, "wts.f32", np.float32)).view(T, k)
    y_gpu = bf16(read(d, "y.bf16", np.uint16), T, H)

    # the router: torch's own choice against the engine's, and the weights torch makes of the engine's choice
    s, choose = scores_of(cfg, x_route, router, bias)
    mine = torch.topk(choose, k, dim=-1, sorted=True).indices
    bad = ties = 0
    for t in range(T):
        if torch.equal(mine[t], ids[t]):
            continue
        if (ids[t] < 0).any():
            bad += 1
            continue
        a, b = choose[t, ids[t]].sort(descending=True).values, choose[t, mine[t]].sort(descending=True).values
        if torch.allclose(a, b, rtol=1e-6, atol=1e-12):
            ties += 1
        else:
            bad += 1
    r_wts = rel(wts, weights_of(cfg, s, ids, escale))

    # the experts one at a time on the tokens that chose them, bf16 throughout; the sum in bf16 as Qwen3.5, Gemma 4 and GLM-5.x add it, in fp32 as NemotronHExperts does (relu squared)
    acc = torch.float32 if cfg["act"] == "relu2" else BF
    y = torch.zeros(T, H, dtype=acc)
    for e in range(E):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = activation(cfg, F.linear(x_exp[tok], up[e]), I)
        out = F.linear(h, down[e])
        y.index_add_(0, tok, (out * wts[tok, slot, None]).to(acc))
    y = y.to(BF)
    # the shared expert, the same activation, times the sigmoid of its gate where it has one
    sh = cfg["shared"]
    if sh is not None:
        si = sh["inter"]
        srows = si if cfg["act"] == "relu2" else 2 * si
        s_up, s_down = bf16(read(d, "s_up.bf16", np.uint16), srows, hr), bf16(read(d, "s_down.bf16", np.uint16), H, si)
        so = F.linear(activation(cfg, F.linear(x_route, s_up), si), s_down)
        gate = bf16(read(d, "s_gate.bf16", np.uint16), 1, hr)
        if gate is not None:
            so = torch.sigmoid(F.linear(x_route, gate)) * so
        y = y + so
    r_y = rel(y_gpu, y)
    exact = (y_gpu == y).float().mean().item()
    ok = bad == 0 and r_wts <= 1e-5 and r_y <= bound and bool(torch.isfinite(y_gpu.float()).all())
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {T} tokens; torch's experts equal the engine's for {T - bad - ties}/{T}" + (f" (+{ties} by a tie the other way)" if ties else "") +
          f", weights {r_wts:.1e} (bound 1e-5); the layer's output {r_y:.2e} of the largest value (bound {bound:.1e}), {100 * exact:.1f}% of its values equal")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("only", nargs="*", help="CASE-LAYOUT directories to run (default: all)")
    ap.add_argument("--bound", type=float, default=2e-2)
    a = ap.parse_args()
    torch.manual_seed(0)
    names = sorted(n for n in os.listdir(a.dump) if os.path.exists(os.path.join(a.dump, n, "cfg.json")) and (not a.only or n in a.only))
    if not names:
        print(f"no case in {a.dump}")
        return 1
    print(f"torch {torch.__version__}, {len(names)} cases")
    bad = [n for n in names if not run_case(os.path.join(a.dump, n), n, a.bound)]
    print("all cases within the bound" if not bad else "FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
