"""A mixture of experts' layer on this GPU by tokens a step: the library's grouped products checked against an fp32
product and timed under CUDA graphs (the kernels' GPU time, not their launches'), each call on routings of its own so a layer's
weights come from memory as in a step through its layers; the route, the gate and up product, the down product with the
router's weights, the three together, and the experts decoded for vLLM's own kernel (the plugin's other route), each with the
share of the weights' streaming rate it reaches.

    python moe_layer.py --synthetic 128,8,2048,768           E, k, hidden, intermediate (Qwen3-30B-A3B's layer); --wild F: its weights at far exponents
    python moe_layer.py --hf ibm-granite/granite-3.1-3b-a800m-instruct --layer 0     a layer's experts from the Hugging Face cache
    python moe_layer.py --file experts.safetensors --k 8      a layer's experts from a file: w13 [E, 2I, H] (gate then up) and w2 [E, H, I] (fetch_tensors.py)
    options: --layout mma|mma12 (default: this GPU's best), --tokens "1 8 32 232 1152", --check, --skew S (routing skew), --gelu (the gate's activation GELU, not SiLU), --json OUT,
             --variants "base:GLYD_GPU_PF=0;pf4:GLYD_GPU_PF=4" (each variant on the same layer and routings, in one process)

Env: REPS (calls a graph, 12: as many routings), ROUNDS (graph replays, 5); the library's GLYD_GPU_* variables as they are set."""
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

import glyd_gpu_dev as g
from glyd_gpu import _lib

BF = torch.bfloat16


def weights(shape, wild=0.0005, gen=None):
    """A trained matrix's spread: normal, 0.02, per-expert scale, and `wild` of the weights at exponents far from the common ones."""
    E = shape[0]
    w = torch.randn(shape, device="cuda", generator=gen) * (0.02 * torch.exp(torch.randn(E, 1, 1, device="cuda", generator=gen) * 0.2))
    m = torch.rand(shape, device="cuda", generator=gen) < wild
    n = int(m.sum())
    w[m] = torch.randn(n, device="cuda", generator=gen) * torch.exp2(torch.randint(-40, 20, (n,), device="cuda", generator=gen).float())
    return w.to(BF)


def hf_layer(repo, layer):
    """(w13 [E, 2I, H] gate then up, w2 [E, H, I], k) of layer `layer`, from the files of the local Hugging Face cache that it needs: the config, the index and the shards that hold the
    layer's tensors, each asked for by its name (hf_cache.cached: a cache that holds only those, as a job's download of one layer leaves it, is whole enough)."""
    from hf_cache import cached
    from safetensors import safe_open

    cfg = json.load(open(cached(repo, "config.json")))
    k = cfg.get("num_experts_per_tok") or cfg.get("moe_top_k") or cfg["num_experts_per_tok"]
    try:
        wm = json.load(open(cached(repo, "model.safetensors.index.json")))["weight_map"]
    except FileNotFoundError:  # (a model in one file)
        wm = {}
    files = {}

    def get(name):
        f = cached(repo, wm[name] if wm else "model.safetensors")
        if f not in files:
            files[f] = safe_open(f, "pt", device="cuda")
        return files[f].get_tensor(name)

    pre = f"model.layers.{layer}."
    if pre + "block_sparse_moe.input_linear.weight" in wm:  # granite: stacked
        return get(pre + "block_sparse_moe.input_linear.weight"), get(pre + "block_sparse_moe.output_linear.weight"), k
    E = cfg.get("num_experts") or cfg.get("num_local_experts")
    w13 = torch.stack([torch.cat([get(f"{pre}mlp.experts.{e}.gate_proj.weight"), get(f"{pre}mlp.experts.{e}.up_proj.weight")]) for e in range(E)])
    w2 = torch.stack([get(f"{pre}mlp.experts.{e}.down_proj.weight") for e in range(E)])
    return w13, w2, k


def routing(T, E, k, skew, gen):
    """T tokens' top k of random logits (skew: a fixed bias of that many sigmas on the experts, uneven loads), int64, and weights."""
    logits = torch.randn(T, E, device="cuda", generator=gen)
    if skew:
        logits = logits + skew * torch.randn(1, E, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    w, ids = torch.topk(torch.softmax(logits, -1), k, -1)
    return ids.contiguous(), (w / w.sum(-1, keepdim=True)).to(torch.float32)


def graph_us(fns, rounds):
    """The calls of fns in one CUDA graph: GPU us a call, the median of `rounds` replays."""
    for f in fns:
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    _lib.local.fresh = True  # workspaces in the graph's pool, as in the plugin's
    try:
        with torch.cuda.graph(graph):
            for f in fns:
                f()
    finally:
        _lib.local.fresh = False
    graph.replay()
    ts = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / len(fns))
    return statistics.median(ts)


def stream_gbs():
    """The GPU's copy rate (read + write), GB/s: what a kernel that streams the weights can hope for, about."""
    n = 1 << 29
    a, b = torch.empty(n, dtype=torch.uint8, device="cuda"), torch.empty(n, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        x, y = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        x.record()
        b.copy_(a)
        y.record()
        y.synchronize()
        ts.append(x.elapsed_time(y))
    return 2 * n / (statistics.median(ts) * 1e6)


ACT = 1  # the gate's activation: 1 SiLU, 2 GELU (tanh); main() sets it from --gelu


def check_variant(one, plan0, h0, hits0, p2, w13, w2, E, I, H, k, T, vname, row):
    """The products of one routing against an fp32 product on the weights as they were (AssertionError where one differs): the plan
    (the pairs sorted by expert, in order within each), gate and up with SiLU, down with the router's weights, the same bits twice."""
    x, ids, wt = one
    P = T * k
    plan = g.moe_way(ids, E)  # (the route's own, under this variant)
    flat = ids.view(-1)
    order = plan[2 + 2 * E : 2 + 2 * E + P].long()
    assert torch.equal(plan[1 : 1 + hits0].long(), flat.unique()), "route: the experts hit"
    assert torch.equal(order, torch.sort(flat, stable=True).indices), "route: the pairs sorted by expert, in order within each"
    expert = flat[order]
    ref13 = torch.empty(P, I, device="cuda")
    ref2 = torch.empty(P, H, device="cuda")
    h = h0.float()
    for e in flat.unique().tolist():
        sel = (expert == e).nonzero().view(-1)
        y = x[order[sel] // k].float() @ w13[e].float().T
        ref13[sel] = (F.gelu(y[:, :I], approximate="tanh") if ACT == 2 else F.silu(y[:, :I])) * y[:, I:]
        ref2[sel] = h[sel] @ w2[e].float().T  # the down product on this run's own gate/up output (the products checked one at a time)
    err13 = ((h - ref13).abs().max() / ref13.abs().max()).item()
    y2 = g.mma_moe(p2, E, h0, plan, ids, 0, None, wt, gather=False)
    ref = torch.zeros(P, H, device="cuda")
    ref[order] = ref2 * wt.view(-1)[order, None]
    err2 = ((y2.float() - ref.view(T, k, H).sum(1)).abs().max() / ref.abs().max()).item()
    row.update(err_gate_up=err13, err_down=err2)
    assert err13 < 1.5e-2 and err2 < 1.5e-2, (T, vname, err13, err2)
    again = g.mma_moe(p2, E, h0, plan, ids, 0, None, wt, gather=False)
    assert torch.equal(y2.view(torch.int16), again.view(torch.int16)), "the same bits twice"
    print(f"T={T:5d} {vname:>12s}: checked, {hits0} experts hit; gate+up {err13:.1e}, down {err2:.1e} (against fp32)", flush=True)


def parse_variants(spec):
    """'name:K=V,K=V;name2:K=V' -> [(name, {K: V})]; no spec: the environment as it is."""
    out = []
    for part in (spec or "").split(";"):
        if not part.strip():
            continue
        name, _, kv = part.partition(":")
        out.append((name.strip(), dict(x.split("=", 1) for x in kv.split(",") if x.strip())))
    return out or [("env", {})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--hf", default="")
    ap.add_argument("--file", default="", help="w13 and w2 of a layer from a safetensors file")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--layout", default="auto", choices=["auto", "mma", "mma12"])
    ap.add_argument("--tokens", default="1 2 4 8 16 32 64 128 232 512 1024 2048 4096 8192")
    ap.add_argument("--gelu", action="store_true", help="the gate's activation GELU (tanh), not SiLU")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--wild", type=float, default=0.0005, help="the synthetic layer's share of weights at exponents far from the common ones (exceptions: 0.0005 is about 0.5 a step of 1,024 weights, 0.02 about 20)")
    ap.add_argument("--json", default="")
    ap.add_argument("--k", type=int, default=0, help="top k, where the layer's own is not wanted")
    ap.add_argument("--bf16", action="store_true", help="vLLM's own bf16 layer (fused_experts, its Triton kernels) on the same weights and routings, timed the same way (needs vLLM)")
    ap.add_argument("--variants", default="", help="NAME:GLYD_GPU_X=V,GLYD_GPU_Y=W;NAME2:... each run on the same layer and routings, its variables set for it (the library reads them at each call); a variable not named is as the environment has it")
    a = ap.parse_args()
    global ACT
    ACT = 2 if a.gelu else 1
    gen = torch.Generator(device="cuda").manual_seed(0)
    if a.hf:
        w13, w2, k = hf_layer(a.hf, a.layer)
        name = f"{a.hf} layer {a.layer}"
    elif a.file:
        from safetensors.torch import load_file

        t = load_file(a.file, device="cuda")
        w13, w2, k, name = t["w13"], t["w2"], 0, a.file
    else:
        E, k, H, I = (int(v) for v in (a.synthetic or "128,8,2048,768").split(","))
        w13, w2 = weights((E, 2 * I, H), a.wild, gen), weights((E, H, I), a.wild, gen)
        name = f"synthetic {E} experts, top {k}, hidden {H}, intermediate {I}"
    k = a.k or k
    E, O13, H = w13.shape
    I = O13 // 2
    layout = a.layout if a.layout != "auto" else g.best_layout(w13.numel() * 4, 0, 1, 0, moe=True)[0]
    pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
    p13, p2 = pack(w13.reshape(E * O13, H)), pack(w2.reshape(E * H, I))
    g.lib()
    for nm in ("mma12_moe", "mma_moe"):
        _lib._counters(nm, 0, None, 0, 1 << 16)
    per13, per2 = p13.nbytes() / E, p2.nbytes() / E  # packed bytes an expert
    gbs = stream_gbs()
    variants = parse_variants(a.variants)
    print(f"{name}: {torch.cuda.get_device_name()}, layout {layout}, {8 * p13.nbytes() / w13.numel():.2f} bits a weight (gate+up), {8 * p2.nbytes() / w2.numel():.2f} (down), an expert {per13 / 1e6:.2f} + {per2 / 1e6:.2f} MB packed, "
          f"copy rate {gbs:.0f} GB/s (read + write), env {', '.join(f'{x}={v}' for x, v in sorted(os.environ.items()) if x.startswith('GLYD_')) or 'none'}", flush=True)
    reps, rounds = int(os.environ.get("REPS", "12")), int(os.environ.get("ROUNDS", "5"))
    out = {"model": name, "gpu": torch.cuda.get_device_name(), "layout": layout, "E": E, "k": k, "H": H, "I": I, "copy_gbs": gbs, "rows": [], "env": {x: v for x, v in os.environ.items() if x.startswith("GLYD_")}}
    buf13, buf2 = torch.empty(E * O13 * H, dtype=BF, device="cuda"), torch.empty(E * H * I, dtype=BF, device="cuda")
    vctx, fused_experts, bad = None, None, set()
    if a.bf16:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.fused_moe import fused_experts

        vctx = set_current_vllm_config(VllmConfig())
        vctx.__enter__()
    for T in (int(t) for t in a.tokens.split()):
        sets = []
        for r in range(reps):
            ids, wt = routing(T, E, k, a.skew, gen)
            x = torch.randn(T, H, dtype=BF, device="cuda", generator=gen) * 0.1
            sets.append((x, ids, wt))
        P = T * k
        plans = [g.moe_way(ids, E) for _, ids, _ in sets]
        hits = [int(pl[0]) for pl in plans]
        bf16_us = None
        if a.bf16 and not a.no_time:
            ids32 = [ids.to(torch.int32) for _, ids, _ in sets]
            bf = [lambda i=i: fused_experts(sets[i][0], w13, w2, sets[i][2], ids32[i]) for i in range(reps)]
            bf16_us = graph_us(bf, rounds)
            print(f"T={T:5d}        bf16: vLLM's layer {bf16_us:8.1f} us (the weights' bf16 bytes {statistics.mean(hits) * (w13[0].numel() + w2[0].numel()) * 2 / (bf16_us * 1e3):6.0f} GB/s)", flush=True)
        for vname, env in variants:
            saved = {x: os.environ.get(x) for x in env}
            os.environ.update(env)
            _lib._need.cache_clear()  # (a workspace's size can depend on a variable: GLYD_GPU_MOE_SLOTS)
            try:
                hs = [g.mma_moe(p13, E, x, pl, ids, ACT) for (x, ids, _), pl in zip(sets, plans)]
                row = {"T": T, "variant": vname, "hits": statistics.mean(hits)}
                if a.check:
                    check_variant(sets[0], plans[0], hs[0], hits[0], p2, w13, w2, E, I, H, k, T, vname, row)
                if not a.no_time:
                    rt = [lambda i=i: g.moe_way(sets[i][1], E) for i in range(reps)]
                    gu = [lambda i=i: g.mma_moe(p13, E, sets[i][0], plans[i], sets[i][1], ACT) for i in range(reps)]
                    dn = [lambda i=i: g.mma_moe(p2, E, hs[i], plans[i], sets[i][1], 0, None, sets[i][2], gather=False) for i in range(reps)]
                    al = [lambda i=i: g.mma_moe(p2, E, g.mma_moe(p13, E, sets[i][0], g.moe_way(sets[i][1], E), sets[i][1], ACT), plans[i], sets[i][1], 0, None, sets[i][2], gather=False) for i in range(reps)]
                    up = [lambda i=i: (g.mma_moe_unpack(p13, E, plans[i], P, buf13), g.mma_moe_unpack(p2, E, plans[i], P, buf2)) for i in range(reps)]
                    if bf16_us:
                        row["bf16_us"] = bf16_us
                    row["route_us"] = graph_us(rt, rounds)
                    row["gate_up_us"] = graph_us(gu, rounds)
                    row["down_us"] = graph_us(dn, rounds)
                    row["all_us"] = graph_us(al, rounds)
                    row["unpack_us"] = graph_us(up, rounds)
                    m = row["hits"]
                    row["gate_up_gbs"] = m * per13 / (row["gate_up_us"] * 1e3)
                    row["down_gbs"] = m * per2 / (row["down_us"] * 1e3)
                    row["all_gbs"] = m * (per13 + per2) / (row["all_us"] * 1e3)
                    print(f"T={T:5d} {vname:>12s}: hits {m:6.1f}  route {row['route_us']:7.1f}  gate+up {row['gate_up_us']:8.1f} ({row['gate_up_gbs']:6.0f} GB/s)  down {row['down_us']:8.1f} ({row['down_gbs']:6.0f} GB/s)  "
                          f"all {row['all_us']:8.1f} ({row['all_gbs']:5.0f} GB/s, {row['all_gbs'] / gbs * 2:4.2f} of the copy rate)  unpack {row['unpack_us']:8.1f}" + (f"  = {row['all_us'] / bf16_us:4.2f}x bf16" if bf16_us else "") + "  us a layer", flush=True)
                out["rows"].append(row)
            except (AssertionError, RuntimeError) as ex:  # (a variant's products differ, or the library refuses them: reported, the others go on)
                bad.add(vname)
                print(f"T={T:5d} {vname:>12s}: FAILED {type(ex).__name__}: {ex}", flush=True)
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
