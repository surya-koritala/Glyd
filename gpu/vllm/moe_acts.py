"""A mixture of experts' model under vLLM, bf16 against Glyd (--quantization glyd), each in a process of its own (vLLM's
default mode: compiled, CUDA graphs): the weights' GiB vLLM logs; decode tokens a second for 1 and 8 users (random
1,024-token prompts, 256 tokens out, greedy, the median of 3); and one of its mixture of experts' layers (LAYER: its
index among them, default 0; -1 the last) by tokens a step, on T random tokens routed to k random experts each: the
GPU time a call (CUDA events, the median of REPS) of bf16's method, and of Glyd's grouped products and its decoded
route, the route Glyd takes at T, the two routes' largest relative difference; the grouped products the other way too
("alt": vLLM's activation on the products' sums where the product applies it itself, and the product's plain SiLU where
the layer's is a SwiGLU with a clamp, which shows what the clamp changes on these inputs); and the rule: Glyd's route no
slower than bf16's at every T.

    python moe_acts.py OUT_DIR MODEL [--no-tps]

Env: TOKENS ("1 2 4 8 16 32 64 128 256 512 1024 2048"), REPS (20), USERS ("1 8"), LAYER (0), UTIL (0.85), XSCALE (0.1: the inputs' scale),
MOE_ACTS_LLM_KW (a JSON object of LLM's arguments added: a slice's {"language_model_only": true}). Results:
OUT_DIR/{bf16,glyd}.json and .log, summary.json."""
import json
import os
import re
import statistics
import subprocess
import sys
import time


def _time(f, reps):
    import torch

    for _ in range(3):
        f()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        f()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def _measure(model):
    """The layer's rows: each T's GPU us a call by route, and Glyd's two routes' largest relative difference."""
    import torch
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    torch.manual_seed(0)
    layers = [m for m in model.modules() if isinstance(m, RoutedExperts)]
    m = layers[int(os.environ.get("LAYER", "0"))]
    qm, E, k, H = m.quant_method, m.global_num_experts, m.top_k, m.moe_config.hidden_dim
    out = {"layer": m.layer_name, "layers": len(layers), "E": E, "k": k, "H": H, "I": m.moe_config.intermediate_size_per_partition, "activation": m.activation.value,
           "swiglu": [getattr(m.moe_config, "swiglu_" + a, None) for a in ("limit", "alpha", "beta")], "method": type(qm).__name__,
           "decode_min": getattr(qm, "decode_min", None), "act": getattr(qm, "act", None), "rows": []}
    reps = int(os.environ.get("REPS", "20"))
    for T in [int(t) for t in os.environ.get("TOKENS", "1 2 4 8 16 32 64 128 256 512 1024 2048").split()]:
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * float(os.environ.get("XSCALE", "0.1"))
        w, ids = torch.topk(torch.softmax(torch.randn(T, E, device="cuda"), dim=-1), k, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(torch.float32)
        ids = ids.to(getattr(qm, "topk_indices_dtype", None) or torch.int32)
        call = lambda: qm.apply(m, x, w, ids, None, None)
        row = {"T": T}
        if getattr(m, "glyd_moe", None) is None:
            row["bf16_us"] = _time(call, reps)
        else:
            keep, act = qm.decode_min, qm.act
            try:
                qm.decode_min = 1 << 30
                row["grouped_us"] = _time(call, reps)
                yg = call().float()
                alt = 0 if act else 1 if m.activation.value == "silu" and out["I"] % 64 == 0 else None
                if alt is not None:
                    qm.act = alt
                    row["alt_us"] = _time(call, reps)
                    row["alt_rel_diff"] = ((call().float() - yg).abs().max() / yg.abs().max()).item()
                    qm.act = act
                if qm.ref is not None:
                    qm.decode_min = 0
                    row["decoded_us"] = _time(call, reps)
                    yd = call().float()
                    row["rel_diff"] = ((yg - yd).abs().max() / yd.abs().max()).item()
            finally:
                qm.decode_min, qm.act = keep, act
            row["route"] = "decoded" if keep is not None and T >= keep else "grouped"
        out["rows"].append(row)
    return out


def child(spec):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=spec["model"], dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=float(os.environ.get("UTIL", "0.85")), enable_prefix_caching=False)
    if spec["mode"] == "glyd":
        kw.update(quantization="glyd", additional_config={"glyd": {"kv": "off"}})
    kw.update(json.loads(os.environ.get("MOE_ACTS_LLM_KW") or "{}"))
    llm = LLM(**kw)
    out = {"mode": spec["mode"], "gpu": torch.cuda.get_device_name(), "glyd": (llm.llm_engine.vllm_config.additional_config or {}).get("glyd")}
    if spec["tps"]:
        g = torch.Generator().manual_seed(0)

        def run(n, new):
            ps = [TokensPrompt(prompt_token_ids=torch.randint(1000, 30000, (1024,), generator=g).tolist()) for _ in range(n)]
            torch.cuda.synchronize()
            t = time.perf_counter()
            llm.generate(ps, SamplingParams(temperature=0, max_tokens=new, ignore_eos=True), use_tqdm=False)
            torch.cuda.synchronize()
            return time.perf_counter() - t

        for n in [int(u) for u in os.environ.get("USERS", "1 8").split()]:
            run(n, 8)
            out[f"tps_{n}"] = statistics.median(n * 255 / (run(n, 256) - run(n, 1)) for _ in range(3))
    out["moe"] = llm.apply_model(_measure)[0]
    return out


def main():
    if sys.argv[1] == "--child":
        spec = json.loads(sys.argv[2])
        json.dump(child(spec), open(spec["out"], "w"), indent=1)
        return
    out_dir, model = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for mode in ("bf16", "glyd"):
        f = os.path.join(out_dir, mode)
        env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", GLYD_KV="off")
        with open(f + ".log", "w") as log:
            rc = subprocess.run([sys.executable, __file__, "--child", json.dumps({"model": model, "mode": mode, "tps": "--no-tps" not in sys.argv, "out": f + ".json"})], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        r = json.load(open(f + ".json")) if rc == 0 and os.path.exists(f + ".json") else {"error": f"exit {rc}: {open(f + '.log').read()[-600:]}"}
        m = re.search(r"Model loading took ([\d.]+) GiB", open(f + ".log").read())
        r["weights_gib"] = float(m.group(1)) if m else None
        res[mode] = r
        print(mode, {k: v for k, v in r.items() if k != "moe"}, flush=True)
    b, g = res["bf16"], res["glyd"]
    rows, slower = [], []
    if "moe" in b and "moe" in g:
        bf = {r["T"]: r["bf16_us"] for r in b["moe"]["rows"]}
        for r in g["moe"]["rows"]:
            r["bf16_us"] = bf.get(r["T"])
            r["route_us"] = r.get("decoded_us") if r["route"] == "decoded" else r["grouped_us"]
            if r["bf16_us"] and r["route_us"] > r["bf16_us"]:
                slower.append(r["T"])
            rows.append(r)
    s = {"model": model, "weights_gib": {"bf16": b.get("weights_gib"), "glyd": g.get("weights_gib")}, "tps": {m: {k: v for k, v in res[m].items() if k.startswith("tps_")} for m in res},
         "layer": {k: v for k, v in g.get("moe", {}).items() if k != "rows"}, "rows": rows, "route_slower_than_bf16_at": slower, "errors": {m: res[m]["error"] for m in res if "error" in res[m]}}
    json.dump(s, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}), flush=True)
    for r in rows:
        print(f"T={r['T']:5d}  bf16 {r['bf16_us'] or 0:8.1f}  grouped {r['grouped_us']:8.1f}" + (f" (alt {r['alt_us']:8.1f}, {r['alt_rel_diff']:.1e})" if "alt_us" in r else "")
              + (f"  decoded {r['decoded_us']:8.1f}  rel diff {r['rel_diff']:.1e}" if "decoded_us" in r else "") + f"  route {r['route']}: {r['route_us'] / r['bf16_us'] if r['bf16_us'] else 0:.2f}x bf16 us", flush=True)
    print("RULE " + ("PASS: Glyd's route no slower than bf16's at every T" if rows and not slower else f"FAIL: slower than bf16 at T {slower}" if rows else "NOT RUN"), flush=True)


if __name__ == "__main__":
    main()
