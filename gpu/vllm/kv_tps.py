"""Decode tokens a second, the lossless KV cache against vLLM's own: each in a vLLM of its own (eager; bf16 weights; prefix caching and chunked prefill off),
batches of B requests of L tokens in all (a prompt of random tokens, then 128 new ones: every request ignores EOS), the time of the decode steps alone: a run of 129 tokens less
a run of 1 token, B x 128 tokens over the difference, the median of 3.

    python kv_tps.py [--out DIR] [--util 0.96] [--new 128] [--tables FILE] [--rounds N] [--graphs] [--configs 4x8192,32x1024,...] [MODEL]

--graphs: vLLM's default mode (compiled, FULL decode graphs; the plugin's decode runs in them) in place of eager.

--rounds N: N times the stock and kv runs, the order alternating (stock first, then kv first, ...), each configuration's tokens a second the median of the rounds' (the GPU's clock and
temperature drift between runs, a minute or more apart: alternating cancels a drift that goes one way). --configs: BxL batches (L tokens in all) and maxL (the largest batch of L tokens a cache holds).

Batches: the same batch on both sides (4 of 8,192 tokens, 32 of 1,024, one of 8,192, 8 of 4,096, 3 of 8,192), and each side at the largest batch its cache holds of 8,192 (and of
1,024) tokens (the KV cache tokens vLLM logs, over a request's blocks; ours has about 30% more). A batch that does not fit its side's cache is not run (its row says so).
The KV cache's size, the GPU's clock and temperature are in the rows."""
import json
import os
import shlex
import statistics
import subprocess
import sys
import time


def child(spec):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=spec["model"], dtype="bfloat16", enforce_eager=not spec.get("graphs"), max_model_len=8704, gpu_memory_utilization=spec["util"], seed=0, max_num_seqs=256, max_num_batched_tokens=8704,
              enable_prefix_caching=False, enable_chunked_prefill=False)
    if spec["kv"]:
        kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless"}})
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    blocks, bs = cfg.cache_config.num_gpu_blocks, cfg.cache_config.block_size
    out = {"kv": spec["kv"], "blocks": blocks, "tokens": blocks * bs, "rows": []}
    g = torch.Generator().manual_seed(0)
    new = spec["new"]

    def batch(B, L):
        return [TokensPrompt(prompt_token_ids=torch.randint(1000, 50000, (L,), generator=g).tolist()) for _ in range(B)]

    def timed(prompts, n):
        sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    smi = lambda: os.popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits").read().strip()
    for B, L in spec["configs"]:  # (L tokens in all: a prompt of L - new, then new)
        free = blocks - 2  # (the pool's null block, and one to spare: a batch that fills the cache would be preempted and recomputed)
        if B == "max":
            B = free // -(-L // bs)
        fits = B * -(-L // bs) <= free
        row = {"B": B, "L": L, "fits": fits}
        if fits and B > 0:
            ps = batch(B, L - new)
            timed(ps, 1)  # (warm)
            ds = []
            for _ in range(3):
                t1, tn = timed(ps, 1), timed(ps, new + 1)
                ds.append((tn - t1, t1, tn))
            d = statistics.median(x[0] for x in ds)
            row.update({"decode_s": d, "tokens_per_s": B * new / d, "ms_per_step": d / new * 1000, "prefill_s": statistics.median(x[1] for x in ds), "runs": [x[0] for x in ds], "smi": smi()})
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    return out


def main():
    a = sys.argv[1:]
    if a[:1] == ["--child"]:
        json.dump(child(json.loads(a[1])), open(a[2], "w"))
        return
    out = a[a.index("--out") + 1] if "--out" in a else "kv_tps_results"
    util = float(a[a.index("--util") + 1]) if "--util" in a else 0.96
    new = int(a[a.index("--new") + 1]) if "--new" in a else 128
    tables = a[a.index("--tables") + 1] if "--tables" in a else None
    rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 1
    graphs = "--graphs" in a  # (vLLM's default mode: compiled, FULL decode graphs; else eager)
    model = next((x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--out", "--util", "--new", "--tables", "--rounds", "--configs"))), "Qwen/Qwen3-8B")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", **({"GLYD_KV_TABLES": tables} if tables else {}))
    configs = [[1, 8192], [1, 1024], [3, 8192], [4, 8192], [32, 1024], [8, 4096], ["max", 8192], ["max", 1024]]
    if "--configs" in a:
        configs = [["max", int(c[3:])] if c.startswith("max") else [int(c.split("x")[0]), int(c.split("x")[1])] for c in a[a.index("--configs") + 1].split(",")]
    res = {"stock": [], "kv": []}
    for rnd in range(rounds):
        for kv in ((False, True) if rnd % 2 == 0 else (True, False)):
            name = "kv" if kv else "stock"
            tag = name if rounds == 1 else f"{name}-r{rnd}"
            spec = {"model": model, "kv": kv, "util": util, "new": new, "configs": configs, "graphs": graphs}
            path = os.path.join(out, f"{tag}.json")
            with open(os.path.join(out, f"{tag}.log"), "w") as log:
                r = subprocess.run(shlex.split(os.environ.get("KV_WRAP", "")) + [sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec), path], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)  # (KV_WRAP: a command each run is under, e.g. one that takes a lock)
            if r.returncode:
                raise RuntimeError(f"{tag}: exit {r.returncode}; see {out}/{tag}.log")
            res[name].append(json.load(open(path)))
    s0, k0 = res["stock"][0], res["kv"][0]
    print(f"KV cache: stock {s0['tokens']:,} tokens, kv {k0['tokens']:,} ({k0['tokens'] / s0['tokens']:.3f}x); decode tokens a second ({new} new tokens, {'default mode: compiled, FULL decode graphs' if graphs else 'eager'}, bf16 weights, util {util}; "
          f"{'one run each' if rounds == 1 else f'the median of {rounds} rounds, the order alternating'})")
    print("| Batch x tokens | Stock | KV | KV / stock |" + ("" if rounds == 1 else " each round's |"))
    print("| :--- | ---: | ---: | ---: |" + ("" if rounds == 1 else " :--- |"))
    for i, rs in enumerate(s0["rows"]):
        sr, kr = [x["rows"][i] for x in res["stock"]], [x["rows"][i] for x in res["kv"]]
        if not all("tokens_per_s" in r for r in sr + kr):
            print(f"| stock {rs['B']} x {rs['L']}, kv {kr[0]['B']} x {kr[0]['L']} | {'' if 'tokens_per_s' in rs else 'does not fit'} | {'' if 'tokens_per_s' in kr[0] else 'does not fit'} | | |")
            continue
        med = lambda rows, key: statistics.median(r[key] for r in rows)
        st, kt = med(sr, "tokens_per_s"), med(kr, "tokens_per_s")
        cell = lambda rows, t: f"{t:.1f} ({med(rows, 'ms_per_step'):.1f} ms a step; {rows[0]['smi']})"
        each = "" if rounds == 1 else " " + ", ".join(f"{k['tokens_per_s'] / s['tokens_per_s']:.3f}" for s, k in zip(sr, kr)) + " |"
        print(f"| stock {rs['B']} x {rs['L']}, kv {kr[0]['B']} x {kr[0]['L']} | {cell(sr, st)} | {cell(kr, kt)} | {kt / st:.3f} |{each}")
    json.dump(res, open(os.path.join(out, "tps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
