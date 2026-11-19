"""One user's tokens a second under vLLM, two Glyd trees (or modes) against each other, alternated: each round runs A then B, each in a vLLM of its own (offline,
vLLM's default mode: compiled, CUDA graphs), on one request of a random prompt of --prompt tokens decoded for --new tokens (the time of the run less the time of a
run of one token, over the new tokens; the median of --reps requests), the GPU's clock and temperature, the weights and the KV cache vLLM logs. The order of a round
alternates, so a drift of the GPU's clock (an L4 slows as it heats, at its power cap) goes both ways.

    python tps_ab.py --a NAME:K=V,K=V --b NAME:K=V,... [--rounds 3] [--prompt 1024] [--new 256] [--reps 3] [--util 0.9] [--out DIR] [MODEL]

An env of a side (K=V, comma apart) is added to the process's: PYTHONPATH and GLYD_GPU_LIB name a tree's sources and its library, GLYD_PACK_EMBED=none leaves the embedding
and the LM head as vLLM runs them, VLLM_CACHE_ROOT a compile cache of its own. Results: DIR/NAME-rN.json and a table."""
import json
import os
import re
import statistics
import subprocess
import sys
import time


def child(spec):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=spec["model"], dtype="bfloat16", seed=0, max_model_len=spec["prompt"] + spec["new"] + 64, gpu_memory_utilization=spec["util"], enable_prefix_caching=False, quantization="glyd",
              additional_config={"glyd": {"kv": "off"}})
    cfg = llm.llm_engine.vllm_config.cache_config
    g = torch.Generator().manual_seed(0)
    mk = lambda: [TokensPrompt(prompt_token_ids=torch.randint(1000, 50000, (spec["prompt"],), generator=g).tolist())]

    def timed(ps, n):
        """(start, end) of a generate() of n tokens."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(ps, SamplingParams(temperature=0, max_tokens=n, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        return t0, time.perf_counter()

    ps = mk()
    timed(ps, 8)  # (warm)
    smi = lambda: os.popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits").read().strip()
    runs, clocks = [], []
    for _ in range(spec["reps"]):
        ps = mk()  # (a prompt of its own each: the prefix cache is off, and the same prompt twice would be no harm)
        a0, a1 = timed(ps, 1)
        b0, b1 = timed(ps, spec["new"] + 1)
        runs.append(spec["new"] / ((b1 - b0) - (a1 - a0)))
        clocks.append(smi())
    json.dump({"tokens_per_s": statistics.median(runs), "runs": runs, "smi": clocks, "blocks": cfg.num_gpu_blocks, "block_size": cfg.block_size}, open(spec["path"], "w"))


def parse(side):
    name, _, env = side.partition(":")
    return name, dict(kv.split("=", 1) for kv in env.split(",") if kv)


def run(side, rnd, a):
    name, env = side
    path = os.path.join(a["out"], f"{name}-r{rnd}.json")
    spec = dict(a, path=path)
    e = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", GLYD_KV="off", **env)
    log = os.path.join(a["out"], f"{name}-r{rnd}.log")
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)], env=e, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    if r.returncode or not os.path.exists(path):
        raise RuntimeError(f"{name} round {rnd}: exit {r.returncode}; see {log}")
    res = json.load(open(path))
    text = open(log).read()
    m = re.search(r"Model loading took ([\d.]+) GiB", text)
    res["weights_gib"] = float(m.group(1)) if m else None
    res["kv_tokens"] = res["blocks"] * res["block_size"]
    return res


def main():
    a = sys.argv[1:]
    if a[:1] == ["--child"]:
        return child(json.loads(a[1]))
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    A, B = parse(opt("--a", "a")), parse(opt("--b", "b"))
    spec = {"rounds": int(opt("--rounds", 3)), "prompt": int(opt("--prompt", 1024)), "new": int(opt("--new", 256)), "reps": int(opt("--reps", 3)), "util": float(opt("--util", 0.9)), "out": opt("--out", "tps_ab_results")}
    spec["model"] = next((x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--a", "--b", "--rounds", "--prompt", "--new", "--reps", "--util", "--out"))), "Qwen/Qwen3-8B")
    os.makedirs(spec["out"], exist_ok=True)
    got = {A[0]: [], B[0]: []}
    for rnd in range(spec["rounds"]):
        for side in ((A, B) if rnd % 2 == 0 else (B, A)):
            r = run(side, rnd, spec)
            got[side[0]].append(r)
            print(f"round {rnd} {side[0]}: {r['tokens_per_s']:.2f} tokens/s (runs {[round(x, 2) for x in r['runs']]}; weights {r['weights_gib']} GiB, KV cache {r['kv_tokens']:,} tokens; GPU {r['smi'][-1]})", flush=True)
    ta, tb = (statistics.median(r["tokens_per_s"] for r in got[n]) for n in (A[0], B[0]))
    each = ", ".join(f"{y['tokens_per_s'] / x['tokens_per_s']:.3f}" for x, y in zip(got[A[0]], got[B[0]]))
    print(f"\n{spec['model']}, one user, {spec['prompt']} tokens in, {spec['new']} out, vLLM's default mode (compiled, CUDA graphs), the median of {spec['rounds']} rounds, alternated:")
    print(f"| | tokens a second | weights GiB | KV cache tokens (the last round's start: vLLM's memory profile of a start that compiles reads other than that of one that does not) |\n| :--- | ---: | ---: | ---: |")
    for n in (A[0], B[0]):
        rounds = ", ".join("%.2f" % r["tokens_per_s"] for r in got[n])
        print(f"| {n} | {statistics.median(r['tokens_per_s'] for r in got[n]):.2f} (rounds {rounds}) | {got[n][-1]['weights_gib']} | {got[n][-1]['kv_tokens']:,} |")
    print(f"\n{B[0]} against {A[0]}: {tb / ta:.3f}x the tokens a second (each round {each})")
    json.dump(got, open(os.path.join(spec["out"], "tps_ab.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
