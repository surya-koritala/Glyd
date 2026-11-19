"""vllm serve on a model with kv lossless (an empty tables cache, the default reserves) or stock: ordinary requests alone, a prompt of the model's heavy tokens, a flood of them at once with ordinary requests sent
while it runs, and the same requests after: every request's status and latency.

    serve_flood.py MODEL OUTDIR [--flood N] [--stock] [--heavy FILE] [--env KEY=VALUE ...]

kv: the tables are made by the first start (in OUTDIR/cache) and the heavy tokens written to OUTDIR/heavy.json; stock takes them from a file (--heavy). Results: OUTDIR/result.json.
"""
import concurrent.futures as cf
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from glyd_gpu import kvcal

args = sys.argv[1:]
model, out = args[0], os.path.abspath(args[1])
nflood = int(args[args.index("--flood") + 1]) if "--flood" in args else 24
stock = "--stock" in args
heavy_file = args[args.index("--heavy") + 1] if "--heavy" in args else None
extra_env = dict(a.split("=", 1) for i, a in enumerate(args) if i and args[i - 1] == "--env")
nord = int(os.environ.get("N_ORDINARY", "6"))
os.makedirs(out, exist_ok=True)
cache = os.path.join(out, "cache")
env = dict(os.environ, GLYD_CACHE=cache, HF_HOME=os.path.expanduser("~/hf"), TOKENIZERS_PARALLELISM="false", **extra_env)
env.pop("GLYD_KV_TABLES", None)
port = int(os.environ.get("PORT", "8011"))
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port), "--max-model-len", "4096", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9"]
cmd += ["--enforce-eager"] if os.environ.get("EAGER") else ["--max-num-batched-tokens=4096"]
if not stock:
    cmd += ["--quantization", "glyd", "--additional-config", json.dumps({"glyd": {"fraction": 0, "kv": "lossless"}})]
t0 = time.time()
log = open(os.path.join(out, "server.log"), "w")
srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def post(path, body, timeout=900):
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400], time.perf_counter() - t


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status
    except Exception:
        return None


def text(r):
    return r["choices"][0]["text"] if isinstance(r, dict) else None


def lat(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median": round(statistics.median(xs), 3), "max": round(xs[-1], 3), "min": round(xs[0], 3)} if xs else {}


res = {"model": model, "stock": stock, "flood": nflood, "env": extra_env}
try:
    while health() != 200:
        if srv.poll() is not None:
            raise RuntimeError(f"the server exited {srv.returncode}: see server.log")
        time.sleep(3)
        if time.time() - t0 > 1800:
            raise RuntimeError("the server did not come up in 30 minutes")
    res["start_seconds"] = round(time.time() - t0, 1)
    if heavy_file:
        heavy_info = json.load(open(heavy_file))
    else:
        tabs = glob.glob(os.path.join(cache, "kv", "kvtab-*.npz"))
        meta = kvcal.Tables(tabs[0]).meta if tabs else {}
        hv = kvcal.heavy(meta)
        text_ids, ids = hv["text"], hv["ids"]
        heavy_info = {"heavy": (text_ids + ids) or [hv["worst"]], "text": len(text_ids), "ids": hv["count"]}
        json.dump(heavy_info, open(os.path.join(out, "heavy.json"), "w"))
    heavy = heavy_info["heavy"]
    res["heavy"] = {k: v for k, v in heavy_info.items() if k != "heavy"} | {"used": len(heavy)}
    prompts = ["The capital of Australia is", "The history of data compression began", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "In quantum mechanics, the uncertainty principle states that",
               "Once upon a time, in a village at the edge of a great forest,", "SELECT name, COUNT(*) FROM orders JOIN customers ON"]
    body = lambda p, n=16: {"model": model, "prompt": p, "max_tokens": n, "temperature": 0}
    # ordinary requests alone, twice (the first warms)
    alone, texts = [], {}
    for rnd in range(2):
        for p in prompts[:nord]:
            c, r, dt = post("/v1/completions", body(p))
            if rnd:
                alone.append(dt)
                texts[p] = (c, text(r))
    res["ordinary_alone"] = lat(alone)
    res["ordinary_alone_codes"] = sorted({c for c, _ in texts.values()})
    c1, r1, dt1 = post("/v1/completions", body([heavy[i % len(heavy)] for i in range(1000)], 8))
    res["heavy_prompt"] = [c1, (r1["choices"][0]["finish_reason"] if c1 == 200 else r1), round(dt1, 2)]
    flood = [[heavy[(i * 7 + j) % len(heavy)] for j in range(1400)] if len(heavy) > 1 else [1000 + i] + [heavy[0]] * 1399 for i in range(nflood)]  # (one heavy token only: behind a token of each prompt's own, so that no two share a block)
    results = {}
    t1 = time.perf_counter()

    def hv(i):
        c, r, dt = post("/v1/completions", body(flood[i], 8))
        results[f"h{i}"] = (c, (r["choices"][0]["finish_reason"] if c == 200 else r), dt, time.perf_counter() - t1)

    def od(i):
        time.sleep(0.3 + 0.1 * i)  # (sent while the flood is under way)
        p = prompts[i % nord]
        c, r, dt = post("/v1/completions", body(p))
        results[f"o{i}"] = (c, text(r), dt, time.perf_counter() - t1, p)

    with cf.ThreadPoolExecutor(nflood + nord + 2) as ex:
        fs = [ex.submit(hv, i) for i in range(nflood)] + [ex.submit(od, i) for i in range(nord)]
        for f in fs:
            f.result()
    wall = time.perf_counter() - t1
    hres = [results[f"h{i}"] for i in range(nflood)]
    ores = [results[f"o{i}"] for i in range(nord)]
    ptok = nflood * 1400 + sum(len(p) // 4 for p in prompts[:nord])
    res["flood_result"] = {"requests": nflood, "served": sum(c == 200 for c, *_ in hres), "codes": sorted({c for c, *_ in hres}), "failed": [(c, str(m)[:160]) for c, m, *_ in hres if c != 200][:3], "seconds": round(wall, 2),
                           "prompt_tokens_per_second": round(nflood * 1400 / wall, 1), "heavy_latency": lat([dt for _, _, dt, _ in hres]),
                           "ordinary_codes": [c for c, *_ in ores], "ordinary_latency_during": lat([dt for _, _, dt, _, _ in ores]),
                           "ordinary_same_text_as_alone": all(t == texts[p][1] for c, t, _, _, p in ores if c == 200 and p in texts), "ordinary_finished_at": [round(e, 2) for _, _, _, e, _ in ores]}
    res["alive_after_flood"] = health() == 200 and srv.poll() is None
    c3, r3, dt3 = post("/v1/completions", body([heavy[i % len(heavy)] for i in range(700)], 8))
    res["heavy_after_flood"] = [c3, (r3["choices"][0]["finish_reason"] if c3 == 200 else r3), round(dt3, 2)]
    after = [post("/v1/completions", body(p)) for p in prompts[:nord]]
    res["after_same"] = all(text(r) == texts[p][1] for (c, r, _), p in zip(after, prompts[:nord]) if c == 200 and p in texts) and all(c == 200 for c, _, _ in after)
    # a prompt of the longest allowed length of one heavy token: served, or refused with a 400 where the reserve could never hold it
    c4, r4, dt4 = post("/v1/completions", body([heavy[0]] * 4000, 8))
    res["longest_heavy_prompt"] = [c4, (r4["choices"][0]["finish_reason"] if c4 == 200 else r4), round(dt4, 2)]
    res["server_alive_end"] = srv.poll() is None
finally:
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except Exception:
        srv.kill()
    log.close()
    lines = [l.strip()[:260] for l in open(os.path.join(out, "server.log")) if "glyd kv:" in l and ("ERROR" in l or "WARNING" in l)]
    res["server_log"] = {"lines": len(lines), "sample": lines[:10]}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
