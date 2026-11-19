"""vllm serve on a model with kv lossless (tables from an earlier start's cache) or stock: ordinary requests at a steady rate for a while, with bursts of prompts of the model's heavy tokens among them; every request's
status and latency, the ordinary ones' latency in the bursts and out of them.

    serve_mixed.py MODEL OUTDIR --heavy FILE [--stock] [--seconds 120] [--rate 3] [--burst 12] [--every 30] [--cache DIR]

kv: GLYD_CACHE is DIR (the tables are there: made by serve_flood.py). Results: OUTDIR/result.json.
"""
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

args = sys.argv[1:]
model, out = args[0], os.path.abspath(args[1])
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
stock = "--stock" in args
heavy = json.load(open(opt("--heavy", "")))["heavy"]
seconds, rate, burst, every = opt("--seconds", 120), opt("--rate", 3.0), opt("--burst", 12), opt("--every", 30)
cache = opt("--cache", "")
os.makedirs(out, exist_ok=True)
env = dict(os.environ, GLYD_CACHE=cache, HF_HOME=os.path.expanduser("~/hf"), TOKENIZERS_PARALLELISM="false")
env.pop("GLYD_KV_TABLES", None)
port = int(os.environ.get("PORT", "8013"))
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port), "--max-model-len", "4096", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9", "--max-num-batched-tokens=4096"]
if not stock:
    cmd += ["--quantization", "glyd", "--additional-config", json.dumps({"glyd": {"fraction": 0, "kv": "lossless"}})]
t0 = time.time()
log = open(os.path.join(out, "server.log"), "w")
srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def post(body, timeout=900):
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status, time.perf_counter() - t
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t
    except Exception as e:
        return type(e).__name__, time.perf_counter() - t


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status
    except Exception:
        return None


prompts = ["The capital of Australia is", "The history of data compression began", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "In quantum mechanics, the uncertainty principle states that",
           "Once upon a time, in a village at the edge of a great forest,", "SELECT name, COUNT(*) FROM orders JOIN customers ON"]
res = {"model": model, "stock": stock, "seconds": seconds, "rate": rate, "burst": burst, "every": every}
try:
    while health() != 200:
        if srv.poll() is not None:
            raise RuntimeError(f"the server exited {srv.returncode}: see server.log")
        time.sleep(3)
        if time.time() - t0 > 1800:
            raise RuntimeError("the server did not come up in 30 minutes")
    res["start_seconds"] = round(time.time() - t0, 1)
    for p in prompts:  # (warm)
        post({"model": model, "prompt": p, "max_tokens": 32, "temperature": 0})
    ordinary, heavies, lock = [], [], threading.Lock()
    start = time.perf_counter()
    bursts = []  # (start, end) of each burst: the first request's start to the last one's end

    def ord_req(i):
        s = time.perf_counter() - start
        c, dt = post({"model": model, "prompt": prompts[i % len(prompts)], "max_tokens": 32, "temperature": 0})
        with lock:
            ordinary.append((s, c, dt))

    def burst_run(k):
        s = time.perf_counter() - start
        with cf.ThreadPoolExecutor(burst) as ex:
            fs = [ex.submit(post, {"model": model, "prompt": [heavy[(i * 7 + j + k) % len(heavy)] for j in range(1400)], "max_tokens": 8, "temperature": 0}) for i in range(burst)]
            rs = [f.result() for f in fs]
        e = time.perf_counter() - start
        with lock:
            heavies.extend((s, c, dt) for c, dt in rs)
            bursts.append((s, e))

    threads, i, k = [], 0, 0
    next_burst = 10.0
    while time.perf_counter() - start < seconds:
        now = time.perf_counter() - start
        if now >= next_burst:
            th = threading.Thread(target=burst_run, args=(k,))
            th.start()
            threads.append(th)
            k += 1
            next_burst += every
        th = threading.Thread(target=ord_req, args=(i,))
        th.start()
        threads.append(th)
        i += 1
        time.sleep(1.0 / rate)
    for th in threads:
        th.join()
    in_burst = lambda s: any(b0 - 0.0 <= s <= b1 for b0, b1 in bursts)
    ob, oq = [dt for s, c, dt in ordinary if in_burst(s) and c == 200], [dt for s, c, dt in ordinary if not in_burst(s) and c == 200]
    pct = lambda xs, q: round(sorted(xs)[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None
    res["ordinary"] = {"requests": len(ordinary), "codes": {str(c): sum(1 for _, x, _ in ordinary if x == c) for c in sorted({x for _, x, _ in ordinary}, key=str)},
                       "quiet": {"n": len(oq), "median": pct(oq, 0.5), "p95": pct(oq, 0.95), "max": round(max(oq), 3) if oq else None},
                       "in_a_burst": {"n": len(ob), "median": pct(ob, 0.5), "p95": pct(ob, 0.95), "max": round(max(ob), 3) if ob else None}}
    res["heavy"] = {"requests": len(heavies), "codes": {str(c): sum(1 for _, x, _ in heavies if x == c) for c in sorted({x for _, x, _ in heavies}, key=str)},
                    "latency": {"median": pct([dt for _, c, dt in heavies if c == 200], 0.5), "max": round(max((dt for _, c, dt in heavies if c == 200), default=0), 3)}, "bursts": [[round(a, 1), round(b, 1)] for a, b in bursts]}
    res["server_alive_end"] = srv.poll() is None and health() == 200
finally:
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except Exception:
        srv.kill()
    log.close()
    lines = [l.strip()[:240] for l in open(os.path.join(out, "server.log")) if "glyd kv:" in l and ("ERROR" in l or "WARNING" in l)]
    res["server_log"] = {"lines": len(lines), "sample": lines[:6]}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
