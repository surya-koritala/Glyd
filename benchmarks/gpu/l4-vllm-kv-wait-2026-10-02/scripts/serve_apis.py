"""vllm serve with kv lossless and several API server processes (--api-server-count N), the API servers optionally given a TMPDIR of their own (the engine's process is started first with the parent's): a prompt that could never fit is refused
with a 400 by every API server on its own, the ones that fit are served.

    serve_apis.py MODEL OUTDIR --cache DIR --heavy FILE [--servers 2] [--tmp-api DIR] [--reserve-2 PERCENT] [--env K=V ...]

The tables are DIR's (made by serve_flood.py). Results: OUTDIR/result.json.
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

args = sys.argv[1:]
model, out = args[0], os.path.abspath(args[1])
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
cache, heavy_file, servers, tmp_api, reserve2 = opt("--cache", ""), opt("--heavy", ""), opt("--servers", 2), opt("--tmp-api", ""), opt("--reserve-2", "")
extra_env = dict(a.split("=", 1) for i, a in enumerate(args) if i and args[i - 1] == "--env")
os.makedirs(out, exist_ok=True)
heavy = json.load(open(heavy_file))["heavy"]
port = int(os.environ.get("PORT", "8014"))
env = dict(os.environ, GLYD_CACHE=cache, HF_HOME=os.path.expanduser("~/hf"), TOKENIZERS_PARALLELISM="false", **extra_env)
for k in ("GLYD_KV_TABLES", "GLYD_KV_SESSION"):
    env.pop(k, None)
if reserve2:
    env["GLYD_KV_RESERVE_2"] = reserve2
if tmp_api:
    os.makedirs(tmp_api, exist_ok=True)
    env["GLYD_TEST_TMP_API"] = tmp_api
launcher = os.path.join(out, "launcher.py")
open(launcher, "w").write('''import os, sys
from vllm.entrypoints.cli import serve as S
from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    tmp = os.environ.pop("GLYD_TEST_TMP_API", "")
    if tmp:  # the API servers are started after the engine's process: they get this TMPDIR, the engine keeps the parent's
        Orig = S.APIServerProcessManager

        class Mgr(Orig):
            def __init__(self, *a, **k):
                os.environ["TMPDIR"] = tmp
                super().__init__(*a, **k)

        S.APIServerProcessManager = Mgr
    sys.argv = ["vllm"] + sys.argv[1:]
    main()
''')
cmd = [sys.executable, launcher, "serve", model, "--port", str(port), "--max-model-len", "4096", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.9", "--max-num-batched-tokens=4096", "--api-server-count", str(servers),
       "--quantization", "glyd", "--additional-config", json.dumps({"glyd": {"fraction": 0, "kv": "lossless"}})]
t0 = time.time()
log = open(os.path.join(out, "server.log"), "w")
srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def post(body, timeout=900):
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600], time.perf_counter() - t


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status
    except Exception:
        return None


res = {"model": model, "api_servers": servers, "tmp_api": tmp_api, "second_reserve_percent": reserve2}
try:
    while health() != 200:
        if srv.poll() is not None:
            raise RuntimeError(f"the server exited {srv.returncode}: see server.log")
        time.sleep(3)
        if time.time() - t0 > 1500:
            raise RuntimeError("the server did not come up in 25 minutes")
    res["start_seconds"] = round(time.time() - t0, 1)
    body = lambda p, n=8: {"model": model, "prompt": p, "max_tokens": n, "temperature": 0}
    c, r, _ = post(body("The capital of Australia is", 16))
    res["ordinary"] = [c, r["choices"][0]["text"] if c == 200 else r]
    res["heavy_700"] = post(body([heavy[i % len(heavy)] for i in range(700)]))[0]
    res["heavy_1400"] = post(body([heavy[i % len(heavy)] for i in range(1400)]))[0]
    long_prompt = [heavy[(i * 7) % len(heavy)] for i in range(4000)]
    c, r, _ = post(body(long_prompt))
    res["heavy_4000"] = [c, r if c != 200 else "served"]
    # many 4,000-token prompts at once, over separate connections: every API server process gets some
    with cf.ThreadPoolExecutor(40) as ex:
        rs = list(ex.map(lambda i: post(body([heavy[(i * 7 + j) % len(heavy)] for j in range(4000)])), range(40)))
    res["many_4000"] = {"codes": sorted({c for c, _, _ in rs}), "n400": sum(c == 400 for c, _, _ in rs), "messages": sorted({re.sub(r"[\d,]+", "N", m)[:200] for c, m, _ in rs if c == 400})}
    res["many_4000"]["rooms"] = sorted({int(re.search(r"against ([\d,]+)\)", m).group(1).replace(",", "")) for c, m, _ in rs if c == 400 and re.search(r"against ([\d,]+)\)", m)})
    # a flood of 1,400-token prompts: all served (they wait for room)
    with cf.ThreadPoolExecutor(24) as ex:
        fl = list(ex.map(lambda i: post(body([heavy[(i * 7 + j) % len(heavy)] for j in range(1400)])), range(12)))
    res["flood_1400"] = sorted({c for c, _, _ in fl})
    res["alive"] = srv.poll() is None and health() == 200
finally:
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except Exception:
        srv.kill()
    log.close()
    text = open(os.path.join(out, "server.log")).read()
    res["access_400_by_process"] = {}
    for m in re.finditer(r"\((\w+) pid=\d+\).*?\"POST /v1/completions HTTP/1\.1\" 400", text):
        res["access_400_by_process"][m.group(1)] = res["access_400_by_process"].get(m.group(1), 0) + 1
    res["engine_second_reserve_units"] = re.findall(r"second reserve [\d.]+ MiB \([\d.]+%, (\d+) units\)", text)[:1]
    res["server_log_glyd"] = [l.strip()[:230] for l in text.splitlines() if "glyd" in l and ("ERROR" in l or "WARNING" in l)][:8]
    res["files_in_tmp"] = [f for d in {"/tmp", tmp_api} if d for f in glob.glob(os.path.join(d, "glyd-kv-*"))]
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
