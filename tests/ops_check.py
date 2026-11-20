"""The operational endpoints of a served model, on a GPU: /health, /ready and /metrics (Prometheus's text format, the families an operator needs, filled by real requests), and a graceful drain: SIGTERM while a request
is streaming stops the server taking requests (/ready 503, a new request 503), lets that request end whole, and the process exits 0 inside the bound.

    python3 tests/ops_check.py ENGINE MODEL [PORT]

Run by the `gpu` CI job (.github/workflows/gpu-tests.yml) on the model it serves for the client checks. Standard library only. Exit 0: all of it held.
"""

import json
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ENGINE, MODEL = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8351
BASE = f"http://127.0.0.1:{PORT}"
DRAIN_BOUND = 60  # seconds the server is told to wait (--drain-timeout)


def fail(msg, proc=None):
    print("FAIL:", msg, flush=True)
    if proc is not None and proc.poll() is None:
        proc.kill()
    try:
        print(open("/tmp/ops-serve.log").read()[-4000:])
    except OSError:
        pass
    sys.exit(1)


def get(path, timeout=10):
    """(status, body) of a GET; the status of an error answer is its own."""
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def chat(max_tokens, stream, prompt="Write a long story about a lighthouse keeper and a ship."):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": stream, "temperature": 0}
    return urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})


def samples(text):
    """{metric line name with labels: float} of a scrape, checking each line is Prometheus's text format."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r'([a-z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?) (\S+)', line)
        assert m, f"not a sample line: {line!r}"
        out[m.group(1)] = float(m.group(2))
    return out


def histogram(s, name):
    """(count, sum) of a histogram, after checking its buckets are cumulative and end at the count."""
    buckets = [(k, v) for k, v in s.items() if k.startswith(name + "_bucket")]
    assert buckets, f"{name} has no buckets"
    vals = [v for _, v in buckets]
    assert vals == sorted(vals), f"{name}'s buckets are not cumulative: {vals}"
    count = s[name + "_count"]
    assert vals[-1] == count and 'le="+Inf"' in buckets[-1][0], f"{name}: +Inf is {vals[-1]}, count {count}"
    return count, s[name + "_sum"]


log = open("/tmp/ops-serve.log", "w")
proc = subprocess.Popen([ENGINE, "serve", MODEL, "--port", str(PORT), "--ctx", "4096", "--drain-timeout", str(DRAIN_BOUND)], stdout=log, stderr=subprocess.STDOUT)

# ---- alive, then ready once the model is loaded ----
t0 = time.time()
while True:
    try:
        if get("/ready", 3)[0] == 200:
            break
    except OSError:
        pass
    if proc.poll() is not None:
        fail("the server ended before it was ready", proc)
    if time.time() - t0 > 1800:
        fail("the server was not ready in 30 minutes", proc)
    time.sleep(2)
print(f"ready after {time.time() - t0:.0f} s")
assert get("/health")[0] == 200, "/health"
st, body = get("/ready")
assert (st, json.loads(body)["status"]) == (200, "ready"), (st, body)

# ---- the metrics, after one request whole and one streamed ----
before = samples(get("/metrics")[1])
with urllib.request.urlopen(chat(32, False), timeout=300) as r:
    assert r.status == 200
    json.load(r)
tokens = 0
with urllib.request.urlopen(chat(64, True), timeout=300) as r:
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data: {"):
            tokens += 1
assert tokens > 5, "the streamed reply had no tokens"
st, text = get("/metrics")
assert st == 200
s = samples(text)
for name in ["glyd_build_info", "glyd_uptime_seconds", "glyd_ready", "glyd_draining", "glyd_requests_in_flight", "glyd_queue_depth", "glyd_sequences_running"]:
    assert any(k.startswith(name) for k in s), f"{name} missing"
assert s["glyd_ready"] == 1 and s["glyd_draining"] == 0
assert s['glyd_requests_total{api="chat_completions"}'] - before['glyd_requests_total{api="chat_completions"}'] >= 2, "the two requests were not counted"
assert s["glyd_generated_tokens_total"] - before["glyd_generated_tokens_total"] >= 20, "tokens were not counted"
assert s["glyd_prompt_tokens_total"] > before["glyd_prompt_tokens_total"], "prompt tokens were not counted"
ttft, tpot = histogram(s, "glyd_time_to_first_token_seconds"), histogram(s, "glyd_time_per_output_token_seconds")
histogram(s, "glyd_request_duration_seconds")
assert ttft[0] >= 2 and 0 < ttft[1] / ttft[0] < 120, f"first-token histogram {ttft}"
assert tpot[0] >= 20 and 0 < tpot[1] / tpot[0] < 5, f"per-token histogram {tpot}"
# (the KV cache's pages are the scheduler's: a lossy level's engine has none and says nothing of them; where they are told they must make sense)
kv = s.get("glyd_kv_cache_pages")
assert kv is None or (kv > 0 and 0 <= s["glyd_kv_cache_usage_ratio"] <= 1), "the KV cache's pages"
gpu = [v for k, v in s.items() if k.startswith("glyd_gpu_memory_used_bytes")]
assert gpu and max(gpu) > 1 << 28, f"the GPU's memory in use: {gpu}"
print(f"metrics ok: first token {1000 * ttft[1] / ttft[0]:.0f} ms mean, {1000 * tpot[1] / tpot[0]:.1f} ms a token, KV {"%.3f used" % s["glyd_kv_cache_usage_ratio"] if kv else "not told (no scheduler at this level)"}, GPU {max(gpu) / 2**30:.1f} GiB")

# ---- the drain: SIGTERM while a reply is streaming ----
result = {}


def stream_long():
    chunks, done, finish = 0, False, None
    try:
        with urllib.request.urlopen(chat(400, True, "Count from 1 to 300, one number to a line, and nothing else."), timeout=300) as r:
            for raw in r:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: {"):
                    chunks += 1
                    if chunks == 3:
                        result["started"] = True
                    ch = json.loads(line[6:])["choices"]
                    if ch and ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
    result.update(chunks=chunks, done=done, finish=finish)


th = threading.Thread(target=stream_long)
th.start()
t = time.time()
while "started" not in result:
    if time.time() - t > 120 or "error" in result:
        fail(f"the long reply did not start: {result}", proc)
    time.sleep(0.05)
proc.send_signal(signal.SIGTERM)
sigterm_at = time.time()
# readiness turns 503 (draining) at once; the process is alive; a new request is refused; the metrics say so; the reply in flight is still coming
for _ in range(100):
    st, body = get("/ready", 3)
    if st == 503:
        break
    time.sleep(0.02)
assert (st, json.loads(body).get("status")) == (503, "draining"), f"/ready while draining: {st} {body}"
assert get("/health", 3)[0] == 200, "/health while draining"
try:
    urllib.request.urlopen(chat(8, False), timeout=10)
    fail("a request was taken while draining", proc)
except urllib.error.HTTPError as e:
    assert e.code == 503 and e.headers.get("Retry-After"), (e.code, dict(e.headers))
s = samples(get("/metrics", 3)[1])
assert s["glyd_draining"] == 1 and s["glyd_ready"] == 0, "the metrics did not say draining"
assert s["glyd_requests_in_flight"] >= 1, "the long reply was over before the drain was looked at: it proves nothing; lengthen it"
print("draining: /ready 503, new request 503, /health 200, in flight", int(s["glyd_requests_in_flight"]))
# the reply that was in flight ends whole, and the process exits 0 inside the bound
th.join(DRAIN_BOUND)
assert not th.is_alive(), "the reply in flight did not end"
assert "error" not in result and result["done"] and result["finish"] in ("stop", "length"), f"the reply in flight was cut: {result}"
try:
    code = proc.wait(DRAIN_BOUND)
except subprocess.TimeoutExpired:
    fail("the server did not exit within the bound after SIGTERM", proc)
log.flush()
text = open("/tmp/ops-serve.log").read()
assert code == 0, f"exit status {code}\n{text[-2000:]}"
assert "drained: every request ended" in text, text[-2000:]
print(f"drained: the reply in flight ended whole ({result['chunks']} chunks, {result['finish']}), exit 0 {time.time() - sigterm_at:.1f} s after SIGTERM")
