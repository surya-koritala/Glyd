#!/usr/bin/env python3
"""Many short requests that end and start, on a server run with GLYD_ENGINE_VERIFY=1 and little room for them (its pages are reused again and again, by other sequences and after being kept for a later prompt):

    serve_churn.py PORT [--requests N] [--clients C] [--seed S]

Each request is a slice of a text (--text FILE: the words of the file from a random place on; else random words) of 40 to 700 tokens (some begin as an earlier one, so the cache of finished prompts is used too) and 8 to 48 tokens out. At the end /stats gives the values the cache returned
compared with a bf16 copy of what was written, and how many differed: the exit status is 0 only where none did and every request got its tokens."""
import http.client
import json
import random
import sys
import threading

a = sys.argv[1:]
port = int(a[0])
opt = lambda n, d: int(a[a.index(n) + 1]) if n in a else d
N, C, rng = opt("--requests", 400), opt("--clients", 4), random.Random(opt("--seed", 1))
FILE = a[a.index("--text") + 1] if "--text" in a else None
WORDS = open(FILE, errors="replace").read().split() if FILE else [f"w{rng.randrange(1, 40000)}x{rng.randrange(1, 900)}" for _ in range(3000)]


def text(n, base=None):
    if FILE:
        at = rng.randrange(0, max(1, len(WORDS) - n))
        return (base or []) + WORDS[at : at + n]
    return (base or []) + [rng.choice(WORDS) for _ in range(n)]


bases = [text(120) for _ in range(6)]
jobs = []
for i in range(N):
    n = rng.randrange(30, 500)
    base = rng.choice(bases)[: rng.randrange(0, 120)] if rng.random() < 0.4 else None
    jobs.append((" ".join(text(n, base)), rng.randrange(8, 49)))
bad, done, lock = [], [0], threading.Lock()


def worker(k):
    for i in range(k, N, C):
        prompt, new = jobs[i]
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        c.request("POST", "/v1/completions", json.dumps({"model": "m", "prompt": prompt, "max_tokens": new, "temperature": 0, "ignore_eos": True}), {"Content-Type": "application/json"})
        r = c.getresponse()
        body = json.loads(r.read())
        ok = r.status == 200 and body["usage"]["completion_tokens"] == new
        with lock:
            done[0] += 1
            if not ok:
                bad.append((i, r.status, str(body)[:200]))
        c.close()


ts = [threading.Thread(target=worker, args=(k,)) for k in range(C)]
[t.start() for t in ts]
[t.join() for t in ts]
c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
c.request("GET", "/stats")
s = json.loads(c.getresponse().read())
print(f"{done[0]} requests, {len(bad)} wrong; prompt tokens {s['prompt_tokens']}, found in the cache {s['prefix_hit_tokens']}, sequences sent back to wait {s['preemptions']}")
print(f"values of the cache checked against the bf16 copy: {s['values_checked']}, differing: {s['values_differ']}")
sys.exit(0 if not bad and s["values_checked"] > 0 and s["values_differ"] == 0 else 1)
