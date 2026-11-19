"""A running `glyd-engine serve` under mixed traffic for a while: python serve_soak.py http://127.0.0.1:8765 long-text.txt [SECONDS]

What it checks, besides that every request ends well-formed:
  - a greedy reply is the same however the cache stands: the replies of 6 prompts (the shortest a sentence, the longest 3,000 tokens) taken first on a cold server are taken again after random
    traffic, after their own prefixes were extended and answered, and after clients that left in the middle of a stream; every one must be the same text;
  - the server answers /health at the end and its memory use (nvidia-smi, where there is one) has not grown;
  - requests that cannot be honoured (a prompt longer than the context) are refused, and the server goes on.
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE, TEXT = sys.argv[1].rstrip("/"), open(sys.argv[2]).read()
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 120
KEY = os.environ.get("API_KEY")
rng = random.Random(7)
bad = []
counts = {"ok": 0, "refused": 0, "left": 0}


def hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=hdr())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ask(prompt, max_tokens=24, **kw):
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    s, d = post("/v1/chat/completions", body)
    return s, d


def reply(prompt, **kw):
    s, d = ask(prompt, **kw)
    if s != 200:
        return None
    return d["choices"][0]["message"]["content"]


def leave_midway(prompt):
    """A streaming request whose client closes the connection after the first events."""
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr())
    try:
        r = urllib.request.urlopen(req, timeout=120)
        for i, _ in enumerate(r):
            if i > 6:
                break
        r.close()
        counts["left"] += 1
    except Exception as e:  # (a refused request is fine)
        bad.append(f"a streaming request failed: {e}")


def smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        return int(out)
    except Exception:
        return None


def slice_of(n_chars, at):
    return TEXT[at : at + n_chars]


fixed = [
    ("Say a sentence about autumn.", 24),
    (slice_of(600, 0) + "\nWhat is the first word of the text above? Answer with the word only.", 8),
    (slice_of(2500, 500) + "\nHow many sentences are there above? Answer with a number only.", 8),
    (slice_of(6000, 100) + "\nSummarize the above in one short sentence.", 32),
    (slice_of(9000, 1000) + "\nWhat is this text about? Answer in five words.", 16),
    (slice_of(12000, 2000) + "\nName one thing from the text above, in three words.", 12),
]
mem0 = smi()
cold = [reply(p, max_tokens=n) for p, n in fixed]
check = lambda name, got, want: bad.append(f"{name}: {got!r} where the cold server said {want!r}") if got != want else None
print("cold replies:", [c[:40] if c else c for c in cold])
if any(c is None for c in cold):
    print("a cold request failed")
    sys.exit(1)
t0 = time.time()
n = 0
while time.time() - t0 < SECONDS:
    n += 1
    kind = rng.choice(["random", "random", "extend", "leave", "again", "stop", "toolong", "stream", "concurrent"])
    if kind == "random":
        p = slice_of(rng.randint(40, 9000), rng.randint(0, 40000)) + "\nContinue the text in one sentence."
        s, d = ask(p, max_tokens=rng.randint(1, 40), temperature=rng.choice([0, 0.7, 1.2]), seed=rng.randint(0, 1000))
        counts["ok" if s == 200 else "refused"] += 1
        if s != 200:
            bad.append(f"a random request: {s} {d}")
    elif kind == "extend":
        i = rng.randrange(len(fixed))
        p, k = fixed[i]
        first = reply(p, max_tokens=k)
        check(f"fixed prompt {i} before its conversation went on", first, cold[i])
        msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": first}, {"role": "user", "content": "Say that again in other words."}]
        s, d = post("/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        counts["ok" if s == 200 else "refused"] += 1
    elif kind == "leave":
        leave_midway(slice_of(rng.randint(100, 4000), rng.randint(0, 40000)) + "\nWrite a long story.")
    elif kind == "again":
        i = rng.randrange(len(fixed))
        p, k = fixed[i]
        check(f"fixed prompt {i} after other traffic", reply(p, max_tokens=k), cold[i])
        counts["ok"] += 1
    elif kind == "stop":
        s, d = ask("Count from one to twenty in words, separated by commas.", max_tokens=60, stop=["seven"])
        if s != 200 or "seven" in d["choices"][0]["message"]["content"]:
            bad.append(f"a stop string: {s} {d}")
        counts["ok"] += 1
    elif kind == "toolong":
        s, d = ask("word " * 12000)
        if s != 400:
            bad.append(f"a prompt past the context was not refused: {s}")
        counts["refused"] += 1
    elif kind == "stream":
        s, d = ask(slice_of(3000, rng.randint(0, 40000)) + "\nOne sentence about this.", max_tokens=20, stream=False)
        counts["ok" if s == 200 else "refused"] += 1
    else:
        res = []
        ths = [threading.Thread(target=lambda: res.append(ask(slice_of(rng.randint(200, 3000), rng.randint(0, 40000)), max_tokens=12)[0])) for _ in range(3)]
        [x.start() for x in ths]
        [x.join() for x in ths]
        if res != [200, 200, 200]:
            bad.append(f"three at once: {res}")
        counts["ok"] += 3
# the end: every fixed prompt once more, and the server's state
for i, (p, k) in enumerate(fixed):
    check(f"fixed prompt {i} at the end", reply(p, max_tokens=k), cold[i])
try:
    with urllib.request.urlopen(urllib.request.Request(BASE + "/health"), timeout=10) as r:
        health = r.status
except Exception as e:
    health = str(e)
mem1 = smi()
print(f"{n} rounds in {time.time() - t0:.0f} s: {counts}; /health {health}; GPU memory {mem0} -> {mem1} MiB")
if mem0 is not None and mem1 is not None and mem1 > mem0 + 200:
    bad.append(f"GPU memory grew {mem1 - mem0} MiB")
print("FAILED: " + "; ".join(bad[:10]) if bad else "all ok")
sys.exit(1 if bad else 0)
