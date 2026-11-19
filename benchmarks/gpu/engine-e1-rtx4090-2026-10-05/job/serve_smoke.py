"""A running `glyd-engine serve`'s API, request by request: python serve_smoke.py http://127.0.0.1:8765 [long-text.txt]. Prints each result; exits 1 if a check fails."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/")
KEY = os.environ.get("API_KEY")  # the server's API key, where it has one: sent with every request that needs it
LONG = open(sys.argv[2]).read() if len(sys.argv) > 2 else "The quick brown fox jumps over the lazy dog. " * 300
bad = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (" -- " + str(detail) if detail != "" else ""))
    if not ok:
        bad.append(name)


def headers(extra=None):
    h = dict(extra or {})
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def get(path, key=True):
    req = urllib.request.Request(BASE + path, headers=headers() if key else {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(path, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=headers({"Content-Type": "application/json"}))
    t = time.time()
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t


def stream(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(dict(body, stream=True)).encode(), headers=headers({"Content-Type": "application/json"}))
    t = time.time()
    first, events = None, []
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if first is None:
                first = time.time() - t
            events.append(p if p == "[DONE]" else json.loads(p))
    return events, time.time() - t, first


s, b = get("/health", key=False)
check("GET /health (no key needed)", s == 200 and json.loads(b).get("status") == "ok", b)
if KEY:
    s, b = get("/v1/models", key=False)
    check("the API refuses a request with no key", s == 401, b[:120])
s, b = get("/v1/models")
m = json.loads(b)
check("GET /v1/models", s == 200 and m["data"], [d["id"] for d in m["data"]])
MODEL = m["data"][0]["id"]
s, b = get("/", key=False)
check("GET / is the chat page", s == 200 and b"<html" in b.lower(), f"{len(b)} bytes")

NT = {"enable_thinking": False}
q = {"model": MODEL, "messages": [{"role": "user", "content": "What is 2+2? Answer with one word."}], "max_tokens": 32, "temperature": 0, "chat_template_kwargs": NT}
s, d, t = post("/v1/chat/completions", q)
text = d["choices"][0]["message"]["content"]
check("chat, greedy", s == 200 and "four" in text.lower() or "4" in text, f"{text!r} in {t:.2f} s, {d['usage']}, {d['choices'][0]['finish_reason']}")
ev, t2, first = stream("/v1/chat/completions", q)
deltas = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e != "[DONE]" and e["choices"])
check("stream = non-stream text", deltas == text, f"{deltas!r}; first event {first:.2f} s, {len(ev)} events, last {ev[-1]}")
q2 = dict(q, max_tokens=4, messages=[{"role": "user", "content": "Count from one to twenty in words."}])
s, d, t = post("/v1/chat/completions", q2)
check("max_tokens ends with length", d["choices"][0]["finish_reason"] == "length" and d["usage"]["completion_tokens"] == 4, d["choices"][0]["finish_reason"])
q3 = dict(q, temperature=1.5, top_p=1.0, top_k=0, seed=7, max_tokens=24, messages=[{"role": "user", "content": "Write one sentence about the sea."}])  # (no nucleus: the model's own defaults would cut a sentence it is sure of to its one choice)
a = post("/v1/chat/completions", q3)[1]["choices"][0]["message"]["content"]
b2 = post("/v1/chat/completions", q3)[1]["choices"][0]["message"]["content"]
c = post("/v1/chat/completions", dict(q3, seed=8))[1]["choices"][0]["message"]["content"]
check("same seed, same text", a == b2, a)
check("another seed, another text", a != c, c)
q4 = dict(q, max_tokens=40, stop=["three"], messages=[{"role": "user", "content": "Count from one to ten in words, separated by commas."}])
s, d, t = post("/v1/chat/completions", q4)
tx = d["choices"][0]["message"]["content"]
check("stop string cuts the text", "three" not in tx and d["choices"][0]["finish_reason"] == "stop", f"{tx!r}")
s, d, t = post("/v1/completions", {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
check("completions", s == 200 and d["choices"][0]["text"], f"{d['choices'][0]['text']!r}")

# a long prompt twice: the second one's prefill is the part of it the first left in the cache
lp = {"model": MODEL, "messages": [{"role": "user", "content": LONG[:9000] + "\nHow many sentences were there? Answer with one number."}], "max_tokens": 1, "temperature": 0, "chat_template_kwargs": NT}
s1, d1, t1 = post("/v1/chat/completions", lp)
s2, d2, t2 = post("/v1/chat/completions", lp)
check("a long prompt twice (prefix reuse)", s1 == 200 and s2 == 200 and t2 < t1 * 0.6, f"{d1['usage']['prompt_tokens']} tokens: {t1:.3f} s, then {t2:.3f} s")
# a conversation grown by a turn
conv = [{"role": "user", "content": LONG[:6000] + "\nSummarize the above in one sentence."}]
s, r1, t1 = post("/v1/chat/completions", dict(lp, messages=conv, max_tokens=24))
conv += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]}, {"role": "user", "content": "Now in three words."}]
s, r2, t2 = post("/v1/chat/completions", dict(lp, messages=conv, max_tokens=12))
check("a conversation's next turn", s == 200, f"first turn {r1['usage']['prompt_tokens']} tokens {t1:.2f} s; second {r2['usage']['prompt_tokens']} tokens {t2:.2f} s: {r2['choices'][0]['message']['content']!r}")

s, d, t = post("/v1/chat/completions", dict(q, messages=[{"role": "user", "content": "x " * 6000}]))
check("a prompt longer than the context is refused", s == 400 and "context" in json.dumps(d), d)
s, d, t = post("/v1/chat/completions", None, raw=b"{not json")
check("bad JSON is refused", s == 400, d)
for what, extra in (("several choices", {"n": 2}), ("tool calls", {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}), ("log-probabilities", {"logprobs": True}), ("a JSON object", {"response_format": {"type": "json_object"}})):
    s, d, t = post("/v1/chat/completions", dict(q, **extra))
    check(f"a request for {what} is refused, not ignored", s == 400 and "not supported" in json.dumps(d), d.get("error", {}).get("message", d) if isinstance(d, dict) else d)

# two at once: one engine, so one after the other
res = []
th = [threading.Thread(target=lambda: res.append(post("/v1/chat/completions", dict(q3, max_tokens=16))[0])) for _ in range(2)]
[x.start() for x in th]
[x.join() for x in th]
check("two requests at once", res == [200, 200], res)
print("FAILED: " + ", ".join(bad) if bad else "all ok")
sys.exit(1 if bad else 0)
