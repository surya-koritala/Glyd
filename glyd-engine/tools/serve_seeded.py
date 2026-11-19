#!/usr/bin/env python3
"""Seeded requests with the server's own sampling defaults (the model's generation_config: temperature, top-k, top-p), a line for each: the seed and a hash of the text it gave.

    serve_seeded.py PORT [--n N] [--new N] [--mode one|many]

--mode one asks the requests one at a time, many four at a time (each on a server just started: a prompt that begins as one the server has answered finds its keys and values in the cache, and those were made in other pieces, so a
second pass on the same server can part from the first). The same lines from two builds of the server say that a draw is the same draw (the sampler's candidates from the GPU against the whole row on the host); the same lines for
one and many say that a reply does not depend on who else is decoded with it."""
import hashlib
import http.client
import json
import sys
import threading

a = sys.argv[1:]
port = int(a[0])
opt = lambda n, d: int(a[a.index(n) + 1]) if n in a else d
N, NEW = opt("--n", 8), opt("--new", 48)
MODE = a[a.index("--mode") + 1] if "--mode" in a else "one"
PROMPTS = ["Write two sentences about the sea.", "Name three uses of a hammer.", "Explain why the sky is blue in one paragraph.", "Give a short recipe for tea.",
           "Describe a small town in the morning.", "List four prime numbers and say why they are prime.", "Say something kind to a friend.", "What does a compiler do?"]


def ask(k):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    body = {"model": "m", "messages": [{"role": "user", "content": PROMPTS[k % len(PROMPTS)]}], "max_tokens": NEW, "seed": 100 + k, "chat_template_kwargs": {"enable_thinking": False}}
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    t = json.loads(r.read())
    c.close()
    return t["choices"][0]["message"]["content"]


texts = [None] * N
if MODE == "one":
    for k in range(N):
        texts[k] = ask(k)
else:
    ts = [threading.Thread(target=lambda k=k: texts.__setitem__(k, ask(k))) for k in range(N)]
    for i in range(0, N, 4):
        grp = ts[i : i + 4]
        [t.start() for t in grp]
        [t.join() for t in grp]
h = lambda s: hashlib.sha1(s.encode()).hexdigest()[:12]
for k in range(N):
    print(f"{100 + k}\t{h(texts[k])}")
print(f"{N} texts, {MODE} at a time" if MODE == "one" else f"{N} texts, four at a time")
