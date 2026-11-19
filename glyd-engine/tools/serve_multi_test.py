#!/usr/bin/env python3
"""Several clients at once against a running glyd-engine server: each prompt alone first, then all together; a client that leaves midway.

    serve_multi_test.py PORT [--clients N] [--new N] [--words N]

Each client streams /v1/completions (greedy, ignore_eos) and counts its tokens from the usage the server sends. Checked: every stream has all of its tokens, and the clients that went
together read what each read alone (the figure is printed: where it is not all, the first place a stream differs from the one that ran alone). A client that closes its connection after a
few chunks leaves nothing behind (/stats shows no sequence running a moment later). Exit status 0 where every stream was whole and nothing was left running."""
import http.client
import json
import sys
import threading
import time

HOST = "127.0.0.1"
SUBJECTS = ["the sea", "a clock", "winter", "the market", "a library", "the engine", "a garden", "the river", "a letter", "the mountain"]


def prompt(k, words):
    s = SUBJECTS[k % len(SUBJECTS)]
    sentence = f"Write a few lines about {s} and what a person notices there in the early morning, number {k}. "
    return (sentence * (words // 20 + 1))[: words * 6]


def stream(port, body, drop_after=None):
    c = http.client.HTTPConnection(HOST, port, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"status {r.status}: {r.read()[:200]}")
    text, usage, first, chunks = [], None, None, 0
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        o = json.loads(data)
        if o.get("usage"):
            usage = o["usage"]
        for ch in o.get("choices", []):
            if ch.get("text"):
                text.append(ch["text"])
                if first is None:
                    first = time.time() - t0
                chunks += 1
        if drop_after and chunks >= drop_after:
            c.close()
            return "".join(text), None, first, time.time() - t0
    c.close()
    return "".join(text), usage, first, time.time() - t0


def stats(port):
    c = http.client.HTTPConnection(HOST, port, timeout=30)
    c.request("GET", "/stats")
    return json.loads(c.getresponse().read())


def main():
    a = sys.argv[1:]
    port = int(a[0])
    opt = lambda n, d: int(a[a.index(n) + 1]) if n in a else d
    clients, new, words = opt("--clients", 4), opt("--new", 128), opt("--words", 120)
    body = lambda k: {"model": "m", "prompt": prompt(k, words), "max_tokens": new, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    s0 = stats(port)
    alone = [stream(port, body(k)) for k in range(clients)]
    out = [None] * clients

    def go(k):
        out[k] = stream(port, body(k))

    ts = [threading.Thread(target=go, args=(k,)) for k in range(clients)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    ok = True
    same = 0
    for k in range(clients):
        (ta, ua, fa, _), (tb, ub, fb, _) = alone[k], out[k]
        whole = ua and ub and ua["completion_tokens"] == new and ub["completion_tokens"] == new
        ok &= bool(whole)
        if ta == tb:
            same += 1
            note = "same text as alone"
        else:
            d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            note = f"differs from alone at character {d} of {len(ta)}"
        print(f"client {k}: {ub['completion_tokens'] if ub else None} of {new} tokens, first text {fb:.2f} s, {note}")
    print(f"{same} of {clients} read what they read alone; {clients} clients together {wall:.1f} s (one at a time {sum(x[3] for x in alone):.1f} s)")
    s1 = stats(port)
    print("stats:", {k: s1[k] for k in ("prompt_tokens", "prefix_hit_tokens", "preemptions", "steps", "rows")}, "since the start:", {k: s1[k] - s0[k] for k in ("preemptions", "steps", "rows")})
    # a client that leaves after a few chunks
    stream(port, body(0) | {"max_tokens": 2000}, drop_after=3)
    for _ in range(100):
        time.sleep(0.1)
        if stats(port)["running"] == 0:
            break
    left = stats(port)
    print(f"after a client left midway: {left['running']} running, {left['waiting']} waiting")
    ok &= left["running"] == 0 and left["waiting"] == 0
    print("ok" if ok else "NOT OK")
    sys.exit(0 if ok else 1)


main()
