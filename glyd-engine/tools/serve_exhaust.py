#!/usr/bin/env python3
"""A server whose cache has too little extra memory for some requests (GLYD_ENGINE_RESERVE_1_UNITS and GLYD_ENGINE_RESERVE_2_UNITS small), what a client sees:

    serve_exhaust.py PORT

A request that the cache cannot hold exactly, even alone, ends with an error in words (HTTP 400 with the reason, or an error event where its stream had begun); a request it can hold is served whole; the server is not
left unfit by the one before (the same small request after the failures gets the same text as before them); /stats counts what the cache lost and what was ended for it. Exit status 0 where all of that is so."""
import http.client
import json
import random
import sys

a = sys.argv[1:]
port = int(a[0])
rng = random.Random(5)
words = lambda n: " ".join(f"w{rng.randrange(1, 40000)}x{rng.randrange(1, 900)}" for _ in range(n))


def post(body, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/completions", json.dumps(body | ({"stream": True} if stream else {})), {"Content-Type": "application/json"})
    r = c.getresponse()
    text = r.read().decode(errors="replace")
    c.close()
    return r.status, text


def stats():
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", "/stats")
    return json.loads(c.getresponse().read())


# a short sentence of ordinary words (a cache whose extra memory is small still holds it), and texts of made-up words, which fill the extra memory: several sizes, one of them must be more than the cache can hold
smalls = ["The capital of France is", "Once upon a time", "Hello"]
bigs = [words(n) for n in (40, 80, 160, 240)]
body = lambda p, n=24: {"model": "m", "prompt": p, "max_tokens": n, "temperature": 0, "ignore_eos": True}
ok = True


def check(what, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("ok   " if cond else "FAIL ") + what + (f" -- {detail}" if detail else ""))


small = before = None
for s in smalls:
    s0, t0 = post(body(s))
    print(f"     a short request {s!r}: {s0}, {t0[:100]!r}")
    if s0 == 200:
        small, before = s, json.loads(t0)["choices"][0]["text"]
        break
check("a short request is served whole", before is not None and json.loads(t0)["usage"]["completion_tokens"] == 24, f"{s0}")
failed = None
for k, big in enumerate(bigs):
    s1, t1 = post(body(big))
    print(f"     a request of {len(big.split())} made-up words (try {k + 1}): {s1}, {t1[:100]!r}")
    if s1 != 200:
        failed = (big, s1, t1)
        break
check("a request the cache cannot hold ends with an error in words", failed is not None and "cannot hold this request" in failed[2], f"{failed[1]}: {failed[2][:220]}" if failed else "none of the tries was too much for the cache")
if failed:
    s2, t2 = post(body(failed[0]), stream=True)
    check("the same, asked as a stream, says so too", "cannot hold this request" in t2, f"{s2}: {t2[:160]!r}")
s3, t3 = post(body(small))
check("the short request after the failures gets the text it got before them", s3 == 200 and json.loads(t3)["choices"][0]["text"] == before)
st = stats()
check("the counts say what happened", st.get("ended_for_loss", 0) >= 1 and st.get("running") == 0 and st.get("waiting") == 0, f"ended_for_loss {st.get('ended_for_loss')}, taken_back {st.get('taken_back')}, lost_blocks {st.get('lost_blocks')}, running {st.get('running')}, waiting {st.get('waiting')}")
print("ok" if ok else "NOT OK")
sys.exit(0 if ok else 1)
