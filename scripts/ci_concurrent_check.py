#!/usr/bin/env python3
"""Several users at once against a served model (the GPU job's check of the scheduler, issue #138).

Asks each prompt alone, one after another (greedy, the reference), then all of them at once, and compares their token ids (log-probabilities with
`return_tokens_as_token_ids`): with --same every reply at once must be the reply alone, token for token (GLYD_ENGINE_BATCH_PREFILL=0); otherwise at least
--least of them must be, and none may fail or come back with no token (prompts run together may take another product route, whose last bits can change a
later greedy token). A reply's text is not what is compared: a random model's tokens can all be ones its tokenizer spells as nothing, alone as at once. It also checks that the server says it runs
that many at once (/v1/glyd/info, limits.concurrency) and that its decode steps had several rows (/stats, steps_by_rows). The times are printed, not judged: a speed bar
needs a GPU of its own (glyd-engine/docs/bench.md).

usage: ci_concurrent_check.py URL [--users 4] [--same] [--least 3] [--tokens 48]
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "What is 17 + 25? Answer with the number only.",
    "Name the three primary colors, separated by commas.",
    "Write one sentence about the moon.",
    "List the first five prime numbers.",
    "Translate 'good morning' into French and Spanish.",
    "What is the capital of Japan? One word.",
    "Count from one to ten in words.",
    "Give a synonym for 'quick'.",
]


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def ask(url, model, prompt, tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 0,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    c = out["choices"][0]
    ids = [x["token"] for x in (c.get("logprobs") or {}).get("content") or []]
    return c["message"]["content"] or "", out["usage"]["completion_tokens"], time.monotonic() - t, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--same", action="store_true")
    ap.add_argument("--least", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    fails = 0
    model = get(url + "/v1/models")["data"][0]["id"]
    info = get(url + "/v1/glyd/info")
    users = info.get("limits", {}).get("concurrency")
    print(f"{model}: the server runs {users} at once")
    if not isinstance(users, int) or users < a.users:
        print(f"FAIL concurrency {users}, where {a.users} were asked for")
        fails += 1
    prompts = PROMPTS[: a.users]
    alone = [ask(url, model, p, a.tokens) for p in prompts]
    before = get(url + "/stats")
    together = [None] * len(prompts)
    go = threading.Barrier(len(prompts))

    def one(i):
        go.wait()
        try:
            together[i] = ask(url, model, prompts[i], a.tokens)
        except Exception as e:  # (a failed request is a failure of the check, with its reason)
            together[i] = e

    t0 = time.monotonic()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    after = get(url + "/stats")
    same = 0
    for i, (p, a1, a2) in enumerate(zip(prompts, alone, together)):
        if isinstance(a2, Exception) or a2[1] == 0 or not a2[3]:
            print(f"FAIL {i}: {p!r}: {a2!r}")
            fails += 1
            continue
        eq = a1[3] == a2[3]
        same += eq
        print(f"{'same' if eq else 'DIFF'} {i}: {a1[1]} tokens alone in {a1[2]:.2f} s, {a2[1]} at once in {a2[2]:.2f} s: {a2[0][:80]!r}")
        if not eq:
            k = next((j for j, (x, y) in enumerate(zip(a1[3], a2[3])) if x != y), min(len(a1[3]), len(a2[3])))
            print(f"     first different token: {k}; alone: {a1[0][:200]!r}\n     once:  {a2[0][:200]!r}")
    rows = [x - y for x, y in zip(after.get("steps_by_rows", []), before.get("steps_by_rows", []))]
    several = sum(rows[2:])
    total = sum(x[1] for x in together if not isinstance(x, Exception))
    print(f"at once: {total} tokens in {wall:.2f} s ({total / wall:.1f} tokens/s); alone one after another: {sum(x[2] for x in alone):.2f} s")
    print(f"decode steps by rows during the burst: {rows}")
    if several == 0:
        print("FAIL no decode step had more than one row")
        fails += 1
    need = len(prompts) if a.same else min(a.least, len(prompts))
    if same < need:
        print(f"FAIL {same} of {len(prompts)} replies at once are the reply alone, {need} asked for")
        fails += 1
    print("PASS" if fails == 0 else f"{fails} failures")
    return fails


if __name__ == "__main__":
    sys.exit(main())
