#!/usr/bin/env python3
"""Four requests at once against a running server (tools/serve.sh, 4 users by default): all four stream together (each one's first
words come before any of the others has finished) and each gets its own whole answer, the one it gets alone. Standard library only.
   tests/test_parallel.py [--url http://127.0.0.1:8080]"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse

# (prompt, text the answer must hold): answers of about 100-200 tokens, each unlike the others, so a mixed-up stream shows
PROMPTS = [("Count from 1 to 60, separated by spaces.", "57 58 59 60"),
           ("List the letters of the English alphabet from A to Z, separated by spaces.", "W X Y Z"),
           ("Count down from 50 to 1, separated by spaces.", "4 3 2 1"),
           ("Write the word banana 40 times, separated by spaces.", "banana banana banana")]


def stream(u, text, out, i, t_start):
    p = urllib.parse.urlparse(u)
    c = http.client.HTTPConnection(p.hostname, p.port, timeout=900)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 300, "temperature": 0, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    first, last, content = None, None, ""
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            for ch in json.loads(line[5:]).get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    last = time.time() - t_start
                    first = last if first is None else first
                content += piece
    out[i] = {"status": r.status, "first_s": round(first or -1, 2), "last_s": round(last or -1, 2), "content": content}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8080"); a = ap.parse_args()
    u, ok = a.url, []

    def check(name, cond, detail=""):
        ok.append(bool(cond)); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"[:600], flush=True)

    # each alone first: the answer a request gets with the server to itself
    alone = [None] * 4
    for i, (text, _) in enumerate(PROMPTS):
        stream(u, text, alone, i, time.time())
    # the four at once
    out, t0 = [None] * 4, time.time()
    ts = [threading.Thread(target=stream, args=(u, PROMPTS[i][0], out, i, t0)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for i, (o, (_, want)) in enumerate(zip(out, PROMPTS)):
        check(f"4 at once: request {i} has its own answer", o and o["status"] == 200 and want in o["content"], json.dumps(o))
    firsts, lasts = [o["first_s"] for o in out], [o["last_s"] for o in out]
    check("4 at once: all four stream together (every first word before any last one)", max(firsts) < min(lasts), f"first words at {firsts} s, last at {lasts} s")
    same = [o["content"] == b["content"] for o, b in zip(out, alone)]
    check("4 at once: each answer is the one it gets alone", all(same), str(same))
    print(f"{sum(ok)} of {len(ok)} passed")


if __name__ == "__main__":
    main()
