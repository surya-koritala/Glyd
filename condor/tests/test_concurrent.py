#!/usr/bin/env python3
"""Several requests at once against a running server (tools/serve.sh): each gets its own whole answer, streamed ones start their stream
(headers and the role chunk) right away while they wait, and a client that gives up while waiting costs the server nothing.
Standard library only.   tests/test_concurrent.py [--url http://127.0.0.1:8080]"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse

PROMPTS = [("Count from 1 to 300, separated by spaces.", "300"), ("What is 6 times 7? Reply with the number only.", "42"),
           ("Name the capital of France in one word.", "Paris")]


def stream(u, text, out, i, max_tokens=60):
    p = urllib.parse.urlparse(u)
    c = http.client.HTTPConnection(p.hostname, p.port, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    first_byte, first_token, content = time.time() - t0, None, ""
    for line in r:
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            for ch in json.loads(line[5:]).get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece and first_token is None:
                    first_token = time.time() - t0
                content += piece
    out[i] = {"status": r.status, "headers_s": round(first_byte, 2), "first_token_s": round(first_token or -1, 2), "total_s": round(time.time() - t0, 2), "content": content}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8080"); a = ap.parse_args()
    u, ok = a.url, []

    def check(name, cond, detail=""):
        ok.append(bool(cond)); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"[:600], flush=True)

    # three at once, streamed: each its own answer; the ones that wait get their headers at once (a client's read timeout does not fire while it waits)
    out = [None] * 3
    ts = [threading.Thread(target=stream, args=(u, PROMPTS[i][0], out, i, 1200 if i == 0 else 60)) for i in range(3)]
    for t in ts:
        t.start(); time.sleep(0.05)
    for t in ts:
        t.join()
    for i, (o, (_, want)) in enumerate(zip(out, PROMPTS)):
        check(f"3 at once: request {i} has its own answer", o and o["status"] == 200 and want in o["content"], json.dumps(o))
    check("3 at once: the waiting ones get headers at once", all(o and o["headers_s"] < 2.0 for o in out), str([o and o["headers_s"] for o in out]))

    # a client that gives up while it waits: the server skips its prompt; the request after it gets its own answer
    out2 = [None]
    long = threading.Thread(target=stream, args=(u, "Write a 400-word story about a lighthouse.", out2, 0, 600))
    long.start(); time.sleep(1.0)
    p = urllib.parse.urlparse(u)
    c = http.client.HTTPConnection(p.hostname, p.port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "Write 300 words about the sea. " * 50}], "max_tokens": 400, "stream": True}),
              {"Content-Type": "application/json"})
    time.sleep(1.0)
    c.sock.close()   # gone before its turn
    long.join()
    out3 = [None]
    stream(u, PROMPTS[1][0], out3, 0)
    check("after a client gave up while waiting", out3[0]["status"] == 200 and "42" in out3[0]["content"], json.dumps(out3[0]))
    print("check the server log: no line for the given-up request's prompt (about 370 tokens): it never ran")
    print(f"{sum(ok)} of {len(ok)} passed")


if __name__ == "__main__":
    main()
