"""bench_conc.py PORT [MODEL] [N ...]: N chats at once, each streaming 200 tokens (thinking off): the aggregate tokens a second, each stream's,
and the first token's time, the median of 3 rounds. The first request warms the server."""
import http.client, json, statistics, sys, threading, time

port = int(sys.argv[1])
ns = [int(x) for x in sys.argv[3:]] or [1, 4, 8, 16]
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
conn.request("GET", "/v1/models")
model = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else json.loads(conn.getresponse().read())["data"][0]["id"]
TOKENS = 200


def one(i, out):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"model": model, "messages": [{"role": "user", "content": f"Write a story about a lighthouse keeper, number {i}."}], "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": TOKENS, "min_tokens": TOKENS, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    first = last = None
    n = 0
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            n = d["usage"]["completion_tokens"]
        for ch in d.get("choices") or ():
            if (ch.get("delta") or {}).get("content"):
                now = time.time()
                first = first or now
                last = now
    out[i] = (t0, first, last, n)


def round_(k):
    out = [None] * k
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(k)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    t0 = min(o[0] for o in out)
    span = max(o[2] for o in out) - min(o[1] for o in out)
    return sum(o[3] for o in out) / span, statistics.median((o[3] - 1) / (o[2] - o[1]) for o in out), statistics.median(o[1] - o[0] for o in out)


one(0, [None])
print(f"model {model}, {TOKENS} tokens a stream")
for k in ns:
    rs = [round_(k) for _ in range(3)]
    agg = statistics.median(r[0] for r in rs)
    each = statistics.median(r[1] for r in rs)
    ttft = statistics.median(r[2] for r in rs)
    print(f"  {k:2d} at once: {agg:6.1f} tokens/s in all, {each:5.1f} each, first token {ttft * 1000:5.0f} ms", flush=True)
