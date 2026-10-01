# tokens/s of one user's and of 8 users' streamed answers through the OpenAI API, greedy and top-p; stdlib only.
#   python chatbench.py [URL] [MODEL] [REPEATS]
import json, statistics, sys, threading, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
PROMPT = "Write a long story about a dragon who learns to code. /no_think"
MODES = {"greedy": {"temperature": 0}, "top-p": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}


def chat(mode, n=256, seed=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": n, "min_tokens": n, "stream": True,
            "stream_options": {"include_usage": True}, **MODES[mode]}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                t = c["delta"].get("content")
                if t:
                    if first is None:
                        first = time.perf_counter()
                    text.append(t)
    t1 = time.perf_counter()
    toks = usage["completion_tokens"]
    return {"tokens": toks, "first_ms": (first - t0) * 1e3, "tps": (toks - 1) / (t1 - first), "text": "".join(text)}


chat("greedy", 16)  # warm
out = {}
for mode in MODES:
    runs = [chat(mode) for _ in range(REPEATS)]
    out[mode] = {"tps_median": round(statistics.median(r["tps"] for r in runs), 2), "tps_all": [round(r["tps"], 2) for r in runs],
                 "first_ms_median": round(statistics.median(r["first_ms"] for r in runs)), "tokens": runs[0]["tokens"]}
    if mode == "greedy":
        out[mode]["same_text"] = len({r["text"] for r in runs}) == 1
        out[mode]["text_sha"] = __import__("hashlib").sha256(runs[0]["text"].encode()).hexdigest()[:12]
    # 8 at once
    res = []
    ths = [threading.Thread(target=lambda: res.append(chat(mode))) for _ in range(8)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    dt = time.perf_counter() - t0
    out[mode]["tps_8_users_total"] = round(sum(r["tokens"] for r in res) / dt, 1)
print(json.dumps(out, indent=1))
