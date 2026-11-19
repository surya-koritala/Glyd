"""One user's speed of an OpenAI-compatible server: time to first token and decode tokens a second, streamed.

    python speed1.py URL MODEL OUT.json [--rounds 3] [--prompt-tokens 1024] [--new 256]

Each round sends the same kind of request: a prompt of about PROMPT words of fixed text (a seed a round so no
round hits a prefix cache), max_tokens NEW with min_tokens NEW (so every reply has exactly NEW tokens), greedy.
TTFT = the first content chunk's arrival; decode rate = (NEW - 1) / (last chunk - first chunk)."""
import json
import random
import sys
import time
import urllib.request

url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
args = dict(rounds=3, prompt_tokens=1024, new=256)
for i, a in enumerate(sys.argv):
    if a.startswith("--") and a[2:].replace("-", "_") in args:
        args[a[2:].replace("-", "_")] = int(sys.argv[i + 1])
WORDS = ("the river carried light across the valley while an old engineer measured each stone of the bridge "
         "and wrote the numbers into a book that nobody would read for a hundred years").split()


def prompt(seed):
    r = random.Random(seed)
    return "Continue this text for as long as you can.\n\n" + " ".join(r.choice(WORDS) for _ in range(args["prompt_tokens"]))


def once(seed):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(seed)}], "max_tokens": args["new"],
            "min_tokens": args["new"], "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = (ch.get("delta") or {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n += 1
    toks = (usage or {}).get("completion_tokens") or n
    return {"ttft_ms": round(1000 * (first - t0), 1), "decode_tok_s": round((toks - 1) / (last - first), 2) if last and last > first else None,
            "completion_tokens": toks, "prompt_tokens": (usage or {}).get("prompt_tokens"), "chunks": n}


once(999)  # a warm-up request, not counted
rows = [once(s) for s in range(args["rounds"])]
med = lambda k: sorted(r[k] for r in rows if r[k] is not None)[len(rows) // 2]
res = {"url": url, "model": model, "each": rows, "ttft_ms_median": med("ttft_ms"), "decode_tok_s_median": med("decode_tok_s"), **args}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("ttft_ms_median", "decode_tok_s_median")}))
