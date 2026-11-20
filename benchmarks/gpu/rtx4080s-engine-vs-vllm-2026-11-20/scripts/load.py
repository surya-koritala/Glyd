#!/usr/bin/env python3
"""Closed-loop load on an OpenAI-compatible chat server (glyd-engine serve, or vllm serve): N streaming requests at once, short and long prompts.

    python3 load.py --url http://127.0.0.1:8000 --label engine --out engine.json [--conc 1,4,8] [--prompts short,2k,4k] [--rounds 3] [--max-tokens 256]

For each (prompt length, concurrency) cell, `rounds` rounds of N simultaneous requests, every prompt different from the first token on (no prefix to share).
Per request: time to first token (from sending), decode tokens/s = (tokens - 1) / (last token - first token), tokens as the server's usage says.
Per cell: the median of the rounds of each; the aggregate decode rate (all requests' tokens after their first / the span from the earliest first token
to the last token); the peak GPU memory of compute processes (nvidia-smi, sampled every 0.1 s) while the cell ran. Standard library only.
"""
import argparse, json, random, statistics, subprocess, threading, time, urllib.request

WORDS = ("river stone market letter garden window engine paper silver winter harbor signal ladder copper meadow lantern bridge orchard compass thunder "
         "village canvas anchor forest ribbon quartz pepper violet cabin marble kettle falcon summit harvest tunnel mirror saddle velvet crystal ember "
         "plains lagoon pigeon ceiling puzzle bottle shadow carpet planet button candle valley hammer island jacket cotton rocket temple orange basket").split()


def prompt(kind, seed):
    rng = random.Random(seed)
    if kind == "short":
        topics = ["how a heat pump works", "why the sky is blue", "how a compiler turns source into machine code", "what makes bread rise",
                  "how vaccines train the immune system", "why tides have two highs a day", "how public-key encryption works", "how a bicycle stays upright"]
        return f"[{seed}] Explain {topics[seed % len(topics)]}, in detail, in about 300 words."
    n = {"2k": 2048, "4k": 4096}[kind]
    body = " ".join(rng.choice(WORDS) for _ in range(int((n - 50) / 1.09)))  # (about 1.09 tokens a word, and 50 for the rest: the prompt_tokens each result carries is the count)
    return f"[{seed}] Here are some notes:\n{body}\n\nIgnoring the notes, write a long, detailed essay about the history of bridges."


def gpu_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return 0


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, gpu_mib())
            time.sleep(0.1)


def one(url, model, kind, seed, max_tokens, out):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(kind, seed)}], "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content") or (ch[0].get("delta") or {}).get("reasoning")):
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
    except Exception as e:
        out.append({"error": str(e)})
        return
    toks = usage.get("completion_tokens") or chunks
    out.append({"ttft": (first - t0) if first else None, "tokens": toks, "prompt_tokens": usage.get("prompt_tokens"), "first": first, "last": last,
                "decode_tps": (toks - 1) / (last - first) if first and last and last > first and toks > 1 else None})


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--prompts", default="short,2k,4k")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    model = a.model or json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    one(a.url, model, "short", 0, 16, [])  # (warm-up)
    one(a.url, model, "2k", 1, 16, [])
    results = {"label": a.label, "model": model, "idle_mib": gpu_mib(), "cells": []}
    seed = 100
    for kind in a.prompts.split(","):
        for n in map(int, a.conc.split(",")):
            s = Sampler()
            s.start()
            rounds = []
            for _ in range(a.rounds):
                outs = [[] for _ in range(n)]
                ts = []
                for i in range(n):
                    seed += 1
                    ts.append(threading.Thread(target=one, args=(a.url, model, kind, seed, a.max_tokens, outs[i])))
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                rs = [o[0] for o in outs if o]
                ok = [r for r in rs if "error" not in r]
                span_first = min((r["first"] for r in ok if r.get("first")), default=None)
                span_last = max((r["last"] for r in ok if r.get("last")), default=None)
                after_first = sum(r["tokens"] - 1 for r in ok)
                rounds.append({"requests": rs, "errors": len(rs) - len(ok),
                               "agg_tps": after_first / (span_last - span_first) if span_first and span_last and span_last > span_first else None})
            s.stop = True
            s.join()
            allr = [r for rd in rounds for r in rd["requests"] if "error" not in r]
            cell = {"prompt": kind, "conc": n, "prompt_tokens": med([r["prompt_tokens"] for r in allr]), "ttft_s": med([r["ttft"] for r in allr]),
                    "decode_tps_per_user": med([r["decode_tps"] for r in allr]), "agg_tps": med([rd["agg_tps"] for rd in rounds]),
                    "tokens_median": med([r["tokens"] for r in allr]), "errors": sum(rd["errors"] for rd in rounds), "peak_mib": s.peak, "rounds": rounds}
            results["cells"].append(cell)
            print(f"{a.label:7s} {kind:5s} x{n}: prompt {cell['prompt_tokens']} tok  ttft {cell['ttft_s'] and round(cell['ttft_s'], 3)} s  "
                  f"decode/user {cell['decode_tps_per_user'] and round(cell['decode_tps_per_user'], 1)} tok/s  total {cell['agg_tps'] and round(cell['agg_tps'], 1)} tok/s  "
                  f"peak {cell['peak_mib']} MiB  errors {cell['errors']}", flush=True)
    results["peak_mib"] = max(c["peak_mib"] for c in results["cells"])
    json.dump(results, open(a.out, "w"), indent=1)


main()
