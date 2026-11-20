#!/usr/bin/env python3
"""Greedy tokens and logprobs of a server for 10 prompts, one at a time (--mode alone) or all at once (--mode together), on a server that has not seen them (cold) and again (warm:
the prompts' pages are found). compare.py puts two such files side by side.

    python3 tokens_check.py --url http://127.0.0.1:8100 --mode together --out together.json
"""
import argparse, json, sys, threading, urllib.request

WORDS = "river stone market letter garden window engine paper silver winter harbor signal ladder copper meadow lantern bridge orchard compass thunder".split()


def prompts():
    p = [f"[{k}] Explain {t}, in detail." for k, t in enumerate(["how a heat pump works", "why the sky is blue", "how a compiler works", "what makes bread rise",
                                                                   "how vaccines work", "why tides happen", "how encryption works", "how a bicycle stays upright"])]
    p.append("[9] Notes: " + " ".join(WORDS[i * 7 % len(WORDS)] for i in range(1500)) + "\nWrite an essay about bridges.")  # (about 2,000 tokens)
    p.append("[10] Notes: " + " ".join(WORDS[i * 3 % len(WORDS)] for i in range(300)) + "\nSummarise the notes.")
    return p


def ask(url, text, out):
    body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 96, "temperature": 0, "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    ch = d["choices"][0]
    lp = [[t.get("token"), t.get("logprob")] for t in (ch.get("logprobs") or {}).get("content") or []]
    out.append({"text": ch["message"]["content"], "lp": lp})


def alone(url, ps):
    r = []
    for t in ps:
        o = []
        ask(url, t, o)
        r.append(o[0])
    return r


def together(url, ps):
    o = [[] for _ in ps]
    ts = [threading.Thread(target=ask, args=(url, t, o[i])) for i, t in enumerate(ps)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return [x[0] for x in o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["alone", "together"], required=True)
    a = ap.parse_args()
    ps = prompts()
    f = alone if a.mode == "alone" else together
    # (a server that has not seen the prompts: their pages are made; then again: the pages are found, and a prompt's last token alone is run)
    json.dump({"cold": f(a.url, ps), "warm": f(a.url, ps)}, open(a.out, "w"))


main()
