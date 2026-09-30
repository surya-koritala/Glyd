"""A multi-turn streaming chat through vLLM's OpenAI API, as one user: each turn's time to the first token (reasoning or
content) and its tokens a second after it (the usage's completion tokens over the stream's time from the first token
to the last); then one longer answer (512 tokens, min_tokens) for the steady tokens a second.
    python chat_test.py [BASE_URL] [MODEL] [OUT.json]"""
import json
import sys
import time

from openai import OpenAI

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"
out = sys.argv[3] if len(sys.argv) > 3 else None
client = OpenAI(base_url=base, api_key="none")
TURNS = ["Hi! In two sentences, what is lossless compression?",
         "Give me one everyday example of it, and one of lossy compression.",
         "Now summarize our conversation so far in one sentence."]


def stream(messages, max_tokens, extra=None):
    t0 = time.perf_counter()
    first = last = None
    text, reasoning, usage = [], [], None
    for ch in client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, stream=True,
                                             stream_options={"include_usage": True}, extra_body=extra or {}):
        if ch.usage:
            usage = ch.usage
        if not ch.choices:
            continue
        d = ch.choices[0].delta
        piece = (d.content or "") + (getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or "")
        if piece:
            now = time.perf_counter()
            first = first or now
            last = now
            (text if d.content else reasoning).append(piece)
    n = usage.completion_tokens if usage else None
    return {"ttft_s": first - t0, "tokens": n, "prompt_tokens": usage.prompt_tokens if usage else None,
            "tok_s": (n - 1) / (last - first) if n and last > first else None, "text": "".join(text), "reasoning": "".join(reasoning)}


res = {"model": model, "turns": []}
messages = [{"role": "system", "content": "You are a helpful assistant."}]
for q in TURNS:
    messages.append({"role": "user", "content": q})
    r = stream(messages, 400, {"chat_template_kwargs": {"enable_thinking": False}})
    messages.append({"role": "assistant", "content": r["text"]})
    res["turns"].append({"user": q, **r})
    print(f"turn {len(res['turns'])}: prompt {r['prompt_tokens']} tokens, first token {r['ttft_s'] * 1e3:.0f} ms, {r['tokens']} tokens at {r['tok_s']:.1f}/s", flush=True)
    print("  assistant:", r["text"][:300].replace("\n", " "), flush=True)
r = stream([{"role": "user", "content": "Write a long story about a lighthouse keeper."}], 512, {"min_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}})
res["long"] = {k: v for k, v in r.items() if k not in ("text", "reasoning")}
print(f"long: first token {r['ttft_s'] * 1e3:.0f} ms, {r['tokens']} tokens at {r['tok_s']:.1f}/s", flush=True)
r = stream([{"role": "user", "content": "What is 17 times 23?"}], 1024, {})
res["thinking"] = {k: v for k, v in r.items() if k not in ("text", "reasoning")}
res["thinking"]["answer"] = r["text"][-200:]
res["thinking"]["reasoning_chars"] = len(r["reasoning"])
print(f"thinking on: first token {r['ttft_s'] * 1e3:.0f} ms, {r['tokens']} tokens at {r['tok_s']:.1f}/s; answer: {r['text'][-120:]!r}", flush=True)
if out:
    json.dump(res, open(out, "w"), indent=1)
