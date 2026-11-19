#!/usr/bin/env python3
"""End-to-end checks of a running server (tools/serve.sh): OpenAI chat (plain, streamed, a tool call), Anthropic messages (plain, streamed,
a tool use) and the chat page. Standard library only.   tests/test_server.py [--url http://127.0.0.1:8080]"""
import argparse
import json
import time
import urllib.request

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def post(url, path, body, stream=False):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    return r if stream else json.load(r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8080"); a = ap.parse_args()
    u, ok = a.url, []

    def check(name, cond, detail=""):
        ok.append(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"[:200], flush=True)
    page = urllib.request.urlopen(u + "/", timeout=30).read().decode()
    check("chat page", "Qwen3.8-Flash-Next" in page)
    r = post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}], "max_tokens": 16})
    check("openai plain", "42" in r["choices"][0]["message"]["content"], r["choices"][0]["message"]["content"])
    t0, chunks, text, times = time.time(), 0, "", []
    for line in post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from 1 to 30, separated by spaces."}], "max_tokens": 120, "stream": True}, True):
        line = line.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            d = json.loads(line[5:])["choices"]
            if d and d[0]["delta"].get("content"):
                chunks += 1; text += d[0]["delta"]["content"]; times.append(time.time() - t0)
    check("openai streamed live", chunks > 5 and times[-1] - times[0] > 0.2 and "30" in text, f"{chunks} chunks over {times[-1] - times[0]:.2f} s")
    r = post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": [WEATHER], "max_tokens": 200})
    tc = r["choices"][0]["message"].get("tool_calls") or []
    check("openai tool call", r["choices"][0]["finish_reason"] == "tool_calls" and tc and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"], json.dumps(tc))
    if tc:
        r = post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}, r["choices"][0]["message"],
                                                          {"role": "tool", "tool_call_id": tc[0]["id"], "content": "18 C and sunny"}], "tools": [WEATHER], "max_tokens": 100})
        check("openai tool result used", "18" in (r["choices"][0]["message"]["content"] or ""), r["choices"][0]["message"]["content"])
    r = post(u, "/v1/messages", {"model": "x", "max_tokens": 16, "system": "Answer with one word.", "messages": [{"role": "user", "content": "What color is the sky on a clear day?"}]})
    check("anthropic plain", r["content"] and "blue" in r["content"][0]["text"].lower(), json.dumps(r["content"]))
    tools = [{"name": "get_weather", "description": "The current weather in a city.", "input_schema": WEATHER["function"]["parameters"]}]
    r = post(u, "/v1/messages", {"model": "x", "max_tokens": 200, "tools": tools, "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]})
    tu = [b for b in r["content"] if b["type"] == "tool_use"]
    check("anthropic tool use", r["stop_reason"] == "tool_use" and tu and tu[0]["input"].get("city", "").lower().startswith("tokyo"), json.dumps(r["content"]))
    events = [l.decode().strip() for l in post(u, "/v1/messages", {"model": "x", "max_tokens": 60, "stream": True, "messages": [{"role": "user", "content": "Say three words."}]}, True)]
    kinds = [l.split(": ", 1)[1] for l in events if l.startswith("event:")]
    check("anthropic streamed", kinds[0] == "message_start" and "content_block_delta" in kinds and kinds[-1] == "message_stop", " ".join(dict.fromkeys(kinds)))
    r = post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a 300-word story."}], "max_tokens": 400, "stream": True}, True)
    for _ in range(5):
        r.readline()
    r.close()   # a client that leaves mid-answer: the next request must get its own answer, not the rest of this one
    r = post(u, "/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 6 times 7? Reply with the number only."}], "max_tokens": 16})
    check("after a dropped stream", "42" in r["choices"][0]["message"]["content"], r["choices"][0]["message"]["content"][:80])
    print(f"{sum(ok)} of {len(ok)} passed")


if __name__ == "__main__":
    main()
