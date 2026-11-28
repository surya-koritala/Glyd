#!/usr/bin/env python3
"""What agent frameworks use, against a running server (tools/serve.sh): streamed tool calls (OpenAI and Anthropic), several calls in one
answer, tool_choice, a long tool loop, tool calls with thinking on, JSON mode, GET /v1/models. Standard library only.
  tests/test_agents.py [--url http://127.0.0.1:8080] [--only NAME,NAME]"""
import argparse
import json
import time
import urllib.error
import urllib.request

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "The current local time in a city.",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
STEP = {"type": "function", "function": {"name": "next_number", "description": "Returns the next number in a secret sequence after the given one.",
                                         "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}


def anth(t):
    f = t["function"]
    return {"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}


def post(url, path, body, stream=False):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:300]}
    return r if stream else json.load(r)


def chat(u, messages, **kw):
    return post(u, "/v1/chat/completions", {"messages": messages, "max_tokens": kw.pop("max_tokens", 400), **kw})


def chat_stream(u, messages, **kw):
    """A streamed chat reply put back together: content, reasoning, tool calls (by index), finish reason, chunk count."""
    r = post(u, "/v1/chat/completions", {"messages": messages, "max_tokens": kw.pop("max_tokens", 400), "stream": True, **kw}, True)
    if isinstance(r, dict):
        return r
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "chunks": 0}
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        for c in json.loads(line[5:]).get("choices", []):
            d = c.get("delta", {})
            out["chunks"] += 1
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = out["calls"].setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                slot["id"] = tc.get("id") or slot["id"]
                f = tc.get("function", {})
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
            out["finish"] = c.get("finish_reason") or out["finish"]
    out["calls"] = [out["calls"][k] for k in sorted(out["calls"])]
    return out


def calls_of(r):
    if "http_error" in r:
        return []
    return r["choices"][0]["message"].get("tool_calls") or []


def args(c):
    try:
        return json.loads(c["function"]["arguments"] if "function" in c else c["arguments"])
    except (ValueError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    u, ok, only = a.url, [], set(filter(None, a.only.split(",")))

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"[:400], flush=True)

    def want(name):
        return not only or name in only

    if want("models"):
        r = json.load(urllib.request.urlopen(u + "/v1/models", timeout=30))
        check("GET /v1/models", r.get("object") == "list" and r["data"] and r["data"][0].get("id"), json.dumps(r))

    q = [{"role": "user", "content": "What's the weather in Paris right now?"}]
    if want("stream_tool"):
        r = chat_stream(u, q, tools=[WEATHER])
        c = r.get("calls") or [{}]
        check("openai streamed tool call", r.get("finish") == "tool_calls" and c[0].get("name") == "get_weather" and c[0].get("id") and (args(c[0]) or {}).get("city", "").startswith("Paris"), json.dumps(r))

    if want("anthropic_stream_tool"):
        lines = [l.decode().strip() for l in post(u, "/v1/messages", {"model": "x", "max_tokens": 300, "stream": True, "tools": [anth(WEATHER)],
                                                                         "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}, True)]
        evs = [json.loads(l[5:]) for l in lines if l.startswith("data:")]
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
        partial = "".join(e["delta"].get("partial_json", "") for e in evs if e["type"] == "content_block_delta")
        stop = [e["delta"].get("stop_reason") for e in evs if e["type"] == "message_delta"]
        try:
            inp = json.loads(partial[partial.index("{"):]) if "{" in partial else {}
        except ValueError:
            inp = {}
        check("anthropic streamed tool_use", any(s["type"] == "tool_use" and s["name"] == "get_weather" for s in starts) and stop == ["tool_use"] and inp.get("city", "").lower().startswith("tokyo"),
              f"blocks {[s['type'] for s in starts]} stop {stop} input {partial!r}")

    two = [{"role": "user", "content": "What's the weather in Paris and in Tokyo right now? Check both."}]
    if want("parallel"):
        r = chat(u, two, tools=[WEATHER])
        c = calls_of(r)
        cities = sorted((args(x) or {}).get("city", "") for x in c)
        check("two calls in one answer", len(c) == 2 and [x.split(",")[0] for x in cities] == ["Paris", "Tokyo"] and len({x["id"] for x in c}) == 2, json.dumps(c))
        r = chat_stream(u, two, tools=[WEATHER])
        cities = sorted((args(x) or {}).get("city", "") for x in r.get("calls", []))
        check("two calls in one answer, streamed", len(cities) == 2 and [x.split(",")[0] for x in cities] == ["Paris", "Tokyo"] and r.get("finish") == "tool_calls", json.dumps(r.get("calls")))
        r = chat(u, two, tools=[WEATHER], parallel_tool_calls=False)
        check("parallel_tool_calls false keeps one", len(calls_of(r)) == 1, json.dumps(calls_of(r)))

    if want("tool_choice"):
        hi = [{"role": "user", "content": "Hi! How are you today?"}]
        r = chat(u, hi, tools=[WEATHER], tool_choice="required")
        check("tool_choice required", r.get("choices") and r["choices"][0]["finish_reason"] == "tool_calls" and calls_of(r) and calls_of(r)[0]["function"]["name"] == "get_weather" and args(calls_of(r)[0]) is not None, json.dumps(r)[:300])
        r = chat(u, q, tools=[WEATHER, TIME], tool_choice="required")
        check("tool_choice required, two tools", calls_of(r) and calls_of(r)[0]["function"]["name"] in ("get_weather", "get_time"), json.dumps(calls_of(r) or r)[:300])
        r = chat(u, q, tools=[WEATHER], tool_choice="none")
        m = r["choices"][0]["message"] if "choices" in r else {}
        check("tool_choice none", "choices" in r and not m.get("tool_calls") and m.get("content") and r["choices"][0]["finish_reason"] == "stop" and "<tool_call>" not in m["content"], json.dumps(r)[:300])
        r = chat(u, q, tools=[WEATHER, TIME], tool_choice={"type": "function", "function": {"name": "get_time"}})
        c = calls_of(r)
        check("tool_choice named", len(c) >= 1 and all(x["function"]["name"] == "get_time" for x in c) and args(c[0]) is not None, json.dumps(c))
        r = chat_stream(u, hi, tools=[WEATHER], tool_choice="required")
        check("tool_choice required, streamed", r.get("finish") == "tool_calls" and r.get("calls") and r["calls"][0]["name"] == "get_weather" and args(r["calls"][0]) is not None, json.dumps(r)[:300])
        names = ("get_weather", "get_time")
        for name, tc, good in [("any", {"type": "any"}, lambda r: r["stop_reason"] == "tool_use" and any(x["type"] == "tool_use" and x["name"] in names for x in r["content"])),
                               ("tool", {"type": "tool", "name": "get_time"}, lambda r: r["stop_reason"] == "tool_use" and any(x["type"] == "tool_use" and x["name"] == "get_time" for x in r["content"])),
                               ("none", {"type": "none"}, lambda r: r["content"] and all(x["type"] == "text" and "<tool_call>" not in x["text"] for x in r["content"]))]:
            r = post(u, "/v1/messages", {"model": "x", "max_tokens": 300, "tools": [anth(WEATHER), anth(TIME)], "tool_choice": tc,
                                         "messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
            check(f"anthropic tool_choice {name}", "content" in r and good(r), json.dumps(r)[:300])

    if want("thinking"):
        r = chat(u, q, tools=[WEATHER], reasoning_effort="medium", max_tokens=2000)
        m = r["choices"][0]["message"] if "choices" in r else {}
        check("tool call with thinking on", m.get("reasoning_content") and calls_of(r) and calls_of(r)[0]["function"]["name"] == "get_weather" and "<tool_call>" not in (m.get("content") or ""),
              f"reasoning {len(m.get('reasoning_content') or '')} chars, content {m.get('content')!r:.80}, calls {json.dumps(calls_of(r))}")
        r = chat_stream(u, q, tools=[WEATHER], reasoning_effort="medium", max_tokens=2000)
        check("tool call with thinking on, streamed", r.get("reasoning") and r.get("calls") and r["calls"][0]["name"] == "get_weather" and r.get("finish") == "tool_calls",
              f"reasoning {len(r.get('reasoning', ''))} chars, content {r.get('content')!r:.80}, calls {json.dumps(r.get('calls'))}")

    if want("loop"):
        # a chain of 12 tool round trips: each answer feeds the next call (the sequence is n*2+1)
        msgs = [{"role": "system", "content": "You are an agent. Use the tools; never guess a tool's result."},
                {"role": "user", "content": "Start from 1. Call next_number with 1, then call it again with what it returned, and so on: 12 calls in all, one at a time. Then tell me the last number it returned."}]
        n, expect, rounds, t0, cached = 1, 1, 0, time.time(), []
        for _ in range(20):
            r = chat(u, msgs, tools=[STEP], max_tokens=600)
            if "choices" not in r:
                break
            cached.append(r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0))
            m = r["choices"][0]["message"]
            c = calls_of(r)
            if not c:
                break
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": c})
            for x in c:
                got = (args(x) or {}).get("n")
                expect = expect * 2 + 1
                rounds += 1
                msgs.append({"role": "tool", "tool_call_id": x["id"], "content": str(int(got) * 2 + 1 if isinstance(got, (int, float)) else "error: n must be an integer")})
        final = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        check("long tool loop (12 round trips)", rounds == 12 and str(expect) in final, f"{rounds} calls, expected {expect}, final {final[-120:]!r}, {time.time() - t0:.0f} s, cached tokens per turn {cached}")

    if want("json"):
        r = chat(u, [{"role": "user", "content": "Give me a JSON object with keys name and age for a made-up person."}], response_format={"type": "json_object"})
        txt = r["choices"][0]["message"]["content"] if "choices" in r else ""
        try:
            good = isinstance(json.loads(txt), dict)
        except ValueError:
            good = False
        check("response_format json_object", good, json.dumps(r)[:300])
        schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"], "additionalProperties": False}
        r = chat(u, [{"role": "user", "content": "The largest city in Japan and its population."}], response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}})
        txt = r["choices"][0]["message"]["content"] if "choices" in r else ""
        try:
            o = json.loads(txt)
            good = isinstance(o, dict) and set(o) == {"city", "population"} and isinstance(o["population"], int) and isinstance(o["city"], str)
        except ValueError:
            good = False
        check("response_format json_schema", good, json.dumps(r)[:300])

    print(f"{sum(ok)} of {len(ok)} passed")


if __name__ == "__main__":
    main()
