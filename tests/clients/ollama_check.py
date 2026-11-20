#!/usr/bin/env python3
"""The `ollama` Python library, and Ollama's wire format with plain HTTP, against a running `glyd serve` (Ollama's API is under /api).

    pip install ollama requests
    python3 tests/clients/ollama_check.py http://127.0.0.1:8238 [--key KEY]

Every check prints PASS or FAIL; the exit status is the number of failures.
"""
import argparse
import json
import sys
import traceback

import ollama
import requests

failures = []
NOTHINK = {"think": False}
WEATHER = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Get the weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
}


def check(fn):
    try:
        fn()
        print(f"PASS {fn.__name__}")
    except Exception:
        failures.append(fn.__name__)
        print(f"FAIL {fn.__name__}")
        traceback.print_exc()
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key", default=None)
    ap.add_argument("--fim", action="store_true", help="the model has fill-in-the-middle tokens (Qwen's)")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    headers = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    c = ollama.Client(host=base, headers=headers)
    ask = [{"role": "user", "content": "Reply with the single word: pong"}]
    opts = {"temperature": 0, "num_predict": 200}

    @check
    def version_tags_ps_show():
        assert requests.get(base + "/api/version", headers=headers).json()["version"]
        tags = c.list()
        assert len(tags.models) == 1 and ":" in tags.models[0].model and tags.models[0].digest
        name = tags.models[0].model
        assert c.ps().models[0].model == name
        s = c.show(name)
        assert "completion" in s.capabilities and s.details is not None
        assert any(k.endswith(".context_length") for k in s.modelinfo), s.modelinfo

    @check
    def chat():
        r = c.chat(model="whatever", messages=ask, options=opts, **NOTHINK)
        assert r.done and r.done_reason in ("stop", "length") and r.message.role == "assistant" and r.message.content
        assert r.eval_count > 0 and r.prompt_eval_count > 0 and r.total_duration > 0

    @check
    def chat_stream():
        parts = list(c.chat(model="whatever", messages=ask, options=opts, stream=True, **NOTHINK))
        assert parts[-1].done and not any(p.done for p in parts[:-1])
        assert "".join(p.message.content for p in parts)

    @check
    def thinking():
        r = c.chat(model="whatever", messages=ask, options={"temperature": 0, "num_predict": 1500}, think=True)
        assert r.message.thinking, r
        r = c.chat(model="whatever", messages=ask, options=opts, think=False)
        assert not r.message.thinking

    @check
    def generate_and_stream():
        r = c.generate(model="whatever", prompt="Reply with the single word: pong", options=opts, **NOTHINK)
        assert r.done and r.response
        parts = list(c.generate(model="whatever", prompt="Reply with the single word: pong", options=opts, stream=True, **NOTHINK))
        assert parts[-1].done and "".join(p.response for p in parts)

    @check
    def tools():
        r = c.chat(model="whatever", messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER], options=opts, **NOTHINK)
        call = r.message.tool_calls[0]
        assert call.function.name == "get_weather" and call.function.arguments["city"].lower().startswith("paris"), r
        msgs = [{"role": "user", "content": "What is the weather in Paris?"}, r.message, {"role": "tool", "tool_name": "get_weather", "content": "18 C, sunny"}]
        r2 = c.chat(model="whatever", messages=msgs, tools=[WEATHER], options=opts, **NOTHINK)
        assert r2.message.content

    @check
    def format_json_and_schema():
        r = c.chat(model="whatever", messages=[{"role": "user", "content": "Paris is 18 degrees. Answer as JSON."}], format="json", options=opts, **NOTHINK)
        assert isinstance(json.loads(r.message.content), dict)
        schema = {"type": "object", "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}}, "required": ["city", "temp_c"]}
        r = c.chat(model="whatever", messages=[{"role": "user", "content": "Paris is 18 degrees. Answer as JSON."}], format=schema, options=opts, **NOTHINK)
        d = json.loads(r.message.content)
        assert set(d) == {"city", "temp_c"} and isinstance(d["temp_c"], int), d

    @check
    def options_seed_and_stop():
        q = lambda: c.generate(model="w", prompt="Name a color.", options={"temperature": 1.0, "seed": 7, "num_predict": 20}, **NOTHINK).response
        assert q() == q()
        r = c.generate(model="w", prompt="Count from 1 to 9 separated by spaces, nothing else.", options={"temperature": 0, "stop": ["5"], "num_predict": 100}, **NOTHINK)
        assert "5" not in r.response and r.done_reason == "stop", r

    if a.fim:
        @check
        def fill_in_the_middle():
            r = c.generate(model="w", prompt="def add(a, b):\n    return ", suffix="\n\nprint(add(1, 2))\n", options={"temperature": 0, "num_predict": 16})
            assert r.done and r.eval_count > 0

    @check
    def wire_format():
        # (the lines of a stream are JSON objects, one per line, as Ollama's server writes them)
        r = requests.post(base + "/api/chat", headers=headers, json={"model": "m", "messages": ask, "options": opts, "think": False}, stream=True)
        assert r.headers["content-type"].startswith("application/x-ndjson"), r.headers
        lines = [json.loads(l) for l in r.iter_lines() if l]
        assert lines[-1]["done"] is True and lines[0]["created_at"] and ":" in lines[0]["model"]
        r = requests.post(base + "/api/generate", headers=headers, json={"model": "m"}).json()
        assert r["done"] is True and r["done_reason"] == "load"
        r = requests.post(base + "/api/pull", headers=headers, json={"model": c.list().models[0].model, "stream": False}).json()
        assert r["status"] == "success"

    @check
    def errors():
        r = requests.post(base + "/api/chat", headers=headers, json={"model": "m", "messages": ask, "format": "xml"})
        assert r.status_code == 400 and "error" in r.json()
        r = requests.post(base + "/api/show", headers=headers, json={"model": "llama3.2"})
        assert r.status_code == 404 and "not found" in r.json()["error"]
        r = requests.post(base + "/api/chat", json={"model": "m", "messages": ask})
        if a.key:
            assert r.status_code == 401 and "error" in r.json()

    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(len(failures))


main()
