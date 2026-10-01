import json, re, sys, time, urllib.request


def call(url, body=None, token=None, timeout=300):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})}
    return urllib.request.urlopen(urllib.request.Request(url, None if body is None else json.dumps(body).encode(), h), timeout=timeout)


def stream(r):
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def turn(base, model, messages, **p):
    t0, first, text, usage = time.perf_counter(), None, [], None
    with call(base + "/chat/completions", {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 400, **p}) as r:
        for d in stream(r):
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                t = c["delta"].get("content")
                if t:
                    first = first or time.perf_counter()
                    text.append(t)
    n = (usage or {}).get("completion_tokens", 0)
    return "".join(text), (f"{(n - 1) / (time.perf_counter() - first):.1f} tokens/s, first token {(first - t0) * 1e3:.0f} ms" if first and n > 1 else "no tokens")


def check(name, good, detail):
    print(("ok   " if good else "FAIL ") + f"{name}: {detail}", flush=True)
    return good


def api(base):
    model = json.load(call(base + "/models"))["data"][0]["id"]
    q1 = [{"role": "user", "content": "What is the capital of France? Answer in one word. /no_think"}]
    a1, s1 = turn(base, model, q1, temperature=0)
    ok = check("OpenAI API, turn 1 (streamed, greedy)", "paris" in a1.lower(), f"{a1.strip()[:60]!r}, {s1}")
    q2 = q1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And of Germany? Answer in one word. /no_think"}]
    a2, s2 = turn(base, model, q2, temperature=0.7, top_p=0.8, top_k=20)
    ok &= check("OpenAI API, turn 2 (streamed, top-p, turn 1 as history)", "berlin" in a2.lower(), f"{a2.strip()[:60]!r}, {s2}")
    tool = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    r = json.load(call(base + "/chat/completions", {"model": model, "tool_choice": "auto", "tools": [tool], "max_tokens": 300, "temperature": 0,
                                                    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool. /no_think"}]}))
    tc = (r["choices"][0]["message"].get("tool_calls") or [{}])[0].get("function", {})
    return ok & check("OpenAI API, a tool call (tool_choice auto)", tc.get("name") == "get_weather" and "paris" in tc.get("arguments", "").lower(), f"{tc}")


def webui(url, model):
    for _ in range(150):
        try:
            if json.load(call(url + "/health", timeout=5)).get("status"):
                break
        except Exception:
            time.sleep(2)
    token = json.load(call(url + "/api/v1/auths/signin", {"email": "", "password": ""}))["token"]
    ids = [m["id"] for m in json.load(call(url + "/api/models", token=token))["data"]]
    ok = check("Open WebUI lists the model", model in ids, f"/api/models: {ids}")

    def chat(msg):  # the browser's request (Open WebUI 0.11): features, params, a session (with which Open WebUI offers the model its built-in tools)
        body = {"stream": True, "model": model, "messages": [{"role": "user", "content": msg}], "params": {}, "tool_servers": [],
                "features": {"voice": False, "image_generation": False, "code_interpreter": False, "web_search": False, "memory": True},
                "variables": {"{{USER_NAME}}": "User", "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d")}, "session_id": "acceptance"}
        text, tools = [], []
        with call(url + "/api/chat/completions", body, token) as r:
            for d in stream(r):
                for c in d.get("choices", []):
                    delta = c.get("delta", {})
                    text.append(delta.get("content") or "")
                    tools += [t["function"]["name"] for t in delta.get("tool_calls") or [] if t.get("function", {}).get("name")]
        return "".join(text), tools

    a, _ = chat("What is the capital of France? Answer in one word. /no_think")
    ok &= check("Open WebUI chat (the browser's request, its tools on)", "paris" in a.lower(), f"{a.strip()[:80]!r}")
    a, tools = chat("What is the current Unix timestamp? Use your tools. /no_think")
    return ok & check("Open WebUI chat, a tool call in the stream", "get_current_timestamp" in tools, f"tool calls {tools}, text {a.strip()[:60]!r}")


sys.exit(0 if (api(sys.argv[2]) if sys.argv[1] == "api" else webui(sys.argv[2], sys.argv[3])) else 1)
