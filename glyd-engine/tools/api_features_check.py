"""The API features of a running `glyd-engine serve` (or `glyd serve`) as the OpenAI Python client sees them, each streamed and not, and, where there is a vLLM 0.30 server of the same model, the same requests against it.

    pip install openai
    python api_features_check.py http://127.0.0.1:8000 [--vllm http://127.0.0.1:8001] [--key KEY] [--out results.json] [--no-guard]
    python api_features_check.py --selftest          (no server, no openai: the script's own helpers)

What it checks, by feature (a line `ok` or `FAIL` for each, `SAME` or `DIFF` for what is compared with vLLM; exit status 1 if one FAILs):
  tools        auto: a call for a question that needs it (the id, the function, the arguments as JSON, finish_reason tool_calls, no text), the same streamed; a question that does not need one is an ordinary reply (what
               Open WebUI's every chat is); the result of a call in the history; tool_choice none, required, a named function; parallel_tool_calls false; the thinking apart from the call
  logprobs     chat logprobs and top_logprobs 0 to 20, streamed and not; a plain completion's logprobs; raw (the model's own, not the draw's): the same token's log-probability at temperature 0 and at top_k 1 with a
               temperature; the end token's entry; the legacy lists; top_logprobs 21 refused
  n            n of 3 and 8: indices, finish reasons, usage (the prompt once, every choice's tokens), choices that differ and are reproduced by the seed (seed + index), streamed equal to whole, the prompt shared
               (the engine's /stats: the prompt run once); n above 8 and n with temperature 0 refused
  logit_bias   -100 on the token a greedy reply begins with changes it, +100 on a token makes the reply of that token, +100 on a token that is not among the 64 largest (the whole row is used), the same with the host
               sampling (GLYD_ENGINE_HOST_SAMPLING=1 on a development build: run this script against that server too)
  openwebui    the requests Open WebUI sends: a chat with its built-in tools offered (an ordinary answer), a question that asks for one, the follow-up after a tool's result, its title request
  other        stop_token_ids, min_tokens, include_stop_str_in_output, skip_special_tokens, echo, prompt as token ids and as a list, request_id, reasoning_effort, allowed_token_ids; response_format json_object and
               json_schema refused in words that name GLYD_ENGINE=0; fields that are not done refused
  loopback     (the engine, unless --no-guard) a Host that is not localhost, 127.0.0.1 or [::1] is 421, another site's Origin 403 and its preflight 403, the page's own origin and a program with none are answered; with
               --key the API is 401 without it, with a wrong one, with another scheme, and open for /health and the page
"""
import argparse
import http.client
import json
import sys
import urllib.parse

RESULTS = []
FAILED = []


def say(kind, name, detail=""):
    print(f"{kind:5} {name}" + (f" -- {detail}" if detail != "" else ""), flush=True)
    RESULTS.append({"kind": kind, "name": name, "detail": str(detail)[:2000]})
    if kind == "FAIL":
        FAILED.append(name)


def check(name, ok, detail=""):
    say("ok" if ok else "FAIL", name, detail)
    return ok


def same(name, a, b):
    say("SAME" if a == b else "DIFF", name, "" if a == b else f"engine {a!r}  vLLM {b!r}")


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Paris"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Get the current time in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
ASK_WEATHER = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]


# ---- what a reply is, put together from the client's objects (whole) or its chunks (streamed) ------------------------------------------------------

def whole_of(resp):
    """A chat completion (the client's object) as plain dicts: {index: {content, reasoning, calls: [(id, name, arguments)], finish, lps: [entry], text}}, and the usage."""
    out = {}
    for c in resp.choices:
        m = c.message
        extra = getattr(m, "model_extra", None) or {}
        out[c.index] = {
            "content": m.content,
            "reasoning": extra.get("reasoning_content") or extra.get("reasoning") or "",
            "calls": [(t.id, t.function.name, t.function.arguments) for t in (m.tool_calls or [])],
            "finish": c.finish_reason,
            "lps": [lp_entry(e) for e in (c.logprobs.content if c.logprobs and c.logprobs.content else [])],
        }
    return out, usage_of(resp)


def lp_entry(e):
    return {"token": e.token, "logprob": e.logprob, "bytes": list(e.bytes) if e.bytes is not None else None, "top": [(t.token, t.logprob) for t in e.top_logprobs]}


def usage_of(resp):
    u = getattr(resp, "usage", None)
    return None if u is None else {"prompt": u.prompt_tokens, "completion": u.completion_tokens, "total": u.total_tokens}


def streamed_of(chunks):
    """The chunks of a streamed chat completion put together, as `whole_of` has them (plus: roles, the order of the chunks): ({index: ...}, usage, roles)."""
    out, usage, roles = {}, None, []
    for ch in chunks:
        if getattr(ch, "usage", None):
            usage = {"prompt": ch.usage.prompt_tokens, "completion": ch.usage.completion_tokens, "total": ch.usage.total_tokens}
        for c in ch.choices:
            o = out.setdefault(c.index, {"content": "", "reasoning": "", "calls": {}, "finish": None, "lps": []})
            d = c.delta
            extra = getattr(d, "model_extra", None) or {}
            if d.role:
                roles.append(c.index)
            o["content"] += d.content or ""
            o["reasoning"] += extra.get("reasoning_content") or extra.get("reasoning") or ""
            for t in d.tool_calls or []:
                call = o["calls"].setdefault(t.index, {"id": None, "name": None, "arguments": ""})
                if t.id:
                    call["id"] = t.id
                if t.function and t.function.name:
                    call["name"] = t.function.name
                if t.function and t.function.arguments:
                    call["arguments"] += t.function.arguments
            if c.logprobs and c.logprobs.content:
                o["lps"].extend(lp_entry(e) for e in c.logprobs.content)
            if c.finish_reason:
                o["finish"] = c.finish_reason
    for o in out.values():
        o["calls"] = [(v["id"], v["name"], v["arguments"]) for _, v in sorted(o["calls"].items())]
    return out, usage, roles


def args_of(call):
    try:
        return json.loads(call[2])
    except (TypeError, ValueError):
        return None


# ---- the servers ----------------------------------------------------------------------------------------------------------------------------------

class Server:
    def __init__(self, base, key, label):
        from openai import OpenAI

        self.base, self.label = base.rstrip("/"), label
        self.client = OpenAI(base_url=self.base + "/v1", api_key=key or "none", timeout=600, max_retries=0)
        self.model = self.client.models.list().data[0].id

    def chat(self, **kw):
        extra = kw.pop("extra_body", None)
        return self.client.chat.completions.create(model=self.model, extra_body=extra, **kw)

    def complete(self, **kw):
        extra = kw.pop("extra_body", None)
        return self.client.completions.create(model=self.model, extra_body=extra, **kw)

    def refused(self, path, body):
        """(status, message) of a request the server refuses (a client's own errors are exceptions: the raw call keeps the words)."""
        conn = http.client.HTTPConnection(*hostport(self.base), timeout=120)
        headers = {"Content-Type": "application/json"}
        if self.client.api_key and self.client.api_key != "none":
            headers["Authorization"] = "Bearer " + self.client.api_key
        conn.request("POST", path, json.dumps(dict(model=self.model, **body)), headers)
        r = conn.getresponse()
        data = r.read().decode()
        try:
            msg = json.loads(data).get("error", {}).get("message", data)
        except ValueError:
            msg = data
        return r.status, msg if isinstance(msg, str) else json.dumps(msg)

    def stats(self):
        conn = http.client.HTTPConnection(*hostport(self.base), timeout=30)
        conn.request("GET", "/stats")
        r = conn.getresponse()
        data = r.read()
        return json.loads(data) if r.status == 200 else None


def hostport(base):
    u = urllib.parse.urlparse(base)
    return u.hostname, u.port or 80


# ---- the features -----------------------------------------------------------------------------------------------------------------------------------

def tools(eng, ref):
    kw = dict(messages=ASK_WEATHER, tools=[WEATHER, TIME], temperature=0, max_tokens=300, extra_body=NOTHINK)
    for srv in [eng] + ([ref] if ref else []):
        r, usage = whole_of(srv.chat(**kw))
        c = r[0]
        ok = check(f"[{srv.label}] tools auto: a call for a question that needs one", c["finish"] == "tool_calls" and len(c["calls"]) >= 1, c)
        if ok:
            cid, name, _ = c["calls"][0]
            a = args_of(c["calls"][0])
            check(f"[{srv.label}] the call has an id, the function, JSON arguments with the city", cid.startswith("chatcmpl-tool-") and name == "get_weather" and isinstance(a, dict) and "paris" in str(a.get("city", "")).lower(), c["calls"])
            check(f"[{srv.label}] the text beside a call is none or white space", not (c["content"] or "").strip(), repr(c["content"]))
        s, usage_s, roles = streamed_of(srv.chat(stream=True, stream_options={"include_usage": True}, **kw))
        sc = s[0]
        check(f"[{srv.label}] streamed: the same call, finish tool_calls, the usage", sc["finish"] == "tool_calls" and [(x[1], args_of(x)) for x in sc["calls"]][:1] == [(c["calls"][0][1], args_of(c["calls"][0]))] and usage_s is not None, sc)
        r2, _ = whole_of(srv.chat(messages=[{"role": "user", "content": "What is 2 plus 3? Answer in one word."}], tools=[WEATHER, TIME], temperature=0, max_tokens=60, extra_body=NOTHINK))
        check(f"[{srv.label}] tools offered and not needed: an ordinary reply", r2[0]["finish"] == "stop" and not r2[0]["calls"] and bool((r2[0]["content"] or "").strip()), r2[0])
        s2, _, _ = streamed_of(srv.chat(stream=True, messages=[{"role": "user", "content": "What is 2 plus 3? Answer in one word."}], tools=[WEATHER, TIME], temperature=0, max_tokens=60, extra_body=NOTHINK))
        check(f"[{srv.label}] the same streamed", s2[0]["finish"] == "stop" and not s2[0]["calls"] and bool(s2[0]["content"].strip()), s2[0])
        # the result of a call, in the history (the arguments as a string, the content null, as Open WebUI sends them)
        history = ASK_WEATHER + [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature_c\": 21, \"sky\": \"sunny\"}"}]
        r3, _ = whole_of(srv.chat(messages=history, tools=[WEATHER, TIME], temperature=0, max_tokens=120, extra_body=NOTHINK))
        text = (r3[0]["content"] or "").lower()
        check(f"[{srv.label}] a tool's result in the history is answered from", r3[0]["finish"] == "stop" and ("21" in text or "sunny" in text), text)
        # the choice among the tools
        r4, _ = whole_of(srv.chat(messages=ASK_WEATHER, tools=[WEATHER, TIME], tool_choice="none", temperature=0, max_tokens=120, extra_body=NOTHINK))
        check(f"[{srv.label}] tool_choice none: no call is read", not r4[0]["calls"], r4[0])
        r5, _ = whole_of(srv.chat(messages=[{"role": "user", "content": "Say hello."}], tools=[WEATHER, TIME], tool_choice="required", temperature=0, max_tokens=300, extra_body=NOTHINK))
        check(f"[{srv.label}] tool_choice required: a call even for a question that needs none", len(r5[0]["calls"]) >= 1 and r5[0]["finish"] == "tool_calls", r5[0])
        r6, _ = whole_of(srv.chat(messages=[{"role": "user", "content": "Tell me about Rome."}], tools=[WEATHER, TIME], tool_choice={"type": "function", "function": {"name": "get_time"}}, temperature=0, max_tokens=300, extra_body=NOTHINK))
        check(f"[{srv.label}] a named function: its call, finish stop as vLLM's", r6[0]["calls"] and r6[0]["calls"][0][1] == "get_time" and r6[0]["finish"] == "stop", r6[0])
        two = [{"role": "user", "content": "Get the weather in Paris and the weather in Rome, both, with two calls."}]
        r7, _ = whole_of(srv.chat(messages=two, tools=[WEATHER], temperature=0, max_tokens=400, extra_body=NOTHINK))
        r8, _ = whole_of(srv.chat(messages=two, tools=[WEATHER], parallel_tool_calls=False, temperature=0, max_tokens=400, extra_body=NOTHINK))
        check(f"[{srv.label}] parallel_tool_calls false: one call at the most", len(r8[0]["calls"]) <= 1 and len(r8[0]["calls"]) <= len(r7[0]["calls"]), (len(r7[0]["calls"]), len(r8[0]["calls"])))
        r9, _ = whole_of(srv.chat(messages=ASK_WEATHER, tools=[WEATHER], temperature=0, max_tokens=1500))
        check(f"[{srv.label}] with the thinking on: the thinking apart, then the call", r9[0]["finish"] in ("tool_calls", "length", "stop") and (r9[0]["reasoning"] or r9[0]["calls"]), {k: str(v)[:200] for k, v in r9[0].items()})
    if ref:
        a, _ = whole_of(eng.chat(**kw))
        b, _ = whole_of(ref.chat(**kw))
        same("tools auto: the function and the arguments", [(x[1], args_of(x)) for x in a[0]["calls"]], [(x[1], args_of(x)) for x in b[0]["calls"]])
        same("tools auto: the text beside the call", (a[0]["content"] or "").strip(), (b[0]["content"] or "").strip())


def logprobs(eng, ref):
    kw = dict(messages=[{"role": "user", "content": "Name three colours."}], temperature=0, max_tokens=24, logprobs=True, top_logprobs=5, extra_body=NOTHINK)
    for srv in [eng] + ([ref] if ref else []):
        r, usage = whole_of(srv.chat(**kw))
        lps = r[0]["lps"]
        check(f"[{srv.label}] chat logprobs: an entry for each token (the end token's too), top_logprobs of 5, the chosen token first (greedy)", len(lps) == usage["completion"] and all(len(e["top"]) == 5 and abs(e["top"][0][1] - e["logprob"]) < 1e-3 and e["logprob"] <= 0 for e in lps), (len(lps), usage))
        check(f"[{srv.label}] the 5 are in order, with their bytes", all(all(a[1] >= b[1] for a, b in zip(e["top"], e["top"][1:])) and e["bytes"] is not None for e in lps))
        s, usage_s, _ = streamed_of(srv.chat(stream=True, stream_options={"include_usage": True}, **kw))
        check(f"[{srv.label}] streamed: the same tokens", [e["token"] for e in s[0]["lps"]] == [e["token"] for e in lps], (len(s[0]["lps"]), len(lps)))
        r0, _ = whole_of(srv.chat(**dict(kw, top_logprobs=0)))
        check(f"[{srv.label}] logprobs with no top_logprobs: the chosen token's alone", all(e["top"] == [] for e in r0[0]["lps"]) and len(r0[0]["lps"]) == len(lps))
        r20, _ = whole_of(srv.chat(**dict(kw, top_logprobs=20, max_tokens=4)))
        check(f"[{srv.label}] top_logprobs 20", all(len(e["top"]) == 20 for e in r20[0]["lps"]))
        # raw: the log-probability of the first token does not depend on the draw (temperature, top_k), as vLLM's default mode gives it
        a, _ = whole_of(srv.chat(**dict(kw, max_tokens=1)))
        b, _ = whole_of(srv.chat(**dict(kw, max_tokens=1, temperature=0.5, top_p=1.0, extra_body=dict(NOTHINK, top_k=1))))
        check(f"[{srv.label}] the log-probability is the model's, not the draw's (top_k 1 at temperature 0.5 is the same number)", a[0]["lps"][0]["token"] == b[0]["lps"][0]["token"] and abs(a[0]["lps"][0]["logprob"] - b[0]["lps"][0]["logprob"]) < 0.05, (a[0]["lps"][0]["logprob"], b[0]["lps"][0]["logprob"]))
        c = srv.complete(prompt="The capital of France is", max_tokens=6, temperature=0, logprobs=3)
        lp = c.choices[0].logprobs
        check(f"[{srv.label}] a plain completion's logprobs: tokens, token_logprobs, top_logprobs of 3, text_offset", len(lp.tokens) == len(lp.token_logprobs) == len(lp.top_logprobs) == len(lp.text_offset) and all(len(t) in (3, 4) for t in lp.top_logprobs) and lp.text_offset[0] == 0, (lp.tokens, lp.text_offset))
        st, msg = srv.refused("/v1/chat/completions", dict(messages=[{"role": "user", "content": "x"}], logprobs=True, top_logprobs=21))
        check(f"[{srv.label}] top_logprobs 21 is refused in vLLM's words", st == 400 and "greater than max allowed: 20" in msg, msg)
    if ref:
        a, _ = whole_of(eng.chat(**kw))
        b, _ = whole_of(ref.chat(**kw))
        same("logprobs: the tokens", [e["token"] for e in a[0]["lps"]], [e["token"] for e in b[0]["lps"]])
        n = min(len(a[0]["lps"]), len(b[0]["lps"]))
        worst = max((abs(x["logprob"] - y["logprob"]) for x, y in zip(a[0]["lps"][:n], b[0]["lps"][:n])), default=0)
        say("SAME" if worst < 0.1 else "DIFF", "logprobs: the largest difference of a token's log-probability (bf16 noise: under 0.1)", f"{worst:.4f}")


def several(eng, ref):
    q = dict(messages=[{"role": "user", "content": "Write one sentence about the sea."}], temperature=0.9, max_tokens=40, logprobs=True, extra_body=dict(NOTHINK, seed=11))
    one, usage1 = whole_of(eng.chat(**dict(q, n=1)))
    for srv in [eng] + ([ref] if ref else []):
        for n in (3, 8):
            r, usage = whole_of(srv.chat(**dict(q, n=n)))
            check(f"[{srv.label}] n of {n}: indices 0 to {n - 1}, a finish reason each", sorted(r) == list(range(n)) and all(c["finish"] in ("stop", "length") for c in r.values()), sorted(r))
            check(f"[{srv.label}] n of {n}: the usage is the prompt once and every choice's tokens", usage["prompt"] == usage1["prompt"] and usage["completion"] == sum(len(c["lps"]) for c in r.values()), (usage, usage1))
            check(f"[{srv.label}] n of {n}: the choices differ", len({c["content"] for c in r.values()}) > 1, [c["content"][:30] for c in r.values()])
        r1, _ = whole_of(srv.chat(**dict(q, n=3)))
        r2, _ = whole_of(srv.chat(**dict(q, n=3)))
        check(f"[{srv.label}] the same seed gives the same choices", {i: c["content"] for i, c in r1.items()} == {i: c["content"] for i, c in r2.items()})
        s, usage_s, roles = streamed_of(srv.chat(stream=True, stream_options={"include_usage": True}, **dict(q, n=3)))
        check(f"[{srv.label}] streamed: a role chunk for each choice first, each ended, the texts the whole ones'", sorted(roles[:3]) == [0, 1, 2] and all(s[i]["finish"] for i in s) and {i: s[i]["content"] for i in s} == {i: c["content"] for i, c in r1.items()} and usage_s and usage_s["completion"] == sum(len(c["lps"]) for c in r1.values()), ({i: s[i]["content"][:20] for i in s}, usage_s))
        st, msg = srv.refused("/v1/chat/completions", dict(messages=q["messages"], n=2, temperature=0, max_tokens=8))
        check(f"[{srv.label}] n of 2 at temperature 0 is refused as vLLM's greedy sampling is", st == 400 and "n must be 1 when using greedy sampling" in msg, msg)
    # the prompt run once: the engine counts the prompt tokens it ran and those it found in its cache
    before = eng.stats()
    if before is not None:
        long = "The quick brown fox jumps over the lazy dog. " * 120
        eng.chat(messages=[{"role": "user", "content": long + "How many sentences were there?"}], temperature=0.9, max_tokens=4, n=4, extra_body=NOTHINK)
        after = eng.stats()
        ran, found = after["prompt_tokens"] - before["prompt_tokens"], after["prefix_hit_tokens"] - before["prefix_hit_tokens"]
        check("[engine] n of 4: the prompt was run once, the other three found its blocks in the cache", found >= 0.7 * ran * 3 / 4, f"{ran} prompt tokens counted (4 prompts), {found} found in the cache")
    st, msg = eng.refused("/v1/chat/completions", dict(messages=q["messages"], n=9))
    check("[engine] n of 9 is refused (the engine runs 8 sequences)", st == 400 and "at most 8" in msg, msg)


def bias(eng, ref):
    base = dict(messages=[{"role": "user", "content": "Name a colour."}], temperature=0, max_tokens=8, logprobs=True, top_logprobs=20, extra_body=dict(NOTHINK, return_tokens_as_token_ids=True))
    for srv in [eng] + ([ref] if ref else []):
        r, _ = whole_of(srv.chat(**base))
        first = r[0]["lps"][0]
        ids = [int(t.split(":")[1]) for t, _ in first["top"]]
        top, second, far = ids[0], ids[1], ids[-1]
        r1, _ = whole_of(srv.chat(**dict(base, logit_bias={str(top): -100}, extra_body=dict(NOTHINK, return_tokens_as_token_ids=True))))
        check(f"[{srv.label}] a bias of -100 on the token the reply begins with: it does not begin with it", r1[0]["lps"][0]["token"] != first["token"], r1[0]["lps"][0]["token"])
        for tid, what in ((second, "a token among the largest"), (far, "a token that is the twentieth")):
            r2, _ = whole_of(srv.chat(**dict(base, max_tokens=5, logit_bias={str(tid): 100}, extra_body=dict(NOTHINK, return_tokens_as_token_ids=True))))
            check(f"[{srv.label}] a bias of +100 on {what}: the reply is that token", {e["token"] for e in r2[0]["lps"][:4]} == {f"token_id:{tid}"}, [e["token"] for e in r2[0]["lps"]])
        r3, _ = whole_of(srv.chat(**dict(base, max_tokens=4, temperature=1.0, logit_bias={"1000": 100}, extra_body=dict(NOTHINK, return_tokens_as_token_ids=True, top_k=20))))
        check(f"[{srv.label}] a bias on a token that is nowhere among the largest, drawn with top_k 20: it is the token", r3[0]["lps"][0]["token"] == "token_id:1000", r3[0]["lps"][0]["token"])
        st, msg = srv.refused("/v1/chat/completions", dict(messages=base["messages"], logit_bias={"x": 1}))
        check(f"[{srv.label}] a key that is no token id is refused", st == 400 and "cannot be converted to integer token IDs" in msg, msg)
        st, msg = srv.refused("/v1/chat/completions", dict(messages=base["messages"], logit_bias={"99999999": 1}))
        check(f"[{srv.label}] a token id out of the vocabulary is refused", st == 400 and "out-of-vocab" in msg, msg)
    if ref:
        a, _ = whole_of(eng.chat(**dict(base, logit_bias={"1000": 3})))
        b, _ = whole_of(ref.chat(**dict(base, logit_bias={"1000": 3})))
        same("logit_bias: the reply with a bias that moves a choice", [e["token"] for e in a[0]["lps"]], [e["token"] for e in b[0]["lps"]])


def other(eng, ref):
    msgs = [{"role": "user", "content": "Count from one to five in words."}]
    for srv in [eng] + ([ref] if ref else []):
        r, _ = whole_of(srv.chat(messages=msgs, temperature=0, max_tokens=40, logprobs=True, extra_body=dict(NOTHINK, return_tokens_as_token_ids=True)))
        ids = [e["token"] for e in r[0]["lps"]]
        stop_id = int(ids[3].split(":")[1])
        r2, u2 = whole_of(srv.chat(messages=msgs, temperature=0, max_tokens=40, extra_body=dict(NOTHINK, stop_token_ids=[stop_id])))
        check(f"[{srv.label}] stop_token_ids: the reply ends before the token, finish stop, the token counted", r2[0]["finish"] == "stop" and u2["completion"] == 4, (r2[0]["content"], u2))
        r3, u3 = whole_of(srv.chat(messages=[{"role": "user", "content": "Say hi."}], temperature=0, max_tokens=30, extra_body=dict(NOTHINK, min_tokens=20, ignore_eos=False)))
        check(f"[{srv.label}] min_tokens 20: at least 20 tokens", u3["completion"] >= 20, u3)
        r4, _ = whole_of(srv.chat(messages=msgs, temperature=0, max_tokens=40, stop=["three"], extra_body=dict(NOTHINK, include_stop_str_in_output=True)))
        check(f"[{srv.label}] include_stop_str_in_output: the text ends with the stop string", (r4[0]["content"] or "").endswith("three"), r4[0]["content"])
        r5 = srv.complete(prompt="Hello", max_tokens=5, temperature=0, echo=True)
        check(f"[{srv.label}] echo: the prompt comes first", r5.choices[0].text.startswith("Hello") and len(r5.choices[0].text) > 5, r5.choices[0].text)
        r6 = srv.complete(prompt=["Hello", "Bonjour", "Hola"], max_tokens=3, temperature=0)
        check(f"[{srv.label}] a list of prompts: a choice each, in order", [c.index for c in r6.choices] == [0, 1, 2], [c.index for c in r6.choices])
        r7 = srv.complete(prompt=[9707, 11, 1879], max_tokens=3, temperature=0)
        check(f"[{srv.label}] a prompt as token ids", len(r7.choices) == 1 and r7.usage.prompt_tokens == 3, r7.usage)
        r8 = srv.complete(prompt="Once upon a time", temperature=0)
        check(f"[{srv.label}] a plain completion with no max_tokens is 16 tokens", r8.usage.completion_tokens == 16, r8.usage)
        r9 = srv.complete(prompt="Hello", max_tokens=3, temperature=0, extra_body={"request_id": "mine-1"})
        check(f"[{srv.label}] request_id is the id", r9.id.endswith("mine-1"), r9.id)
        r10, _ = whole_of(srv.chat(messages=[{"role": "user", "content": "What is 2+2?"}], temperature=0, max_tokens=300, reasoning_effort="none"))
        check(f"[{srv.label}] reasoning_effort none: no thinking", not r10[0]["reasoning"], r10[0]["reasoning"][:80])
        only_id = int(ids[0].split(":")[1])
        r11, _ = whole_of(srv.chat(messages=msgs, temperature=0, max_tokens=6, logprobs=True, extra_body=dict(NOTHINK, allowed_token_ids=[only_id], return_tokens_as_token_ids=True)))
        check(f"[{srv.label}] allowed_token_ids: only that token", {e["token"] for e in r11[0]["lps"]} == {f"token_id:{only_id}"}, [e["token"] for e in r11[0]["lps"]])
    for what, body in (
        ("response_format json_object", dict(response_format={"type": "json_object"})),
        ("response_format json_schema", dict(response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}})),
        ("structured_outputs", dict(structured_outputs={"json": {"type": "object"}})),
        ("bad_words", dict(bad_words=["x"])),
        ("prompt_logprobs", dict(prompt_logprobs=1)),
    ):
        st, msg = eng.refused("/v1/chat/completions", dict(messages=msgs, **body))
        check(f"[engine] {what}: refused in words that name the way to vLLM", st == 400 and "not supported" in msg and "GLYD_ENGINE=0" in msg, msg)


def openwebui(eng, ref):
    """The requests Open WebUI sends (0.11.x): a chat that carries its built-in tools whatever the question is, streamed with the usage asked for; the follow-up after a call (the call in the history with its
    arguments as a string and no content, its result as a tool message); and the title it asks for in a request of its own, not streamed, with no tools."""
    tools_ow = [
        {"type": "function", "function": {"name": "search_web", "description": "Search the web for information.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "count": {"type": "integer", "description": "Results to return", "default": 5}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "get_current_timestamp", "description": "Get the current Unix timestamp in seconds.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "view_knowledge_file", "description": "View a file of a knowledge base.", "parameters": {"type": "object", "properties": {"file_id": {"type": "string"}}, "required": ["file_id"]}}},
    ]
    chat = dict(messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France? Answer in one word."}], tools=tools_ow, temperature=0.7, top_p=0.9, max_tokens=512, extra_body=NOTHINK)
    for srv in [eng] + ([ref] if ref else []):
        s, usage, roles = streamed_of(srv.chat(stream=True, stream_options={"include_usage": True}, **chat))
        c = s[0]
        check(f"[{srv.label}] Open WebUI's chat (its tools offered, streamed, usage asked for): an ordinary answer, finish stop, the usage", c["finish"] == "stop" and not c["calls"] and "paris" in c["content"].lower() and usage is not None, (c["content"][:60], usage))
        r, _ = whole_of(srv.chat(**dict(chat, messages=[{"role": "user", "content": "Search the web for glyd compression and tell me the first result's title."}])))
        check(f"[{srv.label}] a question that asks for a tool: the call is search_web with a query", r[0]["calls"] and r[0]["calls"][0][1] == "search_web" and isinstance((args_of(r[0]["calls"][0]) or {}).get("query"), str), r[0])
        follow = [{"role": "user", "content": "Search the web for glyd compression."}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_ab12cd34", "type": "function", "function": {"name": "search_web", "arguments": "{\"query\": \"glyd compression\"}"}}]}, {"role": "tool", "tool_call_id": "call_ab12cd34", "content": "{\"results\": [{\"title\": \"Glyd: lossless compression\", \"url\": \"https://getglyd.com\"}]}"}]
        s2, _, _ = streamed_of(srv.chat(stream=True, stream_options={"include_usage": True}, **dict(chat, messages=follow)))
        check(f"[{srv.label}] the follow-up after a tool's result is answered from it", s2[0]["finish"] == "stop" and ("glyd" in s2[0]["content"].lower()), s2[0]["content"][:100])
        title = "### Task:\nGenerate a concise, 3-5 word title with an emoji summarizing the chat history.\n### Output:\nJSON format: { \"title\": \"your concise title here\" }\n### Chat History:\n<chat_history>\nUSER: What is the capital of France?\nASSISTANT: Paris.\n</chat_history>"
        r2, u2 = whole_of(srv.chat(messages=[{"role": "user", "content": title}], temperature=0.1, max_tokens=1000, extra_body=NOTHINK))
        check(f"[{srv.label}] its title request: an answer, finish stop", r2[0]["finish"] == "stop" and bool((r2[0]["content"] or "").strip()), r2[0]["content"])


# ---- the loopback checks and the key (the engine's own server, as the Python server's were) ----------------------------------------------------

def raw(base, method, path, headers):
    conn = http.client.HTTPConnection(*hostport(base), timeout=60)
    conn.request(method, path, headers=headers)
    r = conn.getresponse()
    return r.status, r.read().decode(errors="replace"), {k.lower(): v for k, v in r.getheaders()}


def loopback(eng, key):
    base = eng.base
    port = hostport(base)[1]
    auth = {"Authorization": "Bearer " + key} if key else {}
    ok = lambda h, p="/v1/models", m="GET": raw(base, m, p, dict(auth, **h))[0]
    for host in ("localhost", f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"):
        check(f"[engine] Host {host} is answered", ok({"Host": host}) == 200)
    for host in ("evil.example", "evil.example:8000", "localhost.evil.example", "127.0.0.1.evil.example", "192.168.1.5:8000", "0.0.0.0:8000"):
        check(f"[engine] Host {host} (a page whose DNS points here) is 421, the page too", ok({"Host": host}) == 421 and ok({"Host": host}, "/") == 421)
    own = f"http://localhost:{port}"
    check("[engine] the page's own origin is answered", ok({"Host": f"localhost:{port}", "Origin": own}) == 200)
    for origin in ("http://evil.example", "https://localhost:8000", "http://localhost:5173", "null"):
        check(f"[engine] Origin {origin} is 403", ok({"Host": f"localhost:{port}", "Origin": origin}, m="POST", p="/v1/chat/completions") == 403)
    pre = {"Host": f"localhost:{port}", "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    check("[engine] another site's preflight is 403", raw(base, "OPTIONS", "/v1/chat/completions", pre)[0] == 403)
    own_pre = {"Host": f"localhost:{port}", "Origin": own, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    st, body, h = raw(base, "OPTIONS", "/v1/chat/completions", own_pre)
    check("[engine] the page's own preflight is answered with its origin", st == 200 and h.get("access-control-allow-origin") == own, (st, h))
    st, body, h = raw(base, "GET", "/", {"Host": f"localhost:{port}"})
    check("[engine] the page has its headers", st == 200 and "content-security-policy" in h and h.get("x-content-type-options") == "nosniff", h)
    if key:
        check("[engine] the key: no key is 401", raw(base, "GET", "/v1/models", {})[0] == 401)
        check("[engine] a wrong key is 401, and so is another scheme", raw(base, "GET", "/v1/models", {"Authorization": "Bearer wrong"})[0] == 401 and raw(base, "GET", "/v1/models", {"Authorization": "Basic " + key})[0] == 401)
        check("[engine] the right key, in any case of the scheme", raw(base, "GET", "/v1/models", {"Authorization": "bearer " + key})[0] == 200)
        check("[engine] the page and /health are open", raw(base, "GET", "/health", {})[0] == 200 and raw(base, "GET", "/", {})[0] == 200)
        check("[engine] an OPTIONS is not asked for the key", raw(base, "OPTIONS", "/v1/models", {})[0] != 401)


# ---- the script's own helpers, offline ----------------------------------------------------------------------------------------------------------

def selftest():
    class O:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, k):
            return None

    def chunk(index, content=None, calls=None, finish=None, role=None, usage=None, lps=None):
        delta = O(content=content, role=role, tool_calls=calls, model_extra={})
        return O(choices=[O(index=index, delta=delta, finish_reason=finish, logprobs=O(content=lps) if lps else None)], usage=usage)

    call = lambda i, id=None, name=None, args=None: O(index=i, id=id, function=O(name=name, arguments=args))
    e = lambda tok, lp: O(token=tok, logprob=lp, bytes=list(tok.encode()), top_logprobs=[O(token=tok, logprob=lp)])
    chunks = [
        chunk(0, content="", role="assistant"),
        chunk(0, calls=[call(0, "chatcmpl-tool-1", "f", None)]),
        chunk(0, calls=[call(0, None, None, '{"a":')]),
        chunk(0, calls=[call(0, None, None, ' 1}')], lps=[e("x", -0.5)]),
        chunk(0, finish="tool_calls"),
        O(choices=[], usage=O(prompt_tokens=10, completion_tokens=3, total_tokens=13)),
    ]
    out, usage, roles = streamed_of(chunks)
    assert out[0]["calls"] == [("chatcmpl-tool-1", "f", '{"a": 1}')] and out[0]["finish"] == "tool_calls" and roles == [0] and usage == {"prompt": 10, "completion": 3, "total": 13}
    assert args_of(out[0]["calls"][0]) == {"a": 1} and args_of(("i", "f", "not json")) is None
    assert out[0]["lps"][0]["token"] == "x" and out[0]["lps"][0]["top"] == [("x", -0.5)]
    resp = O(choices=[O(index=0, message=O(content=None, tool_calls=[O(id="c", function=O(name="f", arguments="{}"))], model_extra={"reasoning": "think"}), finish_reason="tool_calls", logprobs=None)], usage=O(prompt_tokens=1, completion_tokens=2, total_tokens=3))
    whole, u = whole_of(resp)
    assert whole[0]["calls"] == [("c", "f", "{}")] and whole[0]["reasoning"] == "think" and u == {"prompt": 1, "completion": 2, "total": 3}
    assert hostport("http://127.0.0.1:8123") == ("127.0.0.1", 8123)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", nargs="?", help="the engine's URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--vllm", help="a vLLM 0.30 server of the same model, for the comparisons (its tool-call parser hermes, its reasoning parser qwen3: what glyd serve starts)")
    ap.add_argument("--key", help="the API key the engine was started with (GLYD_API_KEY)")
    ap.add_argument("--out", help="write the results here as JSON")
    ap.add_argument("--no-guard", action="store_true", help="skip the Host and Origin checks (the server was started with --host)")
    ap.add_argument("--only", help="a comma list of: tools, logprobs, n, bias, other, openwebui, loopback")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.engine:
        ap.error("the engine's URL")
    eng = Server(a.engine, a.key, "engine")
    ref = Server(a.vllm, None, "vLLM") if a.vllm else None
    only = set(a.only.split(",")) if a.only else {"tools", "logprobs", "n", "bias", "other", "openwebui", "loopback"}
    for name, fn in (("tools", tools), ("logprobs", logprobs), ("n", several), ("bias", bias), ("other", other), ("openwebui", openwebui)):
        if name in only:
            try:
                fn(eng, ref)
            except Exception as e:  # (one feature's failure does not stop the others)
                say("FAIL", f"{name}: an exception", f"{type(e).__name__}: {e}")
    if "loopback" in only and not a.no_guard:
        try:
            loopback(eng, a.key)
        except Exception as e:
            say("FAIL", "loopback: an exception", f"{type(e).__name__}: {e}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"engine": a.engine, "vllm": a.vllm, "results": RESULTS}, f, indent=1)
    print(f"{len(RESULTS)} lines, {len(FAILED)} FAIL" + (": " + "; ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
