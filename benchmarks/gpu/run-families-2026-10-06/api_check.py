"""One glyd serve's API checks, as a user's client sends them: a tool call (and the turn after it), thinking apart from the answer, and a
prompt of ~90% of the context the server was started with. Standard library only. Usage: api_check.py BASE_URL OUT.json"""
import json, sys, time, urllib.request

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
MARKS = ("<think>", "</think>", "<|channel", "<|think|>", "<tool_call>", "<|tool_call", "[TOOL_CALLS]")


def post(path, body, timeout=900):
    t = time.time()
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), round(time.time() - t, 1), None
    except urllib.error.HTTPError as e:
        return None, round(time.time() - t, 1), f"HTTP {e.code}: {e.read()[:600].decode(errors='replace')}"
    except Exception as e:
        return None, round(time.time() - t, 1), repr(e)


models = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"]
model, max_len = models[0]["id"], models[0].get("max_model_len") or 0
res = {"model": model, "max_model_len": max_len}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city's name"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
r, s, err = post("/v1/chat/completions", {"model": model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 4096, "temperature": 0})
ok, call = False, None
if r:
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    if calls:
        call = calls[0]
        try:
            ok = call["function"]["name"] == "get_weather" and "paris" in json.loads(call["function"]["arguments"])["city"].lower()
        except Exception:
            ok = False
    ok = ok and not any(k in (m.get("content") or "") for k in MARKS)
res["tool_call"] = {"ok": ok, "seconds": s, "error": err, "finish": r and r["choices"][0]["finish_reason"], "message": r and r["choices"][0]["message"]}
if ok:  # the turn after the tool's answer
    msgs += [{"role": "assistant", "content": r["choices"][0]["message"].get("content") or "", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"city": "Paris", "sky": "sunny", "celsius": 21})}]
    r2, s2, err2 = post("/v1/chat/completions", {"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 4096, "temperature": 0})
    c2 = r2 and (r2["choices"][0]["message"].get("content") or "")
    res["tool_result_turn"] = {"ok": bool(c2 and "21" in c2 and not any(k in c2 for k in MARKS)), "seconds": s2, "error": err2, "content": c2}
# thinking apart: the answer in content, the thinking in reasoning(_content), no thinking marks in the answer
r, s, err = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "What is 17 times 23? Think it through, then give the number."}],
                                          "max_tokens": 6000, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}})
m = r["choices"][0]["message"] if r else {}
think, ans = (m.get("reasoning_content") or m.get("reasoning") or ""), (m.get("content") or "")
res["thinking"] = {"ok": bool(think.strip()) and "391" in ans and not any(k in ans for k in MARKS), "seconds": s, "error": err, "reasoning_chars": len(think),
                   "content": ans[:400], "finish": r and r["choices"][0]["finish_reason"]}
# a long prompt: ~90% of the context the server holds (at most 200,000 tokens)
target = min(int(0.9 * (max_len - 512)), 200_000) if max_len else 8000
line = "Line {}: the quick brown fox jumps over the lazy dog near the river bank.\n"
text = "".join(line.format(i) for i in range(target // 21))  # (about 19 tokens a line: under the target, not over the context)
r, s, err = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text + "\nWhich animal jumps over the dog? One word."}],
                                          "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=1500)
pt = r["usage"]["prompt_tokens"] if r else 0
res["long_prompt"] = {"ok": bool(r) and pt >= 0.6 * target, "target_tokens": target, "prompt_tokens": pt, "seconds": s, "error": err,
                      "content": r and (r["choices"][0]["message"].get("content") or "")[:200]}
res["ok"] = all(v.get("ok") for k, v in res.items() if isinstance(v, dict))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in res.items()}))
