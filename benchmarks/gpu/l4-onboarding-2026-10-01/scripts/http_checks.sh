#!/bin/bash
# The server on port $1, over HTTP: the chat page, the API, a streamed chat in two turns, and a conversation longer than the window.
P=$1; B=http://127.0.0.1:$P; PY=${PY:-python3}
echo "-- GET /: $(curl -s -o /tmp/page.html -w '%{http_code} %{content_type} %{size_download} bytes' $B/)  title: $(grep -o '<title>[^<]*' /tmp/page.html)"
echo "-- CSP: $(curl -sI $B/ | grep -i content-security-policy | cut -c1-140)"
echo "-- HEAD /: $(curl -sI $B/ | head -1)"
echo "-- /v1/models: $(curl -s $B/v1/models | $PY -c 'import sys,json; d=json.load(sys.stdin)["data"][0]; print(d["id"], "max_model_len", d["max_model_len"])' 2>&1 | tail -1)"
echo "-- /health: $(curl -s -o /dev/null -w '%{http_code}' $B/health)   /docs: $(curl -s -o /dev/null -w '%{http_code}' $B/docs)   /nope: $(curl -s -o /dev/null -w '%{http_code}' $B/nope)"
M=$(curl -s $B/v1/models | $PY -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
echo "-- streamed chat, two turns (first bytes of each SSE stream, and the fields seen):"
$PY - "$B" "$M" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1:3]
msgs = [{"role": "user", "content": "Say hi in three words."}]
for turn in (1, 2):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 300}).encode(), {"Content-Type": "application/json"})
    fields, text, usage, finish = set(), "", None, None
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or ():
                fields |= set((ch.get("delta") or {}).keys())
                text += (ch.get("delta") or {}).get("content") or ""
                finish = ch.get("finish_reason") or finish
    print(f"   turn {turn}: fields {sorted(fields)} finish {finish} usage {usage} answer {text[:60]!r}")
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Now in Spanish."}]
PY
echo "-- a tool call with tool_choice auto (what Open WebUI sends; the server must have its tool-call parser):"
$PY - "$B" "$M" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1:3]
tools = [{"type": "function", "function": {"name": "get_weather", "description": "The weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
body = {"model": model, "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], "tools": tools, "tool_choice": "auto", "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}
try:
    with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})) as r:
        ch = json.load(r)["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    print(f"   HTTP 200, finish {ch['finish_reason']}, tool calls {[(c['function']['name'], c['function']['arguments']) for c in calls]}")
except urllib.error.HTTPError as e:
    print("   HTTP", e.code, e.read()[:200])
PY
echo "-- a conversation longer than the window, through glyd's own chat client:"
$PY - "$B" "$M" <<'PY' 2>&1 | head -5
import sys
sys.path.insert(0, "/root/.local/share/uv/tools/glyd/lib/python3.12/site-packages")
from glyd.gpu import chat
api = chat.Api(sys.argv[1])
name, window = api.model()
c = chat.Chat(api, name, window)
c.turn("word " * (window + 500))
print("   messages left after the refused turn:", len(c.messages))
PY
