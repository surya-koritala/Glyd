# what a thinking answer looks like through the OpenAI API with and without a reasoning parser: the fields of the stream's deltas,
# whether <think> reaches the content, and whether a tool call still parses with thinking on.
import json, sys, urllib.request

BASE = "http://127.0.0.1:8000/v1"
MODEL = "Qwen/Qwen3-8B"


def post(body):
    return urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)


keys, content, reasoning = {}, [], []
with post({"model": MODEL, "stream": True, "max_tokens": 1500, "temperature": 0, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                d = c["delta"]
                for k, v in d.items():
                    if v:
                        keys[k] = keys.get(k, 0) + 1
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
content, reasoning = "".join(content), "".join(reasoning)
print("thinking on, streamed: delta fields", keys)
print(f"  content has <think>: {'<think>' in content or '</think>' in content}; content: {content.strip()[:100]!r}; reasoning chars: {len(reasoning)}")
tool = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
r = json.load(post({"model": MODEL, "tool_choice": "auto", "tools": [tool], "max_tokens": 1500, "temperature": 0, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]}))
m = r["choices"][0]["message"]
print("thinking on, a tool call:", json.dumps(m.get("tool_calls")), "| finish:", r["choices"][0]["finish_reason"], "| content:", (m.get("content") or "").strip()[:80].replace("\n", " "), "| reasoning:", len(m.get("reasoning") or m.get("reasoning_content") or ""), "chars")
