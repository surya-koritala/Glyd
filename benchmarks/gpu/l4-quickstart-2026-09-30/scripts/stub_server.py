# a stand-in for the vLLM server on :8000 (to try acceptance.sh without a GPU): /v1/models, and /v1/chat/completions that answers
# the script's questions and calls tools as a model with a tool parser does. Prints vLLM's startup line.
import json, http.server, sys, time

MODEL = "Qwen/Qwen3-8B"


def reply(body):
    msgs = body["messages"]
    last = msgs[-1]
    tools = {t["function"]["name"] for t in body.get("tools") or []}
    if last["role"] == "tool":
        return {"content": f"The result is {last['content']}."}
    text = " ".join(str(m.get("content") or "") for m in msgs if m["role"] == "user").lower()
    if "timestamp" in text and "get_current_timestamp" in tools:
        return {"tool_calls": [("get_current_timestamp", {})]}
    if "weather" in text and "get_weather" in tools:
        return {"tool_calls": [("get_weather", {"city": "Paris"})]}
    u = str(last.get("content") or "").lower()
    if "germany" in u:
        return {"content": "Berlin"}
    if "france" in u:
        return {"content": "Paris"}
    return {"content": "Hello."}


class H(http.server.BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self.send(200, json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "vllm", "root": MODEL, "max_model_len": 8192}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        r = reply(body)
        usage = {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
        calls = [{"index": i, "id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(r.get("tool_calls", []))]
        if body.get("stream"):
            def chunk(delta, fin=None, u=None):
                return "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}], **({"usage": u} if u else {})}) + "\n\n"
            out = chunk({"role": "assistant", "content": ""})
            out += chunk({"tool_calls": calls}) if calls else "".join(chunk({"content": w + " "}) for w in r["content"].split())
            out += chunk({}, "tool_calls" if calls else "stop", usage) + "data: [DONE]\n\n"
            self.send(200, out.encode(), "text/event-stream")
        else:
            msg = {"role": "assistant", "content": r.get("content"), **({"tool_calls": [{k: v for k, v in c.items() if k != "index"} for c in calls]} if calls else {})}
            self.send(200, json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": MODEL, "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}], "usage": usage}).encode())

    def log_message(self, *a):
        pass


print("Application startup complete", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
