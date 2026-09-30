"""What the server answers when a chat is longer than --max-model-len: a prompt of N tokens (the word "hello" repeated)
and max_tokens 16, to the server and to Open WebUI in front of it.
    python3 context_limit.py [N] [VLLM_URL] [WEBUI_URL]"""
import json
import sys
import urllib.error
import urllib.request

n = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
vllm = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
webui = sys.argv[3] if len(sys.argv) > 3 else None
body = {"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "hello " * n}], "max_tokens": 16}


def post(url, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), h), timeout=300)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null")


code, r = post(vllm + "/v1/chat/completions")
print(f"vLLM, {n} tokens asked: HTTP {code}: {json.dumps(r)[:400]}")
if webui:
    t = json.load(urllib.request.urlopen(urllib.request.Request(webui + "/api/v1/auths/signin", json.dumps({"email": "", "password": ""}).encode(), {"Content-Type": "application/json"})))["token"]
    code, r = post(webui + "/api/chat/completions", t)
    print(f"Open WebUI, {n} tokens asked: HTTP {code}: {json.dumps(r)[:400]}")
