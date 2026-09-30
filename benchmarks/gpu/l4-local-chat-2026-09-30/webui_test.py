"""Open WebUI's API with no login (WEBUI_AUTH=False): the session it gives, the models it lists (vLLM's, through
OPENAI_API_BASE_URL), and one chat through it, whole and streamed (the first chunk's time from the request, the chunks a
second after it).
    python3 webui_test.py [WEBUI_URL] [MODEL]"""
import json
import sys
import time
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"


def call(path, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url + path, data=data, headers=headers), timeout=300)


print("health:", call("/health").read().decode())
# no login in this mode: a sign-in with empty fields gives the one built-in user's session (and its token)
token = json.load(call("/api/v1/auths/signin", {"email": "", "password": ""}))["token"]
ids = [m["id"] for m in json.load(call("/api/models", token=token))["data"]]
print("models:", ids)
assert model in ids, f"{model} not listed"

msgs = [{"role": "user", "content": "In one sentence, what is lossless compression? /no_think"}]
t0 = time.perf_counter()
r = json.load(call("/api/chat/completions", {"model": model, "messages": msgs, "max_tokens": 120}, token))
dt = time.perf_counter() - t0
print(f"chat, whole: {dt:.2f} s, usage {r.get('usage')}")
print("  assistant:", r["choices"][0]["message"]["content"].strip().replace("\n", " ")[:300])

msgs = [{"role": "user", "content": "Write a short story about a lighthouse keeper. /no_think"}]
t0 = time.perf_counter()
times, text = [], []
for line in call("/api/chat/completions", {"model": model, "messages": msgs, "max_tokens": 300, "stream": True}, token):
    line = line.decode().strip()
    if not line.startswith("data:") or line.endswith("[DONE]"):
        continue
    ch = json.loads(line[5:])
    piece = ch["choices"][0]["delta"].get("content") if ch.get("choices") else None
    if piece:
        times.append(time.perf_counter() - t0)
        text.append(piece)
n = len(times)
print(f"chat, streamed: {n} chunks, first at {times[0] * 1e3:.0f} ms, {(n - 1) / (times[-1] - times[0]):.1f} chunks/s after it")
print("  assistant:", "".join(text).strip().replace("\n", " ")[:300])
