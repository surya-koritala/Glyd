"""One answer from an OpenAI-compatible server, greedy: the chat endpoint, or (where the model has no chat template) a completion.
    python3 chat1.py URL MODEL      prints one JSON line: ok, how, text (exit 1 if there is no answer)"""
import json
import sys
import urllib.error
import urllib.request

url, model = sys.argv[1].rstrip("/"), sys.argv[2]


def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


how, text = "chat", ""
try:
    m = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
                                      "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]
    text = (m.get("content") or "").strip()
    if not text:
        how, text = "chat (reasoning only)", (m.get("reasoning_content") or m.get("reasoning") or "").strip()
except urllib.error.HTTPError as e:
    err = e.read().decode(errors="replace")[:160]
    try:
        how = f"completion (chat: HTTP {e.code})"
        text = post("/v1/completions", {"model": model, "prompt": "The capital of France is", "max_tokens": 24, "temperature": 0})["choices"][0]["text"].strip()
    except Exception as e2:  # noqa: BLE001
        print(json.dumps({"ok": False, "how": "none", "text": f"chat HTTP {e.code}: {err}; completion: {e2}"[:300]}))
        sys.exit(1)
except Exception as e:  # noqa: BLE001
    print(json.dumps({"ok": False, "how": "none", "text": f"{type(e).__name__}: {e}"[:300]}))
    sys.exit(1)
text = " ".join(text.split())
print(json.dumps({"ok": bool(text), "how": how, "paris": "paris" in text.lower(), "text": text[:200]}))
sys.exit(0 if text else 1)
