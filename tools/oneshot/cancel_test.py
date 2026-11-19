"""Stop and reuse on a running server: a long streamed answer dropped after 1 s must end in the engine within a second; a follow-up must reuse the conversation."""
import json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8092"
def post(body, stream=False):
    r = urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    return r if stream else json.load(r)
def status():
    return json.load(urllib.request.urlopen(URL + "/v1/glyd/status"))
msgs = [{"role": "user", "content": "Write a 3000-word story about a lighthouse keeper."}]
r = post({"messages": msgs, "stream": True, "max_tokens": 6000}, stream=True)
t0 = time.time()
while time.time() - t0 < 1.0:
    r.readline()
r.close()
t1 = time.time()
for _ in range(100):
    if status()["engine"]["state"] == "idle":
        break
    time.sleep(0.05)
stop_s = time.time() - t1
last = status()["requests"][0]
print(f"stop: engine idle {stop_s:.2f} s after the drop, {last['output']} tokens made")
assert stop_s < 1.5 and last["output"] < 1000, last
a = post({"messages": [{"role": "user", "content": "My name is Ada. Reply with one word: ok."}], "max_tokens": 20, "temperature": 0})
reply = a["choices"][0]["message"]["content"]
b = post({"messages": [{"role": "user", "content": "My name is Ada. Reply with one word: ok."}, {"role": "assistant", "content": reply}, {"role": "user", "content": "What is my name? One word."}], "max_tokens": 20, "temperature": 0})
row = status()["requests"][0]
print(f"reuse: {row['reused']} of {row['prompt']} prompt tokens reused; answer {b['choices'][0]['message']['content']!r}")
assert row["reused"] > 0 and "Ada" in b["choices"][0]["message"]["content"]
t = post({"messages": [{"role": "user", "content": "What is 17*23?"}], "max_tokens": 2000, "chat_template_kwargs": {"enable_thinking": True}})
m = t["choices"][0]["message"]
print(f"thinking: {len(m.get('reasoning_content') or '')} chars of reasoning, answer {m['content'][-60:]!r}")
assert m.get("reasoning_content") and "391" in m["content"]
n = post({"messages": [{"role": "user", "content": "What is 17*23? Just the number."}], "max_tokens": 50})
print(f"default (no thinking): {n['choices'][0]['message'].get('reasoning_content')!r} / {n['choices'][0]['message']['content']!r}")
assert not n["choices"][0]["message"].get("reasoning_content")
print("ALL OK")
