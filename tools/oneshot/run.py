import json, re, sys, time, urllib.request
RENDERS = ("This chat shows Markdown. A ```mermaid block holding a flowchart (flowchart TD or LR) is drawn as a diagram, and an ```html or ```svg block can be previewed live. "
  "When a flowchart, process or architecture is asked for, draw it as a mermaid flowchart, not as text art. A page, game, animation or chart asked for goes in one ```html block, "
  "self-contained (inline CSS and JavaScript); the preview cannot make network requests. Tables and lists are plain Markdown, never inside a code block.")
tag = sys.argv[1]
body = {"messages": [{"role": "system", "content": RENDERS}, {"role": "user", "content": open("prompt.txt").read()}], "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3000))
dt = time.time() - t0
json.dump(r, open(f"reply-{tag}.json", "w"))
c = r["choices"][0]["message"]["content"]; u = r["usage"]
m = re.search(r"```html\n(.*?)```", c, re.S)
open(f"index-{tag}.html", "w").write(m.group(1) if m else "")
print(f"{u['completion_tokens']} tokens in {dt:.1f} s ({u['completion_tokens']/dt:.0f} tok/s), finish {r['choices'][0]['finish_reason']}, html {m.group(1).count(chr(10)) if m else 0} lines")
