"""The one-shot game prompt N times per sampling setting against a running server: how many finish, how many loop."""
import collections, json, re, sys, time, urllib.request
RENDERS = ("This chat shows Markdown. A ```mermaid block holding a flowchart (flowchart TD or LR) is drawn as a diagram, and an ```html or ```svg block can be previewed live. "
  "When a flowchart, process or architecture is asked for, draw it as a mermaid flowchart, not as text art. A page, game, animation or chart asked for goes in one ```html block, "
  "self-contained (inline CSS and JavaScript); the preview cannot make network requests. Tables and lists are plain Markdown, never inside a code block.")
URL, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = open(sys.argv[4]).read()
CONFIGS = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {"default": {}, "presence1": {"presence_penalty": 1.0}, "rep105": {"repetition_penalty": 1.05}}
for i in range(N):
    for name, extra in CONFIGS.items():
        body = {"messages": [{"role": "system", "content": RENDERS}, {"role": "user", "content": prompt}], "max_tokens": 24000,
                "chat_template_kwargs": {"enable_thinking": False}, **extra}
        t0 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3000))
        dt = time.time() - t0
        c = r["choices"][0]["message"]["content"]
        lines = [l.strip() for l in c.split("\n") if len(l.strip()) > 10]
        top = max(collections.Counter(lines).values(), default=0)
        # (a loop: a line 30 times, or numbered lines that only differ by digits 30 times)
        digits = max(collections.Counter(re.sub(r"\d+", "#", l) for l in lines).values(), default=0)
        html = re.search(r"```html\n(.*?)```", c, re.S)
        rec = {"config": name, "run": i, "tokens": r["usage"]["completion_tokens"], "seconds": round(dt, 1), "finish": r["choices"][0]["finish_reason"],
               "loop": top >= 30 or digits >= 60, "closed": bool(html and "</html>" in html.group(1).lower())}
        open(f"{OUT}/{name}-{i}.json", "w").write(json.dumps(r))
        print(json.dumps(rec), flush=True)
