"""Token ids for the scoring runs: a user's request, then the text as the assistant's reply (a code block), cut to N tokens of the reply."""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
N = 5000
out = {}
for name, req, lang in [("loader", "Show me the source of js-yaml's lib/loader.js.", "js"), ("game", open("prompt.txt").read(), "html")]:
    p = tok.apply_chat_template([{"role": "user", "content": req}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    rid = tok(f"```{lang}\n" + open({"loader": "loader.js", "game": "game.html"}[name]).read(), add_special_tokens=False)["input_ids"][:N]
    out[name] = {"ids": pid + rid, "score_from": len(pid)}
    print(name, len(pid), len(rid))
for k, v in out.items():
    open(f"req-{k}.json", "w").write(json.dumps(v) + "\n")
