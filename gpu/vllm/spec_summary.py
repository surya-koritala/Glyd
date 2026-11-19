"""spec_decode.py's runs (DIR/*.json) as tables: for each run and mix of prompts, output tokens/s for the one user (all
the requests' tokens over their whole times), the mean time to the first token, and speculation's acceptance (the
draft tokens accepted, and the tokens a step: 1 + those accepted a draft); then which runs gave the same tokens,
request by request, and where they first parted.

    python spec_summary.py DIR"""
import json
import os
import sys

D = sys.argv[1]
runs = {os.path.basename(p)[:-5]: json.load(open(os.path.join(D, p))) for p in sorted(os.listdir(D)) if p.endswith(".json")}
ORDER = ["bf16", "bf16-ngram", "bf16-eagle3", "bf16-eager", "glyd", "glyd-ngram", "glyd-eagle3", "glyd-eagle3-packed", "glyd-eagle3-packed-again",
         "bf16-tight", "bf16-eagle3-tight", "glyd-eagle3-tight", "bf16-eager-ngram", "glyd-exact-eager-ngram", "bf16-eager-eagle3",
         "glyd-exact-eager-eagle3", "glyd-exact-eager-eagle3-packed", "bi-bf16-eager", "bi-bf16-eager-ngram", "bi-glyd-exact-eager-ngram"]
names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]
print("| Run | Mix | Output tokens/s | TTFT mean (ms) | Draft tokens accepted | Tokens a step |")
print("| :--- | :--- | ---: | ---: | ---: | ---: |")
for n in names:
    for mix, m in runs[n]["mixes"].items():
        rows = m["rows"]
        toks, t = sum(len(r["tokens"]) for r in rows), sum(r["e2e_s"] for r in rows)
        ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        acc = f"{m['accepted'] / m['draft_tokens'] * 100:.0f}% ({m['accepted']} of {m['draft_tokens']})" if m["draft_tokens"] else ""
        step = f"{1 + m['accepted'] / m['drafts']:.2f}" if m["drafts"] else ""
        print(f"| {n} | {mix} | {toks / t:.1f} | {sum(ttft) / len(ttft) * 1e3:.0f} | {acc} | {step} |")


def same(a, b):
    """Requests with the same tokens, of all; and the first token where the others part (request, position)."""
    eq, first = 0, []
    for mix in runs[a]["mixes"]:
        for i, (x, y) in enumerate(zip(runs[a]["mixes"][mix]["rows"], runs[b]["mixes"][mix]["rows"])):
            x, y = x["tokens"], y["tokens"]
            if x == y:
                eq += 1
            else:
                k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                first.append(f"{mix} {i + 1} at {k}")
    total = sum(len(m["rows"]) for m in runs[a]["mixes"].values())
    return f"{eq} of {total}" + (f" (parted: {', '.join(first)})" if first else "")


print()
print("| Runs compared | Requests with the same tokens |")
print("| :--- | :--- |")
for a, b in [("bf16-ngram", "bf16"), ("bf16-eagle3-tight", "bf16-tight"), ("glyd-ngram", "glyd"), ("glyd-eagle3", "glyd"), ("glyd-eagle3-packed", "glyd"),
             ("glyd-eagle3-packed-again", "glyd-eagle3-packed"), ("glyd-exact-eager-ngram", "bf16-eager-ngram"),
             ("glyd-exact-eager-eagle3", "bf16-eager-eagle3"), ("glyd-exact-eager-eagle3-packed", "bf16-eager-eagle3"),
             ("bf16-eager-ngram", "bf16-eager"), ("bf16-eager-eagle3", "bf16-eager"), ("bi-bf16-eager-ngram", "bi-bf16-eager"), ("bi-glyd-exact-eager-ngram", "bi-bf16-eager-ngram"), ("bi-glyd-exact-eager-ngram", "bi-bf16-eager"),
             ("glyd", "bf16"), ("bf16-tight", "bf16")]:
    if a in runs and b in runs:
        print(f"| {a} against {b} | {same(a, b)} |")
print()
for n in names:
    r = runs[n]
    extra = f"; drafter {r['drafter']}" if "drafter" in r else ""
    print(f"{n}: load {r['load_s']} s; Glyd's key {r.get('glyd')}{extra}")
