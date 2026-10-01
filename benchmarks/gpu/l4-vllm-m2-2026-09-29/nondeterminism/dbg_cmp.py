"""dbg_hash.py's runs compared a pair at a time: the first product whose input or output differs, and the products
given the same input that gave another output. argv: DIR NAME:NAME ..."""
import json
import sys

D = sys.argv[1]
P = ("qkv", "o", "gate_up", "down")
for pair in sys.argv[2:]:
    a, b = (json.load(open(f"{D}/{n}.json")) for n in pair.split(":"))
    ca, cb = a["calls"], b["calls"]
    n = min(len(ca), len(cb))
    fx = next((i for i in range(n) if ca[i][3] != cb[i][3]), None)
    fy = next((i for i in range(n) if ca[i][4] != cb[i][4]), None)
    odd = [i for i in range(n) if ca[i][3] == cb[i][3] and ca[i][4] != cb[i][4]]
    where = lambda i: "none" if i is None else f"#{i} (layer {i // 4} {P[i % 4]}, O {ca[i][0]} K {ca[i][1]} M {ca[i][2]})"
    lp = sum(u != v for u, v in zip(a["long_logprobs"], b["long_logprobs"]))
    at = ""
    if a.get("attn") and b.get("attn"):
        aa, ab = a["attn"], b["attn"]
        fi = next((i for i in range(min(len(aa), len(ab))) if aa[i][1] != ab[i][1]), None)
        fo = next((i for i in range(min(len(aa), len(ab))) if aa[i][2] != ab[i][2]), None)
        oo = [i for i in range(min(len(aa), len(ab))) if aa[i][1] == ab[i][1] and aa[i][2] != ab[i][2]]
        qkv = "" if fi is None else " (query, key, value: " + ", ".join("same" if x == y else "differ" for x, y in zip(aa[fi][1], ab[fi][1])) + ")"
        at = f"; attention ({len(aa)} calls): first input differing {fi}{qkv}, first output differing {fo}, same inputs other output {len(oo)}"
    print(f"{pair}: {len(ca)} and {len(cb)} products; first input differing {where(fx)}; first output differing {where(fy)}; "
          f"same input, other output: {len(odd)} products{' (' + ', '.join(where(i) for i in odd[:4]) + ')' if odd else ''}; "
          f"logprobs differing {lp} of {len(a['long_logprobs'])}{at}; counters not zero {sum(a['counters_nonzero'].values())}, {sum(b['counters_nonzero'].values())}")
