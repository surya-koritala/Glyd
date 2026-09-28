# Each kernel of a generation step, in order, with its GPU time: glyd.from_pretrained (the checkout's package), B sequences,
# a step after the prompt profiled; the Linears' kernels named by their place in a layer (q,k,v; o; gate,up; down; lm_head).
#   python stepprof.py MODEL_DIR B [N]
import json, os, sys, collections, torch
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_SRC", "~/glyd/gpu")))
import glyd_gpu  # noqa: F401 (the kernels: JIT or the library, as the scripts)
import glyd
from transformers import AutoTokenizer
from torch.profiler import profile, ProfilerActivity

d, B = sys.argv[1], int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
tok = AutoTokenizer.from_pretrained(d)
if os.environ.get("BF16"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).cuda().eval()
else:
    model = glyd.from_pretrained(d)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda().repeat(B, 1)
with torch.no_grad():
    model.generate(ids, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        model.generate(ids, max_new_tokens=N + 1, min_new_tokens=N + 1, do_sample=False)
        torch.cuda.synchronize()
ev = [e for e in pr.events() if e.device_type.name == "CUDA"]
ev.sort(key=lambda e: e.time_range.start)
# The steps after the prompt: from the last N+1 lm_heads' ... simply: the kernels after the prompt's lm_head (the first big kernel run).
names = [e.name for e in ev]
tot = collections.defaultdict(float)
cnt = collections.Counter()
for e in ev:
    tot[e.name] += e.time_range.elapsed_us()
    cnt[e.name] += 1
print(f"{'glyd' if not os.environ.get('BF16') else 'bf16'} B={B}: {len(ev)} kernels over the prompt and {N + 1} tokens")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v / 1000:8.3f} ms  {cnt[k]:5d} calls  {v / cnt[k]:8.1f} us avg  {k[:80]}")
# The Linears' kernel (the most time) in order within the decode steps (4 a layer and the lm_head a step).
top = max(tot, key=lambda k: tot[k])
calls = [e for e in ev if e.name == top]
per = 4 * model.config.num_hidden_layers + 1
print("top kernel:", top[:100], len(calls), "calls,", len(calls) / per, "steps of", per)
role = lambda i: "lm_head" if i == per - 1 else ["qkv", "o", "gate_up", "down"][i % 4]
acc = collections.defaultdict(list)
for j, e in enumerate(calls[len(calls) % per:]):
    acc[role(j % per)].append(e.time_range.elapsed_us())
n = len(calls) // per
for r in ["qkv", "o", "gate_up", "down", "lm_head"]:
    v = acc[r]
    if v:
        print(f"  {r:8s} {sum(v) / len(v):8.1f} us avg, {sum(v) / max(n, 1) / 1000:7.3f} ms a step")

if os.environ.get("LAYERS"):
    for r in ["gate_up", "qkv", "o", "down"]:
        k = ["qkv", "o", "gate_up", "down"].index(r)
        v = [[acc[r][st * (per // 4) + l] for st in range(n)] for l in range(per // 4)] if False else None
        seq = acc[r]
        L = (per - 1) // 4
        lay = [sum(seq[st * L + l] for st in range(n)) / n for l in range(L)]
        print(f"  {r} by layer (us):", " ".join(f"{t:.0f}" for t in lay))
