"""Where a vLLM process's GPU memory is when the model is loaded (just before vLLM's profile run): every block the allocator holds, by what owns it:
the model's parameters and buffers by kind (the packs: data, blocks or exceptions, bases; norm weights; rotary caches; the rest), and what is no module's
(the plugin's decode buffer where it is held through the load, vLLM's own baseline, anything else), largest first; and the figure vLLM logs for the load
(its "Model loading took": the allocated bytes at the end of the load less those at its start). One vLLM (eager: nothing else is allocated before the profile
run) with --quantization glyd, then exit.
   ACCT_LAYOUT=mma|mma12 python acct.py MODEL [LABEL]      (the plugin's layout: mma, the smallest, by default)"""
import collections, itertools, os, sys
import torch
model_name, label = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
from vllm import LLM
import vllm.v1.worker.gpu_worker as gw

GiB = 2**30
def blocks():
    snap = torch.cuda.memory_snapshot()
    out = {}
    for seg in snap:
        addr = seg["address"]
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                out[addr] = b["size"]
            addr += b["size"]
    return out

def cat_of(name):
    leaf = name.rsplit(".", 1)[-1]
    if leaf.startswith("glyd_"):
        return "pack: " + ("table (embedding, LM head)" if ("embed_tokens" in name or "lm_head" in name) else "Linears")
    if "cos_sin" in name or "rotary" in name:
        return "rotary cache"
    if "norm" in name:
        return "norm weights"
    return "other tensors: " + leaf

orig = gw.Worker.determine_available_memory
def probe(self):
    model = self.model_runner.model
    torch.cuda.synchronize()
    active = blocks()
    total = torch.cuda.memory_allocated()
    owned = {}
    for name, t in itertools.chain(model.named_parameters(remove_duplicate=False), model.named_buffers(remove_duplicate=False)):
        if t.device.type != "cuda" or t.untyped_storage().nbytes() == 0:
            continue
        p = t.untyped_storage().data_ptr()
        if p not in owned:
            owned[p] = (cat_of(name), name)
    by = collections.Counter()
    for p, (c, _) in owned.items():
        by[c] += active.get(p, 0)
    mine = sum(active.get(p, 0) for p in owned)
    rest = sorted(((s, a) for a, s in active.items() if a not in owned), reverse=True)
    print(f"ACCT {label} {model_name}: allocated {total / GiB:.3f} GiB in {len(active)} blocks; the model's tensors {mine / GiB:.3f} GiB; everything else {(total - mine) / GiB:.3f} GiB", flush=True)
    for c, b in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"ACCT   {c}: {b / GiB:.4f} GiB ({b / 2**20:.1f} MiB)", flush=True)
    print(f"ACCT   no module's: {sum(s for s, _ in rest) / GiB:.4f} GiB in {len(rest)} blocks; the largest: " + ", ".join(f"{s / 2**20:.1f} MiB" for s, _ in rest[:8]), flush=True)
    packed = sum(b for c, b in by.items() if c.startswith("pack"))
    print(f"ACCT   the model's own tensors {mine / GiB:.4f} GiB (packs {packed / GiB:.4f}); bf16's checkpoint 15.26 GiB", flush=True)
    os._exit(0)
gw.Worker.determine_available_memory = probe
LLM(model=model_name, dtype="bfloat16", seed=0, max_model_len=2048, gpu_memory_utilization=0.9, enforce_eager=True, enable_prefix_caching=False, quantization="glyd", additional_config={"glyd": {"kv": "off", "layout": "mma"}})
