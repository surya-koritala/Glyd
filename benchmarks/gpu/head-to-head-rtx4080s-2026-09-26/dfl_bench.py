# DFloat11 on this GPU, by e2e.py's measures: memory, tokens/s generating 64 tokens, and GPU time a step
# after the prompt (16 steps, the prompt's time apart), at each batch size.
import sys, time, torch
from dfloat11 import DFloat11Model
from transformers import AutoTokenizer
from torch.profiler import profile, ProfilerActivity
mid, batches = sys.argv[1], [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "1,8,32,64").split(",")]
model = DFloat11Model.from_pretrained(mid, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained(mid)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda()
torch.cuda.synchronize()
print(f"DFloat11 {mid}: loaded, GPU memory {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
for b in batches:
    x = ids.repeat(b, 1)
    with torch.no_grad():
        model.generate(x, max_new_tokens=4, do_sample=False)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(x, max_new_tokens=64, min_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    print(f"DFloat11: batch {b}: {64 * b / dt:.1f} tokens/s ({64 / dt:.1f} a sequence), peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
n = 16
def run(x, k):
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        model.generate(x, max_new_tokens=k, min_new_tokens=k, do_sample=False)
        torch.cuda.synchronize()
    return {e.key: e.device_time_total for e in pr.key_averages() if e.device_type.name == "CUDA"}
for b in batches:
    x = ids.repeat(b, 1)
    with torch.no_grad():
        one = run(x, 1)
        all_ = run(x, n + 1)
    step = {k: (v - one.get(k, 0)) / n for k, v in all_.items()}
    busy = sum(step.values()) / 1000
    print(f"DFloat11 profile, batch {b}: GPU busy {busy:.2f} ms a step after the prompt ({b / busy * 1000:.0f} tokens/s of GPU time), {sum(one.values()) / 1000:.2f} ms for the prompt and a token", flush=True)
    for k, v in sorted(step.items(), key=lambda kv: -kv[1])[:4]:
        print(f"   {v / 1000:7.3f} ms  {k[:90]}")
