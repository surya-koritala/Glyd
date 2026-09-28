# e2e.py --merge --baseline --profile N's bf16 profile alone (the model on the GPU in bf16, q,k,v and gate,up merged as e2e.py merges
# them): GPU time a step after the prompt at each batch size, as e2e.py computes it. For a model whose bf16 and Glyd copies
# do not fit at once (e2e.py's bf16 profile loads it again beside Glyd's).
#   python bf16prof.py MODEL_DIR 1,8,32,64 N
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_SRC", "~/glyd/gpu")))
import glyd_gpu  # noqa: F401 (the package from this checkout)
from glyd.gpu.model import merge_linears
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile as prof_, ProfilerActivity

d, batches, n = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(d)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda()
model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).eval()
print(f"merged: {merge_linears(model)} groups")
model.cuda()

def run(x, k):
    with prof_(activities=[ProfilerActivity.CUDA]) as pr:
        t = time.perf_counter()
        model.generate(inputs=x, max_new_tokens=k, min_new_tokens=k, do_sample=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    return {e.key: e.device_time_total for e in pr.key_averages() if e.device_type.name == "CUDA"}, wall

for b in batches:
    x = ids.repeat(b, 1)
    with torch.no_grad():
        model.generate(inputs=x, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()
        one, _ = run(x, 1)
        all_, wall = run(x, n + 1)
    step = {k: (v - one.get(k, 0)) / n for k, v in all_.items()}
    busy, prompt = sum(step.values()) / 1000, sum(one.values()) / 1000
    print(f"bf16 profile, batch {b}: {wall / (n + 1) * 1000:.2f} ms a step, GPU busy {busy:.2f} ms a step after the prompt ({b / busy * 1000:.0f} tokens/s of GPU time), {prompt:.2f} ms for the prompt and a token")
    for k, v in sorted(step.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   {v / 1000:7.3f} ms  {k[:90]}")
