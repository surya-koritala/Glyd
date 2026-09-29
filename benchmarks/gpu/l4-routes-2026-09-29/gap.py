"""A long prompt's time to its first token through generate() against a plain forward pass over it (the same prompt,
logits_to_keep=1), in one process, to find what generate() adds. Per length, each the median of REPS after warm-ups:
  forward     model(x, logits_to_keep=1);
  gen1        generate(x, max_new_tokens=1): the time to the first token (a streamer's put);
  gen16       generate(x, max_new_tokens=16): its time to the first token;
  fwd-after   a forward pass right after a generate() of 16 tokens (the GPU as the next call finds it);
then one forward and one gen1 under torch.profiler: CUDA time by kernel, and the host's wall time.

    GLYD_GPU_LIB=LIB PYTHONPATH=TREE/bindings/python python gap.py MODEL --mode glyd|bf16 [--compile 1] [--lengths 2048,8192]"""
import argparse, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

TEXT = ("The history of data compression begins long before computers. Telegraph operators shortened common words to "
        "save time on the wire, and Morse gave the most frequent letters the shortest codes. ")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--mode", default="glyd", choices=["glyd", "bf16"])
ap.add_argument("--compile", type=int, default=1)
ap.add_argument("--lengths", default="2048,8192")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--profile", type=int, default=1)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model)
if args.mode == "bf16":
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
else:
    import glyd
    model = glyd.from_pretrained(args.model, compile=bool(args.compile)).eval()
ids = tok(TEXT * 400, return_tensors="pt").input_ids[0]


class Clock(BaseStreamer):
    def __init__(self):
        self.t = []

    def put(self, value):
        self.t.append(time.perf_counter())

    def end(self):
        pass


def fwd(x):
    torch.cuda.synchronize()
    t = time.perf_counter()
    model(x, logits_to_keep=1)
    torch.cuda.synchronize()
    return time.perf_counter() - t


def gen(x, n):
    c = Clock()
    torch.cuda.synchronize()
    t = time.perf_counter()
    model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=c, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return c.t[1] - t


print(f"{args.model} {args.mode} compile={args.compile} on {torch.cuda.get_device_name()}", flush=True)
with torch.no_grad():
    for L in [int(v) for v in args.lengths.split(",")]:
        x = ids[:L].cuda()[None]
        for _ in range(2):
            fwd(x), gen(x, 1), gen(x, 16)
        row = {"forward": [fwd(x) for _ in range(args.reps)], "gen1": [gen(x, 1) for _ in range(args.reps)], "gen16": [gen(x, 16) for _ in range(args.reps)]}
        after = []
        for _ in range(args.reps):
            gen(x, 16)
            after.append(fwd(x))
        row["fwd-after"] = after
        med = {k: statistics.median(v) * 1e3 for k, v in row.items()}
        print(f"  {L} tokens: " + ", ".join(f"{k} {v:.1f} ms" for k, v in med.items()) + f"  (gen1 over forward {med['gen1'] / med['forward']:.3f}x)", flush=True)
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            for name, f in (("forward", lambda: fwd(x)), ("gen1", lambda: gen(x, 1))):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    wall = f()
                ev = [e for e in p.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0]
                cuda = sum(getattr(e, "self_device_time_total", 0) for e in p.key_averages())
                top = sorted(p.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", 0))[:8]
                print(f"    {name} profiled: wall {wall * 1e3:.1f} ms, CUDA kernels {cuda / 1e3:.1f} ms: " + "; ".join(f"{e.key[:60]} x{e.count} {getattr(e, 'self_device_time_total', 0) / 1e3:.1f}" for e in top), flush=True)
