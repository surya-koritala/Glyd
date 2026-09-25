"""End to end: a Hugging Face causal LM generating with its weights held
compressed in VRAM (glyd_gpu), each matrix decoded into one shared
scratch buffer just before its use; against the same model in bf16.
Logits and generated tokens must be bit-identical.

    python e2e.py MODEL_DIR [new_tokens]
"""
import sys, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import glyd_gpu as g

path = sys.argv[1]
new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128
tok = AutoTokenizer.from_pretrained(path)
prompt = "The history of data compression began"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()


class Scratch:
    buf = None


class GLinear(nn.Module):
    def __init__(self, packed, bias):
        super().__init__()
        self.p, self.bias = packed, bias

    def forward(self, x):
        return F.linear(x, g.unpack(self.p, Scratch.buf), self.bias)


class GEmbedding(nn.Module):
    def __init__(self, packed):
        super().__init__()
        self.p = packed

    def forward(self, ids):
        return F.embedding(ids, g.unpack(self.p, Scratch.buf))


def measure(model, label):
    torch.cuda.synchronize()
    with torch.no_grad():
        logits = model(ids).logits
        model.generate(ids, max_new_tokens=8, do_sample=False)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated()
    print(f"{label}: {new_tokens / t:.1f} tokens/s, peak VRAM {peak / 1e6:.0f} MB")
    return logits, out, peak


model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
weights_bf16 = sum(p.numel() * p.element_size() for p in model.parameters())
logits_a, out_a, peak_a = measure(model, f"bf16 (weights {weights_bf16 / 1e6:.0f} MB)")

# Every Linear and the embedding packed; a tied lm_head shares the embedding's.
packed = {}
biggest = 0
with torch.no_grad():
    for name, m in list(model.named_modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, (nn.Linear, nn.Embedding)):
                w = child.weight
                key = w.data_ptr()
                if key not in packed:
                    packed[key] = g.pack(w.data)
                p = packed[key]
                biggest = max(biggest, p.n)
                new = GLinear(p, child.bias) if isinstance(child, nn.Linear) else GEmbedding(p)
                setattr(m, cname, new)
    Scratch.buf = torch.empty(biggest, dtype=torch.bfloat16, device="cuda")
torch.cuda.empty_cache()
weights_glyd = sum(p.nbytes() for p in packed.values()) + sum(p.numel() * p.element_size() for p in model.parameters())
print(f"packed: weights {weights_glyd / 1e6:.0f} MB + scratch {biggest * 2 / 1e6:.0f} MB ({100 * weights_glyd / weights_bf16:.1f}% of bf16 weights)")
logits_b, out_b, peak_b = measure(model, "glyd")
print("logits bit-identical:", torch.equal(logits_a.view(torch.int16), logits_b.view(torch.int16)))
print("tokens identical:", torch.equal(out_a, out_b))
print("text:", tok.decode(out_b[0][ids.shape[1]:ids.shape[1] + 40]).replace("\n", " "))
