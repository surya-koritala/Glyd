# Plain generate() tokens/s as a user calls it, in a process of its own: glyd.from_pretrained(MODEL), with the
# compiled default ("default") or compile=False ("eager"); B sequences of a P-token prompt, N tokens each; the first
# call timed apart (its compile and capture), then REPS timed calls (3), their median. It prints what the prompt's
# products take (GLinear.kernel(B * P), the B sequences' P tokens in one product: on Hopper mma_gemm_wg to 1024 tokens,
# whose calls past 128 run mma12_wgp_kernel), whether generate() compiled, and the first sequence's text.
#   python fast_gen.py MODEL default|eager B P N
import os, statistics, sys, time
import torch
import glyd
from transformers import AutoTokenizer

name, mode, B, P, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
REPS = int(os.environ.get("REPS", 3))
tok = AutoTokenizer.from_pretrained(name)
text = "The history of data compression began with Morse code, then Huffman's trees and the dictionaries of Lempel and Ziv. "
ids = tok(text * (P // 8 + 1), return_tensors="pt").input_ids[:, :P].cuda().repeat(B, 1)
assert ids.shape == (B, P), ids.shape
m = glyd.from_pretrained(name, compile=mode == "default")
from glyd.gpu import model as gm  # (imported by from_pretrained)

lin = next(x for x in m.modules() if isinstance(x, gm.GLinear))
route = lin.kernel(B * P)
kw = dict(attention_mask=torch.ones_like(ids), max_new_tokens=N, min_new_tokens=N, do_sample=False)
with torch.no_grad():
    torch.cuda.synchronize()
    t = time.perf_counter()
    m.generate(ids, **kw)
    torch.cuda.synchronize()
    first = time.perf_counter() - t
    rates = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = m.generate(ids, **kw)
        torch.cuda.synchronize()
        rates.append(B * N / (time.perf_counter() - t))
compiled = m in gm._COMPILED
print(f"{name.split('/')[-1]} {mode} batch {B}, a prompt of {P} tokens and {N} generated: first generate() {first:.1f} s, "
      f"then {statistics.median(rates):.1f} tokens/s (median of {REPS}: {', '.join(f'{r:.1f}' for r in sorted(rates))}), "
      f"{'compiled' if compiled else 'eager'}; the prompt's products ({B * P} tokens): {route.__name__ if route else 'decoded, then cuBLAS'} "
      f"({m.config.quantization_config.layout}); peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
print("   text:", tok.decode(out[0, P:]).replace("\n", " ")[:160], flush=True)
