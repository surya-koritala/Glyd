# How long glyd.from_pretrained(PATH, layout=LAYOUT) takes to a model on the GPU (CUDA and transformers started
# first): argv PATH LAYOUT. Its peak GPU memory, and the packs' types after it (as saved, or packed there).
import sys, time, torch
import glyd.gpu.hf as hf
from glyd.gpu import model as gm

torch.zeros(1, device="cuda")
torch.cuda.synchronize()
t = time.perf_counter()
m = hf.from_pretrained(sys.argv[1], layout=sys.argv[2])
torch.cuda.synchronize()
dt = time.perf_counter() - t
kinds = {type(x.p).__name__ for x in m.modules() if isinstance(x, gm.GLinear)}
print(f"{sys.argv[1]} layout {sys.argv[2]}: {dt:.2f} s, {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak, packs {sorted(kinds)}", flush=True)
