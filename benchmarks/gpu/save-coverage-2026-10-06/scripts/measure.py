"""python measure.py OUT.json ARGS...: `python -m glyd_gpu ARGS` (pack, verify) in this process, its time and the GPU's peak
memory (PyTorch's allocator) to OUT.json."""
import json, sys, time
import torch
from glyd_gpu import _cli

t0 = time.time()
code = _cli.main(sys.argv[2:])
json.dump({"args": sys.argv[2:], "exit": code, "s": round(time.time() - t0, 1), "gpu_peak_bytes": torch.cuda.max_memory_allocated()}, open(sys.argv[1], "w"))
sys.exit(code)
