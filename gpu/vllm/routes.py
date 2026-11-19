"""The library's route table for a GPU code and a model's Linear shapes: for which counts of tokens a step each product takes which route (up to vLLM's 8,192-token
step), and whether any of them decodes the matrix for cuBLAS (the plugin then needs a buffer for it: _decodes). No GPU work: the table is a function of the GPU's
code (this GPU's: glyd_gpu._lib.gpu(); an L4's is 3089) and the shapes.
   python routes.py [GPU CODE]       (the shapes of Qwen3-8B, Qwen3-4B and Llama-3.1-8B)"""
import sys
from glyd_gpu import _lib, kernels as g
gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 3089
names = {g.DECODE: "decode", g.GEMM: "gemm", g.X2: "x2", g.X3: "x3", g.X1: "prompt kernel", g.EARLY: "decode ahead", g.PART: "part"}
shapes = {"Qwen3-8B": [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 24576, 4096), ("down", 4096, 12288)],
          "Qwen3-4B": [("qkv", 5120, 2560), ("o", 2560, 4096), ("gate_up", 19456, 2560), ("down", 2560, 9728)],
          "Llama-3.1-8B": [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 28672, 4096), ("down", 4096, 14336)]}
for layout, way in (("mma", _lib.mma_way), ("mma12", _lib.mma12_way)):
    for model, sh in shapes.items():
        for nm, O, K in sh:
            runs, M = [], 1
            while M <= 8192:
                w, last = way(gpu, O, K, M)
                last = min(last, 8192)
                if runs and runs[-1][0] == w:
                    runs[-1][2] = last
                else:
                    runs.append([w, M, last])
                M = last + 1
            decodes = any(w in (g.DECODE, g.EARLY) for w, _, _ in runs)
            print(f"{layout:5s} {model:13s} {nm:8s} {O:6d} x {K:5d}: " + ", ".join(f"{names.get(w, w)} {a}-{b}" for w, a, b in runs) + f"   -> buffer {'needed (' + format(O * K * 2 / 2**20, '.0f') + ' MiB)' if decodes else 'not needed'}")
