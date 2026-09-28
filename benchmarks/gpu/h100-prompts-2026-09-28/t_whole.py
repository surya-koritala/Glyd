# The whole-blocks rule (mma12_tma_run: nb rounded down to a whole number of blocks a unit) against the even split, on
# other models' merged attention matrices (the rule's cases: few units, short blocks), synthetic weights, cold (copies
# past the L2), CUDA graphs. GLYD_NO_WHOLE=1 (a test build's switch): the even split.
#   python t_whole.py 17,32,64,96,128,256
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_SRC", "~/glyd/gpu")))
import glyd_gpu as g

Ms = [int(m) for m in sys.argv[1].split(",")]
shapes = [("Gemma-2-9B qkv", 8192, 3584), ("Gemma-2-9B o", 3584, 4096), ("Llama-3.1-8B/Mistral-7B qkv", 6144, 4096), ("Llama-3.1-8B/Mistral-7B o", 4096, 4096),
          ("Qwen3-14B qkv", 7168, 5120), ("Qwen3-14B o", 5120, 5120), ("Phi-3-mini qkv", 9216, 3072), ("Llama-3.2-3B qkv", 5120, 3072), ("Qwen3-8B qkv", 6144, 4096)]

def us(f, n=20, reps=5):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(n):
            f()
    gr.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        gr.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (n * reps)

_w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
t = time.time()
while time.time() - t < 1.0:
    _w @ _w
del _w
torch.manual_seed(0)
print(torch.cuda.get_device_name(), "whole blocks", "off" if os.environ.get("GLYD_NO_WHOLE") else "on", flush=True)
for name, O, K in shapes:
    W = (torch.randn(O, K, device="cuda") * 0.02).bfloat16()
    p = g.pack_mma12(W)
    ps = [p] + [g.pack_mma12(W) for _ in range(max(0, -(-(200 << 20) // p.nbytes()) - 1))]
    row = []
    for M in Ms:
        xs = [torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        i = [0]
        def f():
            i[0] += 1
            return g.mma_gemm_wg(ps[i[0] % len(ps)], xs[i[0] % 2])
        row.append(f"{M}: {us(f):7.1f}")
    print(f"{name:28s} {O}x{K}  " + "  ".join(row), flush=True)
