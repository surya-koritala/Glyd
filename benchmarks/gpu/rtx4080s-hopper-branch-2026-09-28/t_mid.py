# mma_gemm_mid on Qwen2.5-7B's merged matrices (synthetic weights), 17-64 tokens, cold (copies past the L2), CUDA graphs.
#   GLYD_SRC=<its gpu dir> python t_mid.py 17,24,32,48,64
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser(os.environ["GLYD_SRC"]))
import glyd_gpu as g

Ms = [int(m) for m in sys.argv[1].split(",")]
shapes = [("qkv", 4608, 3584), ("o", 3584, 3584), ("gate_up", 37888, 3584), ("down", 3584, 18944)]

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

_w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = time.time()
while time.time() - t < 1.0:
    _w @ _w
del _w
torch.manual_seed(0)
print(torch.cuda.get_device_name(), os.environ["GLYD_SRC"], flush=True)
tot = {M: 0.0 for M in Ms}
for name, O, K in shapes:
    W = (torch.randn(O, K, device="cuda") * 0.02).bfloat16()
    p = g.pack_mma12(W)
    ps = [p] + [g.pack_mma12(W) for _ in range(max(0, -(-(200 << 20) // p.nbytes()) - 1))]
    row = []
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        i = [0]
        def f():
            i[0] += 1
            return g.mma_gemm_mid(ps[i[0] % len(ps)], x)
        v = us(f)
        tot[M] += v
        row.append(f"{M}: {v:7.1f}")
    print(f"{name:8s} {O}x{K}  " + "  ".join(row), flush=True)
    del ps, p, W
    torch.cuda.empty_cache()
print("total   " + "  ".join(f"{M}: {tot[M]:7.1f}" for M in Ms))
