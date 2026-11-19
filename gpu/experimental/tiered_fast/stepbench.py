"""The step kernels (mma_gemm: tiered, 12-bit) on one real matrix, weights hot in L2 (or cold, COLD=1): time per call and per step
of 1024 weights. A real Qwen3-8B tensor, tokens M. For ncu: STEPBENCH_ONCE=1 runs each product once (after a warm-up).
    GLYD_GPU_LIB=lib.so python stepbench.py [--tensor model.layers.10.self_attn.o_proj.weight] [--M 1,16] [--fmt mma,mma12]
"""
import argparse, json, os, sys, time, glob
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import glyd_gpu_dev as g

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/hf/hub/models--Qwen--Qwen3-8B/snapshots"))
ap.add_argument("--tensor", default="model.layers.10.self_attn.o_proj.weight")
ap.add_argument("--M", default="1")
ap.add_argument("--fmt", default="mma,mma12")
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--rows", type=int, default=0, help="use only the first ROWS rows (a multiple of 64)")
ap.add_argument("--cols", type=int, default=0)
ap.add_argument("--kernel", default="gemm", help="gemm | mid | big (mma_gemm_x2 / x1)")
args = ap.parse_args()

snap = sorted(glob.glob(args.model + "/*"))[0] if "snapshots" in args.model else args.model
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(snap, idx[args.tensor]), "pt", device="cuda") as f:
    w = f.get_tensor(args.tensor)
if args.rows:
    w = w[: args.rows].contiguous()
if args.cols:
    w = w[:, : args.cols].contiguous()
O, K = w.shape
cold = os.environ.get("COLD") == "1"
once = os.environ.get("STEPBENCH_ONCE") == "1"
print(torch.cuda.get_device_name(), args.tensor, tuple(w.shape), "cold" if cold else "hot", flush=True)

packs = {"mma": g.pack_mma, "mma12": g.pack_mma12}
steps = O * K // 1024
copies = 1
for name in args.fmt.split(","):
    p = packs[name](w)
    assert torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)), name
    print(f"{name}: {p.bits_per_weight():.3f} bits a weight, {p.nbytes() / 1e6:.1f} MB, {steps} steps", flush=True)
    if cold:
        copies = max(1, -(-(200 << 20) // p.nbytes()))
    ps = [p] + [packs[name](w) for _ in range(copies - 1)]
    scratch = torch.empty(O, K, dtype=torch.bfloat16, device="cuda")
    fn = {"gemm": g.mma_gemm, "mid": g.mma_gemm_x2, "big": g.mma_gemm_x1, "unpack": lambda p, x: g.mma_unpack(p, scratch), "big2": lambda p, x: g.mma_gemm_x1(p, x, None, 2), "big1": lambda p, x: g.mma_gemm_x1(p, x, None, 1)}[args.kernel]
    for M in [int(m) for m in args.M.split(",")]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        if args.kernel == 'unpack':
            assert torch.equal(fn(p, x).view(torch.int16), w.view(torch.int16))
            err = 0.0
        else:
            ref = F.linear(x.float(), w.float())
            y = fn(p, x)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2, (name, M, err)
        if once:
            for _ in range(3):
                fn(p, x)
            torch.cuda.synchronize()
            fn(p, x)
            torch.cuda.synchronize()
            continue
        it = iter(range(1 << 62))
        n = args.calls
        for _ in range(3):
            fn(ps[next(it) % len(ps)], x)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(n):
                fn(ps[next(it) % len(ps)], x)
        gr.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        best = 1e9
        for _ in range(5):
            a.record()
            gr.replay()
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) * 1000 / n)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"  M={M}: {best:.2f} us a call; {best * 1000 / steps * sms:.1f} ns a step an SM ({p.nbytes() / best / 1e3:.0f} GB/s); err {err:.1e}", flush=True)
