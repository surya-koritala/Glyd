# Every candidate kernel for one decoder layer's matrices (merged: qkv, o, gate_up, down), mma12 layout, against
# cuBLAS on bf16, M tokens; checked against the fp32 product.
#   python kbench.py MODEL_DIR LAYER 1,8,16,32,48,64,96,128 [kernel,...]
import json, os, sys, torch
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_SRC", "~/glyd/gpu")))
import glyd_gpu as g

d, layer, Ms = sys.argv[1], int(sys.argv[2]), [int(m) for m in sys.argv[3].split(",")]
only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
def w(n):
    key = n if n.endswith(".weight") else f"model.layers.{layer}.{n}.weight"
    with safe_open(os.path.join(d, index[key]), "pt", device="cuda") as f:
        return f.get_tensor(key)
mats = {"lm_head": w("lm_head.weight")} if os.environ.get("LM_HEAD") else {
    "qkv": torch.cat([w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj")]),
    "o": w("self_attn.o_proj"),
    "gate_up": torch.cat([w("mlp.gate_proj"), w("mlp.up_proj")]),
    "down": w("mlp.down_proj"),
}
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "layer", layer, flush=True)

def us(f, n=20, reps=5):
    # GPU time a call: n calls captured in a CUDA graph (no host time between them), replayed.
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (n * reps)

scratch = torch.empty(max(W.numel() for W in mats.values()), dtype=torch.bfloat16, device="cuda")
def decoded(p, x):
    O, K = p.shape
    return F.linear(x, g.mma_unpack(p, scratch[: O * K]))

hopper = torch.cuda.get_device_capability() == (9, 0)
kern = {
    "mma12": (lambda M: M <= 64, g.mma_gemm),
    "mid": (lambda M: not hopper, g.mma_gemm_mid),
    "wg": (lambda M: hopper and M >= 17, g.mma_gemm_wg),
    "big12": (lambda M: M > 16 and not hopper, g.mma_gemm_big),
    "dec": (lambda M: M > 16, decoded),
}
if only:
    kern = {k: v for k, v in kern.items() if k in only}
# The GPU at its clocks first: a second of products.
_w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
import time
_t = time.time()
while time.time() - _t < 1.0:
    _w @ _w
torch.cuda.synchronize()
del _w
total = {}
# Weights read from memory, as in a model's step (each layer's evicted by the others' by its next step): the calls
# cycle through copies of the matrix and its pack enough to overflow the L2 cache (COLD_MB, 0: one copy).
COLD = int(os.environ.get("COLD_MB", 200)) << 20
for name, W in mats.items():
    O, K = W.shape
    p = g.pack_mma12(W)
    Ws = [W] + [W.clone() for _ in range(max(0, -(-COLD // (W.numel() * 2)) - 1))]
    ps = [p] + [g.pack_mma12(W) for _ in range(max(0, -(-COLD // p.nbytes()) - 1))]
    print(f"{name} {O}x{K} ({len(Ws)} copies bf16, {len(ps)} packed)", flush=True)
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x.float(), W.float())
        it = iter(range(1 << 62))
        row = {"cuBLAS": us(lambda: F.linear(x, Ws[next(it) % len(Ws)]))}
        for k, (ok, f) in kern.items():
            if not ok(M):
                continue
            y = f(p, x)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2 and torch.equal(y, f(p, x)), (k, name, M, err)
            row[k] = us(lambda: f(ps[next(it) % len(ps)], x))
        best = min((v, k) for k, v in row.items() if k != "cuBLAS")
        for k, v in row.items():
            total[(M, k)] = total.get((M, k), 0.0) + v
        total[(M, "best")] = total.get((M, "best"), 0.0) + best[0]
        print(f"  M={M:4d}  " + "  ".join(f"{k} {v:7.1f}" for k, v in row.items()) + f"   best {best[1]} {best[0] / row['cuBLAS']:.2f}x", flush=True)
# GLinear's picks: on Hopper mma_gemm to 16 tokens, mma_gemm_wg to 512, decoded past; on an A100 mma_gemm to 16, mma_gemm_mid to 64, mma_gemm_big past.
for M in Ms:
    for route, k in ((("glyd", "mma12" if M <= 16 else "wg" if M <= 512 else "dec"),) if hopper else (("glyd", "mma12" if M <= 16 else "mid" if M <= 64 else "big12"),)):
        if (M, k) in total:
            total[(M, route)] = total[(M, k)]
print("layer total (us):")
for M in Ms:
    ks = sorted({k for (m, k) in total if m == M} - {"best", "glyd"}, key=lambda k: (k != "cuBLAS", k)) + [k for k in ("glyd",) if (M, k) in total]
    print(f"  M={M:4d}  " + "  ".join(f"{k} {total[(M, k)]:7.1f}" for k in ks) + f"  best {total[(M, 'best')]:7.1f} {total[(M, 'best')] / total[(M, 'cuBLAS')]:.2f}x" + "".join(f"  {r} {total[(M, r)] / total[(M, 'cuBLAS')]:.2f}x" for r in ("glyd",) if (M, r) in total))
