# Every candidate kernel for one decoder layer's matrices (merged: qkv, o, gate_up, down), mma12 layout, against
# cuBLAS on bf16, M tokens; checked against the fp32 product. GPU time: 20 calls in a CUDA graph, replayed; weights
# cold (copies past the L2). big: mma_gemm_big as GLinear calls it (variant 0); big256, big128: its variants 2, 1
# (blocks of 256 tokens by one row block, 128 by two); dec: the matrix decoded, then cuBLAS.
#   python kb.py MODEL_DIR LAYER 1,8,16 [kernel,...]      (LM_HEAD=1: the output layer instead)
import json, os, sys, time, torch
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_TREE", "~/glyd") + "/gpu"))
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
N = int(os.environ.get("KB_N", 20))

def us(f, n=N, reps=5):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(n):
            f()
    gr.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        gr.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (n * reps)

scratch = torch.empty(max(W.numel() for W in mats.values()), dtype=torch.bfloat16, device="cuda")
def decoded(p, x):
    O, K = p.shape
    return F.linear(x, g.mma_unpack(p, scratch[: O * K]))

kern = {
    "mma12": (lambda M: M <= 64, g.mma_gemm),
    "mid": (lambda M: M <= 128, g.mma_gemm_mid),
    "big": (lambda M: M > 16, g.mma_gemm_big),
    "big128": (lambda M: M > 128, lambda p, x: g.mma_gemm_big(p, x, None, 1)),
    "big256": (lambda M: M > 128, lambda p, x: g.mma_gemm_big(p, x, None, 2)),
    "dec": (lambda M: M > 16, decoded),
}
if only:
    kern = {k: v for k, v in kern.items() if k in only}
_w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
_t = time.time()
while time.time() - _t < 1.0:
    _w @ _w
torch.cuda.synchronize()
del _w
total = {}
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
        print(f"  M={M:4d}  " + "  ".join(f"{k} {v:8.1f}" for k, v in row.items()) + f"   best {best[1]} {best[0] / row['cuBLAS']:.2f}x", flush=True)
        del x, ref
print("layer total (us):")
for M in Ms:
    ks = sorted({k for (m, k) in total if m == M} - {"best"}, key=lambda k: (k != "cuBLAS", k))
    print(f"  M={M:4d}  " + "  ".join(f"{k} {total[(M, k)]:8.1f} ({total[(M, k)] / total[(M, 'cuBLAS')]:.2f})" for k in ks) + f"  best {total[(M, 'best')] / total[(M, 'cuBLAS')]:.2f}x", flush=True)
