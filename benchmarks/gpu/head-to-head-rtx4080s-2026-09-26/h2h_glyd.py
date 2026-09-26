# Head to head, Glyd's side: real matrices of a model's middle layer dumped for ZipServ's test (bf16 row major
# [O, K], OUT/NAME.bin), and timed here as ZipServ times itself: the L2 cache flushed before every call, the
# call alone timed by CUDA events, 20 calls. cuBLAS (F.linear) and Glyd's products (tiered, 12-bit) at each token count.
import sys, os, json, torch, torch.nn.functional as F
from safetensors import safe_open
import glyd_gpu as g
model_dir, out, layer = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 18
Ns = [int(n) for n in (sys.argv[4] if len(sys.argv) > 4 else "1,8,16,32,64").split(",")]
os.makedirs(out, exist_ok=True)
idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
names = {n: f"model.layers.{layer}.{p}.weight" for n, p in [("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"), ("o_proj", "self_attn.o_proj"), ("gate_proj", "mlp.gate_proj"), ("down_proj", "mlp.down_proj")]}
flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.uint8, device="cuda")
def cold_ms(f, reps=20):
    for _ in range(3): f()
    t = 0.0
    for _ in range(reps):
        flush.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record()
        torch.cuda.synchronize()
        t += e0.elapsed_time(e1)
    return t / reps
for n, key in names.items():
    with safe_open(f"{model_dir}/{idx[key]}", "pt", device="cuda") as fh:
        w = fh.get_tensor(key).contiguous()
    w.view(torch.int16).cpu().numpy().tofile(f"{out}/{n}.bin")
    O, K = w.shape
    qt, q12 = g.pack_mma(w), g.pack_mma12(w)
    bt, b12 = qt.nbytes() * 8 / w.numel(), q12.nbytes() * 8 / w.numel()
    row = [f"{n} {O}x{K}: bits tiered {bt:.2f} 12-bit {b12:.2f}"]
    for N in Ns:
        x = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        s = f"N={N} cuBLAS {cold_ms(lambda: F.linear(x, w)) * 1000:.1f}"
        if N <= 64:
            s += f" tiered {cold_ms(lambda: g.mma_gemm(qt, x)) * 1000:.1f} 12-bit {cold_ms(lambda: g.mma_gemm(q12, x)) * 1000:.1f}"
        row.append(s + " us")
    print(" | ".join(row), flush=True)
