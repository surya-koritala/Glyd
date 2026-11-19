"""torch's flash attention (FlashAttention 2 on this GPU) on the shapes of attn_bench.cu: Qwen3-8B's 32 query heads and 8 key-value heads of 128 channels, causal,
one prompt of n tokens (the yardstick for the engine's prompt attention).

    python attn_torch.py [n ...]
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

H, KVH, D = 32, 8, 128


def bench(n, reps=10):
    torch.manual_seed(0)
    q = (torch.randn(1, H, n, D, device="cuda", dtype=torch.bfloat16) * 1.5)
    k = (torch.randn(1, KVH, n, D, device="cuda", dtype=torch.bfloat16) * 1.5)
    v = torch.randn(1, KVH, n, D, device="cuda", dtype=torch.bfloat16)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        f = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            f()
        e1.record()
        torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / reps
    flop = 4.0 * n * (n + 1) / 2 * D * H
    print(f"torch flash n={n}: {ms:.3f} ms, {flop / (ms * 1e-3) / 1e12:.1f} TFLOPS (a layer; x36 = {ms * 36:.1f} ms)")


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or [1024, 3500]:
        bench(n)
