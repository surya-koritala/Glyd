"""The LM head's product on this GPU, packed against bf16, for 1 to 1,024 rows and more (the rows of a step: the sequences sampled),
in both layouts, and the embedding's lookup for a few token counts: what vllm_plugin's glyd::vllm_logits and glyd::vllm_embed run.

    python head_sweep.py [MODEL] [--rows 1,2,4,...] [--layouts mma,mma12] [--tokens 1,8,64,...]

Per row count: the library's own route for this GPU and the product it gives (ms, CUDA events over 20 calls, the median of 3 rounds),
bf16's F.linear (what vLLM runs without Glyd), the same packed product when its route is the matrix decoded (what a Linear does: the
head's would take a 1.2 GB buffer: not run; the prompt kernel in its place, as the plugin takes), and the matrix decoded in chunks of
at most 64 MB then cuBLAS on each (the bounded-scratch choice for many rows). The head is MODEL's (default Qwen/Qwen3-8B: 151,936 rows,
4,096 columns; a tied model's: its embedding), read from the Hub cache (the one shard that holds it is downloaded where it is not there).
Needs vLLM's venv (torch, safetensors, huggingface_hub) and the library (GLYD_GPU_LIB); one GPU."""
import glob
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from glyd_gpu import _lib, kernels as g

ROWS = (1, 2, 4, 8, 16, 32, 48, 64, 65, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
TOKENS = (1, 8, 64, 256, 1024, 4096, 8192)


def table(model):
    """The head's matrix of model (its lm_head, else its embedding), bf16 on the GPU."""
    from huggingface_hub import hf_hub_download

    names = ("lm_head.weight", "model.embed_tokens.weight")
    d = model if os.path.isdir(model) else os.path.dirname(hf_hub_download(model, "config.json"))
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx) and not os.path.isdir(model):
        try:
            hf_hub_download(model, "model.safetensors.index.json")
        except Exception:
            pass
    where = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
    for n in names:
        f = where.get(n, "model.safetensors")
        p = os.path.join(d, f)
        if not os.path.exists(p) and not os.path.isdir(model):
            try:
                p = hf_hub_download(model, f)
            except Exception:
                continue
        if os.path.exists(p):
            with safe_open(p, "pt", device="cuda") as sf:
                if n in sf.keys():
                    return sf.get_tensor(n), n
    raise SystemExit(f"{model}: no lm_head or embedding found")


def ms(f, n=20, rounds=3):
    """The median over rounds of the time of f (ms), n calls back to back, CUDA events."""
    for _ in range(3):
        f()
    out = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            f()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / n)
    return statistics.median(out)


def chunked(p, x, buf, y):
    """x W^T by the matrix decoded 64 MB at a time (rows of at most `buf`'s size) and cuBLAS on each piece: y [M, O]."""
    O, K = p.shape
    step = max(64, buf.numel() // K // 64 * 64)
    for r in range(0, O, step):
        n = min(step, O - r)
        w = g.mma_unpack(p, buf[: n * K], r, n)
        y[:, r : r + n] = x @ w.t()
    return y


def main():
    a = sys.argv[1:]
    opt = lambda name, default: a[a.index(name) + 1] if name in a else default
    rows = tuple(int(x) for x in opt("--rows", ",".join(map(str, ROWS))).split(","))
    layouts = opt("--layouts", "mma,mma12").split(",")
    tokens = tuple(int(x) for x in opt("--tokens", ",".join(map(str, TOKENS))).split(","))
    model = next((x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--rows", "--layouts", "--tokens"))), "Qwen/Qwen3-8B")
    w, name = table(model)
    O, K = w.shape
    gpu = _lib.gpu()
    print(f"{model}: {name} {O} x {K} bf16 on {torch.cuda.get_device_name()} (GPU code {gpu}), torch {torch.__version__}", flush=True)
    buf = torch.empty(64 * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    for layout in layouts:
        p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
        way = _lib.mma12_way if layout == "mma12" else _lib.mma_way
        print(f"\n## {layout}: {p.bits_per_weight():.2f} bits a weight, {p.nbytes() / 2**30:.3f} GiB (bf16 {w.numel() * 2 / 2**30:.3f} GiB)", flush=True)
        print("| rows | the library's route | packed ms | bf16 ms | packed / bf16 | decoded in 64 MB pieces + cuBLAS ms | / bf16 |")
        print("| ---: | :--- | ---: | ---: | ---: | ---: | ---: |", flush=True)
        for M in rows:
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            r = way(gpu, O, K, M)[0]
            # (the decode routes' product is the prompt kernel here, as the plugin takes it)
            tp = ms(lambda: g.mma_linear(p, x, way=g.X1 if r in (g.DECODE, g.EARLY) else -1))
            tb = ms(lambda: F.linear(x, w))
            y = torch.empty(M, O, dtype=torch.bfloat16, device="cuda")
            tc = ms(lambda: chunked(p, x, buf, y), n=5)
            names = {g.DECODE: "decode", g.GEMM: "gemm", g.X2: "x2", g.X3: "x3", g.X1: "prompt kernel", g.EARLY: "decode ahead", g.PART: "part"}
            print(f"| {M} | {names.get(r, r)} | {tp:.3f} | {tb:.3f} | {tp / tb:.2f} | {tc:.3f} | {tc / tb:.2f} |", flush=True)
        print(f"\nthe embedding's lookup, {layout} (ms a call; bf16 F.embedding in the last column):")
        print("| tokens | uniform ids | zipf ids | bf16 |\n| ---: | ---: | ---: | ---: |")
        for n in tokens:
            u = torch.randint(0, O, (n,), device="cuda")
            z = (torch.rand(n, device="cuda") ** 6 * O).long().clamp_max(O - 1)
            print(f"| {n} | {ms(lambda: g.mma_rows(p, u)):.4f} | {ms(lambda: g.mma_rows(p, z)):.4f} | {ms(lambda: F.embedding(u, w)):.4f} |", flush=True)
        del p, x, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
