"""Whole-model bits a weight (CPU) of the tiered layout T1 (as pack_mma), T2b (18-bit literal entries), and per-tensor best-of with the
12-bit layout (actual pack_nib sizes, exceptions and exc_base included; block_base's 4 B a step added to the tiered ones).
python v2bits_best.py MODEL_DIR_NAME   (e.g. mistralai--Mistral-7B-v0.3)"""
import glob, json, os, re, sys
import torch
from safetensors import safe_open
import v2bits, pack_v2

m = sys.argv[1]
snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
tot = {"n": 0.0, "t1": 0.0, "t2b": 0.0, "best_t2b": 0.0, "best_t1": 0.0, "nib": 0.0}
heavy = []
for fn in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(fn, "pt", device="cpu") as f:
        for k in f.keys():
            if not re.search(r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$", k):
                continue
            w = f.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.shape[0] % 64 or w.shape[1] % 16:
                continue
            n = w.numel()
            v1, v2, v2b, *_ = v2bits.sizes(v2bits.steps_of(w), 4)
            b1 = 8 * v1.double().sum().item() / n + 0.03125
            b2b = 8 * v2b.double().sum().item() / n + 0.03125
            if min(b1, b2b) > 11.7:  # a candidate for the 12-bit layout: its real size
                data, exc, eb, hb = pack_v2.pack_nib(w)
                bn = 8 * (data.numel() + 4 * exc.numel() + 4 * eb.numel()) / n
                heavy.append((k[13:], n, b1, b2b, bn))
            else:
                bn = 12.04
            tot["n"] += n
            tot["t1"] += n * b1
            tot["t2b"] += n * b2b
            tot["nib"] += n * bn
            tot["best_t2b"] += n * min(b2b, bn)
            tot["best_t1"] += n * min(b1, bn)
n = tot["n"]
print(f"{m}: {n / 1e9:.2f} B weights, bits a weight (block_base included): T1 {tot['t1'] / n:.4f}  T2b {tot['t2b'] / n:.4f}  best(T1, 12-bit) {tot['best_t1'] / n:.4f}  best(T2b, 12-bit) {tot['best_t2b'] / n:.4f}  (12-bit alone, 12.04 where not measured: {tot['nib'] / n:.4f})")
for h in heavy:
    print(f"  {h[0]:34s} {h[1] / 1e6:6.1f} M  T1 {h[2]:.2f}  T2b {h[3]:.2f}  12-bit {h[4]:.2f}")
