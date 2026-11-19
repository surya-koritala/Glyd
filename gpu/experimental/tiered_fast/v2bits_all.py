"""Whole-model v1 / v2 / v2b bits a weight over every Linear tensor of a model's decoder layers (CPU), and the share of weights in
tensors where v2 costs more than 0.1 bit a weight over v1."""
import json, os, sys, glob, re
import torch
from safetensors import safe_open
import v2bits

m = sys.argv[1]
snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
files = sorted(glob.glob(snap + "/*.safetensors"))
tot = torch.zeros(4, dtype=torch.float64)
rows = []
for fn in files:
    with safe_open(fn, "pt", device="cpu") as f:
        for k in f.keys():
            if not re.search(r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$", k):
                continue
            w = f.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.shape[0] % 64 or w.shape[1] % 16:
                continue
            e = v2bits.steps_of(w)
            v1, v2, v2b, t1, t2, t3, L = v2bits.sizes(e, 4)
            n = w.numel()
            b = [8 * x.double().sum().item() / n for x in (v1, v2, v2b)]
            rows.append((k, n, b, float(t1.float().mean()), float(L.float().mean())))
            tot += torch.tensor([n] + [x.double().sum().item() for x in (v1, v2, v2b)])
            if len(rows) % 20 == 0:
                print(len(rows), "tensors", flush=True)
print(f"{m}: {len(rows)} tensors, {tot[0] / 1e9:.2f} B weights: v1 {8 * tot[1] / tot[0]:.4f} v2 {8 * tot[2] / tot[0]:.4f} v2b {8 * tot[3] / tot[0]:.4f} bits a weight (+0.031 for block_base)")
heavy = [r for r in rows if r[2][1] - r[2][0] > 0.1]
print(f"tensors with v2 > v1 + 0.1: {len(heavy)}, {sum(r[1] for r in heavy) / tot[0].item() * 100:.1f}% of the weights; their v1/v2/v2b bits:", sum(r[1] * r[2][0] for r in heavy) / max(1, sum(r[1] for r in heavy)), sum(r[1] * r[2][1] for r in heavy) / max(1, sum(r[1] for r in heavy)), sum(r[1] * r[2][2] for r in heavy) / max(1, sum(r[1] for r in heavy)))
print("tensors above 11.5 bits in v1:", [(r[0][13:], round(r[2][0], 2), round(r[2][1], 2)) for r in rows if r[2][0] > 11.5][:40])
