"""Per-tensor share of cold steps (the T2 flags) over a model's Linear tensors, to pick a tensor with some but not all steps cold.
python coldscan.py MODEL_DIR_NAME"""
import glob, os, re, sys
import torch
from safetensors import safe_open
import pack_v2, v2bits

m = sys.argv[1]
snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
for fn in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(fn, "pt", device="cpu") as f:
        for k in f.keys():
            if not re.search(r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$", k):
                continue
            w = f.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.shape[0] % 64 or w.shape[1] % 16 or w.numel() < 8e6:
                continue
            e = v2bits.steps_of(w)
            h = torch.bincount(e.flatten(), minlength=256)
            order = torch.argsort(h, descending=True, stable=True)
            rank = torch.full((256,), 99, dtype=torch.int64)
            rank[order[:10]] = torch.arange(10)
            r = rank[e.long()]
            t1 = (r >= 3).sum(1); nl = (r >= 10).sum(1); t2 = ((r >= 6) & (r < 10)).sum(1)
            size = ((2 * t2 + 7) >> 3) + ((18 * nl + 7) >> 3) + 4 * ((t1 + 15) >> 4)
            start = 128 + torch.cumsum(size, 0) - size
            o = 8 * (start & 3)
            r0 = (2 * t2 + 7) >> 3
            cold = (t1 > 512) | (nl > 32) | (o + 2 * t2 + 32 >= 992) | ((nl > 0) & (o + 8 * r0 + 24 * (nl - 1) >= 992))
            c = cold.double().mean().item()
            if 0.01 < c < 0.6:
                print(f"{k[13:]:36s} {tuple(w.shape)} cold {100 * c:.1f}%", flush=True)
