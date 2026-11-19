"""How often a step leaves T2's fast path (CPU, every Linear of a model's decoder layers): tier-1 escapes > 512, literals > 32, tier-2 escapes > 107
(tier-3 digits or the literal list leave the 128-byte window: T2 > 460 or r0 + 3 literals > 121), any of them; per tensor class (T2b size within 11.7 bits or not).  python v2limits.py MODEL_DIR_NAME"""
import glob, os, re, sys
import torch
from safetensors import safe_open
import v2bits

m = sys.argv[1]
snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
acc = {"normal": torch.zeros(6, dtype=torch.float64), "heavy": torch.zeros(6, dtype=torch.float64)}
worst = []
for fn in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(fn, "pt", device="cpu") as f:
        for k in f.keys():
            if not re.search(r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$", k):
                continue
            w = f.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.shape[0] % 64 or w.shape[1] % 16:
                continue
            v1, v2, v2b, t1, t2, t3, L = v2bits.sizes(v2bits.steps_of(w), 4)
            bits = 8 * v2b.double().sum().item() / w.numel() + 0.03125
            cls = "heavy" if bits > 11.7 else "normal"
            r0 = (2 * t2 + 7) >> 3
            a, b, c = (t1 > 512), (L > 32), ((t2 > 460) | (r0 + 3 * L > 121))
            acc[cls] += torch.tensor([t1.numel(), a.sum().item(), b.sum().item(), c.sum().item(), (a | b | c).sum().item(), w.numel()], dtype=torch.float64)
            if (a | b | c).any():
                worst.append((k[13:], bits, (a | b | c).double().mean().item(), int(t1.max()), int(L.max()), int(t2.max())))
for cls, v in acc.items():
    if v[0]:
        print(f"{m} {cls:6s}: {int(v[0])} steps ({v[5] / 1e9:.2f} B weights): T1>512 {100 * v[1] / v[0]:.3f}%  lit>32 {100 * v[2] / v[0]:.3f}%  T2/window {100 * v[3] / v[0]:.3f}%  any {100 * v[4] / v[0]:.3f}%")
for w_ in sorted(worst, key=lambda x: -x[2])[:12]:
    print(f"   {w_[0]:34s} bits {w_[1]:.2f}  steps out {100 * w_[2]:.2f}%  max T1 {w_[3]} lit {w_[4]} T2 {w_[5]}")
