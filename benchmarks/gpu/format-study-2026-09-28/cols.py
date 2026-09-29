"""Escape rates of the fixed codes with one window for the tensor, one a row or one a column, on chosen matrices
(in the fragment order a lane's 4 weights of a word are one row's, 4 columns: a base a column would cost the decode
no instruction, a register a k-block).

    python cols.py MODEL_DIR NAME [NAME ...]"""
import json, os, sys, torch
from safetensors import safe_open

torch.set_num_threads(4)


def bw(h, width):
    c = torch.nn.functional.pad(h.cumsum(-1), (1, 0))
    s = c[..., width:] - c[..., :-width]
    return s.max(-1)


d = sys.argv[1]
names = sys.argv[2:]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"] if os.path.exists(os.path.join(d, "model.safetensors.index.json")) else None
for k in names:
    fn = idx[k] if idx else "model.safetensors"
    with safe_open(os.path.join(d, fn), "pt") as f:
        w = f.get_tensor(k)
    O, K = w.shape
    e = ((w.view(torch.int16).to(torch.int32) & 0xFFFF) >> 7) & 0xFF
    n = O * K
    h = torch.bincount(e.flatten(), minlength=256)
    hr = torch.bincount((e + 256 * torch.arange(O, dtype=torch.int32).unsqueeze(1)).flatten(), minlength=256 * O).view(O, 256)
    hc = torch.bincount((e + 256 * torch.arange(K, dtype=torch.int32).unsqueeze(0)).flatten(), minlength=256 * K).view(K, 256)
    out = []
    for nm, width, f2 in (("sb11", 4, True), ("v11", 7, False), ("sb12", 8, True)):
        g = (lambda x: x.view(*x.shape[:-1], 128, 2).sum(-1)) if f2 else (lambda x: x)
        t = n - int(bw(g(h), width)[0]); r = n - int(bw(g(hr), width)[0].sum()); c = n - int(bw(g(hc), width)[0].sum())
        out.append(f"{nm} tensor {1e6*t/n:.0f} row {1e6*r/n:.0f} col {1e6*c/n:.0f}")
    # column scale spread: std of per-column mean exponent, per-row
    me_c = (hc.double() * torch.arange(256).double()).sum(1) / hc.sum(1)
    me_r = (hr.double() * torch.arange(256).double()).sum(1) / hr.sum(1)
    print(f"{k} {O}x{K}: " + " | ".join(out) + f" | mean-exponent std: cols {float(me_c.std()):.2f} rows {float(me_r.std()):.2f}", flush=True)
