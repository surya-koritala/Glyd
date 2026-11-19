"""Exponent rank statistics of real weight tensors: share of weights past rank 3, 6, 9, 12, 15 (by count within the tensor),
the entropy, and the bits a weight of the tiered code and of some variants."""
import json, os, sys, glob, math
import torch
from safetensors import safe_open

def tensors(d, layers):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    for L in layers:
        for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            k = f"model.layers.{L}.{n}.weight"
            with safe_open(os.path.join(d, idx[k]), "pt", device="cuda") as f:
                yield k, f.get_tensor(k)

def stats(w):
    e = ((w.view(torch.int16).to(torch.int32) >> 7) & 0xFF).flatten()
    h = torch.bincount(e, minlength=256).double()
    n = h.sum()
    s, _ = torch.sort(h, descending=True)
    p = s / n
    c = torch.cumsum(p, 0)
    H = -(p[p > 0] * torch.log2(p[p > 0])).sum().item()
    return p.cpu(), c.cpu(), H

def bits(c):
    P = lambda r: 1 - c[r - 1].item() if r > 0 else 1.0
    tiered = 8 + 2 + 2 * P(3) + 2 * P(6) + 8 * P(9)
    two3 = 8 + 2 + 3 * P(3) + 8 * P(10)       # tier 2 with 3-bit fields: ranks 3-9 and a literal
    two4 = 8 + 2 + 4 * P(3) + 8 * P(18)       # tier 2 with 4-bit fields: ranks 3-17
    t3b = 8 + 3 + 8 * P(7)                      # fixed 3 bits (7 symbols), literals by position list ignored
    return tiered, two3, two4, t3b

if __name__ == "__main__":
    m = sys.argv[1]
    snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
    layers = [int(x) for x in sys.argv[2].split(",")]
    tot = None
    print("tensor, H(e), p(rank<3) p(<6) p(<9) p(<12) p(<18), bits: tiered | tier2 3-bit | tier2 4-bit | fixed 3-bit(no list)")
    acc = []
    for k, w in tensors(snap, layers):
        p, c, H = stats(w)
        t = bits(c)
        print(f"{k[13:]:38s} H {H:5.3f}  {c[2]:.4f} {c[5]:.4f} {c[8]:.4f} {c[11]:.4f} {c[17]:.4f}  {t[0]:.3f} | {t[1]:.3f} | {t[2]:.3f} | {t[3]:.3f}")
        acc.append((w.numel(), c, t, H))
    n = sum(a[0] for a in acc)
    avg = lambda f: sum(a[0] * f(a) for a in acc) / n
    print("weighted: H %.3f  p<3 %.4f p<6 %.4f p<9 %.4f p<12 %.4f p<18 %.4f | bits %.3f | %.3f | %.3f | %.3f" % (avg(lambda a: a[3]), avg(lambda a: a[1][2].item()), avg(lambda a: a[1][5].item()), avg(lambda a: a[1][8].item()), avg(lambda a: a[1][11].item()), avg(lambda a: a[1][17].item()), *[avg(lambda a, i=i: a[2][i]) for i in range(4)]))
