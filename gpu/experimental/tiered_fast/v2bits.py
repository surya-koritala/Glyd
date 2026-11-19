"""Bits a weight of the tiered layout (v1, as pack_mma) and of a variant (v2: tier 3 a dense digit of 4 symbols (ranks 6-9), ranks
from 10 on in a list of 3-byte entries), per step, from the real tensors, on the CPU. Steps are 1024 weights in the layout's order;
the order inside a step does not change the counts."""
import json, os, sys, glob
import torch
from safetensors import safe_open

def steps_of(w):
    """exponents of w [O, K] as [steps, 1024] (a step: 64 rows x 16 columns)"""
    O, K = w.shape
    u = w.contiguous().view(torch.int16)
    e = ((u.to(torch.int32) >> 7) & 0xFF)
    RB, KS = O // 64, K // 16
    # [rb, 64 rows, ks, 16 cols] -> [rb, ks, 64, 16]
    return e.view(RB, 64, KS, 16).permute(0, 2, 1, 3).reshape(RB * KS, 1024)

def sizes(e, n_sym3):
    """-> v1 bytes, v2 bytes per step (the blocks' sizes + the fixed 1280), and counts"""
    h = torch.bincount(e.flatten(), minlength=256)
    order = torch.argsort(h, descending=True, stable=True)
    rank = torch.full((256,), 99, dtype=torch.int64)
    rank[order[:10]] = torch.arange(10)
    r = rank[e.long()]
    t1 = (r >= 3).sum(1); t2 = (r >= 6).sum(1)
    # v1: tier 3 has 3 symbols (ranks 6-8), literals from rank 9
    t3 = (r >= 9).sum(1)
    v1 = 1280 + ((2 * t2 + 7) >> 3) + t3 + 4 * ((t1 + 15) >> 4)
    # v2: tier 3 has 4 symbols (ranks 6-9), literals from rank 10 (a list of 3-byte entries, rank-ordered)
    L = (r >= 10).sum(1)
    v2 = 1280 + ((2 * t2 + 7) >> 3) + 3 * L + 4 * ((t1 + 15) >> 4)
    # v2b: the entries in 20 bits (10 bit rank + 8 bit byte + 2 spare...) packed: ceil(20 L / 8)
    v2b = 1280 + ((2 * t2 + 7) >> 3) + ((18 * L + 7) >> 3) + 4 * ((t1 + 15) >> 4)
    return v1, v2, v2b, t1, t2, t3, L

if __name__ == "__main__":
    m = sys.argv[1]
    snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{m}/snapshots/*")))[0]
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    layers = [int(x) for x in sys.argv[2].split(",")]
    tot = torch.zeros(4, dtype=torch.float64); n = 0
    worst = (0, "")
    rows = []
    for L in layers:
        for nm in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            k = f"model.layers.{L}.{nm}.weight"
            with safe_open(os.path.join(snap, idx[k]), "pt", device="cpu") as f:
                w = f.get_tensor(k)
            e = steps_of(w)
            v1, v2, v2b, t1, t2, t3, Lc = sizes(e, 4)
            bits = lambda v: 8 * v.double().sum().item() / w.numel()
            b1, b2, b2b = bits(v1), bits(v2), bits(v2b)
            print(f"{k[13:]:36s} v1 {b1:.3f}  v2 {b2:.3f} ({b2 - b1:+.3f})  v2b {b2b:.3f} ({b2b - b1:+.3f})  mean/step: T1 {t1.float().mean():.0f} T2 {t2.float().mean():.1f} lit1 {t3.float().mean():.1f} lit2 {Lc.float().mean():.1f} (max {int(Lc.max())})")
            tot += torch.tensor([w.numel(), v1.double().sum().item(), v2.double().sum().item(), v2b.double().sum().item()])
    print(f"weighted: v1 {8 * tot[1] / tot[0]:.4f} v2 {8 * tot[2] / tot[0]:.4f} v2b {8 * tot[3] / tot[0]:.4f} bits a weight (blocks and fixed parts; block_base's 4 B a step not counted: +0.031)")
