"""For QAT int4 groups: is there a bf16 scale (the engine's Int4 layout keeps a bf16 scale a run of 32) that reproduces the group exactly, and how is it decoded?
Tries the decode bf16(k * s) with s in bf16 between the group's bounds.   python -I qat_scale.py REPO NAME..."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ceil

def bf16_vals_between(lo, hi):
    a = np.array([lo], np.float32).view(np.uint32)[0] >> 16
    b = np.array([hi], np.float32).view(np.uint32)[0] >> 16
    for w in range(int(a), int(b) + 2):
        v = np.array([w << 16], np.uint32).view(np.float32)[0]
        if lo <= v <= hi: yield v

repo = sys.argv[1]
for name in sys.argv[2:]:
    f = glob.glob(os.path.join(os.environ["CEIL_DATA"], "t", f"{repo.replace('/', '__')}__{name}__r*.npy"))[0]
    a = np.load(f); w = a.reshape(-1, a.shape[-1])[:16]
    cl, ch = ceil.LO[w], ceil.HI[w]
    tot = okb = 0
    for r in range(w.shape[0]):
        for c0 in range(0, w.shape[1], 32):
            g, lo, hi = w[r, c0:c0+32], cl[r, c0:c0+32], ch[r, c0:c0+32]
            v = ceil.f32(g).astype(np.float64); m = np.abs(v).max(); tot += 1
            if m == 0: okb += 1; continue
            hit = False
            for den in (7.0, 8.0):
                k = np.rint(v / (m / den)); nz = k != 0
                if np.abs(k).max() > 8 or (v[~nz] != 0).any(): continue
                s_lo = np.max(np.where(k[nz] > 0, lo[nz] / k[nz], hi[nz] / k[nz])); s_hi = np.min(np.where(k[nz] > 0, hi[nz] / k[nz], lo[nz] / k[nz]))
                for s in bf16_vals_between(s_lo, s_hi):
                    if np.array_equal(ceil.words((k.astype(np.float32) * np.float32(s)).astype(np.float32)), g): hit = True; break
                if hit: break
            okb += hit
    print(f"{name.replace('model.language_model.','')}: a bf16 scale reproduces {okb}/{tot} groups exactly", flush=True)
