"""Exact int-N x group-32 scale fits at 4, 5, 6 and 8 bits (KMAX 8, 16, 32, 128) on cached slices.   python -I qat_widths.py REPO NAME..."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ceil, lattice_int4 as L

def fits(w, kmax, dens):
    cl, ch = ceil.LO[w], ceil.HI[w]
    ok = tot = 0
    for r in range(w.shape[0]):
        for c0 in range(0, w.shape[1], 32):
            g, lo, hi = w[r, c0:c0 + 32], cl[r, c0:c0 + 32], ch[r, c0:c0 + 32]
            v = ceil.f32(g).astype(np.float64); m = np.abs(v).max(); tot += 1
            if m == 0: ok += 1; continue
            for den in dens:
                k = np.rint(v / (m / den)); nz = k != 0
                if np.abs(k).max() > kmax or (v[~nz] != 0).any(): continue
                a = np.where(k[nz] > 0, lo[nz] / k[nz], hi[nz] / k[nz]); b = np.where(k[nz] > 0, hi[nz] / k[nz], lo[nz] / k[nz])
                if a.max() <= b.min(): ok += 1; break
    return 100 * ok / tot

repo = sys.argv[1]
for name in sys.argv[2:]:
    f = glob.glob(os.path.join(os.environ["CEIL_DATA"], "t", f"{repo.replace('/', '__')}__{name}__r*.npy"))[0]
    a = np.load(f); w = a.reshape(-1, a.shape[-1])[:48]
    print(name.split("layers.0.")[1], " | ".join(f"int{b} {fits(w, km, [km - 1, km, km - 0.5]):.0f}%" for b, km in ((4, 8), (5, 16), (6, 32), (8, 128))), flush=True)
