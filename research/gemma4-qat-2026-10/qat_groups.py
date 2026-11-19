"""Which group size makes a tensor exactly int4 x scale: G in 32, 64, 128, 256 and the whole row, on cached slices.   python -I qat_groups.py REPO NAME..."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ceil, lattice_int4 as L

repo = sys.argv[1]
for name in sys.argv[2:]:
    f = glob.glob(os.path.join(os.environ["CEIL_DATA"], "t", f"{repo.replace('/', '__')}__{name}__r*.npy"))[0]
    w = np.load(f).reshape(-1, np.load(f).shape[-1])[:64]
    line = []
    for g in (32, 64, 128, 256, w.shape[1]):
        L.G = g
        cl, ch = ceil.LO[w], ceil.HI[w]
        ok = tot = 0
        for r in range(w.shape[0]):
            for c0 in range(0, w.shape[1] - g + 1, g):
                tot += 1
                ok += L.encode_group(w[r, c0:c0 + g], cl[r, c0:c0 + g], ch[r, c0:c0 + g]) is not None
        line.append(f"G{g if g != w.shape[1] else 'row'} {100 * ok / tot:.0f}%")
    print(name, " | ".join(line), flush=True)
