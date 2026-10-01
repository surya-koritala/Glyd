# pack time and transient memory of the packers by chunk size; the packs' bits against the old chunking's (the fixed tree's kernels.py)
import hashlib, time
import torch
from glyd.gpu import kernels as g

torch.manual_seed(0)
new_hist = g._hist


def old_hist(u):  # the packers' _hist before: widened to int32, 32M weights a pass
    h = torch.zeros(256, dtype=torch.int64, device=u.device)
    for a in range(0, u.numel(), 1 << 25):
        v = u[a : a + (1 << 25)].to(torch.int32) & 0xFFFF
        h += torch.bincount((v >> 7) & 0xFF, minlength=256)
    return h


def sha(p):
    ts = (p.data, p.blocks, p.block_base) if hasattr(p, "blocks") else (p.data, p.exc, p.exc_base)
    m = hashlib.sha256()
    for t in ts:
        m.update(t.cpu().numpy().tobytes())
    return m.hexdigest()[:12]


for name, (O, K) in {"qkv 6144x4096": (6144, 4096), "gate_up 24576x4096": (24576, 4096), "down 4096x12288": (4096, 12288)}.items():
    w = (torch.randn(O, K, device="cuda") * 0.02).to(torch.bfloat16)
    w[::97, ::13] = 0  # zeros, and outliers, as real checkpoints have
    w[::211, ::7] *= 64
    base = torch.cuda.memory_allocated()
    ref = {}
    for label, hist, hc, pc in (("old: hist int32 32M, pack 4M", old_hist, 1 << 25, 1 << 22), ("hist 4M, pack 4M", new_hist, 1 << 22, 1 << 22),
                                ("hist 4M, pack 2M", new_hist, 1 << 22, 1 << 21), ("hist 2M, pack 1M", new_hist, 1 << 21, 1 << 20),
                                ("hist 1M, pack 512K", new_hist, 1 << 20, 1 << 19)):
        g._hist, g.HIST_CHUNK, g.PACK_CHUNK = hist, hc, pc
        for lay, f in (("tiered", lambda w: g.pack_mma(w, chunk=pc)), ("12-bit", g.pack_mma12)):
            f(w); torch.cuda.synchronize()
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter(); p = f(w); torch.cuda.synchronize(); dt = time.perf_counter() - t0
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            h = sha(p)
            same = "same bits as old" if ref.setdefault(lay, h) == h else "DIFFERENT BITS"
            print(f"{name:20s} {label:30s} {lay:7s} {dt * 1e3:7.1f} ms  peak +{peak:6.0f} MiB  {same}", flush=True)
            del p
    del w
