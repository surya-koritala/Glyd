"""CPU emulation (numpy) of the split-byte 12-bit layout: kernels.py's pack_mma12 ported line by line, and
glyd_gpu.cu's decodes written out as the kernels run them (Nib::decode: high1, patch, pairs -- the step, prompt, MoE,
unpack and ws kernels; decode12_rows' stage decode with its scratch pass -- the mid and TMA kernels, and the wgp
kernel's stage12_exc + step12, the same arithmetic). Checked against the mma fragments built straight from W
(lane l = 4g + t of step (rb, ks) holds weights i = 4n + j = W[64 rb + 8n + g][16 ks + 8 (j >> 1) + 2t + (j & 1)]),
bit for bit, and the pack's bytes against the layout written in glyd_gpu.h."""
import numpy as np

rng = np.random.default_rng(0)
M32 = np.uint64(0xFFFFFFFF)


def pack12(w):
    """kernels.pack_mma12, ported: w uint16 [O, K] -> (data uint8 [steps, 1536], exc uint32, exc_base int64, hb)."""
    O, K = w.shape
    RB, KS = O // 64, K // 16
    h = np.bincount(((w.astype(np.int64) >> 7) & 0xFF).ravel(), minlength=256)  # _hist
    c = np.pad(h.reshape(128, 2).sum(1).cumsum(), (1, 0))
    hb = int(np.argmax(c[8:] - c[:-8]))
    v = w.reshape(RB, 8, 8, KS, 2, 4, 2).transpose(0, 3, 2, 5, 1, 4, 6).ravel().astype(np.int64)
    steps = O * K // 1024
    hi = (v >> 8) & 0x7F
    off = hi - hb
    esc = (off < 0) | (off > 7)
    j = np.arange(4, dtype=np.int64)
    s4 = ((v >> 15) & 1).reshape(-1, 32, 4, 2, 4)
    o4 = np.where(esc, 0, off).reshape(-1, 32, 4, 2, 4)
    words = ((s4[..., 0, :] << 7 | o4[..., 0, :]) << (8 * j)).sum(-1) | (s4[..., 1, :] << (8 * j + 3)).sum(-1) | (o4[..., 1, :] << (8 * ((j - 1) % 4) + 4)).sum(-1)
    data = np.empty((steps, 1536), dtype=np.uint8)
    data[:, :512] = words.astype(np.uint32).view(np.uint8).reshape(-1, 512)
    data[:, 512:] = (v & 0xFF).astype(np.uint8).reshape(-1, 32, 2, 16).transpose(0, 2, 1, 3).reshape(-1, 1024)
    m = esc.reshape(steps, 1024)
    idx = np.broadcast_to(np.arange(1024), (steps, 1024))[m]
    ent = idx | (hb ^ hi.reshape(steps, 1024)[m]) << 16
    n = m.sum(1)
    pad = 4 - int(n.sum()) % 4
    exc = np.concatenate([ent, np.zeros(pad, dtype=np.int64)]).astype(np.uint32)
    exc_base = np.concatenate([[0], np.cumsum(n)])
    return data, exc, exc_base, hb


def byte_perm(a, b, s):
    x = (b.astype(np.uint64) << np.uint64(32)) | a.astype(np.uint64)
    out = np.zeros(np.shape(a), dtype=np.uint64)
    for k in range(4):
        sel = (s >> (4 * k)) & 7
        out |= ((x >> np.uint64(8 * sel)) & np.uint64(0xFF)) << np.uint64(8 * k)
    return out.astype(np.uint32)


def rotl4(x):
    x = x.astype(np.uint64)
    return (((x << np.uint64(4)) | (x >> np.uint64(28))) & M32).astype(np.uint32)


def high1(nb, hb4, h):
    return (((rotl4(nb) if h else nb) & np.uint32(0x87878787)).astype(np.uint64) + np.uint64(hb4)).astype(np.uint32)


def words32(b):  # uint8 [..., 4k] -> uint32 [..., k], little endian
    return b.copy().view(np.uint32)


def expected(w):
    """R[step, lane, p] from W: p = 2n + half, half 0: columns 2t, 2t + 1; half 1: 8 + 2t, 9 + 2t (rows 8n + g)."""
    O, K = w.shape
    RB, KS = O // 64, K // 16
    x = w.astype(np.uint32).reshape(RB, 8, 8, KS, 2, 4, 2).transpose(0, 3, 2, 5, 1, 4, 6)  # [rb, ks, g, t, n, half, pair]
    return (x[..., 0] | x[..., 1] << 16).reshape(RB * KS, 32, 16)


def decode_nib(data, exc, exc_base, hb):
    """Nib::load + Nib::decode for every step and lane: R [steps, 32, 16]."""
    steps = data.shape[0]
    hb4 = hb * 0x01010101
    nb = words32(data[:, :512]).reshape(steps, 32, 4)
    sw = np.concatenate([words32(data[:, 512:1024]).reshape(steps, 32, 4), words32(data[:, 1024:]).reshape(steps, 32, 4)], 2)  # [steps, lane, 8]
    H = np.stack([high1(nb[:, :, q >> 1], hb4, q & 1) for q in range(8)], 2)
    for s in range(steps):  # patch: each entry XORs its byte into its lane's word
        for k in range(exc_base[s], exc_base[s + 1]):
            x = int(exc[k])
            i, lane = x & 31, (x >> 5) & 31
            H[s, lane, i >> 2] ^= np.uint32(((x >> 16) & 0xFF) << (8 * (i & 3)))
    R = np.stack([byte_perm(sw[:, :, p >> 1], H[:, :, p >> 1], 0x7362 if p & 1 else 0x5140) for p in range(16)], 2)
    return R


def decode_rows(data, exc, exc_base, hb, KS):
    """decode12_rows (and stage12_exc + step12) for every stage (4 steps of a row block), warp w and lane: A [stages, 4
    warps, 32 lanes, 4 kk, 4]."""
    hb4 = hb * 0x01010101
    stages = data.shape[0] // 4
    A = np.zeros((stages, 4, 32, 4, 4), dtype=np.uint32)
    for st in range(stages):
        sp = data[4 * st : 4 * st + 4].reshape(-1)
        eb = [int(exc_base[4 * st + kk]) for kk in range(5)]
        for w in range(4):
            xw = np.zeros((32, 8, 4), dtype=np.uint8)  # the warp's scratch: a lane's 8 words
            for k in range(eb[0], eb[4]):  # (lanes take the run 32 at a time: the same bytes set)
                x = int(exc[k])
                i = x & 31
                if i >> 3 == w:
                    kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3])
                    xw[(x >> 5) & 31, 2 * kk + ((i >> 2) & 1), i & 3] = (x >> 16) & 0xFF
            xe = xw.view(np.uint32).reshape(32, 8)
            for lane in range(32):
                for kk in range(4):
                    q = kk * 1536
                    nw = np.frombuffer(sp[q + 16 * lane + 4 * w : q + 16 * lane + 4 * w + 4].tobytes(), dtype=np.uint32)
                    o = q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1)
                    sx, sy = np.frombuffer(sp[o : o + 8].tobytes(), dtype=np.uint32)
                    e0 = high1(nw, hb4, 0)[0] ^ xe[lane, 2 * kk]
                    e1 = high1(nw, hb4, 1)[0] ^ xe[lane, 2 * kk + 1]
                    A[st, w, lane, kk] = [byte_perm(np.array([sx]), np.array([e0]), 0x5140)[0], byte_perm(np.array([sy]), np.array([e1]), 0x5140)[0],
                                          byte_perm(np.array([sx]), np.array([e0]), 0x7362)[0], byte_perm(np.array([sy]), np.array([e1]), 0x7362)[0]]
    return A


def layout_bytes(w, data, exc, exc_base, hb):
    """The pack's bytes against glyd_gpu.h's words: weight 8q + j's sign in bit 8j + 7 of word q, its offset in bits
    8j to 8j + 2; weight 8q + 4 + j's sign in bit 8j + 3, its offset in bits 8 ((j + 3) % 4) + 4 to + 6; low byte of
    weight i at 512 + 16 l + i (i < 16) or 1024 + 16 l + i - 16; entries 32 l + i | (hb ^ (exponent >> 1)) << 16,
    ascending within a step; zeros after, to a multiple of 4, at least one."""
    O, K = w.shape
    RB, KS = O // 64, K // 16
    v = w.astype(np.int64).reshape(RB, 8, 8, KS, 2, 4, 2).transpose(0, 3, 2, 5, 1, 4, 6).reshape(RB * KS, 32, 32)
    nb = words32(data[:, :512]).reshape(-1, 32, 4).astype(np.int64)
    for i in range(32):
        q, j = i // 8, i % 4
        sign, hi = (v[:, :, i] >> 15) & 1, (v[:, :, i] >> 8) & 0x7F
        esc = (hi < hb) | (hi > hb + 7)
        off = np.where(esc, 0, hi - hb)
        wd = nb[:, :, q]
        if i % 8 < 4:
            got_s, got_o = (wd >> (8 * j + 7)) & 1, (wd >> (8 * j)) & 7
        else:
            got_s, got_o = (wd >> (8 * j + 3)) & 1, (wd >> (8 * ((j + 3) % 4) + 4)) & 7
        assert (got_s == sign).all() and (got_o == off).all(), i
        lo = data[:, 512 + np.arange(32)[:, None] * 16 + i] if i < 16 else data[:, 1024 + np.arange(32)[:, None] * 16 + i - 16]
        assert (lo.reshape(-1, 32) == (v[:, :, i] & 0xFF)).all(), i
    for s in range(RB * KS):
        e = exc[exc_base[s] : exc_base[s + 1]].astype(np.int64)
        pos = e & 0x3FF
        assert (np.diff(pos) > 0).all() and ((e >> 10) & 0x3F == 0).all() and (e >> 24 == 0).all()
        hi = (v[s].reshape(-1)[pos] >> 8) & 0x7F
        assert ((e >> 16) & 0xFF == hb ^ hi).all() and ((hi < hb) | (hi > hb + 7)).all()
        vv = (v[s].reshape(-1) >> 8) & 0x7F
        assert len(pos) == int(((vv < hb) | (vv > hb + 7)).sum())
    tail = exc[exc_base[-1] :]
    assert 1 <= len(tail) <= 4 and (tail == 0).all() and len(exc) % 4 == 0


def bf16(f):
    u = np.asarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def weights(O, K, wild=0.0):
    f = rng.standard_normal((O, K)).astype(np.float32) * 0.02
    m = rng.random((O, K)) < wild
    f[m] = rng.standard_normal(int(m.sum())).astype(np.float32) * np.exp2(rng.integers(-40, 20, int(m.sum()))).astype(np.float32)
    return bf16(f)


every = (np.arange(65536) - 32768).astype(np.int16).view(np.uint16)
most = rng.random(65536) < 0.9
cases = [("64x64", weights(64, 64)), ("192x128", weights(192, 128)), ("128x1040 wild 0.01", weights(128, 1040, 0.01)), ("256x1024 wild 0.001", weights(256, 1024, 0.001)),
         ("128x512 wild 0.1", weights(128, 512, 0.1)), ("64x256 wild 0.5", weights(64, 256, 0.5)),
         ("every pattern, shuffled", rng.permutation(every).reshape(256, 256)), ("every pattern, in order", every.reshape(256, 256)),
         ("hb 0", np.where(most, every & 0x807F, every).astype(np.uint16).reshape(256, 256)), ("hb 120", np.where(most, every | 0x7F00, every).astype(np.uint16).reshape(256, 256))]
for name, w in cases:
    data, exc, exc_base, hb = pack12(w)
    layout_bytes(w, data, exc, exc_base, hb)
    R = decode_nib(data, exc, exc_base, hb)
    want = expected(w)
    assert (R == want).all(), (name, "Nib::decode")
    rows = ""
    if w.shape[1] % 64 == 0:
        KS = w.shape[1] // 16
        A = decode_rows(data, exc, exc_base, hb, KS)
        W4 = want.reshape(-1, 4, 32, 16)  # [stage, kk, lane, p]
        for wp in range(4):
            got = A[:, wp]  # [stage, lane, kk, 4]
            ref = np.stack([W4[:, :, :, 4 * wp], W4[:, :, :, 4 * wp + 2], W4[:, :, :, 4 * wp + 1], W4[:, :, :, 4 * wp + 3]], -1).transpose(0, 2, 1, 3)
            assert (got == ref).all(), (name, "decode12_rows", wp)
        rows = ", decode12_rows / step12 the same fragments"
    print(f"{name}: hb {hb}, {int(exc_base[-1])} exceptions ({8 * (data.size + 4 * exc.size + 4 * exc_base.size + 16) / w.size:.3f} bits a weight): "
          f"bytes as glyd_gpu.h writes them, Nib::decode the mma fragments of W bit for bit{rows}")
print("all passed")
