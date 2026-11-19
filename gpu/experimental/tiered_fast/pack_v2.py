"""CPU packers of the tiered layout (v1, as glyd_gpu.kernels.pack_mma) and of its v2 variant, and a reference decoder of each.

v2 = the same fixed part of a step (its tier-1 digits, then the sign-and-mantissa bytes: byte for byte v1's `data`) and a
different block:
  - tier 3 is a plain 2-bit digit of 4 symbols (the ranks 6-9: tiers[2] holds four symbols), no escape digit;
  - a weight of rank 10 or more (v1: the exponent's byte as a tier-3 escape) is coded as a tier-1 escape whose tier-2 digit is 0
    (a placeholder) and has an entry in the step's list: its place among the step's tier-1 escapes (10 bits, bits 0-9) and its
    exponent byte (bits 10-17) in 3 bytes, little-endian;
  - a step's block: tier-3 digits of its tier-2 escapes (2 bits each, from its start, to a byte), the list's entries, the
    tier-2 digits of its tier-1 escapes (as v1: words of 16, the c-th word back from the end).
    python pack_v2.py MODEL_DIR TENSOR OUT_PREFIX  (writes OUT_PREFIX.{w,v1.data,v1.blocks,v1.bb,v1.tiers,v2.blocks,v2.bb,v2.tiers,dims})
"""
import json, os, sys, glob
import numpy as np
import torch

CHUNK = 1 << 21


def prep(w):
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0
    O, K = w.shape
    return O, K, O // 64, K // 16, w.contiguous().view(torch.int16)


def symbols(u, n):
    h = torch.bincount(((u.flatten() >> 7) & 0xFF).to(torch.int64), minlength=256)
    return torch.argsort(h, descending=True, stable=True)[:n].tolist()


def pack(w, version):
    O, K, RB, KS, u = prep(w)
    sym = symbols(u, 10)
    if version == 1:
        tiers = [sym[3 * k] | sym[3 * k + 1] << 8 | sym[3 * k + 2] << 16 | 0xFF << 24 for k in range(3)]
        nsym = 9
    else:
        tiers = [sym[0] | sym[1] << 8 | sym[2] << 16 | 0xFF << 24, sym[3] | sym[4] << 8 | sym[5] << 16 | 0xFF << 24, sym[6] | sym[7] << 8 | sym[8] << 16 | sym[9] << 24]
        nsym = 10
    rank = torch.full((256,), nsym, dtype=torch.int64)
    rank[torch.tensor(sym[:nsym])] = torch.arange(nsym)
    steps = O * K // 1024
    data = torch.empty(steps, 1280, dtype=torch.uint8)
    shifts = 2 * torch.arange(16, dtype=torch.int64)
    block_parts, sizes = [], []
    per = max(1, CHUNK // (64 * K))
    for b0 in range(0, RB, per):
        b1 = min(RB, b0 + per)
        v = u[b0 * 64 : b1 * 64].view(b1 - b0, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten().to(torch.int32) & 0xFFFF
        a, z = b0 * 64 * K // 1024, b1 * 64 * K // 1024
        e = (v >> 7) & 0xFF
        r = rank[e]
        words = (r.clamp(max=3).view(-1, 32, 2, 16) << shifts).sum(-1)
        words = (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        data[a:z, :256] = words.transpose(1, 2).contiguous().view(torch.uint8).view(-1, 256)
        pr = v.view(-1, 2)
        sign = (pr >> 15) & 1
        data[a:z, 256:] = (((pr & 0x7F) << 1) | sign.flip(1)).to(torch.uint8).view(-1, 32, 2, 16).transpose(1, 2).reshape(-1, 1024)
        r = r.view(z - a, 1024)
        ev = e.view(z - a, 1024)
        m1 = r >= 3
        t1 = m1.sum(1)
        k1 = torch.cumsum(m1, 1) - m1.to(torch.int64)  # an escape's place among its step's tier-1 escapes
        if version == 1:
            m2, m3 = r >= 6, r >= 9
            t2, t3 = m2.sum(1), m3.sum(1)
            r0 = (2 * t2 + 7) >> 3
            size = r0 + t3 + 4 * ((t1 + 15) >> 4)
            start = torch.cumsum(size, 0) - size
            buf = torch.zeros(int(size.sum()), dtype=torch.int32)
            pos = 8 * (start + size).unsqueeze(1) - 32 * ((k1 >> 4) + 1) + 2 * (k1 & 15)
            buf.index_add_(0, (pos >> 3)[m1], ((r - 3).clamp(max=3) << (pos & 7))[m1].to(torch.int32))
            k = torch.cumsum(m2, 1) - m2.to(torch.int64)
            pos = 8 * start.unsqueeze(1) + 2 * k
            buf.index_add_(0, (pos >> 3)[m2], ((r - 6).clamp(max=3) << (pos & 7))[m2].to(torch.int32))
            k = torch.cumsum(m3, 1) - m3.to(torch.int64)
            buf[((start + r0).unsqueeze(1) + k)[m3]] = ev[m3]
        else:
            lit = r >= 10
            m2 = (r >= 6) & ~lit
            t2, nl = m2.sum(1), lit.sum(1)
            r0 = (2 * t2 + 7) >> 3
            size = r0 + 3 * nl + 4 * ((t1 + 15) >> 4)
            start = torch.cumsum(size, 0) - size
            buf = torch.zeros(int(size.sum()), dtype=torch.int32)
            d2 = torch.where(lit, torch.zeros_like(r), (r - 3).clamp(max=3))  # a literal's tier-2 digit: 0 (a placeholder)
            pos = 8 * (start + size).unsqueeze(1) - 32 * ((k1 >> 4) + 1) + 2 * (k1 & 15)
            buf.index_add_(0, (pos >> 3)[m1], (d2 << (pos & 7))[m1].to(torch.int32))
            k = torch.cumsum(m2, 1) - m2.to(torch.int64)
            pos = 8 * start.unsqueeze(1) + 2 * k
            buf.index_add_(0, (pos >> 3)[m2], ((r - 6) << (pos & 7))[m2].to(torch.int32))
            j = torch.cumsum(lit, 1) - lit.to(torch.int64)  # a literal's place among its step's
            entry = (k1 | ev.to(torch.int64) << 10)
            base = (start + r0).unsqueeze(1) + 3 * j
            for i in range(3):
                buf[(base + i)[lit]] = ((entry >> (8 * i)) & 0xFF)[lit].to(torch.int32)
        block_parts.append(buf.to(torch.uint8))
        sizes.append(size)
    blocks = torch.cat([torch.zeros(128, dtype=torch.uint8)] + block_parts + [torch.zeros(256, dtype=torch.uint8)])
    size = torch.cat([torch.full((1,), 128, dtype=torch.int64)] + sizes)
    return data.flatten(), blocks, torch.cumsum(size, 0).to(torch.int32), tiers


def pack_nib(w):
    """The 12-bit layout (split byte), as glyd_gpu.kernels.pack_mma12 on the CPU: data [steps * 1536], exc (int32), exc_base (int32), hb."""
    O, K, RB, KS, u = prep(w)
    h = torch.bincount(((u.flatten() >> 7) & 0xFF).to(torch.int64), minlength=256)
    c = torch.nn.functional.pad(h.view(128, 2).sum(1).cumsum(0), (1, 0))
    hb = int((c[8:] - c[:-8]).argmax())
    steps = O * K // 1024
    data = torch.empty(steps, 1536, dtype=torch.uint8)
    j = torch.arange(4, dtype=torch.int64)
    exc_parts, counts = [], []
    per = max(1, CHUNK // (64 * K))
    for b0 in range(0, RB, per):
        b1 = min(RB, b0 + per)
        v = u[b0 * 64 : b1 * 64].view(b1 - b0, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten().to(torch.int32) & 0xFFFF
        a, z = b0 * 64 * K // 1024, b1 * 64 * K // 1024
        hi = (v >> 8) & 0x7F
        off = hi - hb
        esc = (off < 0) | (off > 7)
        s4 = ((v >> 15) & 1).view(-1, 32, 4, 2, 4).to(torch.int64)
        o4 = torch.where(esc, 0, off).view(-1, 32, 4, 2, 4).to(torch.int64)
        words = ((s4[..., 0, :] << 7 | o4[..., 0, :]) << (8 * j)).sum(-1) | (s4[..., 1, :] << (8 * j + 3)).sum(-1) | (o4[..., 1, :] << (8 * ((j - 1) % 4) + 4)).sum(-1)
        words = (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        data[a:z, :512] = words.contiguous().view(torch.uint8).view(-1, 512)
        data[a:z, 512:] = (v & 0xFF).to(torch.uint8).view(-1, 32, 2, 16).transpose(1, 2).reshape(-1, 1024)
        m = esc.view(z - a, 1024)
        idx = torch.arange(1024).expand(z - a, 1024)[m]
        exc_parts.append((idx | (hb ^ hi.view(z - a, 1024)[m]).to(torch.int64) << 16).to(torch.int32))
        counts.append(m.sum(1))
    n = torch.cat(counts)
    pad = 4 - int(n.sum()) % 4
    exc = torch.cat(exc_parts + [torch.zeros(pad, dtype=torch.int32)])
    exc_base = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(n, 0)]).to(torch.int32)
    return data.flatten(), exc, exc_base, hb


def unpack(shape, data, blocks, bb, tiers, version):
    """Reference decoder: the bf16 bits of W [O, K] back from a pack (loops over the steps; slow)."""
    O, K = shape
    RB, KS = O // 64, K // 16
    steps = O * K // 1024
    d = data.view(steps, 1280).numpy()
    bl = blocks.numpy()
    base = bb.numpy().astype(np.int64)
    sym = [(tiers[k] >> (8 * j)) & 0xFF for k in range(3) for j in range(4)]
    s1, s2, s3 = sym[0:3], sym[4:7], (sym[8:11] if version == 1 else sym[8:12])
    out = np.zeros((steps, 1024), dtype=np.uint16)  # [step, lane * 32 + i]
    for s in range(steps):
        pd = d[s, :256].view("<u4").reshape(2, 32)  # [word][lane]
        digits = np.zeros((32, 32), dtype=np.int64)
        for wd in range(2):
            for j in range(16):
                digits[:, wd * 16 + j] = (pd[wd] >> (2 * j)) & 3
        sw = d[s, 256:].reshape(2, 32, 16)  # [half][lane][byte]
        byts = np.concatenate([sw[0], sw[1]], axis=1).astype(np.int64)  # [lane][32]
        blk, end = int(base[s]), int(base[s + 1])
        t1 = int((digits == 3).sum())
        # tier-2 digits of the tier-1 escapes, in order (lane by lane)
        d2 = np.zeros(t1, dtype=np.int64)
        for k in range(t1):
            c, j = k >> 4, k & 15
            wv = int(np.frombuffer(bl[end - 4 * (c + 1) : end - 4 * c].tobytes(), "<u4")[0])
            d2[k] = (wv >> (2 * j)) & 3
        m2 = d2 == 3
        t2 = int(m2.sum())
        if version == 1:
            r0 = (2 * t2 + 7) >> 3
            d3 = np.array([(int(bl[blk + (2 * k) // 8]) >> ((2 * k) % 8)) & 3 for k in range(t2)], dtype=np.int64)
            lit_at = blk + r0
            exps = np.zeros(t1, dtype=np.int64)
            n3 = 0
            k2 = 0
            for k in range(t1):
                if d2[k] < 3:
                    exps[k] = s2[d2[k]]
                else:
                    if d3[k2] < 3:
                        exps[k] = s3[d3[k2]]
                    else:
                        exps[k] = bl[lit_at + n3]
                        n3 += 1
                    k2 += 1
        else:
            r0 = (2 * t2 + 7) >> 3
            d3 = np.array([(int(bl[blk + (2 * k) // 8]) >> ((2 * k) % 8)) & 3 for k in range(t2)], dtype=np.int64)
            exps = np.zeros(t1, dtype=np.int64)
            k2 = 0
            for k in range(t1):
                if d2[k] < 3:
                    exps[k] = s2[d2[k]]
                else:
                    exps[k] = s3[d3[k2]]
                    k2 += 1
            words = (t1 + 15) >> 4
            nl = (end - blk - r0 - 4 * words) // 3
            for i in range(nl):
                x = int(bl[blk + r0 + 3 * i]) | int(bl[blk + r0 + 3 * i + 1]) << 8 | int(bl[blk + r0 + 3 * i + 2]) << 16
                exps[x & 0x3FF] = (x >> 10) & 0xFF
        # weights: lane l's weight i has digit; escapes take exps in order
        flat = digits.reshape(-1)
        e = np.zeros(1024, dtype=np.int64)
        esc = flat == 3
        e[~esc] = np.array(s1)[flat[~esc]]
        e[esc] = exps
        e = e.reshape(32, 32)
        # byte of weight i is its mantissa above the sign of its pair's other weight (i ^ 1)
        sgn = np.zeros((32, 32), dtype=np.int64)
        man = (byts >> 1) & 0x7F
        sgn = byts & 1  # the sign of the weight i ^ 1
        sgn_i = sgn.reshape(32, 16, 2)[:, :, ::-1].reshape(32, 32)
        out[s] = ((sgn_i << 15) | (e << 7) | man).reshape(1024).astype(np.uint16)
    # [step = rb * KS + ks, lane = 4g + t, i = 4n + j] -> W[rb * 64 + 8n + g][ks * 16 + 8(j >> 1) + 2t + (j & 1)]
    o = out.reshape(RB, KS, 8, 4, 8, 2, 2)  # [rb, ks, g, t, n, jh, jl]
    W = o.transpose(0, 4, 2, 1, 5, 3, 6).reshape(O, K)  # [rb, n, g, ks, jh, t, jl]
    return W


def cold_flags(w, bb):
    """The steps the T2 hot decode cannot take (the general decode does them), from the weights and the T2 block offsets bb [steps + 1]
    (each step's block start, then the last's end): more than 512 tier-1 escapes, more than 32 literals, or tier-3 digits or literal entries
    that leave the 128-byte window (the hot path reads a block's first 128 bytes, from the word its first byte is in, through shuffles).
    -> bool [steps]. The format carries it in bit 31 of bb[step] (`flag_bb`)."""
    import v2bits
    O, K, RB, KS, u = prep(w)
    e = v2bits.steps_of(w)
    h = torch.bincount(e.flatten(), minlength=256)
    order = torch.argsort(h, descending=True, stable=True)
    rank = torch.full((256,), 99, dtype=torch.int64)
    rank[order[:10]] = torch.arange(10)
    r = rank[e.long()]
    t1 = (r >= 3).sum(1)
    nl = (r >= 10).sum(1)
    t2 = ((r >= 6) & (r < 10)).sum(1)
    start = bb[:-1].to(torch.int64)
    o = 8 * (start & 3)
    r0 = (2 * t2 + 7) >> 3
    cold = (t1 > 512) | (nl > 32) | (o + 2 * t2 + 32 >= 992) | ((nl > 0) & (o + 8 * r0 + 24 * (nl - 1) >= 992))
    return cold


def flag_bb(w, bb):
    """bb with bit 31 of bb[step] set where the step is cold (as int32 for the file)."""
    cold = cold_flags(w, bb)
    b = bb.to(torch.int64).numpy().copy()
    b[: cold.numel()] += cold.numpy().astype(np.int64) * (1 << 31)
    return b.astype(np.uint32).view(np.int32), int(cold.sum())


def dump(prefix, w, vers=(1, 2)):
    O, K, *_ = prep(w)
    np.asarray(w.contiguous().view(torch.int16).numpy()).tofile(prefix + ".w")
    for v in vers:
        data, blocks, bb, tiers = pack(w, v)
        if v == 1:
            data.numpy().tofile(prefix + ".data")
        blocks.numpy().tofile(f"{prefix}.v{v}.blocks")
        bb.numpy().astype("<i4").tofile(f"{prefix}.v{v}.bb")
        if v == 2:
            bbf, ncold = flag_bb(w, bb)
            bbf.tofile(f"{prefix}.v2.bbf")
            print(f"v2: {ncold} of {bb.numel() - 1} steps cold", flush=True)
        open(f"{prefix}.v{v}.tiers", "w").write(" ".join(str(t) for t in tiers))
        print(f"v{v}: {8 * (data.numel() + blocks.numel() + 4 * bb.numel()) / (O * K):.4f} bits a weight, blocks {blocks.numel()} B", flush=True)
    nd, ne, nb, hb = pack_nib(w)
    nd.numpy().tofile(prefix + ".nib.data")
    ne.numpy().astype("<i4").tofile(prefix + ".nib.exc")
    nb.numpy().astype("<i4").tofile(prefix + ".nib.eb")
    open(prefix + ".nib.hb", "w").write(str(hb))
    print(f"12-bit: {8 * (nd.numel() + 4 * ne.numel() + 4 * nb.numel()) / (O * K):.4f} bits a weight, hb {hb}, {int(nb[-1])} exceptions", flush=True)
    open(prefix + ".dims", "w").write(f"{O} {K}")


def load_w(model, tensor):
    snap = sorted(glob.glob(os.path.expanduser(f"~/hf/hub/models--{model}/snapshots/*")))[0] if "/" not in model or not os.path.isdir(model) else model
    from safetensors import safe_open
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snap, idx[tensor]), "pt", device="cpu") as f:
        return f.get_tensor(tensor)


if __name__ == "__main__" and sys.argv[1] != "flag":
    model, tensor, prefix = sys.argv[1:4]
    w = load_w(model, tensor)
    if len(sys.argv) > 4:  # a corner of it: ROWS COLS
        w = w[: int(sys.argv[4]), : int(sys.argv[5])].contiguous()
    dump(prefix, w)
    if os.environ.get("CHECK"):
        for v in (1, 2):
            data, blocks, bb, tiers = pack(w, v)
            W = unpack(tuple(w.shape), data, blocks, bb, tiers, v)
            assert np.array_equal(W.astype(np.uint16), w.contiguous().view(torch.int16).numpy().astype(np.uint16)), f"v{v} reference decode differs"
            print(f"v{v}: the reference decoder gives every weight back")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "flag":  # python pack_v2.py flag PREFIX: PREFIX.v2.bbf from PREFIX.w and PREFIX.v2.bb
    prefix = sys.argv[2]
    O, K = [int(x) for x in open(prefix + ".dims").read().split()]
    w = torch.from_numpy(np.fromfile(prefix + ".w", dtype=np.int16).reshape(O, K).copy()).view(torch.bfloat16)
    bb = torch.from_numpy(np.fromfile(prefix + ".v2.bb", dtype="<i4").copy())
    bbf, n = flag_bb(w, bb)
    bbf.tofile(prefix + ".v2.bbf")
    print(f"{prefix}: {n} of {bb.numel() - 1} steps cold")
