"""GGUF: the tensor table of a file's head, and the blocks of the types a Q4_K_M or Q8_0 file holds unpacked to codes and scales
(llama.cpp's ggml-quants.c layouts). parse() needs no numpy; the unpackers do.

Unpacked, each block type gives codes [rows, columns] in the weight's own order and its side streams, each [rows, blocks a row]:
  Q8_0  32 values a block: d (fp16), 32 int8 codes                                   -> codes int8 (as uint8), d
  Q4_K  256 a block: d, dmin (fp16), 8 six-bit scales and 8 six-bit mins, 4-bit codes -> codes 0..15, d, dmin, sc, mn
  Q5_K  as Q4_K with a fifth bit a code                                              -> codes 0..31, d, dmin, sc, mn
  Q6_K  256 a block: 6-bit codes, 16 int8 scales, d (fp16)                           -> codes 0..63, sc (as uint8), d
"""
import struct

# ggml type -> (name, values a block, bytes a block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 15: ("Q8_K", 256, 292)}
KEEP = ("general.architecture", "general.file_type", "general.alignment", "general.name", "general.quantization_version")


class _R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, fmt):
        n = struct.calcsize(fmt)
        if self.i + n > len(self.b):
            raise IndexError("head too short")
        v = struct.unpack_from(fmt, self.b, self.i)
        self.i += n
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("<Q")
        if self.i + n > len(self.b):
            raise IndexError("head too short")
        s = self.b[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def value(self, t):
        f = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in f:
            return self.take(f[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            if et in f:  # skip a numeric array without decoding it
                sz = struct.calcsize(f[et]) * n
                if self.i + sz > len(self.b):
                    raise IndexError("head too short")
                self.i += sz
                return f"[{n} values]"
            for _ in range(n):
                self.value(et)
            return f"[{n} values]"
        raise ValueError(f"gguf value type {t}")


def parse(head):
    """The head of a GGUF file -> dict(version, alignment, data_start, kv (a few keys), tensors [name, dims, type, offset])."""
    r = _R(head)
    if r.take("<4s") != b"GGUF":
        raise ValueError("not GGUF")
    version, nt, nkv = r.take("<I"), r.take("<Q"), r.take("<Q")
    kv = {}
    for _ in range(nkv):
        k = r.string()
        v = r.value(r.take("<I"))
        if k in KEEP:
            kv[k] = v
    tensors = []
    for _ in range(nt):
        name = r.string()
        nd = r.take("<I")
        dims = [r.take("<Q") for _ in range(nd)]
        tensors.append([name, dims, r.take("<I"), r.take("<Q")])
    al = int(kv.get("general.alignment", 32))
    return {"version": version, "alignment": al, "data_start": -(-r.i // al) * al, "kv": kv, "tensors": tensors}


def nbytes(dims, t):
    _, bv, bb = TYPES[t]
    n = 1
    for d in dims:
        n *= d
    return n // bv * bb


def unpack(raw, t, cols):
    """Raw bytes of whole rows of a tensor of type t with `cols` values a row -> (codes uint8 [r, cols], {stream: array [r, blocks]},
    {stream: bits a value as stored})."""
    import numpy as np
    name, bv, bb = TYPES[t]
    nblk = cols // bv
    B = np.frombuffer(raw, np.uint8).reshape(-1, nblk, bb)
    r = B.shape[0]
    f16 = lambda a: np.ascontiguousarray(a).view("<u2").reshape(r, nblk)
    if name == "Q8_0":
        return B[:, :, 2:].reshape(r, cols).copy(), {"d": f16(B[:, :, 0:2])}, {"d": 16}
    if name in ("Q4_K", "Q5_K"):
        d, dmin, s = f16(B[:, :, 0:2]), f16(B[:, :, 2:4]), B[:, :, 4:16].astype(np.uint8)
        sc = np.empty((r, nblk, 8), np.uint8)
        mn = np.empty((r, nblk, 8), np.uint8)
        sc[..., :4], mn[..., :4] = s[..., 0:4] & 63, s[..., 4:8] & 63
        sc[..., 4:] = (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)
        mn[..., 4:] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
        if name == "Q4_K":
            qs = B[:, :, 16:144].reshape(r, nblk, 4, 32)
            q = np.stack([qs & 15, qs >> 4], axis=3)  # [r, blk, chunk, lo/hi, 32]: chunk j holds values 64j..64j+31 low, +32.. high
        else:
            qh = B[:, :, 16:48]  # [r, blk, 32]
            qs = B[:, :, 48:176].reshape(r, nblk, 4, 32)
            j = np.arange(4)[None, None, :, None]
            lo = (qs & 15) | (((qh[:, :, None, :] >> (2 * j)) & 1) << 4)
            hi = (qs >> 4) | (((qh[:, :, None, :] >> (2 * j + 1)) & 1) << 4)
            q = np.stack([lo, hi], axis=3)
        return (q.reshape(r, cols).astype(np.uint8), {"d": d, "dmin": dmin, "sc": sc.reshape(r, nblk * 8), "mn": mn.reshape(r, nblk * 8)},
                {"d": 16, "dmin": 16, "sc": 6, "mn": 6})
    if name == "Q6_K":
        ql = B[:, :, 0:128].reshape(r, nblk, 2, 64)    # a half of 128 values: ql[0:32], ql[32:64]
        qh = B[:, :, 128:192].reshape(r, nblk, 2, 32)
        sc = B[:, :, 192:208]
        d = f16(B[:, :, 208:210])
        q = np.empty((r, nblk, 2, 4, 32), np.uint8)    # value n + 32g + l of a half, g = 0..3
        q[..., 0, :] = (ql[..., 0:32] & 15) | (((qh >> 0) & 3) << 4)
        q[..., 1, :] = (ql[..., 32:64] & 15) | (((qh >> 2) & 3) << 4)
        q[..., 2, :] = (ql[..., 0:32] >> 4) | (((qh >> 4) & 3) << 4)
        q[..., 3, :] = (ql[..., 32:64] >> 4) | (((qh >> 6) & 3) << 4)
        return q.reshape(r, cols), {"sc": sc.reshape(r, nblk * 16).copy(), "d": d}, {"sc": 8, "d": 16}
    raise ValueError(f"no unpacker for {name}")


def block_of(t):
    """Values a scale value covers, for each side stream of type t."""
    name = TYPES[t][0]
    return {"Q8_0": {"d": 32}, "Q4_K": {"d": 256, "dmin": 256, "sc": 32, "mn": 32}, "Q5_K": {"d": 256, "dmin": 256, "sc": 32, "mn": 32},
            "Q6_K": {"sc": 16, "d": 256}}[name]


def selftest():
    """Hand-packed blocks unpack to the values put in (the layouts above, written out value by value)."""
    import numpy as np
    rng = np.random.default_rng(1)
    # Q4_K: one block
    v = rng.integers(0, 16, 256)
    scs, mns = rng.integers(0, 64, 8), rng.integers(0, 64, 8)
    s = bytearray(12)
    for j in range(4):
        s[j] = scs[j] | ((scs[j + 4] >> 4) << 6)
        s[j + 4] = mns[j] | ((mns[j + 4] >> 4) << 6)
        s[j + 8] = (scs[j + 4] & 15) | ((mns[j + 4] & 15) << 4)
    qs = bytearray(128)
    for c in range(4):
        for l in range(32):
            qs[32 * c + l] = v[64 * c + l] | (v[64 * c + 32 + l] << 4)
    raw = struct.pack("<HH", 0x3C00, 0x3800) + bytes(s) + bytes(qs)
    q, st, _ = unpack(raw, 12, 256)
    assert (q[0] == v).all() and (st["sc"][0] == scs).all() and (st["mn"][0] == mns).all() and st["d"][0, 0] == 0x3C00
    # Q5_K
    v5 = rng.integers(0, 32, 256)
    qh, qs = bytearray(32), bytearray(128)
    for c in range(4):
        for l in range(32):
            a, b = v5[64 * c + l], v5[64 * c + 32 + l]
            qs[32 * c + l] = (a & 15) | ((b & 15) << 4)
            qh[l] |= ((a >> 4) << (2 * c)) | ((b >> 4) << (2 * c + 1))
    q, _, _ = unpack(struct.pack("<HH", 0, 0) + bytes(s) + bytes(qh) + bytes(qs), 13, 256)
    assert (q[0] == v5).all()
    # Q6_K
    v6 = rng.integers(0, 64, 256)
    ql, qh = bytearray(128), bytearray(64)
    for h in range(2):
        for l in range(32):
            a, b, c, d = (v6[128 * h + l + 32 * g] for g in range(4))
            ql[64 * h + l] = (a & 15) | ((c & 15) << 4)
            ql[64 * h + l + 32] = (b & 15) | ((d & 15) << 4)
            qh[32 * h + l] = (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((d >> 4) << 6)
    sc = rng.integers(0, 256, 16).astype(np.uint8)
    q, st, _ = unpack(bytes(ql) + bytes(qh) + bytes(sc) + struct.pack("<H", 7), 14, 256)
    assert (q[0] == v6).all() and (st["sc"][0] == sc).all() and st["d"][0, 0] == 7
    # Q8_0
    v8 = rng.integers(0, 256, 64).astype(np.uint8)
    q, st, _ = unpack(struct.pack("<H", 5) + bytes(v8[:32]) + struct.pack("<H", 6) + bytes(v8[32:]), 8, 64)
    assert (q[0] == v8).all() and list(st["d"][0]) == [5, 6]
    # parse: a two-tensor file head
    def gs(x):
        return struct.pack("<Q", len(x)) + x.encode()
    head = b"GGUF" + struct.pack("<IQQ", 3, 2, 2) + gs("general.architecture") + struct.pack("<I", 8) + gs("qwen35")
    head += gs("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 3) + struct.pack("<3f", 0, 1, 2)
    head += gs("blk.0.ffn_down.weight") + struct.pack("<I2QIQ", 2, 256, 4, 12, 0) + gs("output.weight") + struct.pack("<I2QIQ", 2, 32, 2, 8, 576)
    m = parse(head)
    assert m["kv"]["general.architecture"] == "qwen35" and m["tensors"][1] == ["output.weight", [32, 2], 8, 576] and m["data_start"] % 32 == 0
    try:
        parse(head[:-5])
        raise AssertionError("short head parsed")
    except IndexError:
        pass
    print("gguf selftest ok")


if __name__ == "__main__":
    selftest()
