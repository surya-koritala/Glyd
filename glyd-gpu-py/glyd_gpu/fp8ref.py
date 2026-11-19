"""The lossless FP8 (e4m3) weight format, in numpy: pack(w) and unpack(p), bit for bit. The reference the GPU kernels follow byte for byte,
and what gpu/fp8/README.md's table is measured on. numpy only (no torch); a release build compiles this module, so no name or message in
it says how the format works (check_wheel's words: the layout is described here, in the docstring, which is not compiled).

The matrix and its patches
    w is uint8 [O, K], e4m3fn bytes (sign 1 bit, exponent 4 bits with bias 7, mantissa 3 bits; 0x7F and 0xFF are NaN; no infinity).
    Any byte is a value of the format: NaNs, both zeros and subnormals are coded as the bytes they are, never changed or special-cased.
    w is padded with zero bytes to a multiple of PATCH = (64, 128) (rows, columns) and cut into patches of 64 x 128 values (8192 values,
    8 KiB), numbered in row-major order of the patches: patch p = (row of patches) * KT + (column of patches), KT = ceil(K / 128).
    A patch is decoded on its own, from its head, its core and its part of var; and each of its four row blocks (mb) on its own too, from
    the head's offsets of that block.

A patch holds 4 row blocks mb of 16 rows, each 4 column blocks ks of 32 columns: 16 units u = 4 * mb + ks of 16 x 32 values, a unit 32
lanes of 16 values. Lane l (g = l >> 2, t = l & 3) of a unit holds, for j = 0..15, the value at
        row = 16 * mb + g + 8 * ((j >> 2) & 1),   column = 32 * ks + 4 * t + (j & 3) + 16 * (j >> 3)
(the order in which a lane of a warp holds the A operand, a0..a3, of mma.sync.m16n8k32 with e4m3, 4 bytes a register; the same 16 values
as the B operand's two fragments of two n8 blocks). A lane's values j = 4q..4q+3 are its register q ("group" q).
The patch's "stream" is its values in the order (mb, lane, ks, j): stream index i = 2048 * mb + 64 * l + 16 * ks + j (a lane's 64 values
of a row block are together: its four units' values, unit by unit). PERM[i] is the stream's i-th value's index in the patch's row-major order.

Each value is coded as three parts: its sign and 3 mantissa bits, kept as they are, and its exponent e (0..15) as a rank r in the patch's
own ranking, coded in levels:
    ranking   the 16 exponents by count in the patch, commonest first, ties by the higher exponent first (a count of 0 is a tie too:
              the unused exponents follow in descending order); sym[r] is the exponent of rank r.
    level 1   every value has a 2-bit digit d1 = min(r, 3). d1 < 3: the value's exponent is sym[d1]. d1 = 3: it spills to level 2.
    level 2   the values with d1 = 3 (in stream order, n1 of them) each have a 2-bit digit d2 = min(r - 3, 3); d2 < 3: sym[3 + d2]; d2 = 3: spills.
    level 3   the values with d2 = 3 (n2 of them, in stream order) each have a 3-bit digit d3 = min(r - 6, 7); d3 < 7: sym[6 + d3]; d3 = 7: spills.
    literal   the values with d3 = 7 (n3 of them, in stream order) each have their exponent as 4 bits. (Ranks 0..12 are in sym.)
    A value costs 6 bits in the core, plus 2 more at level 2, 3 at level 3 and 4 at the literal, as it reaches them.

Three arrays make a pack (Packed; the byte order is little-endian throughout; every index below is of a patch p):
    core   uint8 [patches, 6144]. Patch p's core: 16 units of 384 bytes, unit u at u * 384:
              +0     32 words of d1 digits: word l (lane l) holds d1 of its 16 values, value j at bits 4 * (j & 7) + 2 * (j >> 3) (values 0..7
                     in the low two bits of the 8 nibbles, values 8..15 in their high two bits).
              +128   32 lanes x 2 words of signs and mantissas: lane l's words at +128 + 8 l (values 0..7) and +132 + 8 l (values 8..15). In
                     the word of values 8w..8w+7, byte k (k = 0..3) holds: bit 7 the sign of value 8w + k, bits 0..2 its mantissa; bit 3 the
                     sign of value 8w + 4 + k; bits 4..6 the mantissa of value 8w + 4 + ((k + 1) & 3). (So the word masked by 0x87878787 is
                     values 8w..8w+3 as e4m3 bytes with exponent 0, and the word rotated left by 4 bits, masked the same, is values
                     8w+4..8w+7 so.)
    heads  uint32 [patches, 8], patch p's head:
              [0]     base: the word index in var of its variable part;
              [1]     sym[0..7] as nibbles: sym[r] at bits 4r..4r+3;
              [2]     bits 0..19: sym[8..12] (sym[8 + r] at bits 4r); bits 20..31: n3;
              [3]     bits 0..13: n1; bits 14..27: n2; bit 31: 1 if the patch is plain (see below); the other bits 0;
              [4..7]  the row blocks' offsets: for mb = 1, 2, 3, the values of the stream before row block mb that reach level 2 (o2[mb]),
                      level 3 (o3[mb]) and the literal (o4[mb]): [4] = o2[1] | o2[2] << 16, [5] = o2[3] | o3[1] << 16,
                      [6] = o3[2] | o3[3] << 16, [7] = o4[1] | o4[2] << 10 | o4[3] << 20. (Row block 0's are 0.)
    var    uint32 [W + 8]: for each patch, its variable part at heads[p][0], parts in order of p with no gap; then 8 zero words.
              A coded patch: level 2's n1 digits (2 bits each, digit k at word k >> 4, bits 2 * (k & 15), stream order), padded with zero
              bits to a word: w2 = ceil(n1 / 16) words; then level 3's n2 digits, 3 bits each, digit k at bits 3k..3k+2 of that run of
              words read as one little-endian bit string (a digit may cross a word): w3 = ceil(3 n2 / 32) words; then the literals, n3
              nibbles (nibble k at word k >> 3, bits 4 * (k & 7)): w4 = ceil(n3 / 8) words.
    A patch whose variable part would be 512 words or more, or with 1024 literals or more (which the first implies: 1024 literals take 725
    words at least, the 13 ranks before them as common as the rarest of them), is plain (bit 31 of heads[p][3]): its stream bytes,
    as they are (e4m3 bytes), are in core for the first 6144 (row blocks 0..2) and in var, 512 words from heads[p][0], for the last 2048
    (row block 3); the rest of its head is 0. So no patch takes more than 8192 + 32 bytes.

    nbytes() = core + heads + var + 8 (the shape, two words).

pack(w) is deterministic: the same w gives the same bytes (and the GPU's pack, the same bytes as this one).
"""
import numpy as np

PATCH = (64, 128)  # the unit a GPU kernel decodes: a block of 64 rows x 128 columns of the matrix, 16 units of 16 x 32
_R, _C = PATCH
VALUES = _R * _C  # 8192
UNITS, LANES, PER_LANE = 16, 32, 16
BLOCK = VALUES // 4  # a row block's values (2048)
CORE_BYTES = VALUES * 6 // 8  # 6144: 4 bits of sign and mantissa and a 2-bit digit, a value
TAIL_WORDS = (VALUES - CORE_BYTES) // 4  # 512: the words of var a plain patch holds
HEAD_WORDS = 8
MAX_LIT = 1024  # a coded patch has fewer literals than this
GUARD = 8  # words after the last patch's part: a lane's reads of a few words past its digits never leave var
CHUNK = 128  # patches a step of pack and unpack (their scratch is some 60 bytes a value)


def _perm():
    mb, l, ks, j = np.meshgrid(np.arange(4), np.arange(LANES), np.arange(4), np.arange(PER_LANE), indexing="ij")
    row = 16 * mb + (l >> 2) + 8 * ((j >> 2) & 1)
    col = 32 * ks + 4 * (l & 3) + (j & 3) + 16 * (j >> 3)
    return (row * _C + col).reshape(-1)


PERM = _perm()
_D1_SHIFT = (4 * (np.arange(16) & 7) + 2 * (np.arange(16) >> 3)).astype(np.uint32)  # a value's digit in its lane's word
_k = np.arange(4)
_S_SHIFT = np.concatenate([8 * _k + 7, 8 * _k + 3]).astype(np.uint32)  # a sign's bit in its word (the word's values 0..7)
_M_SHIFT = np.concatenate([8 * _k, 8 * ((_k + 3) & 3) + 4]).astype(np.uint32)  # a mantissa's lowest bit


class Packed:
    """A matrix [O, K] of e4m3 bytes in this format: shape, core, heads and var (see the module's docstring)."""

    def __init__(self, shape, core, heads, var):
        self.shape, self.core, self.heads, self.var = tuple(shape), core, heads, var

    def nbytes(self):
        return self.core.nbytes + self.heads.nbytes + self.var.nbytes + 8

    def bits_per_value(self):
        return self.nbytes() * 8 / (self.shape[0] * self.shape[1])

    def plain(self):
        """The patches stored as they are (bit 31 of heads[p][3]): a count."""
        return int((self.heads[:, 3] >> 31).sum())


def _ordinal(ti, c):
    """The place of each of the entries (ti: the patch of each, ascending) among its patch's."""
    n = np.bincount(ti, minlength=c)
    return np.arange(len(ti)) - (np.cumsum(n) - n)[ti]


def _fields(nvar, at_bits, val, width):
    """var words (nvar) with the fields val (width[i] bits each) at the bit positions at_bits, which never overlap."""
    w = at_bits >> 5
    sh = at_bits & 31
    lo = (val.astype(np.uint64) << sh.astype(np.uint64)) & 0xFFFFFFFF
    over = sh + width > 32
    hi = val[over].astype(np.uint64) >> (32 - sh[over]).astype(np.uint64)
    out = np.bincount(np.concatenate([w, w[over] + 1]), weights=np.concatenate([lo, hi]).astype(np.float64), minlength=nvar)
    return out.astype(np.uint64).astype(np.uint32)  # (fields never share a bit: the sums are exact)


def _read(var, at_bits, width):
    """The fields (width bits) of var at the bit positions at_bits."""
    w = at_bits >> 5
    two = var[w].astype(np.uint64) | (var[np.minimum(w + 1, len(var) - 1)].astype(np.uint64) << 32)
    return ((two >> (at_bits & 31).astype(np.uint64)) & ((1 << width) - 1)).astype(np.int64)


def _offsets(m):
    """m [c, 8192] bool (stream order): the counts of the row blocks before row blocks 1..3, [c, 3]."""
    per = m.reshape(len(m), 4, BLOCK).sum(2)
    return np.cumsum(per, axis=1)[:, :3]


def _pack_chunk(x):
    """x [c, 8192] uint8, patches in row-major order -> core [c, 6144], heads [c, 8] (bases from 0), var (uint32)."""
    c = len(x)
    x = x[:, PERM]  # the streams
    ex = ((x >> 3) & 15).astype(np.int64)
    cnt = np.bincount((np.arange(c, dtype=np.int64)[:, None] * 16 + ex).ravel(), minlength=16 * c).reshape(c, 16)
    order = np.argsort(-(cnt * 16 + np.arange(16)), axis=1, kind="stable")  # [c, rank] -> the exponent: commonest first, ties higher first
    r = np.take_along_axis(np.argsort(order, axis=1), ex, axis=1)  # each value's rank
    m1, m2, m3 = r >= 3, r >= 6, r >= 13
    n1, n2, n3 = m1.sum(1), m2.sum(1), m3.sum(1)
    w2, w3, w4 = -(-n1 // 16), -(-3 * n2 // 32), -(-n3 // 8)
    words = w2 + w3 + w4
    plain = (words >= TAIL_WORDS) | (n3 >= MAX_LIT)
    size = np.where(plain, TAIL_WORDS, words)
    base = np.cumsum(size) - size
    nvar = int(size.sum())
    # the variable parts: the digits and literals of the coded patches, each a run of fields in words
    cm1, cm2, cm3 = m1 & ~plain[:, None], m2 & ~plain[:, None], m3 & ~plain[:, None]
    ti1, ti2, ti3 = np.nonzero(cm1)[0], np.nonzero(cm2)[0], np.nonzero(cm3)[0]
    k1, k2, k3 = _ordinal(ti1, c), _ordinal(ti2, c), _ordinal(ti3, c)
    at = np.concatenate([base[ti1] * 32 + 2 * k1, (base + w2)[ti2] * 32 + 3 * k2, (base + w2 + w3)[ti3] * 32 + 4 * k3])
    val = np.concatenate([np.minimum(r[cm1] - 3, 3), np.minimum(r[cm2] - 6, 7), ex[cm3]])
    width = np.concatenate([np.full(len(ti1), 2), np.full(len(ti2), 3), np.full(len(ti3), 4)])
    var = _fields(nvar, at, val, width) if len(at) else np.zeros(nvar, np.uint32)
    p = np.nonzero(plain)[0]
    if len(p):
        var[(base[p][:, None] + np.arange(TAIL_WORDS)).ravel()] = np.ascontiguousarray(x[p, CORE_BYTES:]).view("<u4").reshape(-1)
    # the core: per unit 32 words of d1 digits, then 32 lanes of 2 words of signs and mantissas ([c, mb, l, ks, j] -> [c, u, l, j])
    d1 = np.minimum(r, 3).astype(np.uint32).reshape(c, 4, LANES, 4, PER_LANE).transpose(0, 1, 3, 2, 4).reshape(c, UNITS, LANES, PER_LANE)
    t1 = np.bitwise_or.reduce(d1 << _D1_SHIFT, axis=-1)
    xu = x.reshape(c, 4, LANES, 4, PER_LANE).transpose(0, 1, 3, 2, 4).reshape(c, UNITS, LANES, 2, 8).astype(np.uint32)
    nb = np.bitwise_or.reduce(((xu >> 7) << _S_SHIFT) | ((xu & 7) << _M_SHIFT), axis=-1)
    core = np.ascontiguousarray(np.concatenate([t1, nb.reshape(c, UNITS, 2 * LANES)], axis=2), dtype="<u4").view(np.uint8).reshape(c, CORE_BYTES)
    core[plain] = x[plain, :CORE_BYTES]
    o2, o3, o4 = _offsets(m1).astype(np.uint64), _offsets(m2).astype(np.uint64), _offsets(m3).astype(np.uint64)
    heads = np.zeros((c, HEAD_WORDS), np.uint64)
    heads[:, 0] = base
    heads[:, 1] = np.bitwise_or.reduce(order[:, :8].astype(np.uint64) << (4 * np.arange(8, dtype=np.uint64)), axis=1)
    heads[:, 2] = np.bitwise_or.reduce(order[:, 8:13].astype(np.uint64) << (4 * np.arange(5, dtype=np.uint64)), axis=1) | (n3.astype(np.uint64) << 20)
    heads[:, 3] = n1.astype(np.uint64) | (n2.astype(np.uint64) << 14)
    heads[:, 4] = o2[:, 0] | (o2[:, 1] << 16)
    heads[:, 5] = o2[:, 2] | (o3[:, 0] << 16)
    heads[:, 6] = o3[:, 1] | (o3[:, 2] << 16)
    heads[:, 7] = o4[:, 0] | (o4[:, 1] << 10) | (o4[:, 2] << 20)
    heads[plain, 1:] = 0
    heads[plain, 3] = 1 << 31
    return core, heads.astype(np.uint32), var


def pack(w):
    """w [O, K] uint8 (e4m3 bytes) -> Packed. Exact for every byte pattern, any shape."""
    w = np.asarray(w)
    if w.dtype != np.uint8 or w.ndim != 2 or w.size == 0:
        raise ValueError("w must be a non-empty uint8 matrix")
    O, K = w.shape
    rt, kt = -(-O // _R), -(-K // _C)
    pad = np.zeros((rt * _R, kt * _C), np.uint8)
    pad[:O, :K] = w
    x = pad.reshape(rt, _R, kt, _C).transpose(0, 2, 1, 3).reshape(rt * kt, VALUES)
    cores, heads, vars_, at = [], [], [], 0
    for a in range(0, len(x), CHUNK):
        core, head, var = _pack_chunk(x[a : a + CHUNK])
        if at + len(var) + GUARD >= 2**32:
            raise ValueError("w is too large for one pack")
        head[:, 0] += at
        cores.append(core)
        heads.append(head)
        vars_.append(var)
        at += len(var)
    return Packed((O, K), np.concatenate(cores), np.concatenate(heads), np.concatenate(vars_ + [np.zeros(GUARD, np.uint32)]))


def _unpack_chunk(core, head, var):
    """core [c, 6144], head [c, 8], var -> the patches [c, 8192] in row-major order."""
    c = len(core)
    base = head[:, 0].astype(np.int64)
    plain = (head[:, 3] >> 31) != 0
    n1 = (head[:, 3] & 0x3FFF).astype(np.int64)
    n2 = ((head[:, 3] >> 14) & 0x3FFF).astype(np.int64)
    w2, w3 = -(-n1 // 16), -(-3 * n2 // 32)
    sym = np.concatenate([(head[:, 1:2] >> (4 * np.arange(8, dtype=np.uint32))) & 15, (head[:, 2:3] >> (4 * np.arange(5, dtype=np.uint32))) & 15], axis=1).astype(np.int64)
    cw = np.ascontiguousarray(core).view("<u4").reshape(c, UNITS, 3 * LANES)
    d1 = ((cw[:, :, :LANES, None] >> _D1_SHIFT) & 3).reshape(c, 4, 4, LANES, PER_LANE)  # [c, mb, ks, l, j]
    nb = cw[:, :, LANES:].reshape(c, UNITS, LANES, 2, 1)
    s = ((nb >> _S_SHIFT) & 1).reshape(c, 4, 4, LANES, PER_LANE)
    mt = ((nb >> _M_SHIFT) & 7).reshape(c, 4, 4, LANES, PER_LANE)
    d1 = d1.transpose(0, 1, 3, 2, 4).reshape(c, VALUES).astype(np.int64)  # the streams
    s = s.transpose(0, 1, 3, 2, 4).reshape(c, VALUES)
    mt = mt.transpose(0, 1, 3, 2, 4).reshape(c, VALUES)
    m1 = (d1 == 3) & ~plain[:, None]
    ti1 = np.nonzero(m1)[0]
    k1 = _ordinal(ti1, c)
    d2 = _read(var, base[ti1] * 32 + 2 * k1, 2)
    m2 = d2 == 3
    ti2 = ti1[m2]
    k2 = _ordinal(ti2, c)
    d3 = _read(var, (base + w2)[ti2] * 32 + 3 * k2, 3)
    m3 = d3 == 7
    ti3 = ti2[m3]
    k3 = _ordinal(ti3, c)
    lit = _read(var, (base + w2 + w3)[ti3] * 32 + 4 * k3, 4)
    n3 = (head[:, 2] >> 20).astype(np.int64)
    counts = [np.bincount(t, minlength=c)[~plain] for t in (ti1, ti2, ti3)]
    if not all(np.array_equal(a, b[~plain]) for a, b in zip(counts, (n1, n2, n3))):
        raise ValueError("a pack whose digits are not the counts of its heads")
    e2 = sym[ti2, 6 + np.minimum(d3, 6)]
    e2[m3] = lit
    e1 = sym[ti1, 3 + np.minimum(d2, 2)]
    e1[m2] = e2
    ex = np.take_along_axis(sym, np.minimum(d1, 2), axis=1)
    ex[m1] = e1
    out = ((s << 7) | (ex << 3) | mt).astype(np.uint8)
    p = np.nonzero(plain)[0]
    if len(p):
        out[p, :CORE_BYTES] = core[p]
        out[p, CORE_BYTES:] = var[(base[p][:, None] + np.arange(TAIL_WORDS)).ravel()].astype("<u4").view(np.uint8).reshape(len(p), -1)
    res = np.empty_like(out)
    res[:, PERM] = out
    return res


def unpack(p):
    """Packed -> w [O, K] uint8, the bytes pack was given."""
    O, K = p.shape
    rt, kt = -(-O // _R), -(-K // _C)
    x = np.empty((rt * kt, VALUES), np.uint8)
    for a in range(0, len(x), CHUNK):
        x[a : a + CHUNK] = _unpack_chunk(p.core[a : a + CHUNK], p.heads[a : a + CHUNK], p.var)
    return x.reshape(rt, kt, _R, _C).transpose(0, 2, 1, 3).reshape(rt * _R, kt * _C)[:O, :K].copy()
