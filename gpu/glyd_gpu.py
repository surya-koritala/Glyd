"""Glyd weights on the GPU: bf16 tensors held compressed in VRAM and
decoded on the GPU, bit for bit (glyd_gpu.cu has the layout).

    p = pack(w)          # w: a bf16 CUDA tensor
    w2 = unpack(p)       # the same bits
    p.bits_per_weight()
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_ext = load(
    name="glyd_gpu",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyd_gpu.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_89"] + (["-Xptxas", "-v"] if os.environ.get("GLYD_GPU_PTXAS") else []),
    verbose=bool(os.environ.get("GLYD_GPU_PTXAS")),
)

FLAT_TILE = 16384  # weights a tile when the tensor is not a matrix of rows
ROW_TILE = 8192  # about as many a tile for a matrix: whole rows
MIN_TILES = 2048  # warps a matrix's product should keep busy
MIN_TILE = 4096  # but no tile under this many weights (its offsets cost)
STAGE_MAX = int(os.environ.get("GLYD_GPU_STAGE_MAX", 2048))  # words of a tile's streams a warp stages in shared memory, at most (past that the reader reads in place, a word ahead)
SEG = 1024  # the fast format's escape index is kept a segment of this many weights of a row


def class_code(freq):
    """The cheapest code of the dense format for these exponent counts:
    classes (c zeros, a one, s_c bits) over the ranks by frequency, the
    last one an escape (8 raw bits) when ranks are left. Returns the
    classes as (base, s, escape) and the ranks' exponents."""
    order = [int(e) for e in np.argsort(-np.asarray(freq), kind="stable") if freq[e] > 0]
    p = np.asarray([freq[e] for e in order], dtype=np.float64)
    R = len(order)
    cum = np.concatenate([[0.0], np.cumsum(p)])
    memo = {}

    def f(i, c):  # (cost, classes) covering ranks i.. from class c
        if i >= R:
            return 0.0, ()
        if (i, c) not in memo:
            best = ((cum[R] - cum[i]) * (c + 9), ((i, 8, True),))
            if c + 1 < 16:
                for sbits in range(0, 9):
                    j = min(R, i + (1 << sbits))
                    if j > 32:  # the rank table is 32 entries, one a lane
                        break
                    cost, rest = f(j, c + 1)
                    cost += (cum[j] - cum[i]) * (c + 1 + sbits)
                    if cost < best[0]:
                        best = (cost, ((i, sbits, False),) + rest)
            memo[(i, c)] = best
        return memo[(i, c)]

    return list(f(0, 0)[1]), order


def code_tables(freq):
    """Every exponent's code length and code (MSB first), and the kernel's
    tables: 32 class entries (base << 8 | s << 1 | escape), 32 ranks."""
    classes, order = class_code(freq)
    lengths = np.zeros(256, dtype=np.int64)
    codes = np.zeros(256, dtype=np.int64)
    for r, e in enumerate(order):
        for c, (base, sbits, esc) in enumerate(classes):
            if esc:
                lengths[e], codes[e] = c + 9, (1 << 8) | e
                break
            if r < base + (1 << sbits):
                lengths[e], codes[e] = c + 1 + sbits, (1 << sbits) | (r - base)
                break
    # 32 class entries (code length | (base - 2^s) mod 32 << 8 | escape
    # << 16), 32 ranks' exponents in a float's place (<< 23).
    tables = np.zeros(64, dtype=np.int64)
    for c, (base, sbits, esc) in enumerate(classes):
        tables[c] = (c + 1 + sbits) | (((base - (1 << sbits)) % 32) << 8) | (int(esc) << 16)
    for r, e in enumerate(order[:32]):
        tables[32 + r] = e << 23
    tables = np.where(tables[:64] >= 2**31, tables[:64] - 2**32, tables[:64])
    return lengths, codes, tables


class Packed:
    def __init__(self, shape, n, sm, stream, offs, tables, tw, V, tile_words):
        self.shape, self.n, self.sm, self.stream, self.offs, self.tables = shape, n, sm, stream, offs, tables
        # A warp stages its tile's streams in shared memory when they are
        # small enough not to cut the warps an SM holds (0: read in place).
        self.tw, self.V, self.tile_words = tw, V, tile_words if tile_words <= STAGE_MAX else 0
        self.rows_per_tile = tw // shape[1] if len(shape) == 2 and tw % shape[1] == 0 else 0
        # Split rows (tiles shorter than a row): the product adds each row's
        # parts in fp32 sums, cleared as they are written out.
        self.split = len(shape) == 2 and tw % shape[1] != 0
        self.sum = torch.zeros(shape[0] if self.split else 0, dtype=torch.float32, device=sm.device)
        self.count = torch.zeros(shape[0] if self.split else 0, dtype=torch.int32, device=sm.device)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.sm, self.stream, self.offs, self.tables))

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


CHUNK = 1 << 25  # weights the packers widen to int32 at a time


def _chunks(u):
    for a in range(0, u.numel(), CHUNK):
        yield a, u[a : a + CHUNK].to(torch.int32) & 0xFFFF


def _hist(u):
    h = torch.zeros(256, dtype=torch.int64, device=u.device)
    for _, v in _chunks(u):
        h += torch.bincount((v >> 7) & 0xFF, minlength=256)
    return h


def _sign_mantissa(u):
    sm = torch.empty(u.numel(), dtype=torch.uint8, device=u.device)
    for a, v in _chunks(u):
        sm[a : a + v.numel()] = (((v >> 8) & 0x80) | (v & 0x7F)).to(torch.uint8)
    return sm


_NONE = {}


def _none(dev):
    if dev not in _NONE:
        _NONE[dev] = torch.empty(0, dtype=torch.int64, device=dev)
    return _NONE[dev]


def pack(w):
    """The dense format (glyd_gpu.cu): exponents in a prefix code read by
    counting leading zeros."""
    assert w.dtype == torch.bfloat16 and w.is_cuda
    u = w.contiguous().view(torch.int16).flatten()
    n = u.numel()
    lengths, codes, tables = code_tables(_hist(u).cpu().numpy())
    dev = w.device
    len_t = torch.tensor(lengths, dtype=torch.uint8, device=dev)
    code_t = torch.tensor(codes, dtype=torch.int32, device=dev)
    # A matrix whose rows are a multiple of 128 long goes in tiles of
    # whole rows (the fused product needs them), 16 weights a lane a step
    # where the rows allow; anything else flat.
    if w.dim() == 2 and w.shape[1] % 128 == 0:
        # Rows a tile: about ROW_TILE weights, fewer where that would leave
        # the GPU under MIN_TILES warps (a small matrix: its offsets cost
        # more a weight, on few weights).
        O, K = w.shape
        V = 16 if K % 512 == 0 else 4
        if K > ROW_TILE and K % 512 == 0:
            tw = ROW_TILE  # long rows: tiles split them, the product adds the parts
        else:
            tw = max(1, min(ROW_TILE // K, max(MIN_TILE // K, O // MIN_TILES))) * K
    else:
        tw, V = FLAT_TILE, 4
    bits = _ext.lane_bits(u, len_t, tw, V).to(torch.int64)
    offs64 = torch.cumsum(bits, 0) - bits
    total = int(offs64[-1] + bits[-1])
    assert total < 2**31, "a tensor of more than 2^31 code bits"
    stream = torch.zeros(total // 32 + 4, dtype=torch.int32, device=dev)
    offs = offs64.to(torch.int32)
    _ext.write_codes(u, len_t, code_t, offs, stream, tw, V)
    # The most words a tile's streams span, with the reader's look-ahead:
    # what a warp stages in shared memory.
    starts = torch.cat([offs64[::32], torch.tensor([total], device=dev)])
    tile_words = int(((starts[1:] + 31) // 32 - starts[:-1] // 32).max()) + 3
    tab = torch.tensor(tables, dtype=torch.int32, device=dev)
    return Packed(tuple(w.shape), n, _sign_mantissa(u), stream, offs, tab, tw, V, tile_words)


def decode_tiles(p, tiles, out):
    """Tiles `tiles` of p (all of them: an empty tensor) decoded into out."""
    _ext.decode(p.sm, p.stream, p.offs, p.tables, p.n, p.tw, p.V, p.tile_words, tiles, out.view(torch.int16))


def unpack(p, out=None):
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.sm.device)
    decode_tiles(p, _none(p.sm.device), out)
    return out[: p.n].view(p.shape)


def rows(p, ids):
    """Rows `ids` of a matrix (an embedding's lookup): only their tiles decoded."""
    T, K = p.rows_per_tile, p.shape[1]
    ids = ids.flatten()
    tiles, where = torch.unique(ids // T, return_inverse=True)
    out = torch.empty(tiles.numel() * p.tw, dtype=torch.bfloat16, device=ids.device)
    decode_tiles(p, tiles, out)
    return out.view(-1, T, K)[where, ids % T]


def gemv(p, x, bias=None):
    """W x (+ bias) for one input vector, the weights decoded in registers."""
    O, K = p.shape
    y = torch.empty(O, dtype=torch.bfloat16, device=x.device)
    _ext.gemv(p.sm, p.stream, p.offs, p.tables, O, K, p.tw, p.V, p.tile_words, x.contiguous().view(-1), bias if bias is not None else _none(x.device).to(torch.bfloat16), y, p.sum, p.count)
    return y


class Fast:
    """The fast format of a matrix [O, K] (K a multiple of 128): each
    exponent a 3-bit code into the 7 most common, code 7 an escape to the
    exponent itself; decoded by bit operations, one warp a row."""

    def __init__(self, shape, sm, planes, exc, exc_base, top):
        self.shape, self.sm, self.planes, self.exc, self.exc_base, self.top = shape, sm, planes, exc, exc_base, top
        self.n = shape[0] * shape[1]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.sm, self.planes, self.exc, self.exc_base)) + 8

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


def pack_fast(w):
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2 and w.shape[1] % 128 == 0
    O, K = w.shape
    u = w.contiguous().view(torch.int16).flatten()
    dev = w.device
    top7 = _hist(u).argsort(descending=True)[:7]
    code_of = torch.full((256,), 7, dtype=torch.int32, device=dev)
    code_of[top7] = torch.arange(7, dtype=torch.int32, device=dev)
    shifts = torch.arange(32, dtype=torch.int64, device=dev)
    segs = (K + SEG - 1) // SEG
    planes = torch.empty(u.numel() // 32 * 3, dtype=torch.int32, device=dev)
    exc_parts, seg_counts = [], []
    # Whole rows a chunk, so that segments do not straddle chunks.
    rows = max(1, CHUNK // K)
    for r0 in range(0, O, rows):
        r1 = min(O, r0 + rows)
        v = u[r0 * K : r1 * K].to(torch.int32) & 0xFFFF
        e = (v >> 7) & 0xFF
        c = code_of[e]
        pl = torch.stack([(((c.view(-1, 32) >> b) & 1).to(torch.int64) << shifts).sum(1) for b in range(3)], 1).flatten()
        planes[r0 * K // 32 * 3 : r1 * K // 32 * 3] = (pl - (pl >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        esc = c == 7
        exc_parts.append(e[esc].to(torch.uint8))
        seg_counts.append(F.pad(esc.view(r1 - r0, K), (0, segs * SEG - K)).view(r1 - r0, segs, SEG).sum(2).flatten())
    exc = torch.cat(exc_parts)
    per_seg = torch.cat(seg_counts)
    exc_base = torch.cumsum(per_seg, 0) - per_seg
    assert exc.numel() < 2**31
    top = sum(int(x) << (8 * i) for i, x in enumerate(top7.tolist()))
    return Fast((O, K), _sign_mantissa(u), planes, exc, exc_base.to(torch.int32), top)


def fast_unpack(p, out=None):
    O, K = p.shape
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.sm.device)
    _ext.fast_decode(p.sm, p.planes, p.exc, p.exc_base, p.top, 0, O, _none(p.sm.device), K, out.view(torch.int16))
    return out[: p.n].view(O, K)


def fast_rows(p, ids):
    K = p.shape[1]
    ids = ids.flatten().to(torch.int64)
    out = torch.empty(ids.numel() * K, dtype=torch.bfloat16, device=ids.device)
    _ext.fast_decode(p.sm, p.planes, p.exc, p.exc_base, p.top, 0, 0, ids, K, out.view(torch.int16))
    return out.view(-1, K)


def fast_gemv(p, x, bias=None):
    O, K = p.shape
    y = torch.empty(O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_gemv(p.sm, p.planes, p.exc, p.exc_base, p.top, O, K, x.contiguous().view(-1), bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


def fast_gemm(p, x, bias=None):
    """X W^T (+ bias) for several tokens (x [M, K]), the weights decoded a
    step at a time into shared memory and multiplied on the tensor cores."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_gemm(p.sm, p.planes, p.exc, p.exc_base, p.top, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


def fast_bgemv(p, x, bias=None):
    """X W^T (+ bias) for 2, 4, 8 or 16 tokens (x [M, K]) on the CUDA cores:
    the one-token product's decode, each weight multiplied into M sums."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_bgemv(p.sm, p.planes, p.exc, p.exc_base, p.top, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


class Mma:
    """The mma layout of a matrix [O, K] (O a multiple of 64, K of 16): the
    fast format with its codes and bytes in the order the tensor cores take
    their B operand, its 7 exponents a run from base (see glyd_gpu.cu)."""

    def __init__(self, shape, data, exc, exc_base, base):
        self.shape, self.data, self.exc, self.exc_base, self.base = shape, data, exc, exc_base, base
        self.sm = data  # its device, as the other formats'
        self.n = shape[0] * shape[1]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.data, self.exc, self.exc_base)) + 4

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


def pack_mma(w):
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2 and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0
    O, K = w.shape
    RB, KS, dev = O // 64, K // 16, w.device
    u = w.contiguous().view(torch.int16)
    # The run of 7 exponents holding the most weights (base + 7 must not reach the sign).
    h = _hist(u.flatten())
    base = int(h.unfold(0, 7, 1).sum(1)[:249].argmax())
    steps = O * K // 1024
    data = torch.empty(steps, 1408, dtype=torch.uint8, device=dev)  # a warp step: code words [3][32 lanes], then [32 lanes][32 bytes]
    shifts = torch.arange(32, dtype=torch.int64, device=dev)
    exc_parts, step_counts = [], []
    per = max(1, (1 << 22) // (64 * K))  # row blocks a chunk
    for b0 in range(0, RB, per):
        b1 = min(RB, b0 + per)
        # [rb, n, g, ks, j >> 1, t, j & 1] -> [rb, ks, lane = 4g + t, 4n + j]
        v = u[b0 * 64 : b1 * 64].view(b1 - b0, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten().to(torch.int32) & 0xFFFF
        a, z = b0 * 64 * K, b1 * 64 * K
        e = (v >> 7) & 0xFF
        c = e - base
        esc = (c < 0) | (c > 6)
        c[esc] = 7
        bits = torch.stack([(c >> b) & 1 for b in range(3)], -1).view(-1, 3, 32).to(torch.int64)  # the 96-bit stream, code i at 3i
        words = (bits << shifts).sum(-1)
        words = (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32)  # [groups, 3]
        data[a // 1024 : z // 1024, :384] = words.view(-1, 32, 3).transpose(1, 2).contiguous().view(torch.uint8).view(-1, 384)
        # Weight i's byte: its mantissa above the sign of weight i ^ 1 (a pair decodes by one rotate).
        pr = v.view(-1, 2)
        sign = (pr >> 15) & 1
        data[a // 1024 : z // 1024, 384:] = (((pr & 0x7F) << 1) | sign.flip(1)).to(torch.uint8).view(-1, 1024)
        exc_parts.append(e[esc].to(torch.uint8))
        step_counts.append(esc.view(-1, 1024).sum(1))
    exc = torch.cat(exc_parts + [torch.zeros(16, dtype=torch.uint8, device=dev)])  # padded: the kernels read words past an escape
    per_step = torch.cat(step_counts)
    assert exc.numel() < 2**31
    return Mma((O, K), data.flatten(), exc, (torch.cumsum(per_step, 0) - per_step).to(torch.int32), base)


def mma_unpack(p, out=None, row0=0, rows=None):
    """Rows [row0, row0 + rows) of W (multiples of 64), bf16 [rows, K]."""
    O, K = p.shape
    rows = O - row0 if rows is None else rows
    if out is None:
        out = torch.empty(rows * K, dtype=torch.bfloat16, device=p.sm.device)
    _ext.mma_unpack(p.data, p.exc, p.exc_base, p.base, K, row0, rows, out.view(torch.int16))
    return out[: rows * K].view(rows, K)


def mma_gemm(p, x, bias=None):
    """X W^T (+ bias) for up to 64 tokens (x [M, K]) on the tensor cores,
    the weights decoded in registers straight into their operands."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.mma_gemm(p.data, p.exc, p.exc_base, p.base, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


def mma_gemm_big(p, x, bias=None):
    """X W^T (+ bias) for many tokens (x [M, K]; a prompt): each weight
    decoded once for 128 tokens, into shared memory, on the tensor cores."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.mma_gemm_big(p.data, p.exc, p.exc_base, p.base, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y
