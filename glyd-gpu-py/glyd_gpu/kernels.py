"""Glyd weights on the GPU: bf16 tensors held compressed in VRAM and
decoded on the GPU, bit for bit (gpu/glyd_gpu.cu has the layout).

    p = pack(w)          # w: a bf16 CUDA tensor
    w2 = unpack(p)       # the same bits
    p.bits_per_weight()

The kernels come from the prebuilt library (gpu/build_lib.sh) through
_lib.py: $GLYD_GPU_LIB, else libglyd_gpu_cudaN.so next to this file for
PyTorch's CUDA N, loaded at the first call. gpu/glyd_gpu_dev.py, the scripts'
module, sets _ext to its own choice first (the library or its JIT build).
"""
import os
import types
import numpy as np
import torch
import torch.nn.functional as F
from . import _lib


def library():
    """The prebuilt library's path: $GLYD_GPU_LIB, else libglyd_gpu_cudaN.so
    next to this file for PyTorch's CUDA N; None if neither."""
    if os.environ.get("GLYD_GPU_LIB"):
        return os.environ["GLYD_GPU_LIB"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libglyd_gpu_cuda{(torch.version.cuda or '0').split('.')[0]}.so")
    return path if os.path.exists(path) else None


class _Load:
    """_ext before its first use: the library is loaded then, and _ext becomes _lib (no cost a call after)."""

    def __getattr__(self, name):
        global _ext
        path = library()
        if path is None and not torch.version.cuda:
            raise OSError("no Glyd GPU library for this PyTorch: it is built without CUDA")
        if path is None:
            raise OSError(f"no Glyd GPU library: libglyd_gpu_cuda{torch.version.cuda.split('.')[0]}.so is not next to {__file__} (the Linux wheels carry CUDA 12's and 13's); build it with gpu/build_lib.sh, or name it in GLYD_GPU_LIB")
        _lib.load(path)
        _ext = _lib
        return getattr(_lib, name)


_ext = _Load()  # the kernels: the pybind module's functions, by name


def lib():
    """_lib where the kernels are the prebuilt library's (loaded now if not yet), else None: the JIT build's, or no
    library found (the first kernel's call says so)."""
    if isinstance(_ext, _Load) and library() is not None:
        _ext.cuda_version  # loads it: _ext is _lib from here on
    return _lib if _ext is _lib else None


FLAT_SPAN = 16384  # weights a tile when the tensor is not a matrix of rows
ROW_SPAN = 8192  # about as many a tile for a matrix: whole rows
MIN_SPANS = 2048  # warps a matrix's product should keep busy
MIN_SPAN = 4096  # but no tile under this many weights (its offsets cost)
WORK_MAX = 2048  # words of a tile's streams a warp stages in shared memory, at most (past that the reader reads in place, a word ahead)
# dev: GLYD_GPU_STAGE_MAX
WORK_MAX = int(os.environ.get("GLYD_GPU_STAGE_MAX", WORK_MAX))
# end dev
SEG = 1024  # the fast format's escape index is kept a segment of this many weights of a row


def class_key(freq):
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


def key_tables(freq):
    """Every exponent's code length and code (MSB first), and the kernel's
    tables: 32 class entries (base << 8 | s << 1 | escape), 32 ranks."""
    classes, order = class_key(freq)
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
    def __init__(self, shape, n, core, stream, offs, tables, tw, V, span_words):
        self.shape, self.n, self.core, self.stream, self.offs, self.tables = shape, n, core, stream, offs, tables
        # A warp stages its tile's streams in shared memory when they are
        # small enough not to cut the warps an SM holds (0: read in place).
        self.tw, self.V, self.span_words = tw, V, span_words if span_words <= WORK_MAX else 0
        self.rows_per_span = tw // shape[1] if len(shape) == 2 and tw % shape[1] == 0 else 0
        # Split rows (tiles shorter than a row): the product adds each row's
        # parts in fp32 sums, cleared as they are written out.
        self.split = len(shape) == 2 and tw % shape[1] != 0
        self.sum = torch.zeros(shape[0] if self.split else 0, dtype=torch.float32, device=core.device)
        self.count = torch.zeros(shape[0] if self.split else 0, dtype=torch.int32, device=core.device)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.core, self.stream, self.offs, self.tables))

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


CHUNK = 1 << 25  # weights the packers widen to int32 at a time
# The mma packers run in passes of PACK_CHUNK weights, _hist of HIST_CHUNK, sized for PyTorch's allocator as vLLM's load
# sets it up (max_split_size_mb 20: a block past 20 MiB is never split, so each larger temporary keeps a block of its own,
# which only the same size reuses, and a long-lived pack takes such a block whole, up to 20 MiB over its size). An int64
# temporary of PACK_CHUNK weights is 16 MiB: past 10 MiB, where the allocator cuts a block to size (below it each request
# takes a 20 MiB segment: at 1 << 20 weights a load logged 5,312 allocator warnings on a 16 GB card), and under 20 MiB.
# _hist counts int16 as it is, 8 MiB a pass. The packs' bits do not depend on either (benchmarks/gpu/l4-quickstart-2026-09-30:
# the same bits at 1 << 19 to 1 << 25 weights a pass).
HIST_CHUNK = 1 << 22
PACK_CHUNK = 1 << 21


def _chunks(u):
    for a in range(0, u.numel(), CHUNK):
        yield a, u[a : a + CHUNK].to(torch.int32) & 0xFFFF


def _hist(u):
    """The 256 exponent counts of u (bf16's bits, int16), HIST_CHUNK weights a pass, in int16: no widened copy."""
    h = torch.zeros(256, dtype=torch.int64, device=u.device)
    for a in range(0, u.numel(), HIST_CHUNK):
        h += torch.bincount((u[a : a + HIST_CHUNK] >> 7) & 0xFF, minlength=256)
    return h


def _pieces(u):
    core = torch.empty(u.numel(), dtype=torch.uint8, device=u.device)
    for a, v in _chunks(u):
        core[a : a + v.numel()] = (((v >> 8) & 0x80) | (v & 0x7F)).to(torch.uint8)
    return core


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
    lengths, codes, tables = key_tables(_hist(u).cpu().numpy())
    dev = w.device
    len_t = torch.tensor(lengths, dtype=torch.uint8, device=dev)
    key_t = torch.tensor(codes, dtype=torch.int32, device=dev)
    # A matrix whose rows are a multiple of 128 long goes in tiles of
    # whole rows (the fused product needs them), 16 weights a lane a step
    # where the rows allow; anything else flat.
    if w.dim() == 2 and w.shape[1] % 128 == 0:
        # Rows a tile: about ROW_TILE weights, fewer where that would leave
        # the GPU under MIN_TILES warps (a small matrix: its offsets cost
        # more a weight, on few weights).
        O, K = w.shape
        V = 16 if K % 512 == 0 else 4
        if K > ROW_SPAN and K % 512 == 0:
            tw = ROW_SPAN  # long rows: tiles split them, the product adds the parts
        else:
            tw = max(1, min(ROW_SPAN // K, max(MIN_SPAN // K, O // MIN_SPANS))) * K
    else:
        tw, V = FLAT_SPAN, 4
    bits = _ext.pass_a(u, len_t, tw, V).to(torch.int64)
    offs64 = torch.cumsum(bits, 0) - bits
    total = int(offs64[-1] + bits[-1])
    assert total < 2**31, "a tensor too large for this layout (past 2^31 bits packed)"
    stream = torch.zeros(total // 32 + 4, dtype=torch.int32, device=dev)
    offs = offs64.to(torch.int32)
    _ext.pass_b(u, len_t, key_t, offs, stream, tw, V)
    # The most words a tile's streams span, with the reader's look-ahead:
    # what a warp stages in shared memory.
    starts = torch.cat([offs64[::32], torch.tensor([total], device=dev)])
    span_words = int(((starts[1:] + 31) // 32 - starts[:-1] // 32).max()) + 3
    tab = torch.tensor(tables, dtype=torch.int32, device=dev)
    return Packed(tuple(w.shape), n, _pieces(u), stream, offs, tab, tw, V, span_words)


def decode_spans(p, spans, out):
    """Tiles `tiles` of p (all of them: an empty tensor) decoded into out."""
    _ext.decode(p.core, p.stream, p.offs, p.tables, p.n, p.tw, p.V, p.span_words, spans, out.view(torch.int16))


def unpack(p, out=None):
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.core.device)
    decode_spans(p, _none(p.core.device), out)
    return out[: p.n].view(p.shape)


def rows(p, ids):
    """Rows `ids` of a matrix (an embedding's lookup): only their tiles decoded."""
    T, K = p.rows_per_span, p.shape[1]
    ids = ids.flatten()
    spans, where = torch.unique(ids // T, return_inverse=True)
    out = torch.empty(spans.numel() * p.tw, dtype=torch.bfloat16, device=ids.device)
    decode_spans(p, spans, out)
    return out.view(-1, T, K)[where, ids % T]


def gemv(p, x, bias=None):
    """W x (+ bias) for one input vector, the weights decoded in registers."""
    O, K = p.shape
    y = torch.empty(O, dtype=torch.bfloat16, device=x.device)
    _ext.gemv(p.core, p.stream, p.offs, p.tables, O, K, p.tw, p.V, p.span_words, x.contiguous().view(-1), bias if bias is not None else _none(x.device).to(torch.bfloat16), y, p.sum, p.count)
    return y


class Fast:
    """The fast format of a matrix [O, K] (K a multiple of 128): each
    exponent a 3-bit code into the 7 most common, code 7 an escape to the
    exponent itself; decoded by bit operations, one warp a row."""

    def __init__(self, shape, core, planes, exc, exc_base, top):
        self.shape, self.core, self.planes, self.exc, self.exc_base, self.top = shape, core, planes, exc, exc_base, top
        self.n = shape[0] * shape[1]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.core, self.planes, self.exc, self.exc_base)) + 8

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


def pack_fast(w):
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2 and w.shape[1] % 128 == 0
    O, K = w.shape
    u = w.contiguous().view(torch.int16).flatten()
    dev = w.device
    top7 = _hist(u).argsort(descending=True)[:7]
    key_of = torch.full((256,), 7, dtype=torch.int32, device=dev)
    key_of[top7] = torch.arange(7, dtype=torch.int32, device=dev)
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
        c = key_of[e]
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
    return Fast((O, K), _pieces(u), planes, exc, exc_base.to(torch.int32), top)


def fast_unpack(p, out=None):
    O, K = p.shape
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.core.device)
    _ext.fast_decode(p.core, p.planes, p.exc, p.exc_base, p.top, 0, O, _none(p.core.device), K, out.view(torch.int16))
    return out[: p.n].view(O, K)


def fast_rows(p, ids):
    K = p.shape[1]
    ids = ids.flatten().to(torch.int64)
    out = torch.empty(ids.numel() * K, dtype=torch.bfloat16, device=ids.device)
    _ext.fast_decode(p.core, p.planes, p.exc, p.exc_base, p.top, 0, 0, ids, K, out.view(torch.int16))
    return out.view(-1, K)


def fast_gemv(p, x, bias=None):
    O, K = p.shape
    y = torch.empty(O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_gemv(p.core, p.planes, p.exc, p.exc_base, p.top, O, K, x.contiguous().view(-1), bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


def fast_gemm(p, x, bias=None):
    """X W^T (+ bias) for several tokens (x [M, K]), the weights decoded a
    step at a time into shared memory and multiplied on the tensor cores."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_gemm(p.core, p.planes, p.exc, p.exc_base, p.top, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


def fast_bgemv(p, x, bias=None):
    """X W^T (+ bias) for 2, 4, 8 or 16 tokens (x [M, K]) on the CUDA cores:
    the one-token product's decode, each weight multiplied into M sums."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    _ext.fast_bgemv(p.core, p.planes, p.exc, p.exc_base, p.top, O, K, x, bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
    return y


MMA_TAG = 2  # the top byte of tiers[0] of a pack in this layout (a pack of the earlier layout, glyd 0.20 to 0.27, has 0xFF there)
COMPACT_TAG = 3  # the compact tiered layout's: weights of three fraction bits (Candidate A's values), each sign-and-mantissa byte a nibble (glyd_gpu.cu's Compact)
OLD_TAG = 0xFF
FLAG_N1, FLAG_LIST, FLAG_BITS = 512, 32, 992  # what a step's block may hold for the shorter decode (a step's flag: bit 31 of its block_base entry)
COLD_SHARE = (1, 4)  # the share of a matrix's steps that may have the flag: past it the 12-bit layout holds the matrix


class Mma:
    """The mma layout of a matrix [O, K] (O a multiple of 64, K of 16): signs
    and mantissas, and exponents coded in tiers of 2-bit digits (its 10
    commonest in 3 tiers, others as a list of entries in the step's block), in
    the order the tensor cores take their B operand (see glyd_gpu.cu). The
    top byte of tiers[0] is MMA_TAG; a pack of the earlier layout (0xFF) is
    decoded and packed again here."""

    def __init__(self, shape, data, blocks, block_base, tiers):
        if tiers[0] >> 24 == OLD_TAG:
            old = types.SimpleNamespace(shape=shape, data=data, blocks=blocks, block_base=block_base, tiers=tiers, core=data, n=shape[0] * shape[1])
            new = pack_mma(mma_unpack(old))
            shape, data, blocks, block_base, tiers = new.shape, new.data, new.blocks, new.block_base, new.tiers
        elif tiers[0] >> 24 not in (MMA_TAG, COMPACT_TAG):
            raise ValueError(f"an mma pack's words {[hex(t) for t in tiers]}: not a layout this package reads")
        self.shape, self.data, self.blocks, self.block_base, self.tiers = shape, data, blocks, block_base, tiers
        self.core = data  # its device, as the other formats'
        self.n = shape[0] * shape[1]

    def compact(self):
        """The compact tiered layout's (Candidate A's values: lossy, decoded as they were packed)."""
        return self.tiers[0] >> 24 == COMPACT_TAG

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.data, self.blocks, self.block_base)) + 12

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n

    def flagged(self):
        """The steps with the flag (bit 31 of their block_base entry): a count."""
        return int((self.block_base[:-1] < 0).sum())


def _symbols(tiers):
    """The 10 exponents by rank of a pack's tiers: tier 1's three, tier 2's, tier 3's four."""
    return [(tiers[k] >> (8 * j)) & 0xFF for k in range(3) for j in range(3)] + [(tiers[2] >> 24) & 0xFF]


def _mma_arrays(u, tiers, chunk):
    """The mma layout's arrays (data, blocks, block_base) and words of the bits u [O, K] (int16, on any device)."""
    O, K = u.shape
    RB, KS, dev = O // 64, K // 16, u.device
    # Exponents by count: ranks 0-2 tier 1's digits 0-2, 3-5 tier 2's, 6-9 tier 3's; digit 3 of a tier goes on to the next; past the tenth: a literal.
    if tiers is None:
        sym = torch.argsort(_hist(u.flatten()), descending=True, stable=True)[:10].tolist()
        tiers = [sym[0] | sym[1] << 8 | sym[2] << 16 | MMA_TAG << 24, sym[3] | sym[4] << 8 | sym[5] << 16 | 0xFF << 24, sym[6] | sym[7] << 8 | sym[8] << 16 | sym[9] << 24]
    else:
        if tiers[0] >> 24 != MMA_TAG:
            raise ValueError("words of another layout")
        sym = _symbols(tiers)
    rank = torch.full((256,), 10, dtype=torch.int64, device=dev)
    rank[torch.tensor(sym, device=dev)] = torch.arange(10, device=dev)
    steps = O * K // 1024
    data = torch.empty(steps, 1280, dtype=torch.uint8, device=dev)  # a warp step: tier-1 digits [2 words][32 lanes], then [32 lanes][32 bytes] as two halves
    shifts = 2 * torch.arange(16, dtype=torch.int64, device=dev)
    block_parts, sizes, flags = [], [], []
    at = 128  # where a chunk's blocks start in the blocks array (its first 128 bytes are padding)
    per = max(1, chunk // (64 * K))  # row blocks a chunk
    for b0 in range(0, RB, per):
        b1 = min(RB, b0 + per)
        # [rb, n, g, ks, j >> 1, t, j & 1] -> [rb, ks, lane = 4g + t, 4n + j]
        v = u[b0 * 64 : b1 * 64].view(b1 - b0, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten().to(torch.int32) & 0xFFFF
        a, z = b0 * 64 * K // 1024, b1 * 64 * K // 1024
        e = (v >> 7) & 0xFF
        r = rank[e]
        words = (r.clamp(max=3).view(-1, 32, 2, 16) << shifts).sum(-1)  # [steps, lanes, 2]: weight i at bits 2(i mod 16) of word i / 16
        words = (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        data[a:z, :256] = words.transpose(1, 2).contiguous().view(torch.uint8).view(-1, 256)
        # Weight i's byte: its mantissa above the sign of weight i ^ 1 (a pair decodes by one rotate).
        pr = v.view(-1, 2)
        sign = (pr >> 15) & 1
        # As two halves: every lane's first 16 bytes, then every lane's last 16 (each 16-byte load of a warp: 512 contiguous bytes).
        data[a:z, 256:] = (((pr & 0x7F) << 1) | sign.flip(1)).to(torch.uint8).view(-1, 32, 2, 16).transpose(1, 2).reshape(-1, 1024)
        # A step's block: tier-3 digits of its weights of ranks 6-9 (from its start), the list of its literals (ranks 10 on), and, at its end,
        # tier-2 digits of its tier-1 escapes (a literal's: 0), words of 16 back from it.
        r = r.view(z - a, 1024)
        lit = r >= 10
        m1, m2 = r >= 3, (r >= 6) & ~lit
        t1, t2, nl = m1.sum(1), m2.sum(1), lit.sum(1)
        r0 = (2 * t2 + 7) >> 3
        size = r0 + ((18 * nl + 7) >> 3) + 4 * ((t1 + 15) >> 4)
        start = torch.cumsum(size, 0) - size
        o = 8 * ((at + start) & 3)  # the bit offset of the block's start in its word
        flags.append((t1 > FLAG_N1) | (nl > FLAG_LIST) | (o + 2 * t2 + 32 >= FLAG_BITS) | ((nl > 0) & (o + 8 * r0 + 18 * (nl - 1) >= FLAG_BITS)))
        buf = torch.zeros(int(size.sum()) if z > a else 0, dtype=torch.int32, device=dev)
        k = torch.cumsum(m1, 1) - m1.to(torch.int64)  # an escape's place among its step's
        d2 = torch.where(lit, 0, (r - 3).clamp(max=3))
        pos = 8 * (start + size).unsqueeze(1) - 32 * ((k >> 4) + 1) + 2 * (k & 15)
        buf.index_add_(0, (pos >> 3)[m1], (d2 << (pos & 7))[m1].to(torch.int32))
        j = torch.cumsum(m2, 1) - m2.to(torch.int64)
        pos = 8 * start.unsqueeze(1) + 2 * j
        buf.index_add_(0, (pos >> 3)[m2], ((r - 6) << (pos & 7))[m2].to(torch.int32))
        j = torch.cumsum(lit, 1) - lit.to(torch.int64)  # a literal's place in its step's list; its entry (its escape's place, its exponent) 18 bits from a byte's start
        pos = 8 * (start + r0).unsqueeze(1) + 18 * j
        entry = (k | e.view(z - a, 1024).to(torch.int64) << 10) << (pos & 7)
        for i in range(4):
            part = (entry >> (8 * i)) & 0xFF
            used = lit & (part != 0)
            buf.index_add_(0, ((pos >> 3) + i)[used], part[used].to(torch.int32))
        block_parts.append(buf.to(torch.uint8))
        sizes.append(size)
        at += int(size.sum())
    # Padded: a step's first and last 128 bytes are read whole, and fields up to 8 bytes past.
    blocks = torch.cat([torch.zeros(128, dtype=torch.uint8, device=dev)] + block_parts + [torch.zeros(256, dtype=torch.uint8, device=dev)])
    assert blocks.numel() < 2**31
    base = torch.cumsum(torch.cat([torch.full((1,), 128, dtype=torch.int64, device=dev)] + sizes), 0)  # steps + 1: each step's block start, the last's end
    base[:-1] += torch.cat(flags).to(torch.int64) << 31
    return data.flatten(), blocks, torch.where(base >= 2**31, base - 2**32, base).to(torch.int32), tiers


def pack_mma(w, tiers=None, chunk=PACK_CHUNK):
    """w [O, K] in the mma layout; tiers: another pack's (its exponents by
    count), else this matrix's own. Packed about `chunk` weights at a time
    (its scratch: some 100 bytes a weight). A step's flag (bit 31 of its
    block_base entry) is set where its block is more than the shorter decode
    takes: more than 512 escapes or 32 literals, or a field past the first
    124 bytes of the word-aligned window its block starts in."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0
    return Mma(tuple(w.shape), *_mma_arrays(w.contiguous().view(torch.int16), tiers, chunk))


def pack_smallest(w):
    """w [O, K] in the smaller of its layouts: the mma layout, unless the 12-bit one takes fewer bytes or more than a
    quarter of the mma pack's steps have the flag (a matrix whose exponents are spread wide)."""
    p = pack_mma(w)
    O, K = w.shape
    c = F.pad(_hist(w.contiguous().view(torch.int16).flatten()).view(128, 2).sum(1).cumsum(0), (1, 0))  # the high bytes' 7 bits (exponent >> 1), counted
    n = O * K - int((c[8:] - c[:-8]).max())  # the weights outside the 8 values of the best base: the 12-bit layout's exceptions
    steps = O * K // 1024
    if p.nbytes() <= 1536 * steps + 4 * (n + 4 - n % 4) + 4 * (steps + 1) + 16 and p.flagged() * COLD_SHARE[1] <= steps * COLD_SHARE[0]:
        return p
    return pack_mma12(w)


class Mma12(Mma):
    """The 12-bit mma layout of a matrix [O, K], split byte: the same order, a
    weight's low byte (the exponent's lowest bit and the mantissa) as it is
    and its high byte (sign and exponent >> 1) a 4-bit code, the sign and an
    offset 0-7 from the matrix's base hb (0-120: the 8 values of exponent >> 1
    from hb, 16 exponents, holding the most weights); any other weight (zeros
    and specials among them) in its step's exception list. A lighter decode
    than the tiered code's, at 12 bits a weight (see glyd_gpu.cu, Nib). sym:
    the C API's four words, hb in each byte of the first."""

    def __init__(self, shape, data, exc, exc_base, hb):
        if type(hb) is not int or not 0 <= hb <= 120:
            raise ValueError(f"an mma12 pack's hb {hb!r}: an int 0-120")
        self.shape, self.data, self.exc, self.exc_base, self.hb = shape, data, exc, exc_base, hb
        self.sym = [hb * 0x01010101, 0, 0, 0]
        self.core = data
        self.n = shape[0] * shape[1]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.data, self.exc, self.exc_base)) + 16


def pack_mma12(w):
    """w [O, K] (O a multiple of 64, K of 16) in the 12-bit layout. A warp
    step's 1536 bytes: lane l's 16 bytes of codes (words q = 0-3, weights 8q
    to 8q + 7: n-tile 2q's sign and offset in bits {7, 2, 1, 0} of byte j,
    n-tile 2q + 1's rotated by 4: its sign in bit 3 of byte j, its offset in
    bits 4-6 of byte j - 1 mod 4), then the low bytes, every lane's first 16
    and then every lane's last 16. An exception's code has offset 0, its entry
    its weight (lane * 32 + i) and the byte to XOR into its high byte, hb ^
    (exponent >> 1), << 16."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0
    O, K = w.shape
    RB, KS, dev = O // 64, K // 16, w.device
    u = w.contiguous().view(torch.int16)
    c = F.pad(_hist(u.flatten()).view(128, 2).sum(1).cumsum(0), (1, 0))  # the high bytes' 7 bits (exponent >> 1), counted
    hb = int((c[8:] - c[:-8]).argmax())  # the first window of 8 holding the most
    steps = O * K // 1024
    data = torch.empty(steps, 1536, dtype=torch.uint8, device=dev)  # a warp step: [32 lanes][16 bytes] of codes, then [32 lanes][32 bytes] as two halves
    j = torch.arange(4, dtype=torch.int64, device=dev)
    exc_parts, counts = [], []
    per = max(1, PACK_CHUNK // (64 * K))  # row blocks a chunk
    for b0 in range(0, RB, per):
        b1 = min(RB, b0 + per)
        # [rb, n, g, ks, j >> 1, t, j & 1] -> [rb, ks, lane = 4g + t, i = 4n + j]
        v = u[b0 * 64 : b1 * 64].view(b1 - b0, 8, 8, KS, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).flatten().to(torch.int32) & 0xFFFF
        a, z = b0 * 64 * K // 1024, b1 * 64 * K // 1024
        hi = (v >> 8) & 0x7F
        off = hi - hb
        esc = (off < 0) | (off > 7)
        s4 = ((v >> 15) & 1).view(-1, 32, 4, 2, 4).to(torch.int64)  # [steps, lanes, word q, n-tile 2q + h, j]
        o4 = torch.where(esc, 0, off).view(-1, 32, 4, 2, 4).to(torch.int64)
        words = ((s4[..., 0, :] << 7 | o4[..., 0, :]) << (8 * j)).sum(-1) | (s4[..., 1, :] << (8 * j + 3)).sum(-1) | (o4[..., 1, :] << (8 * ((j - 1) % 4) + 4)).sum(-1)
        words = (words - (words >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
        data[a:z, :512] = words.contiguous().view(torch.uint8).view(-1, 512)
        data[a:z, 512:] = (v & 0xFF).to(torch.uint8).view(-1, 32, 2, 16).transpose(1, 2).reshape(-1, 1024)
        m = esc.view(z - a, 1024)
        idx = torch.arange(1024, device=dev).expand(z - a, 1024)[m]  # lane * 32 + i
        exc_parts.append((idx | (hb ^ hi.view(z - a, 1024)[m]).to(torch.int64) << 16).to(torch.int32))
        counts.append(m.sum(1))
    n = torch.cat(counts)
    pad = 4 - int(n.sum()) % 4  # at least one entry, to a multiple of 4 (mma_gemm_wg copies runs widened to 16 bytes)
    exc = torch.cat(exc_parts + [torch.zeros(pad, dtype=torch.int32, device=dev)])
    exc_base = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(n, 0)]).to(torch.int32)
    return Mma12((O, K), data.flatten(), exc, exc_base, hb)


def best_layout(linear_bytes, other_bytes=0, gpus=1, device=0, moe=False):
    """The layout for this GPU, and why: "mma" (tiered, 10.80 bits a weight)
    or "mma12" (12.04 bits, the lighter decode), for Linears of
    linear_bytes in bf16 and other_bytes besides (moe: a mixture of
    experts' among them). Measured (benchmarks/gpu, 2026-09-26): on Ada (RTX
    40, L40S) the tiered decode is the faster one at 1-32 sequences; on an
    A10, an A100 and an H100 the 12-bit one. A mixture of experts
    (2026-09-27), whose steps read a few experts' matrices each: on an A10
    the tiered layout as fast as the 12-bit one (OLMoE-1B-7B and
    granite-3.1-3b-a800m, 1-6% less GPU time a step at 1 and 8 sequences)
    and 11% smaller, so there as on Ada the tiered one (granite on an RTX
    4080 SUPER: 6-7% less GPU time a step, 10% smaller; its prompts 4-11%
    slower); on an H100 PCIe the 12-bit one (Qwen3-30B-A3B, 26.2 and 191.0
    tokens/s at 1 and 8 sequences against the tiered layout's 15.8 and
    111.8), and so on an A100 (not measured with one); Blackwell as for
    dense Linears until measured. Either way the tiered layout when only
    it fits (2 GiB a GPU kept for activations and the KV cache)."""
    p = torch.cuda.get_device_properties(device)
    room = gpus * (p.total_memory - 2 * 2**30)
    banded, twelve = linear_bytes * 10.80 / 16 + other_bytes, linear_bytes * 12.04 / 16 + other_bytes
    if twelve > room >= banded:
        return "mma", f"only the mma layout fits (the smallest: {banded / 1e9:.1f} GB; mma12 {twelve / 1e9:.1f} GB, room {room / 1e9:.1f} GB)"
    if (p.major, p.minor) == (8, 9):
        return "mma", "Ada: mma, the smallest layout, is as fast as mma12 (measured)"
    if moe and (p.major, p.minor) == (8, 6):
        return "mma", "a mixture of experts on an A10: mma, the smallest layout, is as fast as mma12 (measured)"
    return "mma12", "mma12, the faster layout, on this GPU (measured)"


def mma_cat(a, b):
    """Two packs with the same tiers and columns as one: a's rows, then b's."""
    assert a.tiers == b.tiers and a.shape[1] == b.shape[1]
    body = int(a.block_base[-1]) - 128  # a's blocks without their padding
    blocks = torch.cat([a.blocks[: 128 + body], b.blocks[128:]])
    return Mma((a.shape[0] + b.shape[0], a.shape[1]), torch.cat([a.data, b.data]), blocks, torch.cat([a.block_base[:-1], b.block_base + body]), a.tiers)


def mma_unpack(p, out=None, row0=0, rows=None, crews=0):
    """Rows [row0, row0 + rows) of W (multiples of 64), bf16 [rows, K]. warps: a
    warp a step (0), or that many in all, each taking every so many steps (a
    decode beside a product running on another stream)."""
    O, K = p.shape
    rows = O - row0 if rows is None else rows
    if out is None:
        out = torch.empty(rows * K, dtype=torch.bfloat16, device=p.core.device)
    if isinstance(p, Mma12):
        _ext.mma12_unpack(p.data, p.exc, p.exc_base, p.sym, K, row0, rows, out.view(torch.int16), crews)
    else:
        _ext.mma_unpack(p.data, p.blocks, p.block_base, p.tiers, K, row0, rows, out.view(torch.int16), crews)
    return out[: rows * K].view(rows, K)


def mma_rows(p, ids, out=None):
    """Rows ids (int64, on the pack's GPU) of W, bf16 [len(ids), K]: an embedding's lookup, each row's step decoded and only
    that row written, one launch for any ids (an id outside [0, rows): a row of zeros)."""
    O, K = p.shape
    if out is None:
        out = torch.empty(ids.numel(), K, dtype=torch.bfloat16, device=ids.device)
    if isinstance(p, Mma12):
        _ext.mma12_rows(p.data, p.exc, p.exc_base, p.sym, K, O, ids, out)
    else:
        _ext.mma_rows(p.data, p.blocks, p.block_base, p.tiers, K, O, ids, out)
    return out


def hold(ns):
    """The current stream held ns nanoseconds (a kernel of one thread): a
    decode ahead launched after it, beside a product launched as it is,
    starts once the product has placed its blocks."""
    _ext.hold(ns)


# The library's routes (glyd_gpu.h's GLYD_GPU_ROUTE_*): how a product for M tokens is taken on a GPU. DECODE: the
# matrix decoded, then cuBLAS; AHEAD: so, decoded ahead beside the products before it (model.Ahead).
DECODE, GEMM, X2, X3, X1, EARLY, PART, GEMM_CC12 = range(8)  # (GEMM_CC12: opt-in, never a GPU's own; GEMM on a GPU other than compute capability 12)
GEFORCE, A10, L4, L40S, PCIE, GH200, H100 = 1000, 2000, 3000, 4000, 5000, 6000, 7000  # a GPU's classes by name in its code (glyd_gpu.h's GLYD_GPU_GEFORCE, _A10, _L4, _L40S, _PCIE, _GH200, _H100)
WITH_PART = 1 << 20  # in a GPU's code: its routes with SPLIT (glyd_gpu.h's GLYD_GPU_WITH_SPLIT: opt-in, asked by GLinear, which runs the ring)


def way(p, gpu, M):
    """(route, last): the library's route for M tokens of pack p (the mma layouts) on gpu (its code, model.gpu_code:
    compute capability and class; plus WITH_SPLIT for the route SPLIT, opt-in), and the last token count from M on that
    takes it."""
    O, K = p.shape
    return (_ext.mma12_way if isinstance(p, Mma12) else _ext.mma_way)(gpu, O, K, M)


def part_cus(p, gpu, M):
    """The route SPLIT's SMs for the decode for M tokens of pack p on gpu (its code with WITH_SPLIT; 0: another route,
    or no flag; the 12-bit layout's alone)."""
    O, K = p.shape
    return _ext.mma12_part_cus(gpu, O, K, M) if isinstance(p, Mma12) else 0


def mma_unpack_part(p, cus, out=None, row0=0, rows=None):
    """Rows [row0, row0 + rows) of a 12-bit W by the route SPLIT's decode, a grid for sms SMs: bf16 [rows, K]."""
    O, K = p.shape
    rows = O - row0 if rows is None else rows
    if out is None:
        out = torch.empty(rows * K, dtype=torch.bfloat16, device=p.core.device)
    _ext.mma12_unpack_part(p.data, p.exc, p.exc_base, p.sym, K, row0, rows, out.view(torch.int16), cus)
    return out[: rows * K].view(rows, K)


def mma_linear(p, x, bias=None, way=-1):
    """X W^T (+ bias) by a route (-1: this GPU's for M), as a Linear's one-call path takes it: its kernel; DECODE and
    AHEAD the prompt kernel, on every GPU (refused where K is not a multiple of 64: decode W there)."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    b = bias if bias is not None else _none(x.device).to(torch.bfloat16)
    if isinstance(p, Mma12):
        _ext.mma12_linear(p.data, p.exc, p.exc_base, p.sym, O, K, x, b, y, way)
    else:
        _ext.mma_linear(p.data, p.blocks, p.block_base, p.tiers, O, K, x, b, y, way)
    return y


def mma_gemm(p, x, bias=None):
    """X W^T (+ bias) for up to 64 tokens (x [M, K]) on the tensor cores,
    the weights decoded in registers straight into their operands."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    b = bias if bias is not None else _none(x.device).to(torch.bfloat16)
    if isinstance(p, Mma12):
        _ext.mma12_gemm(p.data, p.exc, p.exc_base, p.sym, O, K, x, b, y)
    else:
        _ext.mma_gemm(p.data, p.blocks, p.block_base, p.tiers, O, K, x, b, y)
    return y


def mma_gemm_x3(p, x, bias=None):
    """X W^T (+ bias) for many tokens on Hopper (x [M, K], K a multiple of
    64): the TMA copies the compressed weights and X's tiles into shared
    memory, consumer warpgroups decode the weights into wgmma's operands
    and multiply. The tiered layout's to 64 tokens (its route WG)."""
    if not isinstance(p, Mma12):
        return mma_linear(p, x, bias, way=X3)
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    b = bias if bias is not None else _none(x.device).to(torch.bfloat16)
    _ext.mma12_gemm_x3(p.data, p.exc, p.exc_base, p.sym, O, K, x, b, y)
    return y


def mma_gemm_x2(p, x, bias=None):
    """X W^T (+ bias) for many tokens on Ampere and Ada (x [M, K], K a
    multiple of 64; the 12-bit layout): a producer warp copies the
    compressed weights and X's tiles into shared memory with cp.async,
    consumer warps decode the weights into mma.sync's registers and
    multiply (mma_gemm_wg's plan, this generation's instructions). On an
    A100, producer warps decode the weights into shared memory for the
    consumers' products (mma_gemm_big's split, its reads staged)."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    b = bias if bias is not None else _none(x.device).to(torch.bfloat16)
    assert isinstance(p, Mma12), "this product takes the mma12 layout"
    _ext.mma12_gemm_x2(p.data, p.exc, p.exc_base, p.sym, O, K, x, b, y)
    return y


def mma_gemm_x1(p, x, bias=None, variant=0):
    """X W^T (+ bias) for many tokens (x [M, K]; a prompt; K a multiple of
    64): a tiled GEMM, each weight decoded once for 128 or 256 tokens
    (variant 1 or 2; 3, the 12-bit layout: 256 tokens by 128 rows, an
    A100's past 128 tokens but where blocks of 128 fit the prompt; 0: by M
    and the GPU) into shared memory by warps of its own while others
    multiply on the tensor cores."""
    O, K = p.shape
    x = x.contiguous()
    y = torch.empty(x.shape[0], O, dtype=torch.bfloat16, device=x.device)
    b = bias if bias is not None else _none(x.device).to(torch.bfloat16)
    if isinstance(p, Mma12):
        _ext.mma12_gemm_x1(p.data, p.exc, p.exc_base, p.sym, O, K, x, b, y, variant)
    else:
        _ext.mma_gemm_x1(p.data, p.blocks, p.block_base, p.tiers, O, K, x, b, y, variant)
    return y



def moe_way(ids, E):
    """A mixture-of-experts layer's pairs (ids [T, k], int64: each token's k experts) sorted by expert on the GPU:
    the plan mma_moe takes."""
    ids = ids.reshape(-1)
    plan = torch.empty(2 + 2 * E + ids.numel(), dtype=torch.int32, device=ids.device)
    _ext.moe_way(ids, E, plan)
    return plan


def mma_moe(p, E, x, plan, ids, act=0, bias=None, weights=None, gather=True):
    """A mixture-of-experts layer's product, p its E experts' matrices [O, K]
    stacked ([E O, K]), for the pairs of plan (moe_route(ids, E); ids [T, k]):
    X's rows (gather: the tokens', [T, K]; else the pairs' in the plan's order,
    [T k, K]) by each pair's expert. act 0: [T k, O], the pairs in the plan's
    order (+ bias [E, O]); act 1 (SiLU) or 2 (GELU, tanh): [T k, O / 2], act(gate)
    up, the gate an expert's first O / 2 rows; weights [T, k] (act 0): [T, O],
    each token's k rows times their weights, added."""
    O, K = p.shape[0] // E, p.shape[1]
    T, k = ids.shape
    x = x.contiguous()
    y = torch.empty(T if weights is not None else T * k, O // 2 if act else O, dtype=torch.bfloat16, device=x.device)
    none = _none(x.device)
    w, i = (weights.reshape(-1).contiguous(), ids.reshape(-1)) if weights is not None else (none, none)
    b = bias if bias is not None else none
    if isinstance(p, Mma12):
        _ext.mma12_moe(p.data, p.exc, p.exc_base, p.sym, E, O, K, x, k, int(gather), plan, act, b, w, i, y)
    else:
        _ext.mma_moe(p.data, p.blocks, p.block_base, p.tiers, E, O, K, x, k, int(gather), plan, act, b, w, i, y)
    return y


def mma_moe_unpack(p, E, plan, P, out):
    """Exact, a mixture-of-experts layer (p its E experts' matrices [O, K]
    stacked): the experts the plan of P pairs hits decoded into their rows of
    out ([E O, K] bf16), the rest of out left as it is. out, viewed [E O, K]."""
    O, K = p.shape[0] // E, p.shape[1]
    if isinstance(p, Mma12):
        _ext.mma12_moe_unpack(p.data, p.exc, p.exc_base, p.sym, E, O, K, P, plan, out.view(torch.int16))
    else:
        _ext.mma_moe_unpack(p.data, p.blocks, p.block_base, p.tiers, E, O, K, P, plan, out.view(torch.int16))
    return out[: E * O * K].view(E * O, K)


# ---- the lossless FP8 (e4m3) weights: glyd_gpu.cu's f8 kernels; fp8ref.py makes the same bytes in numpy, its docstring is the layout ----
F8_BYTES = (torch.uint8, getattr(torch, "float8_e4m3fn", torch.uint8))  # what a matrix of e4m3 bytes may be held as
F8_E4M3 = F8_BYTES[1]


class F8Packed:
    """A matrix [O, K] of e4m3 bytes in the lossless FP8 layout, all of it held in core [patches, 6144] (uint8), heads [patches, 8] and var (int32, as the library's
    uint32 words), on one GPU."""

    def __init__(self, shape, core, heads, var):
        self.shape, self.core, self.heads, self.var = tuple(shape), core, heads, var
        self.n = shape[0] * shape[1]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.core, self.heads, self.var)) + 8

    def bits_per_value(self):
        return self.nbytes() * 8 / self.n

    def plain(self):
        """The patches held as they are (bit 31 of their head's last word): a count."""
        return int((self.heads[:, 3] < 0).sum())


def _f8_rows(w, what):
    """(rows, columns, the distance of the rows in bytes) of a CUDA matrix of e4m3 bytes whose rows are contiguous."""
    _lib._check(w.dim() == 2 and w.dtype in F8_BYTES and w.is_cuda and w.stride(1) == 1 and w.size(0) > 0 and w.size(1) > 0, f"{what}: a non-empty CUDA matrix of uint8 or float8_e4m3fn with rows contiguous")
    O, K = w.shape
    ld = w.stride(0) if O > 1 else K
    _lib._check(ld >= K, f"{what}: rows that overlap")
    return O, K, ld


def f8_pack(w):
    """w [O, K] (a CUDA matrix of e4m3 bytes, uint8 or float8_e4m3fn, rows of any distance) in the lossless FP8 layout: the bytes fp8ref.pack gives, for any bytes."""
    O, K, ld = _f8_rows(w, "f8_pack")
    patches = -(-O // 64) * -(-K // 128)
    heads = torch.empty(patches * 8, dtype=torch.int32, device=w.device)
    words = _ext.f8_size(w, ld, O, K, heads)
    core = torch.empty(patches * 6144, dtype=torch.uint8, device=w.device)
    var = torch.empty(words, dtype=torch.int32, device=w.device)
    _ext.f8_pack(w, ld, O, K, heads, core, var)
    return F8Packed((O, K), core.view(patches, 6144), heads.view(patches, 8), var)


def f8_unpack_into(p, out):
    """p's matrix, bit for bit, into out [O, K] (CUDA, uint8 or float8_e4m3fn, rows of any distance, each contiguous): one launch on the current stream, nothing allocated or
    waited for, so that a CUDA graph can hold it."""
    O, K = p.shape
    _lib._check(out.dim() == 2 and out.shape[0] == O and out.shape[1] == K and out.dtype in F8_BYTES and out.stride(1) == 1 and out.stride(0) >= K and out.get_device() == p.core.get_device(), "f8_unpack: out [O, K] of e4m3 bytes on the pack's GPU")
    _ext.f8_unpack(p.core, p.heads, p.var, O, K, out, out.stride(0))


def f8_unpack(p, out=None):
    """p's matrix as float8_e4m3fn [O, K] (into out where given: f8_unpack_into)."""
    if out is None:
        out = torch.empty(p.shape, dtype=F8_E4M3, device=p.core.device)
    f8_unpack_into(p, out)
    return out


def f8_unpack_experts(p, E, ids, out):
    """The matrices ids (int32 on the GPU, an id outside [0, E) skipped) of a stack of E matrices [O, K], p the pack of the matrix [E O, K] (O a multiple of 64), into out [len(ids), O, K]
    (CUDA, e4m3 bytes, rows of any distance, each matrix O rows after the one before): one launch for any ids, nothing allocated or waited for."""
    O, K = p.shape[0] // E, p.shape[1]
    _lib._check(p.shape[0] == E * O and O % 64 == 0 and ids.dtype == torch.int32 and ids.is_contiguous() and ids.get_device() == p.core.get_device(), "f8_unpack_experts: ids int32 on the pack's GPU, E matrices of a multiple of 64 rows")
    _lib._check(out.dim() == 3 and out.size(0) >= ids.numel() and out.size(1) == O and out.size(2) == K and out.dtype in F8_BYTES and out.stride(2) == 1 and out.stride(0) == O * out.stride(1) and out.stride(1) >= K
                and out.get_device() == p.core.get_device(), "f8_unpack_experts: out [len(ids), O, K] of e4m3 bytes on the pack's GPU, the matrices one after the other")
    _ext.f8_unpack_experts(p.core, p.heads, p.var, E, O, K, ids, out, out.stride(1))
    return out


F8_SCALES = {"block": 0, "tensor": 1, "mx": 2, "ds32": 3, "channel": 4}  # a checkpoint's FP8 scales: fp32 a 128 x 128 block, one fp32, e8m0 a 1 x 32 block, e8m0 a 32 x 32 block, fp32 a row

# The most tokens a step at which f8_linear (from the pack, the weight never written out) was measured faster than the decode into a scratch followed by
# vLLM 0.30's own FP8 kernel (both in one CUDA graph; per-shape sums of every 2026 FP8 checkpoint and tensor-parallel size run, at 1, 8, 32 and 64 tokens),
# by GPU (torch.cuda.get_device_name) and kind of scales; a GPU or kind not here is not measured (the L40S's) or not entered (the RTX 5090's block scales:
# faster than the route at 1 and 8 tokens, but above stock vLLM's own time in every per-shape sum; its tensor scale: no served run confirms vLLM's kernel
# there). Opt-in data for a caller's fused path, not a default: both are slower than stock vLLM's kernel (benchmarks/gpu/fp8-speed-consumer-2026-10-06).
F8_LINEAR_TOKENS = {"NVIDIA GeForce RTX 4090": {"block": 1, "tensor": 8, "mx": 1, "ds32": 1, "channel": 8}, "NVIDIA GeForce RTX 5090": {"channel": 8}}



def f8_linear(p, kind, scale, x, out=None):
    """out [M, N] = x [M, K] W^T for M of 1 to 64 (bf16, rows contiguous), W [N, K] the matrix p holds (never written out) times its scales: kind one of F8_SCALES,
    scale as the checkpoint holds it (block: float32 [ceil(N / 128), ceil(K / 128)]; tensor: float32, one; mx: uint8 [N, ceil(K / 32)]; ds32: uint8
    [ceil(N / 32), ceil(K / 32)]; rows of any distance; channel: float32 [N] or [N, 1], contiguous). Nothing allocated or waited for once its workspace is
    made, so a CUDA graph can hold it."""
    N, K = p.shape
    code = F8_SCALES.get(kind)
    _lib._check(code is not None, f"f8_linear: kind one of {', '.join(F8_SCALES)}")
    if code == 4:
        _lib._check(scale.is_cuda and scale.get_device() == p.core.get_device() and scale.dtype == torch.float32 and scale.numel() == N and scale.is_contiguous(),
                    "f8_linear: scale float32 [N] (a row's), contiguous, on the pack's GPU")
        scale = scale.reshape(N)
    want = (torch.float32, (-(-N // 128), -(-K // 128))) if code == 0 else (torch.float32, None) if code == 1 else (torch.uint8, (N, -(-K // 32))) if code == 2 else (torch.uint8, (-(-N // 32), -(-K // 32))) if code == 3 else (torch.float32, (N,))
    _lib._check(scale.is_cuda and scale.get_device() == p.core.get_device() and scale.dtype == want[0] and (want[1] is None and scale.numel() == 1 or want[1] is not None and scale.dim() == len(want[1])
                and tuple(scale.shape) == want[1] and scale.stride(-1) == 1), f"f8_linear: scale {want[0]} {want[1] or 'of one value'} on the pack's GPU")
    _lib._check(x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.size(1) == K and 1 <= x.size(0) <= 64 and x.stride(1) == 1 and x.get_device() == p.core.get_device(), "f8_linear: x bf16 [M, K], M of 1 to 64, rows contiguous")
    if out is None:
        out = torch.empty(x.size(0), N, dtype=torch.bfloat16, device=x.device)
    _lib._check(out.dtype == torch.bfloat16 and out.dim() == 2 and out.size(0) == x.size(0) and out.size(1) == N and out.stride(1) == 1, "f8_linear: out bf16 [M, N], rows contiguous")
    _ext.f8_linear(p.core, p.heads, p.var, N, K, code, scale, scale.stride(0) if scale.dim() == 2 else 1, x, out)
    return out
