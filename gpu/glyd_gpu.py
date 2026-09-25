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
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)

FLAT_TILE = 16384  # weights a tile when the tensor is not a matrix of rows
ROW_TILE = 8192  # about as many a tile for a matrix: whole rows
MAX_LEN = 12  # the longest code; the decode table has 2^MAX_LEN entries
SEG = 1024  # the fast format's escape index is kept a segment of this many weights of a row


def limited_lengths(freq, limit):
    """Optimal code lengths no longer than `limit` (package-merge)."""
    syms = [s for s in range(len(freq)) if freq[s] > 0]
    lengths = np.zeros(len(freq), dtype=np.int64)
    if len(syms) == 1:
        lengths[syms[0]] = 1
        return lengths
    leaves = sorted(((int(freq[s]), np.eye(1, len(freq), s, dtype=np.int64)[0]) for s in syms), key=lambda x: x[0])
    current = leaves
    for _ in range(limit - 1):
        packages = [(current[i][0] + current[i + 1][0], current[i][1] + current[i + 1][1]) for i in range(0, len(current) - 1, 2)]
        current = sorted(leaves + packages, key=lambda x: x[0])
    for _, c in current[: 2 * len(syms) - 2]:
        lengths += c
    return lengths


def canonical_codes(lengths):
    """Canonical codes, bit-reversed for an LSB-first stream."""
    codes = np.zeros(len(lengths), dtype=np.int64)
    code = 0
    prev = 0
    for l, s in sorted((int(l), s) for s, l in enumerate(lengths) if l > 0):
        code <<= l - prev
        prev = l
        codes[s] = int(format(code, f"0{l}b")[::-1], 2)
        code += 1
    return codes


def decode_table(lengths, codes, L):
    table = np.zeros(1 << L, dtype=np.int64)
    for s, l in enumerate(lengths):
        if l == 0:
            continue
        step = 1 << l
        table[codes[s] :: step] = (l << 8) | s
    return table


class Packed:
    def __init__(self, shape, n, sm, stream, offs, lut, L, tw, tile_words):
        self.shape, self.n, self.sm, self.stream, self.offs, self.lut, self.L, self.tw = shape, n, sm, stream, offs, lut, L, tw
        self.tile_words = tile_words
        self.rows_per_tile = tw // shape[1] if len(shape) == 2 and tw % shape[1] == 0 else 0

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.sm, self.stream, self.offs, self.lut))

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


def pack(w):
    assert w.dtype == torch.bfloat16 and w.is_cuda
    u = w.contiguous().view(torch.int16).flatten()
    n = u.numel()
    hist = _hist(u).cpu().numpy()
    lengths = limited_lengths(hist, MAX_LEN)
    codes = canonical_codes(lengths)
    dev = w.device
    len_t = torch.tensor(lengths, dtype=torch.uint8, device=dev)
    code_t = torch.tensor(codes, dtype=torch.int16, device=dev)
    sm = _sign_mantissa(u)
    # A matrix whose rows are a multiple of 128 long goes in tiles of
    # whole rows (the fused product needs them); anything else flat.
    if w.dim() == 2 and w.shape[1] % 128 == 0:
        tw = max(1, ROW_TILE // w.shape[1]) * w.shape[1]
    else:
        tw = FLAT_TILE
    bits = _ext.lane_bits(u, len_t, tw).to(torch.int64)
    offs64 = torch.cumsum(bits, 0) - bits
    total = int(offs64[-1] + bits[-1])
    assert total < 2**31, "a tensor of more than 2^31 code bits"
    offs = offs64.to(torch.int32)
    # The most words a tile's streams span (from its first lane's word),
    # with the reader's look-ahead: the shared memory a warp stages.
    starts = torch.cat([offs64[::32], torch.tensor([total], device=dev)])
    tile_words = int(((starts[1:] + 31) // 32 - starts[:-1] // 32).max()) + 3
    stream = torch.zeros(total // 32 + 4, dtype=torch.int32, device=dev)
    _ext.write_codes(u, len_t, code_t, offs, stream, tw)
    lut = torch.tensor(decode_table(lengths, codes, MAX_LEN), dtype=torch.int16, device=dev)
    return Packed(tuple(w.shape), n, sm, stream, offs, lut, MAX_LEN, tw, tile_words)


_NONE = {}


def _none(dev):
    if dev not in _NONE:
        _NONE[dev] = torch.empty(0, dtype=torch.int64, device=dev)
    return _NONE[dev]


def unpack(p, out=None):
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.sm.device)
    _ext.decode(p.sm, p.stream, p.offs, p.lut, p.L, p.n, p.tw, _none(p.sm.device), out.view(torch.int16))
    return out[: p.n].view(p.shape)


def rows(p, ids):
    """Rows `ids` of a matrix (an embedding's lookup): only their tiles decoded."""
    T, K = p.rows_per_tile, p.shape[1]
    ids = ids.flatten()
    tiles, where = torch.unique(ids // T, return_inverse=True)
    out = torch.empty(tiles.numel() * p.tw, dtype=torch.bfloat16, device=ids.device)
    _ext.decode(p.sm, p.stream, p.offs, p.lut, p.L, p.n, p.tw, tiles, out.view(torch.int16))
    return out.view(-1, T, K)[where, ids % T]


def gemv(p, x, bias=None):
    """W x (+ bias) for one input vector, the weights decoded in registers."""
    O, K = p.shape
    y = torch.empty(O, dtype=torch.bfloat16, device=x.device)
    _ext.gemv(p.sm, p.stream, p.offs, p.lut, p.L, O, K, p.tw, p.tile_words, x.contiguous().view(-1), bias if bias is not None else _none(x.device).to(torch.bfloat16), y)
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
