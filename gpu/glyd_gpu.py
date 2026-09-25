"""Glyd weights on the GPU: bf16 tensors held compressed in VRAM and
decoded on the GPU, bit for bit (glyd_gpu.cu has the layout).

    p = pack(w)          # w: a bf16 CUDA tensor
    w2 = unpack(p)       # the same bits
    p.bits_per_weight()
"""
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

_ext = load(
    name="glyd_gpu",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "glyd_gpu.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)

LANE = 512
TILE = 32 * LANE
MAX_LEN = 12  # the longest code; the decode table has 2^MAX_LEN entries


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
    def __init__(self, shape, n, sm, stream, offs, lut, L):
        self.shape, self.n, self.sm, self.stream, self.offs, self.lut, self.L = shape, n, sm, stream, offs, lut, L

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.sm, self.stream, self.offs, self.lut))

    def bits_per_weight(self):
        return self.nbytes() * 8 / self.n


def pack(w):
    assert w.dtype == torch.bfloat16 and w.is_cuda
    u = w.contiguous().view(torch.int16).flatten()
    n = u.numel()
    v = u.to(torch.int32) & 0xFFFF
    hist = torch.bincount((v >> 7) & 0xFF, minlength=256).cpu().numpy()
    lengths = limited_lengths(hist, MAX_LEN)
    codes = canonical_codes(lengths)
    dev = w.device
    len_t = torch.tensor(lengths, dtype=torch.uint8, device=dev)
    code_t = torch.tensor(codes, dtype=torch.int16, device=dev)
    sm = (((v >> 8) & 0x80) | (v & 0x7F)).to(torch.uint8)
    bits = _ext.lane_bits(u, len_t).to(torch.int64)
    offs = torch.cumsum(bits, 0) - bits
    total = int(offs[-1] + bits[-1])
    stream = torch.zeros(total // 32 + 4, dtype=torch.int32, device=dev)
    _ext.write_codes(u, len_t, code_t, offs, stream)
    lut = torch.tensor(decode_table(lengths, codes, MAX_LEN), dtype=torch.int16, device=dev)
    return Packed(tuple(w.shape), n, sm, stream, offs, lut, MAX_LEN)


def unpack(p, out=None):
    if out is None:
        out = torch.empty(p.n, dtype=torch.bfloat16, device=p.sm.device)
    _ext.decode(p.sm, p.stream, p.offs, p.lut, p.L, p.n, out.view(torch.int16))
    return out[: p.n].view(p.shape)
