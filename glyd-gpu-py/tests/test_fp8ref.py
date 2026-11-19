"""glyd_gpu/fp8ref.py, the lossless FP8 (e4m3) format's numpy reference: every byte it is given comes back bit for bit.
numpy only (no GPU, no torch).

    python test_fp8ref.py

Covers: the layout's coordinates; all 256 byte patterns (NaNs 0x7F and 0xFF, both zeros) alone, as constant patches and in one matrix;
matrices of 1x1, 17x33, 128x128 and 4096x7168 and shapes that are not multiples of a patch; a patch that is all escapes (stored
plain: no more than its 8 bits a value and its head) beside coded ones, the plain threshold at 511 and 512 words; an all-zero
patch; the sizes the layout says; every field of a realistic patch read back by an independent decoder of the docstring; and every sample
gpu/fp8/fetch_samples.py fetched (GLYD_FP8_SAMPLES: the directory with its manifest.json, else ~/.cache/glyd-fp8-samples; skipped, loudly, where there is none)."""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fp8ref", os.path.join(HERE, "..", "glyd_gpu", "fp8ref.py"))
f8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8)

# e4m3fn's magnitudes by byte 0x00..0x7E (0x7F is NaN)
_B = np.arange(127)
MAG = np.where(_B >> 3 == 0, (_B & 7) / 8 * 2.0**-6, (1 + (_B & 7) / 8) * 2.0 ** ((_B >> 3).astype(np.float64) - 7))


def quantize(x):
    """float array -> e4m3 bytes, to the nearest value (past 448: 448)."""
    idx = np.searchsorted((MAG[1:] + MAG[:-1]) / 2, np.abs(x))
    return (idx | ((x < 0) << 7)).astype(np.uint8)


def weights(O, K, seed=0):
    """Gaussian weights quantized as a block-scaled checkpoint holds them: each 128 x 128 block's largest value at 448 (e4m3's largest)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((O, K), dtype=np.float32)
    ob, kb = -(-O // 128), -(-K // 128)
    pad = np.zeros((ob * 128, kb * 128), np.float32)
    pad[:O, :K] = x
    top = np.abs(pad).reshape(ob, 128, kb, 128).max((1, 3))
    scale = np.repeat(np.repeat(448 / top, 128, axis=0), 128, axis=1)
    return quantize(pad * scale)[:O, :K]


def part_words(heads):
    """Each patch's variable part's words, from its head as the docstring says."""
    h = heads.astype(np.int64)
    n1, n2, n3 = h[:, 3] & 0x3FFF, (h[:, 3] >> 14) & 0x3FFF, h[:, 2] >> 20
    return np.where(h[:, 3] >> 31, 512, -(-n1 // 16) + -(-3 * n2 // 32) + -(-n3 // 8))


def roundtrip(w, plain=None):
    p = f8.pack(w)
    assert np.array_equal(f8.unpack(p), w), w.shape
    size = part_words(p.heads)
    assert np.array_equal(p.heads[:, 0], np.cumsum(size) - size) and p.var.size == size.sum() + f8.GUARD, "the parts follow one another, and the guard words after them"
    assert (p.var[-f8.GUARD :] == 0).all()
    if plain is not None:
        assert p.plain() == plain, (p.plain(), plain)
    return p


def test_layout():
    """The documented coordinates of a lane's values (the A operand of mma.m16n8k32: rows g and g + 8, columns 4t.. and 16 + 4t..), the stream's
    order (row block, lane, unit, value), and PERM a permutation."""
    assert f8.PATCH == (64, 128) and f8.VALUES == 8192 and f8.CORE_BYTES == 6144 and f8.HEAD_WORDS == 8
    assert np.array_equal(np.sort(f8.PERM), np.arange(8192))
    row = lambda i: f8.PERM[i] // 128
    col = lambda i: f8.PERM[i] % 128
    lane = 5  # g = 1, t = 1
    got = [(row(64 * lane + j), col(64 * lane + j)) for j in range(16)]
    assert got == [(1, 4), (1, 5), (1, 6), (1, 7), (9, 4), (9, 5), (9, 6), (9, 7), (1, 20), (1, 21), (1, 22), (1, 23), (9, 20), (9, 21), (9, 22), (9, 23)], got
    assert [(row(64 * lane + 16 * ks), col(64 * lane + 16 * ks)) for ks in range(4)] == [(1, 4), (1, 36), (1, 68), (1, 100)]  # a lane's four units
    assert (row(2048 + 32), col(2048 + 32)) == (16, 64)  # row block 1, lane 0, unit 2, value 0: row 16, column 64
    assert sorted(int(f8.PERM[i]) for i in range(2048)) == list(range(2048)), "row block 0 is the patch's first 16 rows"


def test_every_byte_pattern():
    allb = np.arange(256, dtype=np.uint8)
    for v in (0x7F, 0xFF, 0x00, 0x80):
        assert v in allb
    roundtrip(allb.reshape(16, 16))
    roundtrip(allb.reshape(1, 256))
    roundtrip(allb.reshape(256, 1))
    # each byte value as a patch of its own (a constant patch: its one exponent has all the values; a NaN patch included)
    const = np.repeat(allb, 64 * 128).reshape(256, 64, 128).transpose(1, 0, 2).reshape(64, 256 * 128)
    p = roundtrip(const, plain=0)
    assert p.var.size == f8.GUARD, "a constant patch has no spill"
    # every byte at every place of a patch (a shifted ramp: patch rows hold all 256 values at each of 256 shifts)
    ramp = ((np.arange(64)[:, None] * 7 + np.arange(128)[None, :] * 3) % 256).astype(np.uint8)
    roundtrip(ramp)
    roundtrip(np.tile(allb, (64, 2)))
    # random bytes: every patch is plain (16 exponents with equal counts)
    rng = np.random.default_rng(1)
    p = roundtrip(rng.integers(0, 256, (128, 256), dtype=np.uint8), plain=4)
    assert p.nbytes() == 4 * (8192 + 32) + 4 * f8.GUARD + 8, "a plain patch is its 8192 bytes and its head"


def test_shapes():
    for shape in ((1, 1), (1, 300), (300, 1), (17, 33), (63, 127), (64, 128), (65, 129), (128, 128), (200, 300), (130, 1000)):
        w = weights(*shape, seed=2)
        roundtrip(w)
        rng = np.random.default_rng(3)
        roundtrip(rng.integers(0, 256, shape, dtype=np.uint8))
        roundtrip(np.zeros(shape, np.uint8))
        roundtrip(np.full(shape, 0x7F, np.uint8))
    t = time.time()
    w = weights(4096, 7168, seed=4)
    p = roundtrip(w, plain=0)
    assert p.bits_per_value() < 6.8, p.bits_per_value()  # more than 15% under FP8's 8 bits
    print(f"  4096x7168: {p.bits_per_value():.4f} bits a value, {time.time() - t:.1f} s with the data")
    q = f8.pack(f8.unpack(p))  # pack is deterministic: the same bytes from the same matrix
    assert all(np.array_equal(getattr(p, n), getattr(q, n)) for n in ("core", "heads", "var")) and p.shape == q.shape


def test_all_escapes_is_plain():
    """A patch whose exponents are all alike in count spills almost every value past the levels: it is stored plain, beside coded ones."""
    rng = np.random.default_rng(5)
    flat = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    coded = weights(64, 128, seed=6)
    two = np.concatenate([coded, flat, coded, flat], axis=1)  # patches 0..3, in a row
    p = roundtrip(two, plain=2)
    assert list(p.heads[:, 3] >> 31) == [0, 1, 0, 1]
    assert (p.heads[1, 1:3] == 0).all() and (p.heads[1, 3] & 0x7FFFFFFF) == 0 and (p.heads[1, 4:] == 0).all(), "a plain patch's head says only that"
    # the bound: the variable part is less than 512 words in a coded patch, 512 in a plain one
    sizes = np.diff(np.append(p.heads[:, 0], p.var.size - f8.GUARD))
    assert list(sizes[[1, 3]]) == [512, 512] and (sizes[[0, 2]] < 512).all(), sizes
    # one exponent spread over 13 of the 16 (and no more): still coded if its variable part is under 512 words, plain if not; both exact
    for n_exp in (5, 8, 11, 13, 16):
        ex = rng.integers(0, n_exp, (64, 128))
        roundtrip(((ex << 3) | rng.integers(0, 8, (64, 128)) | (rng.integers(0, 2, (64, 128)) << 7)).astype(np.uint8))
    # a coded patch next to the limit: spill values until its words reach 512
    base = np.full((64, 128), 0x40, np.uint8)  # one exponent: all rank 0
    for spill in (1000, 3000, 5000, 6000, 7000, 8000, 8192):
        w = base.copy().reshape(-1)
        k = rng.permutation(8192)[:spill]
        w[k] = (rng.integers(0, 16, spill) << 3 | rng.integers(0, 8, spill)).astype(np.uint8)
        p = roundtrip(w.reshape(64, 128))
        if spill <= 3000:  # all the levels and the literals are used
            n1, n2, n3 = p.heads[0, 3] & 0x3FFF, (p.heads[0, 3] >> 14) & 0x3FFF, p.heads[0, 2] >> 20
            assert n1 > n2 > n3 > 0 and p.plain() == 0, (spill, n1, n2, n3)
    # the threshold: a variable part of exactly 511 words is coded, 512 plain (the patch's words searched by its counts)
    for want, plain in ((511, 0), (512, 1)):
        w = _patch_of_words(want, rng)
        p = roundtrip(w, plain=plain)
        assert p.var.size - f8.GUARD == (512 if plain else want), (want, p.var.size)


def _patch_of_words(want, rng):
    """A 64 x 128 patch whose variable part would be `want` words: ranks 0..5 about a values each, 6..12 b each, 13..15 the n3 literals
    (exponents in descending order by rank, so that ties keep this order)."""
    words = lambda n1, n2, n3: -(-n1 // 16) + -(-3 * n2 // 32) + -(-n3 // 8)
    for b in range(8, 1200):
        for n3 in range(0, min(3 * b, 300)):
            a = (8192 - n3 - 7 * b) // 6
            r0 = 8192 - n3 - 7 * b - 5 * a
            n1, n2 = 3 * a + 7 * b + n3, 7 * b + n3
            if a >= b and words(n1, n2, n3) == want:
                lit = [n3 // 3 + (i < n3 % 3) for i in range(3)]
                counts = [r0, a, a, a, a, a] + [b] * 7 + lit
                ex = np.repeat(15 - np.arange(16), counts)
                ex = ex[rng.permutation(8192)]
                return ((ex << 3) | rng.integers(0, 8, 8192)).astype(np.uint8).reshape(64, 128)
    raise AssertionError(want)


def test_mixed_across_steps():
    """Coded and plain patches in one matrix, past the 128 patches a step of pack and unpack: the bases of the parts run on across steps."""
    rng = np.random.default_rng(8)
    w = weights(192, 128 * 100, seed=9)  # 300 patches
    pick = rng.random((3, 100)) < 0.4
    for r, c in zip(*np.nonzero(pick)):
        w[r * 64 : (r + 1) * 64, c * 128 : (c + 1) * 128] = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    p = roundtrip(w, plain=int(pick.sum()))
    assert 0 < p.plain() < 300 and len(p.heads) > 2 * f8.CHUNK


def test_fuzz():
    """Random shapes, exponent spreads (geometric around a random peak) and shares of junk bytes (NaNs and every other byte among them)."""
    rng = np.random.default_rng(7)
    for _ in range(80):
        O, K = int(rng.integers(1, 300)), int(rng.integers(1, 300))
        ex = np.clip(np.rint(rng.integers(0, 16) - rng.exponential(rng.uniform(0.3, 6), (O, K))), 0, 15).astype(np.int64)
        w = ((rng.integers(0, 2, (O, K)) << 7) | (ex << 3) | rng.integers(0, 8, (O, K))).astype(np.uint8)
        junk = rng.random((O, K)) < rng.choice([0, 0, 0.01, 0.1, 0.5])
        w[junk] = rng.integers(0, 256, int(junk.sum()), dtype=np.uint8)
        roundtrip(w)


def test_all_zero_patch():
    for shape in ((64, 128), (128, 256), (100, 200)):
        p = roundtrip(np.zeros(shape, np.uint8), plain=0)
        n = len(p.heads)
        assert p.var.size == f8.GUARD and p.nbytes() == n * (6144 + 32) + 4 * f8.GUARD + 8, (p.nbytes(), n)
        assert (p.core == 0).all()


def test_layout_of_the_bytes():
    """Patches of known matrices: their heads, digits, signs and mantissas, the row blocks' offsets and the variable part read back by hand as the
    module's docstring says (an independent decoder of the docstring's layout, which the GPU kernels follow)."""
    w = np.full((64, 128), 0x40, np.uint8)  # exponent 8, mantissa 0, positive: rank 0
    w[1, 4] = 0xC9  # lane 5's value j = 0 (row 1, column 4): negative, exponent 9, mantissa 1; exponent 9 is rank 1
    w[9, 23] = 0x4B  # lane 5's value j = 15 (row 9, column 23): exponent 9 again, mantissa 3
    p = roundtrip(w)
    assert p.heads[0, 0] == 0 and (p.heads[0, 1] & 0xF) == 8 and ((p.heads[0, 1] >> 4) & 0xF) == 9
    assert ((p.heads[0, 1] >> 8) & 0xF) == 15 and ((p.heads[0, 1] >> 12) & 0xF) == 14, "the unused exponents in descending order (the tie rule)"
    words = p.core[0].view("<u4")
    assert words[5] == (1 << 0) | (1 << (4 * 7 + 2)), hex(words[5])  # value 0's digit at bit 0, value 15's at 4 * 7 + 2
    assert words[32 + 2 * 5] == (1 << 7) | 1, hex(words[32 + 2 * 5])  # value 0: sign at bit 7, mantissa 1 at bits 0..2
    assert words[32 + 2 * 5 + 1] == 3 << (8 * 2 + 4), hex(words[32 + 2 * 5 + 1])  # value 15 = 8 + 4 + 3: mantissa in byte (3 + 3) & 3 = 2, bits 4..6
    assert words[6] == 0 and p.var.size == f8.GUARD and (p.heads[0, 3] & 0x3FFF) == 0
    # every level, its offsets and fields, by an independent read of a realistic patch with literals (and 3-bit digits that cross words)
    rng = np.random.default_rng(11)
    ex = np.clip(np.rint(12 - rng.exponential(2.5, (64, 128))), 0, 12).astype(np.int64)
    for e in (13, 14, 15):  # the three rarest: ranks 13..15, the literals
        ex[rng.random((64, 128)) < 0.002] = e
    w = ((rng.integers(0, 2, (64, 128)) << 7) | (ex << 3) | rng.integers(0, 8, (64, 128))).astype(np.uint8)
    p = roundtrip(w, plain=0)
    h = p.heads[0].astype(np.int64)
    sym = [(h[1] >> (4 * r)) & 15 for r in range(8)] + [(h[2] >> (4 * r)) & 15 for r in range(5)]
    n1, n2, n3 = h[3] & 0x3FFF, (h[3] >> 14) & 0x3FFF, h[2] >> 20
    assert n3 > 0 and 3 * n2 > 64, (n1, n2, n3)
    bits = np.unpackbits(p.var.view(np.uint8), bitorder="little")
    field = lambda at, n: int(sum(int(bits[at + b]) << b for b in range(n)))
    w2, w3 = -(-n1 // 16), -(-3 * n2 // 32)
    cw = p.core[0].view("<u4")
    k1 = k2 = k3 = 0
    o = {}
    for mb in range(4):
        o[mb] = (k1, k2, k3)
        for l in range(32):
            for ks in range(4):
                u = 4 * mb + ks
                for j in range(16):
                    d1 = (cw[96 * u + l] >> (4 * (j & 7) + 2 * (j >> 3))) & 3
                    nbw = cw[96 * u + 32 + 2 * l + (j >> 3)]
                    k = j & 7
                    sgn = (nbw >> (8 * k + 7 if k < 4 else 8 * (k - 4) + 3)) & 1
                    man = (nbw >> (8 * k if k < 4 else 8 * ((k - 4 + 3) & 3) + 4)) & 7
                    if d1 < 3:
                        e = sym[d1]
                    else:
                        d2 = field(2 * k1, 2)
                        k1 += 1
                        if d2 < 3:
                            e = sym[3 + d2]
                        else:
                            d3 = field(32 * w2 + 3 * k2, 3)
                            k2 += 1
                            if d3 < 7:
                                e = sym[6 + d3]
                            else:
                                e = field(32 * (w2 + w3) + 4 * k3, 4)
                                k3 += 1
                    row = 16 * mb + (l >> 2) + 8 * ((j >> 2) & 1)
                    col = 32 * ks + 4 * (l & 3) + (j & 3) + 16 * (j >> 3)
                    assert w[row, col] == (sgn << 7) | (e << 3) | man, (mb, l, ks, j)
    assert (k1, k2, k3) == (n1, n2, n3)
    assert (h[4] & 0xFFFF, h[4] >> 16, h[5] & 0xFFFF) == (o[1][0], o[2][0], o[3][0])
    assert (h[5] >> 16, h[6] & 0xFFFF, h[6] >> 16) == (o[1][1], o[2][1], o[3][1])
    assert (h[7] & 0x3FF, (h[7] >> 10) & 0x3FF, h[7] >> 20) == (o[1][2], o[2][2], o[3][2])


def test_rejected():
    for bad in (np.zeros((4, 4), np.int8), np.zeros(4, np.uint8), np.zeros((0, 4), np.uint8), np.zeros((2, 2, 2), np.uint8), np.zeros((4, 4), np.float32)):
        try:
            f8.pack(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad.dtype} {bad.shape}")


def test_real_samples():
    d = os.environ.get("GLYD_FP8_SAMPLES", os.path.expanduser("~/.cache/glyd-fp8-samples"))
    path = os.path.join(d, "manifest.json")
    if not os.path.exists(path):
        print(f"  SKIPPED: no samples at {d} (python gpu/fp8/fetch_samples.py DIR; GLYD_FP8_SAMPLES=DIR)")
        return
    manifest = json.load(open(path))
    assert len(manifest) >= 50, len(manifest)
    by = {}
    for m in manifest:
        a = np.load(os.path.join(d, m["file"]))
        assert a.dtype == np.uint8 and list(a.shape) == m["shape"]
        p = roundtrip(a)
        b = by.setdefault(m["repo"], [0, 0])
        b[0] += p.nbytes()
        b[1] += a.size
    for repo, (nb, n) in by.items():
        print(f"  {repo}: {nb * 8 / n:.4f} bits a value")
        assert nb * 8 / n < 6.8, repo  # at least 15% under FP8's
    print(f"  {len(manifest)} samples of {len(by)} repos, bit for bit")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            test()
            print(name, f"ok ({time.time() - t:.1f} s)")
