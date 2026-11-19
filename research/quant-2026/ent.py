"""Entropies from histograms, and the bits a few simple layouts spend on a symbol histogram.

Everything takes counts (numpy arrays of any shape) and returns bits a symbol.
"""
import heapq
import numpy as np

LN2 = np.log(2.0)


def H(c):
    """Entropy (bits) of a histogram of any shape."""
    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.sum()
    if n == 0:
        return 0.0
    c = c[c > 0]
    return float(np.log2(n) - (c * np.log2(c)).sum() / n)


def Hc(joint, mm=True):
    """H(Y | X) in bits for a joint histogram J[x, y] (X the leading axis), plug-in plus, if mm, the
    Miller-Madow correction (nonzero cells of (x, y) minus nonzero x, over 2 N ln 2: the plug-in's bias)."""
    j = np.asarray(joint, dtype=np.float64)
    j = j.reshape(j.shape[0], -1)
    n = j.sum()
    if n == 0:
        return 0.0
    h = H(j) - H(j.sum(1))
    if mm:
        h += ((j > 0).sum() - (j.sum(1) > 0).sum()) / (2 * n * LN2)
    return h


def huffman_bits(c):
    """Expected length (bits a symbol) of an optimal prefix code for the histogram."""
    c = [int(x) for x in np.asarray(c).ravel() if x > 0]
    n = sum(c)
    if len(c) <= 1:
        return 0.0
    heap = [(x, i) for i, x in enumerate(c)]
    heapq.heapify(heap)
    tot, k = 0, len(c)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        tot += a[0] + b[0]  # every merge adds one bit to all its leaves' weight
        heapq.heappush(heap, (a[0] + b[0], k))
        k += 1
    return tot / n


def ranked(c):
    """Counts sorted most common first."""
    return np.sort(np.asarray(c, dtype=np.float64).ravel())[::-1]


def fast_bits(c, raw_bits, k=3):
    """A k-bit code into the 2^k - 1 commonest symbols, the last code an escape followed by the symbol raw
    (raw_bits) in a side stream (no escape if 2^k symbols or fewer occur): returns (bits a symbol, escape rate)."""
    r = ranked(c)
    n = r.sum()
    if (r > 0).sum() <= 2 ** k:
        return float(k), 0.0
    esc = r[2 ** k - 1:].sum() / n
    return k + esc * raw_bits, esc


def tiered_bits(c, raw_bits, tiers=3, digit=2):
    """Glyd's tiered code: digits of `digit` bits; the commonest 2^digit - 1 symbols take one digit, the next
    2^digit - 1 two digits (the first one the all-ones digit going on), and so on for `tiers` tiers; the rest
    take the `tiers` going-on digits and then the symbol raw. Returns (bits a symbol, share coded raw)."""
    r = ranked(c)
    n = r.sum()
    per = 2 ** digit - 1
    bits = 0.0
    for t in range(tiers):
        bits += r[per * t:per * (t + 1)].sum() / n * digit * (t + 1)
    rest = r[per * tiers:].sum() / n
    return bits + rest * (digit * tiers + raw_bits), rest


def limited_huffman_bits(c, maxlen):
    """Expected length (bits a symbol) of an optimal prefix code whose codes are at most maxlen bits (package-merge)."""
    w = sorted(int(x) for x in np.asarray(c).ravel() if x > 0)
    n = len(w)
    if n <= 1:
        return 0.0
    if n > 2 ** maxlen:
        raise ValueError("too many symbols for the length limit")
    # package-merge: each level's list holds (weight, set of symbols as counts of each symbol it contains)
    leaves = [(x, [1 if j == i else 0 for j in range(n)]) for i, x in enumerate(w)]
    cur = list(leaves)
    for _ in range(maxlen - 1):
        pk = [(cur[i][0] + cur[i + 1][0], [a + b for a, b in zip(cur[i][1], cur[i + 1][1])]) for i in range(0, len(cur) - 1, 2)]
        cur = sorted(leaves + pk, key=lambda t: t[0])
    take = cur[:2 * n - 2]
    length = [sum(t[1][j] for t in take) for j in range(n)]
    return sum(l * x for l, x in zip(length, w)) / sum(w)
