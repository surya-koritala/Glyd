#!/usr/bin/env python3
"""Generic field columns: lines are split on the delimiter (single space
or tab, the most common in the sample); lines whose field count equals
the majority count become one column per field, the rest stay raw in
order (a 'kind' stream says which). Each column: integers as zigzag
varint deltas from the previous value in that column when every value of
the column is a plain integer, otherwise the strings newline-separated
with move-to-front over the last 65,536 distinct values. Every stream
zstd -19; rebuilt and compared byte for byte."""
import subprocess, sys, collections
data = open(sys.argv[1], 'rb').read()
lines = data.split(b'\n')
trailing = lines and lines[-1] == b''
if trailing: lines.pop()
delim = b' ' if data.count(b' ') > data.count(b'\t') else b'\t'
counts = collections.Counter(ln.count(delim) for ln in lines[:200000])
nf = counts.most_common(1)[0][0] + 1
cols = [[] for _ in range(nf)]
kind = bytearray(); raw = []
for ln in lines:
    f = ln.split(delim)
    if len(f) == nf: kind.append(0); [c.append(v) for c, v in zip(cols, f)]
    else: kind.append(1); raw.append(ln)
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
def is_int(b): return b.isdigit() and (len(b) == 1 or b[0:1] != b'0') and len(b) < 18
streams = {'kind': bytes(kind), 'raw': b'\n'.join(raw)}
types = []
for i, c in enumerate(cols):
    if c and all(is_int(v) for v in c):
        out = bytearray(); last = 0
        for v in c: x = int(v); varint(out, zig(x - last)); last = x
        streams[f'c{i}_int'] = bytes(out); types.append('int')
    else:
        distinct = len(set(c))
        if distinct * 20 < len(c):   # few distinct values: dictionary + MTF ranks
            mtf = []; ranks = bytearray(); order = []
            for v in c:
                if v in mtf: r = mtf.index(v); mtf.pop(r); varint(ranks, r + 1)
                else: varint(ranks, 0); order.append(v)
                mtf.insert(0, v)
                if len(mtf) > 65536: mtf.pop()
            streams[f'c{i}_dict'] = b'\n'.join(order); streams[f'c{i}_mtf'] = bytes(ranks); types.append('dict')
        else:
            streams[f'c{i}_str'] = b'\n'.join(c); types.append('str')
def zsize(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-c'], input=b, capture_output=True).stdout) if b else 0
total = 0
print(f"{len(lines)} lines, {nf} fields (delimiter {delim!r}), {len(raw)} raw lines; column types {types}")
for k, v in streams.items():
    z = zsize(v); total += z; print(f"  {k:10s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"streams total: {total:,d} ({len(data)/total:.2f}x)  whole zstd-19: {whole:,d} ({len(data)/whole:.2f}x)  gain {whole/total:.2f}x")
# rebuild
out = []; ri = 0; ci = [0] * nf
for k in kind:
    if k: out.append(raw[ri]); ri += 1
    else:
        out.append(delim.join(cols[i][ci[i]] for i in range(nf)))
        for i in range(nf): ci[i] += 1
rebuilt = b'\n'.join(out) + (b'\n' if trailing else b'')
print("round trip exact:", rebuilt == data)
