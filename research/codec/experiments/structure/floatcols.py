#!/usr/bin/env python3
"""Delimited numeric records (telemetry CSV): every column as text vs
typed. Integers: zigzag varint deltas. Decimals with a fixed number of
places ("43.04"): scaled integers, deltas, one stream per column; the
text must print back exactly. Few-valued columns: dictionary + MTF.
zstd -19 on every stream; compared with zstd -19 on the whole file
(fieldcols.py's layout, plus the decimal type). Rebuilt byte for byte."""
import subprocess, sys, collections
data = open(sys.argv[1], 'rb').read()[:int(sys.argv[2]) if len(sys.argv) > 2 else 64 << 20]
data = data[:data.rfind(b'\n') + 1]
sep = sys.argv[3].encode() if len(sys.argv) > 3 else b','
def z(b): return len(subprocess.run(['zstd', '-q', '-19', '-T1', '-c'], input=b, capture_output=True).stdout) if b else 0
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
lines = data.split(b'\n')[:-1]
rows = [l.split(sep) for l in lines]
width = collections.Counter(len(r) for r in rows).most_common(1)[0][0]
frame = bytearray()  # lines of another width stay as text
cols = [[] for _ in range(width)]
for l, r in zip(lines, rows):
    if len(r) == width:
        frame += b'\x00'
        for i, f in enumerate(r): cols[i].append(f)
    else: frame += l + b'\n'
def is_int(b): return (b.isdigit() and (len(b) == 1 or b[0:1] != b'0')) or (b[:1] == b'-' and b[1:].isdigit() and b[1:2] != b'0' and len(b) > 1)
def dec_places(b):
    if b.count(b'.') != 1: return None
    a, f = b.split(b'.')
    if not f.isdigit() or not (a.isdigit() or (a[:1] == b'-' and a[1:].isdigit())): return None
    if a.lstrip(b'-') != b'0' and a.lstrip(b'-')[:1] == b'0': return None
    return len(f)
streams = {'frame': bytes(frame)}; types = []
for i, c in enumerate(cols):
    name = f"c{i}"
    if all(is_int(v) for v in c):
        out = bytearray(); last = 0
        for v in c: x = int(v); varint(out, zig(x - last)); last = x
        streams[name] = bytes(out); types.append('int'); continue
    # Decimal column: ints or decimals with up to P places, mixed; each
    # value is (its places, scaled by 10^P) so "43.1", "43.10" and "43"
    # print back as written.
    def places_of(v):
        if v == b'': return 0
        if is_int(v): return 0
        d = dec_places(v)
        return d
    pl = [places_of(v) for v in c]
    distinct = set(c)
    if all(q is not None for q in pl) and max(pl) <= 6 and sum(1 for v in c if v != b'') > 0:
        P = max(pl); out = bytearray(); pls = bytearray(); last = 0
        for v, q in zip(c, pl):
            if v == b'': varint(out, 0); pls.append(255); continue
            a, _, f = v.partition(b'.')
            x = int(a) * 10 ** P + (int(f) * 10 ** (P - len(f)) if f else 0) * (-1 if a.startswith(b'-') else 1)
            varint(out, zig(x - last) + 1); last = x; pls.append(q)
        as_text = z(b'\n'.join(c)); as_dec = z(bytes(out)) + z(bytes(pls))
        print(f"   column {i}: {len(distinct):,} distinct of {len(c):,}, {P} places; as text {as_text:,}, as scaled-int deltas {as_dec:,}")
        streams[name] = bytes(out); streams[name + '_p'] = bytes(pls); types.append(f'dec{P}'); continue
    if len(distinct) * 3 < len(c):
        mtf = []; ranks = bytearray(); order = []
        for v in c:
            if v in mtf: r = mtf.index(v); mtf.pop(r); varint(ranks, r + 1)
            else: varint(ranks, 0); order.append(v)
            mtf.insert(0, v)
            if len(mtf) > 4096: mtf.pop()
        streams[name + '_d'] = b'\n'.join(order); streams[name + '_r'] = bytes(ranks); types.append('dict'); continue
    streams[name] = b'\n'.join(c); types.append('text')
# the same layout with decimals left as text, for the comparison
text_variant = sum(z(b'\n'.join(c)) for c, t in zip(cols, types) if t.startswith('dec'))
dec_variant = sum(z(streams[f"c{i}"]) for i, t in enumerate(types) if t.startswith('dec'))
total = sum(z(s) for s in streams.values()); whole = z(data)
print(f"{len(lines):,} rows x {width} columns, types {types}")
print(f"typed streams {total:,} vs whole zstd -19 {whole:,}: {whole / total:.2f}x smaller; decimals as scaled ints {dec_variant:,} vs as text {text_variant:,}")
# rebuild
out = bytearray(); k = 0
for b in bytes(frame).split(b'\x00'):
    pass
# decimal columns print back from their streams
for i, t in enumerate(types):
    if not t.startswith('dec'): continue
    P = int(t[3:]); vals = []; last = 0; b = streams[f"c{i}"]; pls = streams[f"c{i}_p"]; k = 0; j = 0
    while k < len(b):
        v = 0; sh = 0
        while True:
            byte = b[k]; k += 1; v |= (byte & 127) << sh
            if byte < 128: break
            sh += 7
        q = pls[j]; j += 1
        if v == 0: vals.append(b''); continue
        v -= 1; d = (v >> 1) ^ -(v & 1); x = last + d; last = x
        neg = x < 0; x = abs(x); a, f = divmod(x, 10 ** P)
        if q == 0: txt = str(a).encode()
        else: txt = str(a).encode() + b'.' + str(f).zfill(P)[:q].encode()
        vals.append((b'-' if neg else b'') + txt)
    cols[i] = vals
idx = 0; res = bytearray(); i = 0; fb = bytes(frame)
while i < len(fb):
    j = fb.find(b'\x00', i)
    if j < 0: res += fb[i:]; break
    res += fb[i:j]
    res += sep.join(cols[c][idx] for c in range(width)) + b'\n'; idx += 1
    i = j + 1
print("round trip exact:", bytes(res) == data)
