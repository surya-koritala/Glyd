#!/usr/bin/env python3
"""Typed columns for a MySQL dump: the VALUES tuples of INSERT statements
are parsed (quoted strings with backslash escapes, NULL, numbers) into
one column per position; everything else (statements, comments, tuple
punctuation layout) is kept as a frame stream so the bytes rebuild
exactly. Integer columns: zigzag varint delta from the previous row;
other columns: newline-separated text (dictionary + MTF when few
distinct values). zstd -19 per stream; rebuilt and compared."""
import subprocess, sys, collections
data = open(sys.argv[1], 'rb').read()
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
# Parse: find every "VALUES (" ... ";" region; inside, tuples "(...)" separated by "," or ",\n".
frame = bytearray()      # the bytes outside tuples, with \x00 marking a tuple
cols = collections.defaultdict(list)   # (arity, position) -> values
arity_stream = bytearray()
i = 0; n = len(data)
tuples = 0
while i < n:
    j = data.find(b'VALUES', i)
    if j < 0: frame += data[i:]; break
    k = j + 6
    while k < n and data[k] in b' \n': k += 1
    if k >= n or data[k] != ord('('): frame += data[i:k]; i = k; continue
    frame += data[i:k]; i = k
    # tuples until ';'
    while i < n and data[i] == ord('('):
        # parse one tuple
        p = i + 1; fields = []; cur = bytearray()
        while p < n:
            c = data[p]
            if c == ord("'"):
                q = p + 1
                while True:
                    if data[q] == ord('\\'): q += 2; continue
                    if data[q] == ord("'"): break
                    q += 1
                cur += data[p:q + 1]; p = q + 1
            elif c == ord(','): fields.append(bytes(cur)); cur = bytearray(); p += 1
            elif c == ord(')'): fields.append(bytes(cur)); p += 1; break
            else: cur.append(c); p += 1
        tuples += 1
        ar = len(fields)
        varint(arity_stream, ar)
        for pos, f in enumerate(fields): cols[(ar, pos)].append(f)
        frame += b'\x00'
        i = p
        # separator between tuples
        s = i
        while i < n and data[i] in b',\n ': i += 1
        if i < n and data[i] == ord('('): frame += data[s:i]
        else: i = s; break
def is_int(b): return (b.isdigit() and (len(b) == 1 or b[0:1] != b'0') and len(b) < 18) or (b[:1] == b'-' and b[1:].isdigit() and b[1:2] != b'0')
streams = {'frame': bytes(frame), 'arity': bytes(arity_stream)}
types = {}
for key, c in cols.items():
    name = f"c{key[0]}_{key[1]}"
    if all(is_int(v) for v in c):
        out = bytearray(); last = 0
        for v in c: x = int(v); varint(out, zig(x - last)); last = x
        streams[name + '_int'] = bytes(out); types[name] = 'int'
    elif len(set(c)) * 20 < len(c):
        mtf = []; ranks = bytearray(); order = []
        for v in c:
            if v in mtf: r = mtf.index(v); mtf.pop(r); varint(ranks, r + 1)
            else: varint(ranks, 0); order.append(v)
            mtf.insert(0, v)
            if len(mtf) > 4096: mtf.pop()
        streams[name + '_dict'] = b'\n'.join(order); streams[name + '_mtf'] = bytes(ranks); types[name] = 'dict'
    else:
        streams[name + '_str'] = b'\n'.join(c); types[name] = 'str'
def zsize(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-c'], input=b, capture_output=True).stdout) if b else 0
total = 0
print(f"{tuples} tuples; column types {types}")
for k, v in streams.items():
    z = zsize(v); total += z; print(f"  {k:12s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"streams total: {total:,d} ({len(data)/total:.2f}x)  whole zstd-19: {whole:,d} ({len(data)/whole:.2f}x)  gain {whole/total:.2f}x")
# rebuild
idx = collections.defaultdict(int)
ar_list = []
v = 0; shift = 0
for x in arity_stream:
    v |= (x & 127) << shift
    if x < 128: ar_list.append(v); v = 0; shift = 0
    else: shift += 7
out = bytearray(); t = 0
for b in frame:
    if b == 0:
        ar = ar_list[t]; t += 1
        out += b'(' + b','.join(cols[(ar, pos)][idx[(ar, pos)]] for pos in range(ar)) + b')'
        for pos in range(ar): idx[(ar, pos)] += 1
    else: out.append(b)
print("round trip exact:", bytes(out) == data)
