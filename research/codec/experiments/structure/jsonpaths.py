#!/usr/bin/env python3
"""jsonnums.py with key paths (actor.id, repo.id, payload.issue.id are
different columns) and recency coding: a value seen in the column's last
256 distinct values is one byte (its rank), others a varint delta from
the column's last new value. Skeleton with one marker byte per hole
(zstd -19 --long=27), streams (zstd -19); rebuilt byte for byte."""
import subprocess, sys, collections, datetime, re
data = open(sys.argv[1], 'rb').read()[:int(sys.argv[2]) if len(sys.argv) > 2 else 50 << 20]
data = data[:data.rfind(b'\n') + 1]  # whole lines
def z(b, long=False):
    a = ['zstd', '-q', '-19', '-T1', '-c'] + (['--long=27'] if long else [])
    return len(subprocess.run(a, input=b, capture_output=True).stdout) if b else 0
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
ts_re = re.compile(rb'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ')
skeleton = bytearray(); cols = collections.defaultdict(list); order = []
n = len(data); i = 0; stack = []; key = None; pos = 0
# A small scanner: strings, numbers, structure; strings right before ':' are keys.
while i < n:
    c = data[i]
    if c == 0x22:  # string
        j = i + 1
        while True:
            if data[j] == 0x5c: j += 2; continue
            if data[j] == 0x22: break
            j += 1
        s = data[i + 1:j]; i = j + 1
        k = i
        while k < n and data[k] in b' ': k += 1
        if k < n and data[k] == 0x3a:  # a key
            key = s; i = k + 1; continue
        # a string value: quoted digits or a timestamp become holes
        path = b'.'.join(stack + [key or b''])
        if s.isdigit() and 5 <= len(s) <= 18 and s[0:1] != b'0':
            skeleton += data[pos:i - len(s) - 2] + b'\x01'; cols[(path, b'q')].append(int(s)); order.append((path, b'q')); pos = i
        elif len(s) == 20 and ts_re.fullmatch(s):
            t = int(datetime.datetime.strptime(s.decode(), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp())
            skeleton += data[pos:i - 22] + b'\x02'; cols[(path, b't')].append(t); order.append((path, b't')); pos = i
        continue
    if c in b'-0123456789':
        j = i
        while j < n and data[j] in b'-0123456789.eE+': j += 1
        tok = data[i:j]
        if tok.lstrip(b'-').isdigit() and len(tok) <= 18 and str(int(tok)).encode() == tok:
            path = b'.'.join(stack + [key or b''])
            skeleton += data[pos:i] + b'\x00'; cols[(path, b'n')].append(int(tok)); order.append((path, b'n')); pos = j
        i = j; continue
    if c == 0x7b: stack.append(key or b''); key = None
    elif c == 0x7d: stack.pop() if stack else None; key = None
    elif c == 0x0a: stack.clear(); key = None
    i += 1
skeleton += data[pos:]
streams = {}
for col, vals in cols.items():
    out = bytearray(); recent = []; last = 0
    for v in vals:
        if v in recent:
            r = recent.index(v); recent.pop(r); out.append(r); recent.insert(0, v)
        else:
            out.append(255); varint(out, zig(v - last)); last = v
            recent.insert(0, v)
            if len(recent) > 255: recent.pop()
    streams[col] = bytes(out)
# the column order stream (which column each hole belongs to) is implied by the skeleton's marker position and path in a real decoder; here we count it: one byte per hole, zstd'ed
order_ids = {c: i for i, c in enumerate(cols)}
order_stream = bytes(order_ids[c] % 256 for c in order)
sk = z(bytes(skeleton), long=True); st = sum(z(s) for s in streams.values()); whole = z(data, long=True)
print(f"{len(order):,} holes in {len(cols)} columns; skeleton -> {sk:,}; streams -> {st:,} (column order stream {z(order_stream):,} if the decoder needed one)")
print(f"total {sk + st:,} vs whole file {whole:,}: {100 * (whole - sk - st) / whole:.1f}% smaller ({len(data) / (sk + st):.2f}x vs {len(data) / whole:.2f}x)")
for col, s in sorted(streams.items(), key=lambda x: -z(x[1]))[:8]:
    print(f"   {col[0].decode()[-30:]:30} {col[1].decode()} {len(cols[col]):>9,} values -> {z(s):>8,} bytes ({8 * z(s) / len(cols[col]):.1f} bits each)")
# rebuild
idx = collections.defaultdict(int); out = bytearray(); p = 0; h = 0
for m in re.finditer(rb'[\x00-\x02]', bytes(skeleton)):
    col = order[h]; h += 1
    v = cols[col][idx[col]]; idx[col] += 1
    out += skeleton[p:m.start()]
    if col[1] == b'n': out += str(v).encode()
    elif col[1] == b'q': out += b'"' + str(v).encode() + b'"'
    else: out += b'"' + datetime.datetime.fromtimestamp(v, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ').encode() + b'"'
    p = m.end()
out += skeleton[p:]
print("round trip exact:", bytes(out) == data)
