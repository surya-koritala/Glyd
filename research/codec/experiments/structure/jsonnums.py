#!/usr/bin/env python3
"""Partial shredding of JSON lines: numbers (JSON numbers, and quoted
digit strings of 5+ digits) and ISO timestamps leave the text for typed
per-key streams (zigzag varint deltas per key name); everything else
stays in a skeleton with one marker byte per hole, so the strings keep
matching across fields and records. zstd -19 --long=27 on the skeleton
(the stand-in for a 128 MB-window coder), zstd -19 on the streams;
compared with the untouched file. Rebuilt and compared byte for byte."""
import re, subprocess, sys, collections, datetime
data = open(sys.argv[1], 'rb').read()[:int(sys.argv[2]) if len(sys.argv) > 2 else 50 << 20]
def z(b, long=False):
    a = ['zstd', '-q', '-19', '-T1', '-c'] + (['--long=27'] if long else [])
    return len(subprocess.run(a, input=b, capture_output=True).stdout) if b else 0
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
# holes: "key":123 | "key":"12345" | "key":"2024-01-15T12:00:01Z"
pat = re.compile(rb'"([A-Za-z_]+)":(?:(-?\d{1,18})(?=[,}\]])|"(\d{5,18})"|"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)")')
cols = collections.defaultdict(list)   # key -> values (ints, or epoch seconds)
kinds = {}
skeleton = bytearray(); pos = 0; holes = 0
for m in pat.finditer(data):
    k = m.group(1)
    if m.group(2) is not None: v = int(m.group(2)); kind = b'n'
    elif m.group(3) is not None: v = int(m.group(3)); kind = b'q'
    else:
        t = m.group(4).decode(); kind = b't'
        v = int(datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp())
    key = (k, kind)
    # canonical only: the text must be what we would print back
    if kind == b'n' and str(v).encode() != m.group(2): continue
    if kind == b'q' and str(v).encode() != m.group(3): continue
    skeleton += data[pos:m.start()] + b'"' + k + b'":' + {b'n': b'\x00', b'q': b'\x01', b't': b'\x02'}[kind]
    cols[key].append(v)
    pos = m.end(); holes += 1
skeleton += data[pos:]
streams = {}
for (k, kind), vals in cols.items():
    out = bytearray(); last = 0
    for v in vals: varint(out, zig(v - last)); last = v
    streams[(k, kind)] = bytes(out)
sk = z(bytes(skeleton), long=True)
st = sum(z(s) for s in streams.values())
whole = z(data, long=True)
print(f"{holes:,} holes in {len(cols)} key streams; skeleton {len(skeleton):,} -> {sk:,}; streams {sum(len(s) for s in streams.values()):,} -> {st:,}")
print(f"total {sk + st:,} vs whole file {whole:,}: {100 * (whole - sk - st) / whole:.1f}% smaller ({len(data) / (sk + st):.2f}x vs {len(data) / whole:.2f}x)")
for (k, kind), s in sorted(streams.items(), key=lambda x: -z(x[1]))[:6]:
    print(f"   {k.decode():14} {kind.decode()} {len(cols[(k, kind)]):>9,} values -> {z(s):>9,} bytes")
# rebuild
idx = collections.defaultdict(int); out = bytearray(); i = 0
hole_re = re.compile(rb'"([A-Za-z_]+)":([\x00-\x02])')
p = 0
for m in hole_re.finditer(bytes(skeleton)):
    k = m.group(1); kind = {b'\x00': b'n', b'\x01': b'q', b'\x02': b't'}[m.group(2)]
    v = cols[(k, kind)][idx[(k, kind)]]; idx[(k, kind)] += 1
    out += skeleton[p:m.start()] + b'"' + k + b'":'
    if kind == b'n': out += str(v).encode()
    elif kind == b'q': out += b'"' + str(v).encode() + b'"'
    else: out += b'"' + datetime.datetime.fromtimestamp(v, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ').encode() + b'"'
    p = m.end()
out += skeleton[p:]
print("round trip exact:", bytes(out) == data)
