#!/usr/bin/env python3
"""Size estimate of a richer column model for the access log (encode side;
the transforms are invertible: MTF over first-appearance ids, and '=' for
'same size as the last time this path was served')."""
import re, subprocess, sys
from datetime import datetime
data = open(sys.argv[1], 'rb').read()
lines = data.split(b'\n')
pat = re.compile(rb'^(\S+) (\S+) (\S+) \[(\d\d)/(\w{3})/(\d{4}):(\d\d):(\d\d):(\d\d) ([+-]\d{4})\] "(.*)" (\d{3}|-) (\d+|-)$')
months = {b'Jan':1,b'Feb':2,b'Mar':3,b'Apr':4,b'May':5,b'Jun':6,b'Jul':7,b'Aug':8,b'Sep':9,b'Oct':10,b'Nov':11,b'Dec':12}
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
hosts, paths = {}, {}
host_mtf, path_mtf, ts_delta, status, nbytes, raw, kind = bytearray(), bytearray(), bytearray(), bytearray(), bytearray(), [], bytearray()
method, proto = bytearray(), bytearray()
mtf_h, mtf_p = [], []
last_ts = 0; last_size = {}
def mtf_code(mtf, key, out, limit=8192):
    if key in mtf:
        i = mtf.index(key); mtf.pop(i); varint(out, i + 1)
    else:
        varint(out, 0)
    mtf.insert(0, key)
    if len(mtf) > limit: mtf.pop()
for ln in lines:
    m = pat.match(ln)
    if not m: kind.append(1); raw.append(ln); continue
    kind.append(0)
    host, ident, user, dd, mon, yyyy, hh, mi, ss, zone, req, st, nb = m.groups()
    if host not in hosts: hosts[host] = len(hosts)
    mtf_code(mtf_h, host, host_mtf)
    ts = int(datetime(int(yyyy), months[mon], int(dd), int(hh), int(mi), int(ss)).timestamp())
    varint(ts_delta, zig(ts - last_ts)); last_ts = ts
    parts = req.split(b' ')
    p = parts[1] if len(parts) == 3 else req
    method += (parts[0] if len(parts) == 3 else b'?') + b'\n'; proto += (parts[2] if len(parts) == 3 else b'?') + b'\n'
    if p not in paths: paths[p] = len(paths)
    mtf_code(mtf_p, p, path_mtf, 65536)
    status += st + b'\n'
    if last_size.get(p) == nb: nbytes += b'=\n'
    else: nbytes += nb + b'\n'; last_size[p] = nb
cols = {'kind': bytes(kind), 'host_dict': b'\n'.join(hosts), 'host_mtf': bytes(host_mtf), 'path_dict': b'\n'.join(paths), 'path_mtf': bytes(path_mtf),
        'ts_delta': bytes(ts_delta), 'method': bytes(method), 'proto': bytes(proto), 'status': bytes(status), 'bytes': bytes(nbytes), 'raw': b'\n'.join(raw)}
def zsize(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-c'], input=b, capture_output=True).stdout) if b else 0
total = 0
for k, v in cols.items():
    z = zsize(v); total += z; print(f"  {k:10s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"columns total: {total:,d} ({len(data)/total:.2f}x)  whole zstd-19: {whole:,d} ({len(data)/whole:.2f}x)  gain {whole/total:.2f}x")
