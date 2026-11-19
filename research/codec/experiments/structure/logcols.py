#!/usr/bin/env python3
"""Structure-aware experiment on a Common Log Format access log.

Each line `host ident user [timestamp] "request" status bytes` is split
into typed columns; lines that do not parse go to a raw column. Every
column is compressed with zstd -19 on its own and the sizes summed, then
the lines are rebuilt from the columns and compared byte for byte.
"""
import re, subprocess, sys, struct, time
from datetime import datetime

path = sys.argv[1]
data = open(path, 'rb').read()
lines = data.split(b'\n')
tail = lines.pop() if lines and lines[-1] == b'' else None  # trailing newline handling
pat = re.compile(rb'^(\S+) (\S+) (\S+) \[(\d\d)/(\w{3})/(\d{4}):(\d\d):(\d\d):(\d\d) ([+-]\d{4})\] "(.*)" (\d{3}|-) (\d+|-)$')
months = {b'Jan':1,b'Feb':2,b'Mar':3,b'Apr':4,b'May':5,b'Jun':6,b'Jul':7,b'Aug':8,b'Sep':9,b'Oct':10,b'Nov':11,b'Dec':12}

# columns
kind = bytearray()          # 0 = parsed, 1 = raw line
hosts, host_ids = {}, bytearray()   # dictionary-coded host as varint id
ts_delta = bytearray()      # timestamp delta seconds (zigzag varint)
tz = bytearray()
req_method, req_path, req_proto = [], [], []
status = bytearray()
nbytes = bytearray()
raw = []

def varint(out, v):
    while v >= 128:
        out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1

last_ts = 0
fields = []
t0 = time.time()
for ln in lines:
    m = pat.match(ln)
    if not m:
        kind.append(1); raw.append(ln); continue
    kind.append(0)
    host, ident, user, dd, mon, yyyy, hh, mi, ss, zone, req, st, nb = m.groups()
    hid = hosts.setdefault(host, len(hosts)); varint(host_ids, hid)
    ts = int(datetime(int(yyyy), months[mon], int(dd), int(hh), int(mi), int(ss)).timestamp())
    varint(ts_delta, zig(ts - last_ts)); last_ts = ts
    tz += zone + b'\n'
    parts = req.split(b' ')
    # request split into method / path / protocol when it has the usual shape; else the whole string as path with an empty method
    if len(parts) == 3:
        req_method.append(parts[0]); req_path.append(parts[1]); req_proto.append(parts[2])
    else:
        req_method.append(b''); req_path.append(req); req_proto.append(b'\x00')
    status += st + b'\n'
    nbytes += nb + b'\n'
    fields.append((ident, user))
parse_s = time.time() - t0

# ident/user are almost always "- -": store as a dictionary column too
iu = {}
iu_ids = bytearray()
for iden, usr in fields:
    varint(iu_ids, iu.setdefault(iden + b' ' + usr, len(iu)))

cols = {
    'kind': bytes(kind),
    'host_dict': b'\n'.join(hosts.keys()),
    'host_ids': bytes(host_ids),
    'iu_dict': b'\n'.join(iu.keys()), 'iu_ids': bytes(iu_ids),
    'ts_delta': bytes(ts_delta), 'tz': bytes(tz),
    'method': b'\n'.join(req_method), 'path': b'\n'.join(req_path), 'proto': b'\n'.join(req_proto),
    'status': bytes(status), 'bytes': bytes(nbytes),
    'raw': b'\n'.join(raw),
}

def zsize(b, args=('-19',)):
    if not b: return 0
    p = subprocess.run(['zstd', '-q', *args, '-c'], input=b, capture_output=True)
    return len(p.stdout)

total = 0
print(f"{len(lines)} lines, {len(raw)} unparsed; parse {parse_s:.1f}s")
for k, v in cols.items():
    z = zsize(v); total += z
    print(f"  {k:10s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"columns total zstd-19: {total:,d}  ({len(data)/total:.2f}x)   whole file zstd-19: {whole:,d} ({len(data)/whole:.2f}x)   gain {whole/total:.2f}x")

# rebuild and verify
t0 = time.time()
out = []
hi = mi_ = 0
host_list = list(hosts.keys()); iu_list = list(iu.keys())
def read_varints(b):
    v = 0; shift = 0; res = []
    for x in b:
        v |= (x & 127) << shift
        if x < 128: res.append(v); v = 0; shift = 0
        else: shift += 7
    return res
hids = read_varints(host_ids); ius = read_varints(iu_ids); tds = read_varints(ts_delta)
tzs = tz.split(b'\n'); sts = status.split(b'\n'); nbs = nbytes.split(b'\n')
ri = pi = 0; last = 0
for k in kind:
    if k == 1:
        out.append(raw[ri]); ri += 1; continue
    d = tds[pi]; d = -(d >> 1) if d & 1 else d >> 1; last += d
    t = datetime.fromtimestamp(last)
    mon = [m for m, n in months.items() if n == t.month][0]
    m = req_method[pi]
    req = req_path[pi] if m == b'' else m + b' ' + req_path[pi] + b' ' + req_proto[pi]
    out.append(host_list[hids[pi]] + b' ' + iu_list[ius[pi]] + b' [' + f"{t.day:02d}/".encode() + mon + f"/{t.year}:{t.hour:02d}:{t.minute:02d}:{t.second:02d} ".encode() + tzs[pi] + b'] "' + req + b'" ' + sts[pi] + b' ' + nbs[pi])
    pi += 1
rebuilt = b'\n'.join(out) + (b'\n' if tail is not None else b'')
print("round trip exact:", rebuilt == data, f"(rebuild {time.time()-t0:.1f}s)")
if rebuilt != data:
    i = next(i for i in range(min(len(rebuilt), len(data))) if rebuilt[i] != data[i])
    print(data[i-80:i+40]); print(rebuilt[i-80:i+40])
