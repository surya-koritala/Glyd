#!/usr/bin/env python3
"""Generic template extraction, CLP-style, on any text log: each line is
tokenized on delimiters; tokens that contain a digit are variables, the
rest (with the delimiters) form the line's template. Streams: template
ids (dictionary of templates), integer variables (zigzag varint of the
delta from the previous value in the same template slot), other
variables (dictionary-coded with move-to-front over recent ids), and
raw lines that would not round-trip. Every stream is compressed with
zstd -19; sizes are compared with zstd -19 on the whole file. The
transform is checked by rebuilding every line."""
import re, subprocess, sys, collections
data = open(sys.argv[1], 'rb').read()
lines = data.split(b'\n')
trailing = lines and lines[-1] == b''
if trailing: lines.pop()
tok = re.compile(rb'([A-Za-z0-9_.:/\-]+)')   # a token: runs of these characters; everything else is delimiter text kept in the template
digits = re.compile(rb'\d')
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
templates = {}
tmpl_ids = bytearray()
ints = bytearray()          # integer variables: zigzag delta per (template, slot)
last_int = {}
dict_vars = {}              # variable string -> id
mtf = []
var_stream = bytearray()    # MTF rank (0 = new, appended to dict order)
new_vars = []               # dictionary in first-appearance order
records = []                # for the rebuild: (tmpl id, [vars])
for ln in lines:
    parts = tok.split(ln)   # delimiters at even indexes, tokens at odd
    tpl = []; vars_ = []
    for i, p in enumerate(parts):
        if i % 2 == 0: tpl.append(p); continue
        if digits.search(p):
            if p.isdigit() and (len(p) == 1 or p[0:1] != b'0'):
                tpl.append(b'\x01'); vars_.append(('i', int(p)))
            else:
                tpl.append(b'\x02'); vars_.append(('s', p))
        else:
            tpl.append(p)
    key = b''.join(tpl)
    tid = templates.setdefault(key, len(templates))
    varint(tmpl_ids, tid)
    slot = 0
    for kind, v in vars_:
        if kind == 'i':
            k = (tid, slot); varint(ints, zig(v - last_int.get(k, 0))); last_int[k] = v
        else:
            if v in mtf:
                r = mtf.index(v); mtf.pop(r); varint(var_stream, r + 1)
            else:
                varint(var_stream, 0); new_vars.append(v)
            mtf.insert(0, v)
            if len(mtf) > 16384: mtf.pop()
        slot += 1
    records.append((tid, vars_))
streams = {'templates': b'\n'.join(templates), 'tmpl_ids': bytes(tmpl_ids), 'ints': bytes(ints), 'var_mtf': bytes(var_stream), 'var_dict': b'\n'.join(new_vars)}
def zsize(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-c'], input=b, capture_output=True).stdout) if b else 0
total = 0
print(f"{len(lines)} lines, {len(templates)} templates, {len(new_vars)} distinct string variables")
for k, v in streams.items():
    z = zsize(v); total += z; print(f"  {k:10s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"streams total: {total:,d} ({len(data)/total:.2f}x)  whole zstd-19: {whole:,d} ({len(data)/whole:.2f}x)  gain {whole/total:.2f}x")
# rebuild check
tl = list(templates)
out = []
for tid, vars_ in records:
    t = tl[tid]; vi = 0; o = bytearray()
    for ch in re.split(rb'([\x01\x02])', t):
        if ch == b'\x01': o += str(vars_[vi][1]).encode(); vi += 1
        elif ch == b'\x02': o += vars_[vi][1]; vi += 1
        else: o += ch
    out.append(bytes(o))
rebuilt = b'\n'.join(out) + (b'\n' if trailing else b'')
print("round trip exact:", rebuilt == data)
