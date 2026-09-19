#!/usr/bin/env python3
"""Columnar JSON experiment: every line is an object; it is flattened to
(path -> value) in key order; the ordered path list is the record's
schema (dictionary-coded); each path's values form a column. Columns
are laid back to back into three streams (schema ids, strings, numbers)
and compressed with zstd -19; lines that do not re-serialize exactly go
raw. Sizes are compared with zstd -19 on the whole file."""
import json, subprocess, sys, collections
data = open(sys.argv[1], 'rb').read()
lines = data.split(b'\n')
if lines and lines[-1] == b'': lines.pop()
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items(): flatten(v, prefix + '.' + k if prefix else k, out)
    elif isinstance(obj, list):
        out.append((prefix + '[]', ('len', len(obj))))
        for i, v in enumerate(obj): flatten(v, f"{prefix}[{i}]", out)
    else:
        out.append((prefix, obj))
schemas = {}
schema_ids = bytearray()
cols = collections.defaultdict(list)   # path -> values
raw = []
kind = bytearray()
for ln in lines:
    try:
        obj = json.loads(ln)
        if json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode() != ln or not isinstance(obj, dict): raise ValueError
    except Exception:
        kind.append(1); raw.append(ln); continue
    kind.append(0)
    pairs = []
    flatten(obj, '', pairs)
    sig = tuple(p for p, _ in pairs)
    sid = schemas.setdefault(sig, len(schemas))
    varint(schema_ids, sid)
    for p, v in pairs: cols[p].append(v)
# streams: strings (varint length + bytes), numbers (zigzag varints), others (bool/null/len) as bytes
strings, numbers, small = bytearray(), bytearray(), bytearray()
for p in sorted(cols):
    for v in cols[p]:
        if isinstance(v, tuple): varint(small, v[1])
        elif v is None: small.append(0)
        elif v is True: small.append(1)
        elif v is False: small.append(2)
        elif isinstance(v, int): varint(numbers, (v << 1) ^ (v >> 63) if v < 0 else v << 1)
        elif isinstance(v, float): b = repr(v).encode(); varint(strings, len(b)); strings += b
        else: b = v.encode(); varint(strings, len(b)); strings += b
schema_dict = '\n'.join('\t'.join(s) for s in schemas).encode()
streams = {'kind': bytes(kind), 'schema_dict': schema_dict, 'schema_ids': bytes(schema_ids), 'strings': bytes(strings), 'numbers': bytes(numbers), 'small': bytes(small), 'raw': b'\n'.join(raw)}
def zsize(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-c'], input=b, capture_output=True).stdout) if b else 0
total = 0
print(f"{len(lines)} records, {len(raw)} raw, {len(schemas)} distinct schemas, {len(cols)} paths")
for k, v in streams.items():
    z = zsize(v); total += z; print(f"  {k:12s} raw {len(v):11,d}  zstd-19 {z:10,d}")
whole = zsize(data)
print(f"columns total: {total:,d} ({len(data)/total:.2f}x)  whole zstd-19: {whole:,d} ({len(data)/whole:.2f}x)  gain {whole/total:.2f}x")
long = len(subprocess.run(['zstd', '-q', '-19', '--long=27', '-c'], input=data, capture_output=True).stdout)
print(f"whole zstd-19 --long=27: {long:,d} ({len(data)/long:.2f}x)  gain over it {long/total:.2f}x")
