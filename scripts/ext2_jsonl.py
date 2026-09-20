#!/usr/bin/env python3
"""CSV rows as JSON objects, one per line, 128 MB: numbers stay numbers,
empty fields become null. ext2_jsonl.py <csv> <jsonl> <name,name,...>"""
import json, sys
src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
out = open(dst, 'wb'); w = 0
for line in open(src, 'rb'):
    f = line.rstrip(b'\n').split(b',')
    if len(f) != len(names):
        continue
    obj = {}
    for k, v in zip(names, f):
        if v == b'':
            obj[k] = None
            continue
        s = v.decode()
        try:
            obj[k] = int(s) if s.lstrip('-').isdigit() else float(s)
        except ValueError:
            obj[k] = s
    b = (json.dumps(obj, separators=(',', ':')) + '\n').encode()
    out.write(b); w += len(b)
    if w >= 128 << 20:
        break
