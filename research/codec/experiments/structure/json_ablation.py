#!/usr/bin/env python3
"""Where do the bytes of compressed JSON events go? Each token class is
replaced by a constant in turn and the slice recompressed (zstd -19
--long=27, the stand-in for a 128 MB-window coder); the drop is the
most that a perfect model of that class could save. Byte counts are of
the compressed output."""
import re, subprocess, sys
data = open(sys.argv[1], 'rb').read()[:int(sys.argv[2]) if len(sys.argv) > 2 else 50 << 20]
def z(b):
    return len(subprocess.run(['zstd', '-q', '-19', '--long=27', '-T1', '-c'], input=b, capture_output=True).stdout)
classes = [
    ("40-hex SHAs", rb'\b[0-9a-f]{40}\b', b'0' * 40),
    ("timestamps", rb'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', b'2024-01-01T00:00:00Z'),
    ("numbers (ints, quoted or not)", rb'(?<=[:,\[" /])-?\d{1,20}(?=[,}\]" /?\n])', b'0'),
    ("node_id base64 tokens", rb'"node_id":"[A-Za-z0-9_=-]+"', b'"node_id":"X"'),
    ("free text: message/title/body", rb'"(message|title|body|description)":"(?:[^"\\]|\\.)*"', b'"t":""'),
    ("logins/names (user and repo)", rb'"(login|display_login|name|full_name)":"[^"]*"', b'"n":"x"'),
]
base = z(data)
print(f"slice {len(data):,} bytes -> zstd -19 --long=27 {base:,} bytes ({len(data)/base:.2f}x)")
for name, pat, rep in classes:
    v = re.sub(pat, rep, data)
    s = z(v)
    print(f"  {name:34} removed {len(data)-len(v):>11,} raw bytes; compressed drops {base - s:>10,} bytes = {100*(base-s)/base:5.1f}% of the output")
