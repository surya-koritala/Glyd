#!/usr/bin/env python3
"""How much of a new version of an object is really new. For a pair
(old, new): content-defined chunking (a gear rolling hash, mean chunk
CHUNK bytes) of both; the chunks of `new` already present in `old` are
references, the rest is stored, compressed with zstd -19 (the stand-in
for the level). Also `zstd -19 --patch-from=old new`, which is delta
compression with the old version as the window. Every variant is
rebuilt and compared byte for byte. Sizes: new alone (zstd -19), new
as chunk references + stored chunks, new as a patch, and the share of
bytes new."""
import hashlib, os, subprocess, sys, struct
old_p, new_p = sys.argv[1], sys.argv[2]
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
old = open(old_p, 'rb').read(); new = open(new_p, 'rb').read()
import random
random.seed(1)
GEAR = [random.getrandbits(64) for _ in range(256)]
MASK = (1 << 64) - 1
def chunks(data):
    """Content-defined boundaries: gear hash, cut when the top bits are
    zero (mean CHUNK), min CHUNK/4, max CHUNK*4."""
    bits = CHUNK.bit_length() - 1
    mask = ((1 << bits) - 1) << (64 - bits)
    out = []; start = 0; h = 0; n = len(data); i = 0
    lo, hi = CHUNK // 4, CHUNK * 4
    while i < n:
        h = ((h << 1) + GEAR[data[i]]) & MASK
        i += 1
        if (i - start >= lo and (h & mask) == 0) or i - start >= hi:
            out.append((start, i)); start = i; h = 0
    if start < n: out.append((start, n))
    return out
def z(b):
    return len(subprocess.run(['zstd', '-q', '-19', '-T1', '-c'], input=b, capture_output=True).stdout) if b else 0
oc = chunks(old); nc = chunks(new)
have = {hashlib.sha256(old[a:b]).digest(): (a, b) for a, b in oc}
refs = []; stored = bytearray(); new_bytes = 0
for a, b in nc:
    h = hashlib.sha256(new[a:b]).digest()
    if h in have: refs.append(('old', have[h]))
    else:
        refs.append(('new', (len(stored), len(stored) + b - a))); stored += new[a:b]; new_bytes += b - a
        have[h] = None
# rebuild from refs
out = bytearray()
for kind, (a, b) in refs:
    out += old[a:b] if kind == 'old' else stored[a:b]
assert bytes(out) == new, "rebuild differs"
ref_bytes = len(refs) * 12  # a reference: kind, offset, length as varints/ints, before compression
manifest = b''.join(struct.pack('<BQI', 0 if k == 'old' else 1, a, b - a) for k, (a, b) in refs)
z_new = z(new); z_stored = z(bytes(stored)); z_manifest = z(manifest)
# delta with the old version as the window (zstd --patch-from)
wl = max(20, min(31, (max(len(old), len(new)) - 1).bit_length()))
patch = subprocess.run(['zstd', '-q', '-19', '-T1', f'--long={wl}', '--patch-from=' + old_p, '-c', new_p], capture_output=True).stdout
back = subprocess.run(['zstd', '-q', '-d', f'--long={wl}', '--patch-from=' + old_p, '-c'], input=patch, capture_output=True).stdout
assert back == new, "patch rebuild differs"
name = f"{os.path.basename(old_p)} -> {os.path.basename(new_p)}"
print(f"{name}: new version {len(new):,} B; chunks {len(nc):,} (mean {len(new)//max(1,len(nc)):,} B); bytes not in old version: {100*new_bytes/len(new):.1f}%")
print(f"   new alone, zstd -19:            {z_new:>13,} B  ({len(new)/z_new:.1f}x)")
print(f"   chunk dedup vs old + zstd -19:  {z_stored + z_manifest:>13,} B  ({len(new)/(z_stored+z_manifest):.1f}x; stored chunks {z_stored:,}, manifest {z_manifest:,})  -> {z_new/(z_stored+z_manifest):.1f}x smaller than alone")
print(f"   zstd -19 --patch-from old:      {len(patch):>13,} B  ({len(new)/len(patch):.1f}x)  -> {z_new/len(patch):.1f}x smaller than alone")
