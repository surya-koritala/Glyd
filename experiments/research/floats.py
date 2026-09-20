#!/usr/bin/env python3
"""Lever F: binary float columns (data lakes). Double columns read from
the taxi Parquet file, as raw little-endian f64 arrays, compressed with
zstd -19 (what Parquet's zstd codec sees, its PLAIN pages) against a
decimal-aware coding: the values as scaled integers (a column of 2-place
dollars is an int32 of cents) delta-coded, then zstd -19; and the same
integers with a simple bit-packing proxy. Byte-exact: the doubles are
rebuilt and compared."""
import struct, subprocess, sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
def z(b): return len(subprocess.run(['zstd', '-q', '-19', '-T1', '-c'], input=b, capture_output=True).stdout)
def varint(out, v):
    while v >= 128: out.append((v & 127) | 128); v >>= 7
    out.append(v)
def zig(v): return (v << 1) ^ (v >> 63) if v < 0 else v << 1
n = 2_000_000
for col in ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'passenger_count']:
    if col not in t.column_names: continue
    vals = t.column(col).to_pylist()[:n]
    vals = [v for v in vals if v is not None]
    raw = struct.pack(f'<{len(vals)}d', *vals)
    z_raw = z(raw)
    # decimal scaling: find the smallest scale that reproduces every value exactly
    scale = None
    for s in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        if all(abs(round(v * s) / s - v) < 1e-12 for v in vals[:200000]):
            scale = s; break
    if scale is None:
        print(f"{col:16} not decimal (skipped)"); continue
    ints = [round(v * scale) for v in vals]
    ok = all(i / scale == v for i, v in zip(ints, vals))
    delta = bytearray(); last = 0
    for i in ints: varint(delta, zig(i - last)); last = i
    fixed = struct.pack(f'<{len(ints)}i', *ints)
    z_delta = z(bytes(delta)); z_fixed = z(fixed)
    print(f"{col:16} {len(vals):>9} values  f64+zstd-19 {z_raw:>10} B ({8*len(vals)/z_raw:5.2f}x)  scaled int32+zstd-19 {z_fixed:>9} B  zigzag-delta+zstd-19 {z_delta:>9} B  -> best {z_raw/min(z_fixed, z_delta):.2f}x smaller  exact={ok}")
