#!/usr/bin/env python3
"""Lever: model checkpoints. Sizes of a safetensors file (fp32 or bf16
weights) as it is and through lossless transforms — byte planes (the k-th
byte of every element together), the XOR with the previous checkpoint of
the same run — under zstd -3/-19 and Glyd --max; plus Glyd base mode on
the raw pair. What a tensor-aware level would gain, before writing one.
  experiments/research/tensors.py corpus/ai/a.safetensors [corpus/ai/b.safetensors ...]
Consecutive files are treated as consecutive checkpoints."""
import json, os, subprocess, struct, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLYD = os.path.join(ROOT, "target/release/glyd")

def tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = np.fromfile(f, dtype=np.uint8)
    return header, data

def compressed_size(cmd, data):
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(data.tobytes() if hasattr(data, "tobytes") else data)
        name = t.name
    try:
        t0 = time.time()
        out = subprocess.run(cmd + [name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        return len(out), time.time() - t0
    finally:
        os.unlink(name)

def planes(data, width):
    """Byte k of every element, planes concatenated."""
    a = data[: len(data) // width * width].reshape(-1, width)
    return np.ascontiguousarray(a.T).reshape(-1)

def report(name, raw, variants):
    print(f"== {name} ({len(raw)} bytes)")
    for label, data in variants:
        row = [f"   {label:<34}"]
        for codec, cmd in [("zstd -3", ["zstd", "-3", "-T10", "-c"]), ("zstd -19", ["zstd", "-19", "-T10", "-c"]), ("Glyd --max", [GLYD, "--max", "-c"])]:
            size, s = compressed_size(cmd, data)
            row.append(f"{codec} {len(raw)/size:5.2f}x ({size/1e6:7.1f} MB, {len(raw)/s/1e6:5.0f} MB/s)")
        print("  ".join(row))

files = sys.argv[1:]
prev = None
for path in files:
    header, data = tensors(path)
    dtypes = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + (v["data_offsets"][1] - v["data_offsets"][0])
    main = max(dtypes, key=dtypes.get)
    width = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}[main]
    print(f"{os.path.basename(path)}: {dtypes} -> {main}, {width} bytes per element")
    variants = [("as it is", data), (f"byte planes ({width})", planes(data, width))]
    if prev is not None and len(prev) == len(data):
        x = np.bitwise_xor(prev, data)
        same = np.count_nonzero(np.frombuffer(x.tobytes(), dtype=np.uint8 if width == 1 else {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]) == 0)
        print(f"   elements identical to the previous checkpoint: {same / (len(data) // width) * 100:.1f}%")
        variants.append(("xor with previous checkpoint", x))
        variants.append(("xor, byte planes", planes(x, width)))
    report(os.path.basename(path), data, variants)
    if prev is not None and len(prev) == len(data):
        with tempfile.NamedTemporaryFile(delete=False) as a, tempfile.NamedTemporaryFile(delete=False) as b:
            a.write(prev.tobytes()); b.write(data.tobytes())
        t0 = time.time()
        out = subprocess.run([GLYD, "--base", a.name, b.name, "-c"], stdout=subprocess.PIPE).stdout
        print(f"   Glyd --max --base (previous checkpoint as base): {len(data)/len(out):5.2f}x ({len(out)/1e6:.1f} MB, {len(data)/(time.time()-t0)/1e6:.0f} MB/s)")
        os.unlink(a.name); os.unlink(b.name)
    prev = data
