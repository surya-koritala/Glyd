"""What a generic compressor (zstd -19, xz -6) gets on the raw bytes of a few real tensors, for scale: the entropy
numbers are what a model of the codes can reach; a byte-oriented compressor sees none of it.

    python generic.py OUT.json  LABEL=FILE.safetensors::TENSOR ...

Each tensor's stored bytes (the first 64 MB of it) compressed whole, as a share of the stored size.
"""
import json, lzma, subprocess, sys
import numpy as np
from st import ST


def zstd19(b):
    return len(subprocess.run(["zstd", "-19", "--long=27", "-q", "-c", "-T0"], input=b, capture_output=True, check=True).stdout)


if __name__ == "__main__":
    rows = []
    for spec in sys.argv[2:]:
        label, rest = spec.split("=", 1)
        path, name = rest.split("::")
        b = np.asarray(ST(path).get(name)).tobytes()[:64 << 20]
        rows.append({"tensor": label, "bytes": len(b), "zstd19_pct": round(100 * zstd19(b) / len(b), 2), "xz6_pct": round(100 * len(lzma.compress(b, preset=6)) / len(b), 2)})
        print(rows[-1], flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
