"""A safetensors file read through numpy's memmap: no torch, any dtype as raw bytes.

    f = ST(path); f.keys(); f.u8(name) -> (uint8 array of the stored bytes' dtype-sized words, dtype string)
bf16 comes back as uint16, fp8 as uint8, fp4's packed bytes as uint8, scales as uint8 or float32.
"""
import json, struct
import numpy as np

NP = {"U8": np.uint8, "I8": np.int8, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8, "BF16": np.uint16, "F16": np.float16,
      "F32": np.float32, "I64": np.int64, "I32": np.int32, "BOOL": np.bool_}


class ST:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return list(self.hdr)

    def info(self, name):
        h = self.hdr[name]
        return h["dtype"], tuple(h["shape"])

    def get(self, name):
        h = self.hdr[name]
        a, b = h["data_offsets"]
        return self.mm[self.base + a:self.base + b].view(NP[h["dtype"]]).reshape(h["shape"])
