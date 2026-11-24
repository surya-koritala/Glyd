"""The FP8 kernels (gpu/glyd_gpu_kernels.h, which glyd_gpu.cu includes) on the CPU (emu_f8.cpp: thread for thread, no GPU), as the backend of check_f8.py's --emu: the kernels' layout and
logic checked against fp8ref.py on a machine with no GPU. `Emu()` builds the library into a temporary directory from the text of glyd_gpu_kernels.h
between its `f8c begin` and `f8c end` lines, and `pack`, `unpack` and `experts` run the kernels' threads.

    python3 emu.py      the cases of check_f8.py's smoke test through it"""
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "..", "glyd_gpu_kernels.h")  # (the kernels' source the library and the engine both compile: R17)
GUARD = 8
HEAD = 8


def aligned(n, dtype=np.uint8):
    """A zeroed array of n elements whose data is 64-byte aligned (a cudaMalloc's are 256)."""
    raw = np.zeros(n * np.dtype(dtype).itemsize + 64, np.uint8)
    off = (-raw.ctypes.data) % 64
    return raw[off : off + n * np.dtype(dtype).itemsize].view(dtype)


class Emu:
    def __init__(self, build=None):
        text = open(CU).read()
        m = re.search(r"^// f8c begin\n(.*?)^// f8c end\n", text, flags=re.S | re.M)
        assert m, "glyd_gpu_kernels.h has no f8c block"
        self.dir = build or tempfile.mkdtemp(prefix="f8emu")
        open(os.path.join(self.dir, "f8_block.inc"), "w").write(m.group(1))
        so = os.path.join(self.dir, "libemu.so")
        cxx = shutil.which("clang++") or shutil.which("g++") or "c++"
        cmd = [cxx, "-std=c++20", "-O1", "-pthread", "-fno-strict-aliasing", "-Wall", "-Wno-unused-function", "-shared", "-fPIC", "-I", self.dir, os.path.join(HERE, "emu_f8.cpp"), "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("the emulation does not build:\n" + r.stderr[-4000:])
        if r.stderr.strip():
            print(r.stderr[-2000:], file=sys.stderr)
        lib = ctypes.CDLL(so)
        P, I64, I = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
        lib.emu_size.argtypes = [P, I64, I64, I64, P, P, I]
        lib.emu_pack.argtypes = [P, I64, I64, I64, P, P, P, I64, I]
        lib.emu_unpack.argtypes = [P, P, P, I64, I64, I64, P, I64, P, I64, I]
        lib.emu_linear.argtypes = [P, P, P, I64, I64, I, P, I64, P, I64, I64, P, I64, P, P, I, I, I, I, I]
        self.lib = lib
        self.vec = -1  # 1 or 0: 16-byte accesses forced on or off (-1: as the library chooses)

    def pack(self, w):
        """w [O, K] uint8 (rows of any stride) -> (core uint8 [patches, 6144], heads uint32 [patches, 8], var uint32 [words])."""
        O, K = w.shape
        assert w.dtype == np.uint8 and w.strides[1] == 1
        ld = w.strides[0]  # (copied into an aligned buffer with its own row stride)
        buf = aligned(ld * (O - 1) + K)
        np.lib.stride_tricks.as_strided(buf, (O, K), (ld, 1))[:] = w
        patches = -(-O // 64) * -(-K // 128)
        heads = aligned(patches * HEAD, np.uint32)
        words = ctypes.c_int64()
        self.lib.emu_size(buf.ctypes.data, ld, O, K, heads.ctypes.data, ctypes.byref(words), self.vec)
        core, var = aligned(patches * 6144), aligned(words.value, np.uint32)
        self.lib.emu_pack(buf.ctypes.data, ld, O, K, heads.ctypes.data, core.ctypes.data, var.ctypes.data, words.value, self.vec)
        return core.reshape(patches, 6144), heads.reshape(patches, HEAD), var

    def unpack(self, core, heads, var, shape, ld=None):
        """The matrix [O, K] back (rows ld bytes apart in the buffer the kernel writes; the bytes between rows are 0xA5 and must not change)."""
        O, K = shape
        ld = ld or K
        core, heads, var = (aligned_copy(x) for x in (core, heads, var))
        out = aligned(ld * O)
        out[:] = 0xA5
        self.lib.emu_unpack(core.ctypes.data, heads.ctypes.data, var.ctypes.data, 1, O, K, None, 1, out.ctypes.data, ld, self.vec)
        m = out.reshape(O, ld)
        assert (m[:, K:] == 0xA5).all(), "bytes between the rows were written"
        return m[:, :K].copy()

    def experts(self, core, heads, var, E, O, K, ids, ld=None):
        """The matrices ids of a stack of E [O, K] -> [n, O, K] (a place whose id is outside [0, E) stays 0xA5)."""
        ld = ld or K
        core, heads, var = (aligned_copy(x) for x in (core, heads, var))
        ids = np.ascontiguousarray(ids, np.int32)
        out = aligned(len(ids) * O * ld)
        out[:] = 0xA5
        self.lib.emu_unpack(core.ctypes.data, heads.ctypes.data, var.ctypes.data, E, O, K, ids.ctypes.data, len(ids), out.ctypes.data, ld, self.vec)
        return out.reshape(len(ids), O, ld)[:, :, :K].copy()


    def linear(self, core, heads, var, N, K, kind, scale, x, ntl, ctas, ahead=None, dense=0, a8=0):
        """glyd_gpu_f8_linear's kernel with the plan given: x uint16 (bf16 bits) [M, K], scale as the kind holds it (lds: its rows' length) -> y uint16 [M, N]
        (a8: X as e4m3, kinds 0, 1 and 4)."""
        M = x.shape[0]
        core, heads, var = (aligned_copy(a) for a in (core, heads, var))
        xa = aligned_copy(np.ascontiguousarray(x, np.uint16)).reshape(M, K)
        sc = np.ascontiguousarray(scale)
        lds = sc.shape[-1] if sc.ndim == 2 else 1
        sca = aligned_copy(sc)
        y = aligned(M * N, np.uint16)
        y[:] = 0xFFFF
        parts = aligned(max(1, ctas * 2 * M * 128), np.float32)
        RG = -(-(-(-N // 64) * 4) // 8)
        done = aligned(RG, np.int32)
        self.lib.emu_linear(core.ctypes.data, heads.ctypes.data, var.ctypes.data, N, K, kind, sca.ctypes.data, lds, xa.ctypes.data, K, M, y.ctypes.data, N,
                            parts.ctypes.data, done.ctypes.data, ntl, ctas, 0 if ahead is None else int(ahead), int(dense), int(a8))
        assert (done == 0).all(), "a counter left past its product"
        return y.reshape(M, N)


def aligned_copy(x):
    a = aligned(x.size, x.dtype)
    a[:] = x.reshape(-1)
    return a


if __name__ == "__main__":
    import time

    sys.path.insert(0, os.path.join(HERE, "..", "..", "glyd-gpu-py", "glyd_gpu"))
    import fp8ref

    rng = np.random.default_rng(0)
    e = Emu()
    for shape in ((1, 1), (17, 33), (64, 128), (70, 200)):
        w = rng.integers(0, 256, shape, dtype=np.uint8)
        w = ((w & 0x87) | (rng.integers(5, 12, shape) << 3)).astype(np.uint8)  # a few exponents: coded
        t = time.time()
        core, heads, var = e.pack(w)
        ref = fp8ref.pack(w)
        assert np.array_equal(core, ref.core) and np.array_equal(heads, ref.heads) and np.array_equal(var, ref.var), shape
        assert np.array_equal(e.unpack(core, heads, var, shape), w)
        print(shape, "ok", f"{time.time() - t:.1f} s")
