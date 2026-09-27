"""glyd_gpu.cu's kernels from the prebuilt library (libglyd_gpu_cudaN.so,
build_lib.sh) through its C API: the pybind module's functions, by the same
names and arguments, for glyd_gpu.py where nvcc is not at hand.

    import glyd_gpu_lib as ext
    ext.load("libglyd_gpu_cuda13.so")
    ext.mma_gemm(...)                    # as the JIT-built module's

Each runs on the current stream of its tensors' device and allocates what
the C++ allocates: its outputs, a product's workspace (the bytes the library
asks for) and its done counters (zeroed once and kept, one set a device, as
there)."""
import contextlib
import ctypes
import torch

_lib = None
_P, _I64, _U64, _SZ, _W = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)
_PACK = [_P, _P, _P, _W]  # data, blocks (exc), block_base (exc_base), tiers[3] (sym[4])
_FAST = [_P, _P, _P, _P, _U64]  # sm, planes, exc, exc_base, top
_DENSE = [_P, _P, _I64, _P, _P]  # sm, stream (and its words), offs, tables
_ARGS = {  # each function's arguments before its stream
    "lane_bits": [_P, _I64, _P, _I64, _I64, _P],
    "write_codes": [_P, _I64, _P, _P, _P, _P, _I64, _I64],
    "decode": _DENSE + [_I64, _I64, _I64, _I64, _P, _I64, _P],
    "gemv": _DENSE + [_I64, _I64, _I64, _I64, _I64, _P, _P, _P, _P, _P],
    "fast_gemv": _FAST + [_I64, _I64, _P, _P, _P],
    "fast_decode": _FAST + [_I64, _I64, _P, _I64, _I64, _P],
    "fast_gemm": _FAST + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ],
    "fast_bgemv": _FAST + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ],
    "mma_gemm": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_gemm": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma_gemm_big": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ],
    "mma12_gemm_big": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ],
    "mma12_gemm_mid": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_gemm_wg": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma_unpack": _PACK + [_I64, _I64, _I64, _P],
    "mma12_unpack": _PACK + [_I64, _I64, _I64, _P],
    "attn_decode": [_P, _I64, _P, _P, _P, _W, _P, _P, _P, _W, _P, _P, _I64, _I64, _I64, _I64, ctypes.c_double, _P, _P, _SZ, _P],
}
_SIZES = {"fast_gemm": 3, "fast_bgemv": 3, "mma_gemm": 3, "mma12_gemm": 3, "mma_gemm_big": 4, "mma12_gemm_big": 4, "mma12_gemm_mid": 3, "mma12_gemm_wg": 3, "attn_decode": 4}  # their workspace queries' sizes


def load(path):
    """The library at path, for the functions below."""
    global _lib
    lib = ctypes.CDLL(path)
    for name, args in _ARGS.items():
        f = getattr(lib, "glyd_gpu_" + name)
        f.argtypes, f.restype = args + [_P], ctypes.c_int
    for name, n in _SIZES.items():
        f = getattr(lib, f"glyd_gpu_{name}_workspace")
        f.argtypes, f.restype = [_I64] * n + [ctypes.POINTER(_SZ)], ctypes.c_int
    lib.glyd_gpu_error_string.argtypes, lib.glyd_gpu_error_string.restype = [ctypes.c_int], ctypes.c_char_p
    lib.glyd_gpu_cuda_version.restype = ctypes.c_int
    _lib = lib


def cuda_version():
    """The CUDA runtime the library was built with, e.g. 13000."""
    return _lib.glyd_gpu_cuda_version()


def _check(ok, why):
    if not ok:
        raise RuntimeError(why)


def _fail(name, r):
    if r:
        raise RuntimeError(f"{name}: {_lib.glyd_gpu_error_string(r).decode()}")


# A small product's host time counts in generation (a call a Linear a token): the device switched only where
# it is not current, the stream read without building its Stream object, the sizes and words kept.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_here = contextlib.nullcontext()


def _on(t):
    """t's device made current for the call (as the C++'s CUDAGuard), where it is not already."""
    i = t.get_device()
    return _here if i == torch.cuda.current_device() else torch.cuda.device(i)


def _stream():
    """The current device's current stream: torch.cuda.current_stream().cuda_stream."""
    return _raw_stream(torch.cuda.current_device()) if _raw_stream else torch.cuda.current_stream().cuda_stream


def _call(name, *args):
    """glyd_gpu_<name>(args, the current stream), tensors as their data; the device made current by the caller."""
    _fail(name, getattr(_lib, "glyd_gpu_" + name)(*[a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args], _stream()))


_sizes = {}


def _workspace(name, dev, *sizes):
    """A product's workspace on dev (None where it needs none): the library's size for these sizes on this
    device (asked once)."""
    key = (name, dev.index, sizes)
    n = _sizes.get(key)
    if n is None:
        b = _SZ()
        _fail(name, getattr(_lib, f"glyd_gpu_{name}_workspace")(*sizes, ctypes.byref(b)))
        n = _sizes[key] = b.value
    return torch.empty(n, dtype=torch.uint8, device=dev) if n else None


def _bytes(ws):
    return ws.numel() if ws is not None else 0


def _opt(t):
    """A bias: none when empty."""
    return t if t.numel() else None


_done = {}


def _counters(name, like, n, least):
    """A product's done counters on like's device (at least n; least when first made): zero between products."""
    key = (name, like.get_device())
    t = _done.get(key)
    if t is None or t.numel() < n:
        t = _done[key] = torch.zeros(max(n, least), dtype=torch.int32, device=like.device)
    return t


_arrays = {}


def _words(v, n, what):
    """tiers (3) or sym (4) as the C API's words (a pack's are the same every call: kept)."""
    _check(len(v) == n, what)
    key = tuple(v)
    a = _arrays.get(key)
    if a is None:
        a = _arrays[key] = (ctypes.c_uint32 * n)(*[x & 0xFFFFFFFF for x in v])
    return a


def lane_bits(w, len_, tw, V):
    with _on(w):
        bits = torch.empty((w.numel() + tw - 1) // tw * 32, dtype=torch.int32, device=w.device)
        _call("lane_bits", w, w.numel(), len_, tw, V, bits)
    return bits


def write_codes(w, len_, code, offs, out, tw, V):
    with _on(w):
        _call("write_codes", w, w.numel(), len_, code, offs, out, tw, V)


def decode(sm, stream, offs, tables, n, tw, V, tile_words, tile_ids, out):
    ids = tile_ids.numel()
    _check(out.numel() >= (ids * tw if ids else n), "the output is too small")
    with _on(sm):
        _call("decode", sm, stream, stream.numel(), offs, tables, n, tw, V, tile_words, tile_ids, ids, out)


def gemv(sm, stream, offs, tables, O, K, tw, V, tile_words, x, bias, y, sum_, count):
    split = tw % K != 0
    _check(K % (32 * V) == 0 and tw % (32 * V) == 0 and (not split or sum_.numel() >= O), "gemv needs K and tiles multiples of 32 V, and row sums for split rows")
    with _on(sm):
        _call("gemv", sm, stream, stream.numel(), offs, tables, O, K, tw, V, tile_words, x, _opt(bias), y, sum_, count)


def fast_gemv(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 128 == 0, "rows a multiple of 128 long")
    with _on(sm):
        _call("fast_gemv", sm, planes, exc, exc_base, top, O, K, x, _opt(bias), y)


def fast_decode(sm, planes, exc, exc_base, top, row0, rows, row_ids, K, out):
    ids = row_ids.numel()
    _check(K % 128 == 0 and out.numel() >= (ids or rows) * K, "rows a multiple of 128 long, room for them")
    with _on(sm):
        _call("fast_decode", sm, planes, exc, exc_base, top, row0, rows, row_ids, ids, K, out)


def fast_gemm(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 64 == 0 and x.is_contiguous() and x.size(1) == K, "K a multiple of 64, X contiguous [M, K]")
    M = x.size(0)
    with _on(sm):
        ws = _workspace("fast_gemm", x.device, O, K, M)
        _call("fast_gemm", sm, planes, exc, exc_base, top, O, K, x, M, _opt(bias), y, ws, _bytes(ws))


def fast_bgemv(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    M = x.size(0)
    _check(K % 512 == 0 and x.is_contiguous() and x.size(1) == K and M in (2, 4, 8, 16), "K a multiple of 512, X contiguous [M, K], M 2, 4, 8 or 16")
    with _on(sm):
        ws = _workspace("fast_bgemv", x.device, O, K, M)
        _call("fast_bgemv", sm, planes, exc, exc_base, top, O, K, x, M, _opt(bias), y, ws, _bytes(ws))


def _small(name, pack, words, O, K, x, bias, y):
    """mma_gemm, mma12_gemm: up to 64 tokens."""
    M = x.size(0)
    _check(O % 64 == 0 and K % 16 == 0 and M <= 64 and x.is_contiguous() and x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]")
    with _on(pack[0]):
        ws = _workspace(name, x.device, O, K, M)
        _call(name, *pack, words, O, K, x, M, _opt(bias), y, ws, _bytes(ws), _counters(name, pack[0], O // 64, 1 << 16))


def mma_gemm(data, blocks, block_base, tiers, O, K, x, bias, y):
    _small("mma_gemm", (data, blocks, block_base), _words(tiers, 3, "three tiers"), O, K, x, bias, y)


def mma12_gemm(data, exc, exc_base, sym, O, K, x, bias, y):
    _small("mma12_gemm", (data, exc, exc_base), _words(sym, 4, "four words of symbols"), O, K, x, bias, y)


def _big(name, pack, words, O, K, x, bias, y, variant):
    """mma_gemm_big, mma12_gemm_big: a prompt."""
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    M = x.size(0)
    with _on(pack[0]):
        ws = _workspace(name, x.device, O, K, M, variant)
        _call(name, *pack, words, O, K, x, M, _opt(bias), y, variant, ws, _bytes(ws))


def mma_gemm_big(data, blocks, block_base, tiers, O, K, x, bias, y, variant):
    _big("mma_gemm_big", (data, blocks, block_base), _words(tiers, 3, "three tiers"), O, K, x, bias, y, variant)


def mma12_gemm_big(data, exc, exc_base, sym, O, K, x, bias, y, variant):
    _big("mma12_gemm_big", (data, exc, exc_base), _words(sym, 4, "four words of symbols"), O, K, x, bias, y, variant)


def _staged(name, data, exc, exc_base, sym, O, K, x, bias, y):
    """mma12_gemm_mid, mma12_gemm_wg: many tokens, the 12-bit layout copied a stage at a time (the GPU checked by the library)."""
    words = _words(sym, 4, "four words of symbols")
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    _check(data.data_ptr() % 16 == 0 and exc.data_ptr() % 16 == 0 and exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)")
    M = x.size(0)
    with _on(data):
        ws = _workspace(name, data.device, O, K, M)
        _call(name, data, exc, exc_base, words, O, K, x, M, _opt(bias), y, ws, _bytes(ws), _counters(name, data, O // 64, 1 << 16))


def mma12_gemm_mid(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_mid", data, exc, exc_base, sym, O, K, x, bias, y)


def mma12_gemm_wg(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_wg", data, exc, exc_base, sym, O, K, x, bias, y)


def mma_unpack(data, blocks, block_base, tiers, K, row0, rows, out):
    words = _words(tiers, 3, "three tiers")
    _check(row0 % 64 == 0 and rows % 64 == 0 and out.numel() >= rows * K, "rows a multiple of 64")
    with _on(data):
        _call("mma_unpack", data, blocks, block_base, words, K, row0, rows, out)


def mma12_unpack(data, exc, exc_base, sym, K, row0, rows, out):
    words = _words(sym, 4, "four words of symbols")
    _check(row0 % 64 == 0 and rows % 64 == 0 and out.numel() >= rows * K, "rows a multiple of 64")
    with _on(data):
        _call("mma12_unpack", data, exc, exc_base, words, K, row0, rows, out)


def attn_decode(q, kd, kb, kbb, kt, vd, vb, vbb, vt, tk, tv, tlen, pairs, G, P, scale, out):
    k3, v3 = _words(kt, 3, "three tiers"), _words(vt, 3, "three tiers")
    D = q.size(-1)
    _check(D in (64, 128) and 1 <= G <= 16 and q.is_contiguous() and tlen < 64 and P + (tlen > 0) > 0, "head_dim 64 or 128, up to 16 queries a KV head")
    with _on(q):
        ws = _workspace("attn_decode", q.device, D, tlen, pairs, P)
        _call("attn_decode", q, D, kd, kb, kbb, k3, vd, vb, vbb, v3, tk, tv, tlen, pairs, G, P, scale, out, ws, _bytes(ws), _counters("attn_decode", q, pairs, 1 << 12))
