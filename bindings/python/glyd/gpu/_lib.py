"""gpu/glyd_gpu.cu's kernels from the prebuilt library (libglyd_gpu_cudaN.so,
gpu/build_lib.sh) through its C API: the pybind module's functions, by the
same names and arguments, for kernels.py where nvcc is not at hand.

    from glyd.gpu import _lib as ext
    ext.load("libglyd_gpu_cuda13.so")
    ext.mma_gemm(...)                    # as the JIT-built module's

Each runs on the current stream of its tensors' device and allocates what
the C++ allocates: its outputs, and a product's done counters (zeroed once
and kept, a set a device, as there). A product's workspace (the bytes the
library asks for) is kept for its stream and reused in stream order, up to
16 MB (a prompt's larger one is allocated for the call, as there). A
generation step calls a product for every Linear, so a call's host time is
its C call and a few lookups: no Stream object, no device switch where the
device is current, no allocation."""
import ctypes
import functools
import threading
import torch

_lib = None
_fn, _query = {}, {}  # the C API's functions, argument types set; the workspace queries
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
    "mma_gemm_big": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "mma12_gemm_big": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "mma12_gemm_mid": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_gemm_wg": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "hold": [_I64],
    "mma_unpack": _PACK + [_I64, _I64, _I64, _P, _I64],
    "mma12_unpack": _PACK + [_I64, _I64, _I64, _P, _I64],
    "attn_decode": [_P, _I64, _P, _P, _P, _W, _P, _P, _P, _W, _P, _P, _I64, _I64, _I64, _I64, ctypes.c_double, _P, _P, _SZ, _P],
    "moe_route": [_P, _I64, _I64, _P],
    "mma_moe_unpack": _PACK + [_I64, _I64, _I64, _I64, _P, _P],
    "mma12_moe_unpack": _PACK + [_I64, _I64, _I64, _I64, _P, _P],
    "mma_moe": _PACK + [_I64, _I64, _I64, _P, _I64, _I64, _I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_moe": _PACK + [_I64, _I64, _I64, _P, _I64, _I64, _I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _SZ, _P],
}
_SIZES = {"fast_gemm": 3, "fast_bgemv": 3, "mma_gemm": 3, "mma12_gemm": 3, "mma_gemm_big": 4, "mma12_gemm_big": 4, "mma12_gemm_mid": 3, "mma12_gemm_wg": 3, "attn_decode": 4, "mma_moe": 7, "mma12_moe": 7}  # their workspace queries' sizes


def load(path):
    """The library at path, for the functions below."""
    global _lib
    lib = ctypes.CDLL(path)
    for name, args in _ARGS.items():
        f = _fn[name] = getattr(lib, "glyd_gpu_" + name)
        f.argtypes, f.restype = args + [_P], ctypes.c_int
    for name, n in _SIZES.items():
        f = _query[name] = getattr(lib, f"glyd_gpu_{name}_workspace")
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
    raise RuntimeError(f"{name}: {_lib.glyd_gpu_error_string(r).decode()}")


_device = torch._C._cuda_getDevice
# A device's current stream, torch.cuda.current_stream(d).cuda_stream, read without making a Stream object.
_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None) or (lambda d: torch.cuda.current_stream(d).cuda_stream)


def _there(f, d, *args):
    """f(*args) with device d made current (as the C++'s CUDAGuard), where it was not."""
    with torch.cuda.device(d):
        return f(*args)


@functools.lru_cache(maxsize=1024)
def _need(name, d, sizes):
    """A product's workspace bytes for these sizes on device d (the current one), as the library gives them."""
    b = _SZ()
    r = _query[name](*sizes, ctypes.byref(b))
    if r:
        _fail(name, r)
    return b.value


_kept = {}  # (device, stream): a workspace kept for the stream, (buffer, address, bytes)
_KEEP = 16 << 20
_NONE = (None, None, 0)


class _Local(threading.local):
    fresh = False  # a torch.compile graph's node runs on this thread (model.py's ops): workspaces for the call alone, never kept


local = _Local()


def _workspace(name, d, s, *sizes):
    """A product's workspace on device d for stream s, as (buffer, address, bytes): the library's size
    for these sizes; the stream's kept buffer, reused in stream order (grown where too small), up to
    _KEEP bytes, else (or while local.fresh) one for the call; none for 0 bytes. A CUDA graph keeps the
    addresses it was captured with, so its calls must not take a buffer that is later replaced, nor
    keep one made in its memory pool."""
    n = _need(name, d, sizes)
    if n == 0:
        return _NONE
    if n > _KEEP or local.fresh:
        t = torch.empty(n, dtype=torch.uint8, device=torch.device("cuda", d))
        return t, t.data_ptr(), n
    w = _kept.get((d, s))
    if w is None or w[2] < n:
        t = torch.empty(n, dtype=torch.uint8, device=torch.device("cuda", d))
        w = _kept[(d, s)] = (t, t.data_ptr(), n)
    return w


_done, _replaced = {}, []


def _counters(name, d, n, least):
    """The address of a product's done counters on device d (at least n; least when first made): zero
    between products. One replaced by a bigger set is kept (a step's call, or a CUDA graph, holds its
    address)."""
    c = _done.get((name, d))
    if c is None or c[2] < n:
        if c is not None:
            _replaced.append(c)
        t = torch.zeros(max(n, least), dtype=torch.int32, device=torch.device("cuda", d))
        c = _done[(name, d)] = (t, t.data_ptr(), t.numel())
    return c[1]


_arrays = {}


def _words(v, n, what):
    """tiers (3) or sym (4) as the C API's words (a pack's are the same every call: kept)."""
    key = (n, *v)
    a = _arrays.get(key)
    if a is None:
        _check(len(v) == n, what)
        a = _arrays[key] = (ctypes.c_uint32 * n)(*[x & 0xFFFFFFFF for x in v])
    return a


def lane_bits(w, len_, tw, V):
    d = w.get_device()
    if d != _device():
        return _there(lane_bits, d, w, len_, tw, V)
    bits = torch.empty((w.numel() + tw - 1) // tw * 32, dtype=torch.int32, device=w.device)
    r = _fn["lane_bits"](w.data_ptr(), w.numel(), len_.data_ptr(), tw, V, bits.data_ptr(), _stream(d))
    if r:
        _fail("lane_bits", r)
    return bits


def write_codes(w, len_, code, offs, out, tw, V):
    d = w.get_device()
    if d != _device():
        return _there(write_codes, d, w, len_, code, offs, out, tw, V)
    r = _fn["write_codes"](w.data_ptr(), w.numel(), len_.data_ptr(), code.data_ptr(), offs.data_ptr(), out.data_ptr(), tw, V, _stream(d))
    if r:
        _fail("write_codes", r)


def decode(sm, stream, offs, tables, n, tw, V, tile_words, tile_ids, out):
    d = sm.get_device()
    if d != _device():
        return _there(decode, d, sm, stream, offs, tables, n, tw, V, tile_words, tile_ids, out)
    ids = tile_ids.numel()
    _check(out.numel() >= (ids * tw if ids else n), "the output is too small")
    r = _fn["decode"](sm.data_ptr(), stream.data_ptr(), stream.numel(), offs.data_ptr(), tables.data_ptr(), n, tw, V, tile_words, tile_ids.data_ptr(), ids, out.data_ptr(), _stream(d))
    if r:
        _fail("decode", r)


def gemv(sm, stream, offs, tables, O, K, tw, V, tile_words, x, bias, y, sum_, count):
    d = sm.get_device()
    if d != _device():
        return _there(gemv, d, sm, stream, offs, tables, O, K, tw, V, tile_words, x, bias, y, sum_, count)
    split = tw % K != 0
    _check(K % (32 * V) == 0 and tw % (32 * V) == 0 and (not split or sum_.numel() >= O), "gemv needs K and tiles multiples of 32 V, and row sums for split rows")
    r = _fn["gemv"](sm.data_ptr(), stream.data_ptr(), stream.numel(), offs.data_ptr(), tables.data_ptr(), O, K, tw, V, tile_words, x.data_ptr(), bias.data_ptr() if bias.numel() else None, y.data_ptr(), sum_.data_ptr(), count.data_ptr(), _stream(d))
    if r:
        _fail("gemv", r)


def fast_gemv(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    d = sm.get_device()
    if d != _device():
        return _there(fast_gemv, d, sm, planes, exc, exc_base, top, O, K, x, bias, y)
    _check(K % 128 == 0, "rows a multiple of 128 long")
    r = _fn["fast_gemv"](sm.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, O, K, x.data_ptr(), bias.data_ptr() if bias.numel() else None, y.data_ptr(), _stream(d))
    if r:
        _fail("fast_gemv", r)


def fast_decode(sm, planes, exc, exc_base, top, row0, rows, row_ids, K, out):
    d = sm.get_device()
    if d != _device():
        return _there(fast_decode, d, sm, planes, exc, exc_base, top, row0, rows, row_ids, K, out)
    ids = row_ids.numel()
    _check(K % 128 == 0 and out.numel() >= (ids or rows) * K, "rows a multiple of 128 long, room for them")
    r = _fn["fast_decode"](sm.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, row0, rows, row_ids.data_ptr(), ids, K, out.data_ptr(), _stream(d))
    if r:
        _fail("fast_decode", r)


def _fast_product(name, sm, planes, exc, exc_base, top, O, K, x, bias, y):
    """fast_gemm, fast_bgemv: several tokens (the checks done)."""
    d = sm.get_device()
    if d != _device():
        return _there(_fast_product, d, name, sm, planes, exc, exc_base, top, O, K, x, bias, y)
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](sm.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], s)
    if r:
        _fail(name, r)


def fast_gemm(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 64 == 0 and x.is_contiguous() and x.size(1) == K, "K a multiple of 64, X contiguous [M, K]")
    _fast_product("fast_gemm", sm, planes, exc, exc_base, top, O, K, x, bias, y)


def fast_bgemv(sm, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 512 == 0 and x.is_contiguous() and x.size(1) == K and x.size(0) in (2, 4, 8, 16), "K a multiple of 512, X contiguous [M, K], M 2, 4, 8 or 16")
    _fast_product("fast_bgemv", sm, planes, exc, exc_base, top, O, K, x, bias, y)


def _small(name, data, a, b, words, O, K, x, bias, y):
    """mma_gemm, mma12_gemm: up to 64 tokens (a generation step's product)."""
    d = data.get_device()
    if d != _device():
        return _there(_small, d, name, data, a, b, words, O, K, x, bias, y)
    M = x.size(0)
    _check(O % 64 == 0 and K % 16 == 0 and M <= 64 and x.is_contiguous() and x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]")
    s = _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], _counters(name, d, O // 64, 1 << 16), s)
    if r:
        _fail(name, r)


def mma_gemm(data, blocks, block_base, tiers, O, K, x, bias, y):
    _small("mma_gemm", data, blocks, block_base, _words(tiers, 3, "three tiers"), O, K, x, bias, y)


def mma12_gemm(data, exc, exc_base, sym, O, K, x, bias, y):
    _small("mma12_gemm", data, exc, exc_base, _words(sym, 4, "four words of symbols"), O, K, x, bias, y)


def _units(O, M):
    """A prompt's product's done counters: its units (tiles of 128 or 256 tokens by pairs of row blocks) at most."""
    return (M + 127) // 128 * (O // 64)


_UNITS = 1 << 18  # a prompt's product's done counters when first made: to some 100K tokens of the largest matrices none made again (in a CUDA graph's capture)


def _big(name, data, a, b, words, O, K, x, bias, y, variant):
    """mma_gemm_big, mma12_gemm_big: a prompt."""
    d = data.get_device()
    if d != _device():
        return _there(_big, d, name, data, a, b, words, O, K, x, bias, y, variant)
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M, variant)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), variant, ws[1], ws[2], _counters(name, d, _units(O, M), _UNITS), s)
    if r:
        _fail(name, r)


def mma_gemm_big(data, blocks, block_base, tiers, O, K, x, bias, y, variant):
    _big("mma_gemm_big", data, blocks, block_base, _words(tiers, 3, "three tiers"), O, K, x, bias, y, variant)


def mma12_gemm_big(data, exc, exc_base, sym, O, K, x, bias, y, variant):
    _big("mma12_gemm_big", data, exc, exc_base, _words(sym, 4, "four words of symbols"), O, K, x, bias, y, variant)


def _staged(name, data, exc, exc_base, sym, O, K, x, bias, y):
    """mma12_gemm_mid, mma12_gemm_wg: many tokens, the 12-bit layout copied a stage at a time (the GPU checked by the library)."""
    d = data.get_device()
    if d != _device():
        return _there(_staged, d, name, data, exc, exc_base, sym, O, K, x, bias, y)
    words = _words(sym, 4, "four words of symbols")
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    _check(data.data_ptr() % 16 == 0 and exc.data_ptr() % 16 == 0 and exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)")
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](data.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], _counters(name, d, O // 64, 1 << 16), s)
    if r:
        _fail(name, r)


def mma12_gemm_mid(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_mid", data, exc, exc_base, sym, O, K, x, bias, y)


def mma12_gemm_wg(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_wg", data, exc, exc_base, sym, O, K, x, bias, y)


def hold(ns):
    """The current stream held ns nanoseconds."""
    r = _fn["hold"](ns, _stream(_device()))
    if r:
        _fail("hold", r)


def _unpack(name, data, a, b, words, K, row0, rows, out, warps):
    """mma_unpack, mma12_unpack."""
    d = data.get_device()
    if d != _device():
        return _there(_unpack, d, name, data, a, b, words, K, row0, rows, out, warps)
    _check(row0 % 64 == 0 and rows % 64 == 0 and out.numel() >= rows * K, "rows a multiple of 64")
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, K, row0, rows, out.data_ptr(), warps, _stream(d))
    if r:
        _fail(name, r)


def mma_unpack(data, blocks, block_base, tiers, K, row0, rows, out, warps):
    _unpack("mma_unpack", data, blocks, block_base, _words(tiers, 3, "three tiers"), K, row0, rows, out, warps)


def mma12_unpack(data, exc, exc_base, sym, K, row0, rows, out, warps):
    _unpack("mma12_unpack", data, exc, exc_base, _words(sym, 4, "four words of symbols"), K, row0, rows, out, warps)


def attn_decode(q, kd, kb, kbb, kt, vd, vb, vbb, vt, tk, tv, tlen, pairs, G, P, scale, out):
    d = q.get_device()
    if d != _device():
        return _there(attn_decode, d, q, kd, kb, kbb, kt, vd, vb, vbb, vt, tk, tv, tlen, pairs, G, P, scale, out)
    k3, v3 = _words(kt, 3, "three tiers"), _words(vt, 3, "three tiers")
    D = q.size(-1)
    _check(D in (64, 128) and 1 <= G <= 16 and q.is_contiguous() and tlen < 64 and P + (tlen > 0) > 0, "head_dim 64 or 128, up to 16 queries a KV head")
    s = _stream(d)
    ws = _workspace("attn_decode", d, s, D, tlen, pairs, P)
    r = _fn["attn_decode"](q.data_ptr(), D, kd.data_ptr(), kb.data_ptr(), kbb.data_ptr(), k3, vd.data_ptr(), vb.data_ptr(), vbb.data_ptr(), v3, tk.data_ptr(), tv.data_ptr(), tlen, pairs, G, P, scale, out.data_ptr(), ws[1], ws[2], _counters("attn_decode", d, pairs, 1 << 12), s)
    if r:
        _fail("attn_decode", r)


def step(data, a, b, words, n_words, shape, bias, names, big=None, big_max=0):
    """A generation step's product over one pack in the mma layouts as one C call: what does not change between
    calls made once (the pack's addresses and words, O and K, the bias; each M's function, workspace bytes and done
    counters at its first call), the checks that hold by the pack's making left out. words: its n_words tiers (3)
    or words of symbols (4). names[M]: the function for M tokens (mma_gemm, mma12_gemm, mma12_gemm_mid,
    mma12_gemm_wg), or None; big: a prompt's (mma_gemm_big, mma12_gemm_big, its own blocks of tokens) past them to
    big_max tokens, or None. run(x): Y [..., O] for X contiguous [..., K] of M rows on the pack's device (made
    current for the call where it is not: a layer on another GPU), where a function takes M; else None (the checked
    path)."""
    O, K = shape
    d, dev = data.get_device(), data.device
    head = (data.data_ptr(), a.data_ptr(), b.data_ptr(), _words(words, n_words, "three tiers or four words of symbols"), O, K)
    bias = bias.data_ptr() if bias is not None else None
    for name in set(names) - {None}:
        _counters(name, d, O // 64, 1 << 16)  # made now: never in a CUDA graph's memory pool
    if big:
        _counters(big, d, 0, _UNITS)
    plans = [None] * len(names)
    bf16 = torch.bfloat16
    prompt = _fn[big] if big else None

    def run(x):
        if x.shape[-1] != K or not x.is_contiguous() or x.get_device() != d:
            return None
        if _device() != d:  # accelerate's device map leaves the first GPU current
            with torch.cuda.device(d):
                return run(x)
        M = x.numel() // K
        if M >= len(plans) and M < big_max:  # a prompt
            s = _stream(d)
            w = _workspace(big, d, s, O, K, M, 0)
            y = torch.empty(*x.shape[:-1], O, dtype=bf16, device=dev)
            r = prompt(*head, x.data_ptr(), M, bias, y.data_ptr(), 0, w[1], w[2], _counters(big, d, _units(O, M), _UNITS), s)
            if r:
                _fail(big, r)
            return y
        plan = plans[M] if M < len(plans) else False
        if plan is None:
            name = names[M]
            plan = plans[M] = name is not None and (_fn[name], name, _need(name, d, (O, K, M)), _counters(name, d, O // 64, 1 << 16))
        if not plan:
            return None
        fn, name, need, done = plan
        s = _stream(d)
        w = _kept.get((d, s))
        if local.fresh or w is None or w[2] < need:
            w = _workspace(name, d, s, O, K, M)
        y = torch.empty(*x.shape[:-1], O, dtype=bf16, device=dev)
        r = fn(*head, x.data_ptr(), M, bias, y.data_ptr(), w[1], w[2], done, s)
        if r:
            _fail(name, r)
        return y

    return run


def lookup(sm, planes, exc, exc_base, top, K):
    """An embedding's lookup in the fast format as one C call (fast_decode): run(ids) -> its rows, bf16
    [*ids.shape, K], for ids int64 and contiguous on the pack's device (made current for the call where it is
    not); else None."""
    d, dev, i64 = sm.get_device(), sm.device, torch.int64
    fn, head = _fn["fast_decode"], (sm.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, 0, 0)

    def run(ids):
        n = ids.numel()
        if not n or ids.dtype is not i64 or not ids.is_contiguous() or ids.get_device() != d:
            return None
        if _device() != d:
            with torch.cuda.device(d):
                return run(ids)
        out = torch.empty(*ids.shape, K, dtype=torch.bfloat16, device=dev)
        r = fn(*head, ids.data_ptr(), n, K, out.data_ptr(), _stream(d))
        if r:
            _fail("fast_decode", r)
        return out

    return run


def moe_route(ids, E, plan):
    d = ids.get_device()
    if d != _device():
        return _there(moe_route, d, ids, E, plan)
    _check(ids.dtype == torch.int64 and ids.is_contiguous() and plan.dtype == torch.int32 and plan.numel() >= 2 + 2 * E + ids.numel() and plan.get_device() == d, "ids int64, plan int32 [2 + 2E + P] on the same GPU")
    r = _fn["moe_route"](ids.data_ptr(), ids.numel(), E, plan.data_ptr(), _stream(d))
    if r:
        _fail("moe_route", r)


def _moe(name, data, a, b, words, E, O, K, x, k, gather, plan, act, bias, w, ids, y):
    """mma_moe, mma12_moe: a mixture-of-experts layer's product (w, ids: empty for none)."""
    d = data.get_device()
    if d != _device():
        return _there(_moe, d, name, data, a, b, words, E, O, K, x, k, gather, plan, act, bias, w, ids, y)
    weighted = w.numel() > 0
    _check(x.is_contiguous() and x.size(1) == K and plan.dtype == torch.int32, "X contiguous [., K], plan int32")
    _check(not weighted or (w.dtype in (torch.float32, torch.bfloat16) and ids.dtype == torch.int64), "weights bf16 or fp32, ids int64")
    _check(not bias.numel() or (bias.dtype == torch.bfloat16 and bias.is_contiguous() and bias.numel() >= E * O and bias.get_device() == d), "bias bf16, contiguous [E, O], on the pack's GPU")
    _check(x.get_device() == d and plan.get_device() == d and y.get_device() == d and (not weighted or (w.get_device() == d and ids.get_device() == d)), "every tensor on the pack's GPU")
    T, s = x.size(0) if gather else x.size(0) // k, _stream(d)
    ws = _workspace(name, d, s, E, O, K, T, k, act, int(weighted))
    done = _counters(name, d, (O // 128 if act else O // 64) * min(E, T * k), 1 << 16)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, E, O, K, x.data_ptr(), T, k, gather, plan.data_ptr(), act, bias.data_ptr() if bias.numel() else None, w.data_ptr() if weighted else None, int(w.dtype == torch.float32), ids.data_ptr() if weighted else None, y.data_ptr(), ws[1], ws[2], done, s)
    if r:
        _fail(name, r)


def mma_moe(data, blocks, block_base, tiers, E, O, K, x, k, gather, plan, act, bias, w, ids, y):
    _moe("mma_moe", data, blocks, block_base, _words(tiers, 3, "three tiers"), E, O, K, x, k, gather, plan, act, bias, w, ids, y)


def mma12_moe(data, exc, exc_base, sym, E, O, K, x, k, gather, plan, act, bias, w, ids, y):
    _moe("mma12_moe", data, exc, exc_base, _words(sym, 4, "four words of symbols"), E, O, K, x, k, gather, plan, act, bias, w, ids, y)


def _moe_unpack(name, data, a, b, words, E, O, K, P, plan, out):
    """mma_moe_unpack, mma12_moe_unpack."""
    d = data.get_device()
    if d != _device():
        return _there(_moe_unpack, d, name, data, a, b, words, E, O, K, P, plan, out)
    _check(plan.dtype == torch.int32 and out.numel() >= E * O * K, "plan int32, out [E O, K]")
    _check(plan.get_device() == d and out.get_device() == d, "every tensor on the pack's GPU")
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, E, O, K, P, plan.data_ptr(), out.data_ptr(), _stream(d))
    if r:
        _fail(name, r)


def mma_moe_unpack(data, blocks, block_base, tiers, E, O, K, P, plan, out):
    _moe_unpack("mma_moe_unpack", data, blocks, block_base, _words(tiers, 3, "three tiers"), E, O, K, P, plan, out)


def mma12_moe_unpack(data, exc, exc_base, sym, E, O, K, P, plan, out):
    _moe_unpack("mma12_moe_unpack", data, exc, exc_base, _words(sym, 4, "four words of symbols"), E, O, K, P, plan, out)
