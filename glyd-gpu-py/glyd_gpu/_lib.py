"""gpu/glyd_gpu.cu's kernels from the prebuilt library (libglyd_gpu_cudaN.so,
gpu/build_lib.sh) through its C API: the pybind module's functions, by the
same names and arguments, for kernels.py where nvcc is not at hand.

    from glyd_gpu import _lib as ext
    ext.load("libglyd_gpu_cuda13.so")
    ext.mma_gemm(...)                    # as the JIT-built module's

Each runs on the current stream of its tensors' device and allocates what
the C++ allocates: its outputs, and a product's done counters (zeroed once
and kept, a set a stream of a device, as there). A product's workspace (the bytes the
library asks for) is kept for its stream and reused in stream order, up to
16 MB (a prompt's larger one is allocated for the call, as there). A
generation step calls a product for every Linear, so a call's host time is
its C call and a few lookups: no Stream object, no device switch where the
device is current, no allocation."""
import ctypes
import functools
import threading
import torch

_lib = _error_string = _cuda_version = None
_fn, _query = {}, {}  # the C API's functions, argument types set; the workspace queries
_P, _I64, _U64, _SZ, _W = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)
_PACK = [_P, _P, _P, _W]  # data, blocks (exc), block_base (exc_base), tiers[3] (sym[4])
_FAST = [_P, _P, _P, _P, _U64]  # sm, planes, exc, exc_base, top
_DENSE = [_P, _P, _I64, _P, _P]  # sm, stream (and its words), offs, tables
_ARGS = {  # each function's arguments before its stream
    "pass_a": [_P, _I64, _P, _I64, _I64, _P],
    "pass_b": [_P, _I64, _P, _P, _P, _P, _I64, _I64],
    "decode": _DENSE + [_I64, _I64, _I64, _I64, _P, _I64, _P],
    "gemv": _DENSE + [_I64, _I64, _I64, _I64, _I64, _P, _P, _P, _P, _P],
    "fast_gemv": _FAST + [_I64, _I64, _P, _P, _P],
    "fast_decode": _FAST + [_I64, _I64, _P, _I64, _I64, _P],
    "fast_gemm": _FAST + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ],
    "fast_bgemv": _FAST + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ],
    "mma_gemm": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_gemm": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma_gemm_x1": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "mma12_gemm_x1": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "mma12_gemm_x2": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_gemm_x3": _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _SZ, _P],
    "hold": [_I64],
    "mma_unpack": _PACK + [_I64, _I64, _I64, _P, _I64],
    "mma12_unpack": _PACK + [_I64, _I64, _I64, _P, _I64],
    "mma12_unpack_part": _PACK + [_I64, _I64, _I64, _P, _I64],
    "mma_rows": _PACK + [_I64, _I64, _P, _I64, _P],  # K, rows, ids, n, out
    "mma12_rows": _PACK + [_I64, _I64, _P, _I64, _P],
    "attn_decode": [_P, _I64, _P, _P, _P, _W, _P, _P, _P, _W, _P, _P, _I64, _I64, _I64, _I64, ctypes.c_double, _P, _P, _SZ, _P],
    "moe_way": [_P, _I64, _I64, _P],
    "mma_moe_unpack": _PACK + [_I64, _I64, _I64, _I64, _P, _P],
    "mma12_moe_unpack": _PACK + [_I64, _I64, _I64, _I64, _P, _P],
    "mma_moe": _PACK + [_I64, _I64, _I64, _P, _I64, _I64, _I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma_moe_placed": _PACK + [_I64, _I64, _I64, _I64, _P, _I64, _I64, _I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma12_moe": _PACK + [_I64, _I64, _I64, _P, _I64, _I64, _I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _SZ, _P],
    "mma_linear": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "mma12_linear": _PACK + [_I64, _I64, _P, _I64, _P, _P, _I64, _P, _SZ, _P],
    "kv_write": [_P, _I64, _P, _I64, _P, _I64, _P, _I64, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _P, _I64, _I64],  # key, value, slots, pool, bases, H, D, A, escape pool, nchunks, raw pool, nslabs, layer, status, lost bitmap, its bits, the rings' places
    "kv_release": [_P, _I64, _I64, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _P, _I64],  # pool, blocks, H, D, A, escape pool, nchunks, raw pool, nslabs, layer, status, lost bitmap, its bits
    "kv_unpack": [_P, _I64, _P, _I64, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _P, _P],
    "kv_save": [_P, _I64, _I64, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _P],  # pool, nblocks, pages, H, D, A, escape pool, nchunks, raw pool, nslabs, layer, the images, their bytes
    "kv_load": [_P, _P, _I64, _P, _I64, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _I64],  # the images, their blocks, pool, nblocks, H, D, A, escape pool, nchunks, raw pool, nslabs, layer, status, the rings' places
    "kv_gather": [_P, _I64, _P, _P, _I64, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _P],
    "kv_work_put": [_P, _I64, _P, _I64, _P, _I64, _I64, _I64, _P, _I64],
    "kv_decode": [_P, _I64, _P, _I64, _P, _P, _P, _I64, _P, _I64, _I64, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, ctypes.c_double, _P, _I64, _P, _SZ],
    "kv_decode_from": [_P, _I64, _P, _I64, _P, _P, _P, _I64, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, ctypes.c_double, _P, _I64, _P, _SZ],
    "kv_decode_bf16": [_P, _I64, _P, _P, _P, _I64, _I64, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, ctypes.c_double, _P, _I64, _P, _SZ],  # q, kc, vc, table, its stride, cap, lens, first, round, B, splits, H, G, D, scale, out, workspace
    "kv_prefill_bf16": [_P, _I64, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, _I64, ctypes.c_double, _I64, ctypes.c_double, _P, _P, _I64],  # q, kc, vc, table, cap, n, start, window, H, G, D, scale, round, softcap, sinks, out
    "kv_prefill": [_P, _I64, _P, _I64, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _P, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _I64, ctypes.c_double, _I64, ctypes.c_double, _P, _P, _I64],  # q, key, value (and strides), n, start, window, pool, bases, table, H, G, D, A, escape pool, raw pool, scale, round, softcap, sinks, out
    "f8_size": [_P, _I64, _I64, _I64, _P, _P, _SZ, _P],  # w, ld, O, K, heads, workspace and its bytes, the words var needs (host)
    "f8_pack": [_P, _I64, _I64, _I64, _P, _P, _P, _I64],  # w, ld, O, K, heads, core, var and its words
    "f8_unpack": [_P, _P, _P, _I64, _I64, _P, _I64],  # core, heads, var, O, K, out, ld
    "f8_unpack_experts": [_P, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64],  # core, heads, var, E, O, K, ids, n, out, ld
    "f8_linear": [_P, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _I64, _P, _SZ, _P],  # core, heads, var, N, K, kind, scale, lds, x, ldx, M, y, ldy, workspace and its bytes, done
    "f8_moe": [_P, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _SZ, _P],  # core, heads, var, E, N, K, scale, lds, x, ldx, plan, k, gather, M, slots, y, ldy, workspace and its bytes, done
    "f8h_size": [_P, _I64, _I64, _I64, _P, _P, _SZ, _P],  # layout H: as f8_'s
    "f8h_pack": [_P, _I64, _I64, _I64, _P, _P, _P, _I64],  # layout H: as f8_'s
    "f8h_unpack": [_P, _P, _P, _I64, _I64, _P, _I64],  # layout H: as f8_'s
    "f8h_linear": [_P, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _I64, _P, _I64, _P, _SZ, _P],  # layout H: as f8_'s
    "f8h_moe": [_P, _P, _P, _I64, _I64, _I64, _P, _I64, _P, _I64, _P, _I64, _I64, _I64, _I64, _P, _I64, _P, _SZ, _P],  # layout H: as f8_'s
}
_SIZES = {"fast_gemm": 3, "fast_bgemv": 3, "mma_gemm": 3, "mma12_gemm": 3, "mma_gemm_x1": 4, "mma12_gemm_x1": 4, "mma12_gemm_x2": 3, "mma12_gemm_x3": 3, "attn_decode": 4, "mma_moe": 7, "mma12_moe": 7, "mma_linear": 4, "mma12_linear": 4, "kv_decode": 5, "f8_size": 2, "f8_linear": 3, "f8_moe": 4}  # their workspace queries' sizes
_PLAIN = {"gpu": [_P], "mma_way": [_I64] * 4 + [_P, _P], "mma12_way": [_I64] * 4 + [_P, _P], "mma12_part_cus": [_I64] * 4 + [_P], "kv_leaf_bytes": [_I64] * 3 + [_P], "kv_image_bytes": [_I64] * 3 + [_P], "kv_extra_bytes": [_I64, _P], "kv_second_bytes": [_I64, _I64, _P], "kv_decode_shares": [_I64] * 3 + [_P], "kv_floor_set": [ctypes.c_int], "f8h_plain": [ctypes.c_int]}  # the calls with no stream
_AUX = {  # the route SPLIT's ring (glyd_gpu.h), each call's arguments whole (a stream last where it takes one)
    "aux_create": [_P, _SZ, _SZ, _P],
    "aux_destroy": [_P],
    "aux_part": [_P, _I64, _P, _P],
    "aux_reset": [_P, _P],
    "mma12_aux_queue": [_P, _I64] + _PACK + [_I64, _I64],
    "mma12_aux_linear": [_P, _I64] + _PACK + [_I64, _I64, _P, _I64, _P, _P, _P, _P],
}


API_VERSION = 26  # the C API these calls are written for (glyd_gpu_api_version; 0.21.0's library has none: 1)
KV_AUX = 2040  # places of each of the status's two rings, where a caller does not size them
KV_AUX_AT = 16  # the void ring's first place in the status
KV_STATUS_WORDS = KV_AUX_AT + 2 * KV_AUX  # a lossless KV cache's status words (glyd_gpu.h) at KV_RING: counters, then the ring of lost blocks and the ring of blocks with a raw page
KV_SECOND_AUX_AT = KV_AUX_AT + KV_AUX  # the raw ring's first place at KV_RING
KV_AUX_HEAD = 12  # the status word that counts the pages made void (the ring's head)
KV_SECOND_AUX_HEAD = 13  # the status word that counts the blocks that have a raw page


def kv_status_words(aux=KV_AUX):
    """The words of a status with `ring` places in each of its two rings."""
    return KV_AUX_AT + 2 * int(aux)


def kv_aux_of(status):
    """The places of each ring of a status tensor (its size says: kv_status_words)."""
    return (status.numel() - KV_AUX_AT) // 2


def kv_second_aux_at(aux=KV_AUX):
    """Where the raw ring's first place is in a status with `ring` places in each ring."""
    return KV_AUX_AT + int(aux)
X1 = 4  # glyd_gpu.h's GLYD_GPU_ROUTE_BIG: the prompt kernel


def load(path):
    """The library at path, for the functions below: refused where its C API is another version (ctypes does not
    check a call's arguments). Every function is the library's, by its name."""
    global _lib, _error_string, _cuda_version
    lib = ctypes.CDLL(path)
    v = getattr(lib, "glyd_gpu_api_version", None)
    v = v() if v is not None else 1
    if v != API_VERSION:
        raise RuntimeError(f"{path}: its C API is version {v}, this package's is {API_VERSION}: build the library from this package's release (gpu/build_lib.sh), or unset GLYD_GPU_LIB")

    def find(name, args, result=ctypes.c_int):
        try:
            return ctypes.CFUNCTYPE(result, *args)(("glyd_gpu_" + name, lib))
        except AttributeError:
            raise RuntimeError(f"{path}: no function glyd_gpu_{name}: a library of another build") from None

    for name, args in _ARGS.items():
        _fn[name] = find(name, args + [_P])
    for name, n in _SIZES.items():
        _query[name] = find(name + "_workspace", [_I64] * n + [ctypes.POINTER(_SZ)])
    for name, args in (*_PLAIN.items(), *_AUX.items()):
        _fn[name] = find(name, args)
    _error_string = find("error_string", [ctypes.c_int], ctypes.c_char_p)
    _cuda_version = find("cuda_version", [])
    _lib = lib  # (kept: the functions above are its)


def cuda_version():
    """The CUDA runtime the library was built with, e.g. 13000."""
    return _cuda_version()


def kv_floor_set(floor):
    """The CTAs a sequence's decode plan has at least, 1 to 64 (0: the GPU's own): what a batch-invariant server asks for
    (glyd_gpu_kv_floor_set)."""
    r = _fn["kv_floor_set"](floor)
    if r:
        _fail("kv_floor_set", r)


def _check(ok, why):
    if not ok:
        raise RuntimeError(why)


def _fail(name, r):
    raise RuntimeError(f"{name}: {_error_string(r).decode()}")


_device = getattr(torch._C, "_cuda_getDevice", None) or torch.cuda.current_device  # (a PyTorch without CUDA's bindings, a Mac's, imports: nothing there calls it)
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


def _counters(name, d, s, n, least):
    """The address of a product's done counters on device d for stream s (at least n; least when first made):
    zero between products, a set a stream (the C API's: one stream's products at a time on a set). Not made in a
    torch.compile graph's node or a CUDA graph's capture (a graph's memory pool): there the device's own set (s
    None: made with a stream's first set, or a step; ponytail: a GPU's graphs then share one set, replayed one at
    a time). One replaced by a bigger set is kept (a CUDA graph holds its address)."""
    c = _done.get((name, d, s))
    if c is None or c[2] < n:
        if s is not None and (local.fresh or torch.cuda.is_current_stream_capturing()):
            return _counters(name, d, None, n, least)
        if c is not None:
            _replaced.append(c)
        t = torch.zeros(max(n, least), dtype=torch.int32, device=torch.device("cuda", d))
        c = _done[(name, d, s)] = (t, t.data_ptr(), t.numel())
        if s is not None:
            _counters(name, d, None, n, least)
    return c[1]


_arrays = {}


def _words(v, n, what):
    """tiers (3) or sym (4: a 12-bit pack's base) as the C API's words (a pack's are the same every call: kept)."""
    key = (n, *v)
    a = _arrays.get(key)
    if a is None:
        _check(len(v) == n, what)
        a = _arrays[key] = (ctypes.c_uint32 * n)(*[x & 0xFFFFFFFF for x in v])
    return a


def pass_a(w, len_, tw, V):
    d = w.get_device()
    if d != _device():
        return _there(pass_a, d, w, len_, tw, V)
    bits = torch.empty((w.numel() + tw - 1) // tw * 32, dtype=torch.int32, device=w.device)
    r = _fn["pass_a"](w.data_ptr(), w.numel(), len_.data_ptr(), tw, V, bits.data_ptr(), _stream(d))
    if r:
        _fail("pass_a", r)
    return bits


def pass_b(w, len_, code, offs, out, tw, V):
    d = w.get_device()
    if d != _device():
        return _there(pass_b, d, w, len_, code, offs, out, tw, V)
    r = _fn["pass_b"](w.data_ptr(), w.numel(), len_.data_ptr(), code.data_ptr(), offs.data_ptr(), out.data_ptr(), tw, V, _stream(d))
    if r:
        _fail("pass_b", r)


def decode(core, stream, offs, tables, n, tw, V, span_words, span_ids, out):
    d = core.get_device()
    if d != _device():
        return _there(decode, d, core, stream, offs, tables, n, tw, V, span_words, span_ids, out)
    ids = span_ids.numel()
    _check(out.numel() >= (ids * tw if ids else n), "the output is too small")
    r = _fn["decode"](core.data_ptr(), stream.data_ptr(), stream.numel(), offs.data_ptr(), tables.data_ptr(), n, tw, V, span_words, span_ids.data_ptr(), ids, out.data_ptr(), _stream(d))
    if r:
        _fail("decode", r)


def gemv(core, stream, offs, tables, O, K, tw, V, span_words, x, bias, y, sum_, count):
    d = core.get_device()
    if d != _device():
        return _there(gemv, d, core, stream, offs, tables, O, K, tw, V, span_words, x, bias, y, sum_, count)
    split = tw % K != 0
    _check(K % (32 * V) == 0 and tw % (32 * V) == 0 and (not split or sum_.numel() >= O), "gemv: shapes this product does not take")
    r = _fn["gemv"](core.data_ptr(), stream.data_ptr(), stream.numel(), offs.data_ptr(), tables.data_ptr(), O, K, tw, V, span_words, x.data_ptr(), bias.data_ptr() if bias.numel() else None, y.data_ptr(), sum_.data_ptr(), count.data_ptr(), _stream(d))
    if r:
        _fail("gemv", r)


def fast_gemv(core, planes, exc, exc_base, top, O, K, x, bias, y):
    d = core.get_device()
    if d != _device():
        return _there(fast_gemv, d, core, planes, exc, exc_base, top, O, K, x, bias, y)
    _check(K % 128 == 0, "rows a multiple of 128 long")
    r = _fn["fast_gemv"](core.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, O, K, x.data_ptr(), bias.data_ptr() if bias.numel() else None, y.data_ptr(), _stream(d))
    if r:
        _fail("fast_gemv", r)


def fast_decode(core, planes, exc, exc_base, top, row0, rows, row_ids, K, out):
    d = core.get_device()
    if d != _device():
        return _there(fast_decode, d, core, planes, exc, exc_base, top, row0, rows, row_ids, K, out)
    ids = row_ids.numel()
    _check(K % 128 == 0 and out.numel() >= (ids or rows) * K, "rows a multiple of 128 long, room for them")
    r = _fn["fast_decode"](core.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, row0, rows, row_ids.data_ptr(), ids, K, out.data_ptr(), _stream(d))
    if r:
        _fail("fast_decode", r)


def _fast_product(name, core, planes, exc, exc_base, top, O, K, x, bias, y):
    """fast_gemm, fast_bgemv: several tokens (the checks done)."""
    d = core.get_device()
    if d != _device():
        return _there(_fast_product, d, name, core, planes, exc, exc_base, top, O, K, x, bias, y)
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](core.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], s)
    if r:
        _fail(name, r)


def fast_gemm(core, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 64 == 0 and x.is_contiguous() and x.size(1) == K, "K a multiple of 64, X contiguous [M, K]")
    _fast_product("fast_gemm", core, planes, exc, exc_base, top, O, K, x, bias, y)


def fast_bgemv(core, planes, exc, exc_base, top, O, K, x, bias, y):
    _check(K % 512 == 0 and x.is_contiguous() and x.size(1) == K and x.size(0) in (2, 4, 8, 16), "K a multiple of 512, X contiguous [M, K], M 2, 4, 8 or 16")
    _fast_product("fast_bgemv", core, planes, exc, exc_base, top, O, K, x, bias, y)


def _small(name, data, a, b, words, O, K, x, bias, y):
    """mma_gemm, mma12_gemm: up to 64 tokens (a generation step's product)."""
    d = data.get_device()
    if d != _device():
        return _there(_small, d, name, data, a, b, words, O, K, x, bias, y)
    M = x.size(0)
    _check(O % 64 == 0 and K % 16 == 0 and M <= 64 and x.is_contiguous() and x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]")
    s = _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], _counters(name, d, s, O // 64, 1 << 16), s)
    if r:
        _fail(name, r)


def mma_gemm(data, blocks, block_base, tiers, O, K, x, bias, y):
    _small("mma_gemm", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), O, K, x, bias, y)


def mma12_gemm(data, exc, exc_base, sym, O, K, x, bias, y):
    _small("mma12_gemm", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), O, K, x, bias, y)


def _units(O, M):
    """A prompt's product's done counters: its units (tiles of 128 or 256 tokens by pairs of row blocks) at most."""
    return (M + 127) // 128 * (O // 64)


_UNITS = 1 << 18  # a prompt's product's done counters when first made: to some 100K tokens of the largest matrices none made again (in a CUDA graph's capture)


def _x1(name, data, a, b, words, O, K, x, bias, y, variant):
    """mma_gemm_big, mma12_gemm_big: a prompt."""
    d = data.get_device()
    if d != _device():
        return _there(_x1, d, name, data, a, b, words, O, K, x, bias, y, variant)
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M, variant)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), variant, ws[1], ws[2], _counters(name, d, s, _units(O, M), _UNITS), s)
    if r:
        _fail(name, r)


def mma_gemm_x1(data, blocks, block_base, tiers, O, K, x, bias, y, variant):
    _x1("mma_gemm_x1", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), O, K, x, bias, y, variant)


def mma12_gemm_x1(data, exc, exc_base, sym, O, K, x, bias, y, variant):
    _x1("mma12_gemm_x1", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), O, K, x, bias, y, variant)


def _loaded(name):
    """The C API's function name (_fn's), the library loaded first where nothing has loaded it yet, as the kernels'
    first call does (kernels._Load: GLYD_GPU_LIB, else the one beside the package; OSError, saying so, where there is
    none): the calls that are not the kernels' (gpu, the routes) can come first."""
    if name not in _fn:
        from . import kernels  # (kernels imports this module: here, at the call)

        if isinstance(kernels._ext, kernels._Load):
            kernels._ext.cuda_version  # loads it, or raises its OSError
        if name not in _fn:  # (the kernels are gpu/glyd_gpu_dev.py's JIT build's: no library to ask)
            raise OSError(f"glyd_gpu_{name}: the Glyd GPU library is not loaded (the kernels are the JIT build's)")
    return _fn[name]


def gpu():
    """The current device as the library's routes take it: its code (compute capability, major * 10 + minor, plus its
    class by name, glyd_gpu.h); the library loaded first where it is not yet."""
    g = ctypes.c_int()
    r = _loaded("gpu")(ctypes.byref(g))
    if r:
        _fail("gpu", r)
    return g.value


@functools.lru_cache(maxsize=4096)
def _way(name, gpu, O, K, M):
    """mma_route, mma12_route: (route, last), the route for M tokens on gpu and the last token count that takes it (the
    library loaded first where it is not yet)."""
    way, last = ctypes.c_int(), ctypes.c_int64()
    r = _loaded(name)(gpu, O, K, M, ctypes.byref(way), ctypes.byref(last))
    if r:
        _fail(name, r)
    return way.value, last.value


def mma_way(gpu, O, K, M):
    return _way("mma_way", gpu, O, K, M)


def mma12_way(gpu, O, K, M):
    return _way("mma12_way", gpu, O, K, M)


def _linear(name, data, a, b, words, O, K, x, bias, y, way):
    """mma_linear, mma12_linear: a product by a route (-1: this GPU's for M)."""
    d = data.get_device()
    if d != _device():
        return _there(_linear, d, name, data, a, b, words, O, K, x, bias, y, way)
    _check(O % 64 == 0 and K % 16 == 0 and x.is_contiguous() and x.size(1) == K, "O a multiple of 64, K of 16, X contiguous [M, K]")
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M, way)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), way, ws[1], ws[2], _counters(name, d, s, _units(O, M), _UNITS), s)
    if r:
        _fail(name, r)


def mma_linear(data, blocks, block_base, tiers, O, K, x, bias, y, way):
    _linear("mma_linear", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), O, K, x, bias, y, way)


def mma12_linear(data, exc, exc_base, sym, O, K, x, bias, y, way):
    _linear("mma12_linear", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), O, K, x, bias, y, way)


def _staged(name, data, exc, exc_base, sym, O, K, x, bias, y):
    """mma12_gemm_mid, mma12_gemm_wg: many tokens, the 12-bit layout copied a stage at a time (the GPU checked by the library)."""
    d = data.get_device()
    if d != _device():
        return _there(_staged, d, name, data, exc, exc_base, sym, O, K, x, bias, y)
    words = _words(sym, 4, "the mma12 pack's parameters")
    _check(O % 64 == 0 and K % 64 == 0 and x.is_contiguous() and x.size(1) == K and x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]")
    _check(data.data_ptr() % 16 == 0 and exc.data_ptr() % 16 == 0 and exc.numel() % 4 == 0, "the pack's arrays 16-byte aligned (pack_mma12)")
    M, s = x.size(0), _stream(d)
    ws = _workspace(name, d, s, O, K, M)
    r = _fn[name](data.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), words, O, K, x.data_ptr(), M, bias.data_ptr() if bias.numel() else None, y.data_ptr(), ws[1], ws[2], _counters(name, d, s, O // 64, 1 << 16), s)
    if r:
        _fail(name, r)


def mma12_gemm_x2(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_x2", data, exc, exc_base, sym, O, K, x, bias, y)


def mma12_gemm_x3(data, exc, exc_base, sym, O, K, x, bias, y):
    _staged("mma12_gemm_x3", data, exc, exc_base, sym, O, K, x, bias, y)


def hold(ns):
    """The current stream held ns nanoseconds."""
    r = _fn["hold"](ns, _stream(_device()))
    if r:
        _fail("hold", r)


def _unpack(name, data, a, b, words, K, row0, rows, out, crews):
    """mma_unpack, mma12_unpack."""
    d = data.get_device()
    if d != _device():
        return _there(_unpack, d, name, data, a, b, words, K, row0, rows, out, crews)
    _check(row0 % 64 == 0 and rows % 64 == 0 and out.numel() >= rows * K, "rows a multiple of 64")
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, K, row0, rows, out.data_ptr(), crews, _stream(d))
    if r:
        _fail(name, r)


def mma_unpack(data, blocks, block_base, tiers, K, row0, rows, out, crews):
    _unpack("mma_unpack", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), K, row0, rows, out, crews)


def mma12_unpack(data, exc, exc_base, sym, K, row0, rows, out, crews):
    _unpack("mma12_unpack", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), K, row0, rows, out, crews)


def _rows(name, data, a, b, words, K, rows, ids, out):
    """mma_rows, mma12_rows: row ids[i] of W [rows, K] as out[i] (bf16, bit for bit; an id outside [0, rows) is a row of zeros)."""
    d = data.get_device()
    if d != _device():
        return _there(_rows, d, name, data, a, b, words, K, rows, ids, out)
    _check(K % 16 == 0 and rows % 64 == 0 and ids.dtype == torch.int64 and ids.is_contiguous() and ids.get_device() == d and out.is_contiguous() and out.get_device() == d and out.numel() >= ids.numel() * K,
           "ids int64 and out [n, K] on the pack's GPU, K a multiple of 16, rows of 64")
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, K, rows, ids.data_ptr(), ids.numel(), out.data_ptr(), _stream(d))
    if r:
        _fail(name, r)


def mma_rows(data, blocks, block_base, tiers, K, rows, ids, out):
    _rows("mma_rows", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), K, rows, ids, out)


def mma12_rows(data, exc, exc_base, sym, K, rows, ids, out):
    _rows("mma12_rows", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), K, rows, ids, out)


def mma12_unpack_part(data, exc, exc_base, sym, K, row0, rows, out, cus):
    """The route SPLIT's decode: rows [row0, row0 + rows) into out, a grid for sms SMs (K a multiple of 64)."""
    _check(K % 64 == 0 and out.data_ptr() % 16 == 0, "K a multiple of 64, out 16-byte aligned")
    _unpack("mma12_unpack_part", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), K, row0, rows, out, cus)


@functools.lru_cache(maxsize=4096)
def mma12_part_cus(gpu, O, K, M):
    """The route SPLIT's SMs for the decode for M tokens of W [O, K] on gpu (its code with WITH_SPLIT; 0: another route)."""
    cus = ctypes.c_int64()
    r = _loaded("mma12_part_cus")(gpu, O, K, M, ctypes.byref(cus))
    if r:
        _fail("mma12_part_cus", r)
    return cus.value


class Blas(ctypes.Structure):
    """glyd_gpu.h's glyd_gpu_blas: a cuBLAS handle and its functions' addresses (the caller's cuBLAS: PyTorch's)."""
    _fields_ = [("handle", _P), ("gemm_ex", _P), ("set_stream", _P), ("get_stream", _P), ("set_workspace", _P), ("set_cap", _P),
                ("get_cap", _P), ("workspace", _P), ("workspace_bytes", _SZ)]


def aux_create(buffer, cell_bytes):
    """A ring over buffer (uint8, on the current device) in slots of slot_bytes: its handle."""
    h = ctypes.c_void_p()
    r = _fn["aux_create"](buffer.data_ptr(), buffer.numel(), cell_bytes, ctypes.byref(h))
    if r:
        _fail("aux_create", r)
    return h.value


def aux_part(aux, cus):
    """The split for a decode of sms SMs, made where it is not yet: (status, the decode's SMs, the products')."""
    a, b = ctypes.c_int64(), ctypes.c_int64()
    r = _fn["aux_part"](aux, cus, ctypes.byref(a), ctypes.byref(b))
    return r, a.value, b.value


def aux_destroy(aux):
    """The ring let go, its streams' work waited for first: the status."""
    return _fn["aux_destroy"](aux)


def aux_reset(aux):
    """The queue dropped, the current stream waiting for what it had queued: the status."""
    return _fn["aux_reset"](aux, _stream(_device()))


def mma12_aux_queue(aux, cus, data, exc, exc_base, sym, O, K):
    """W queued for decoding ahead: the status."""
    return _fn["mma12_aux_queue"](aux, cus, data.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), _words(sym, 4, "the mma12 pack's parameters"), O, K)


def mma12_aux_linear(aux, cus, data, exc, exc_base, sym, O, K, x, bias, y, blas):
    """Y = X W^T (+ bias) through the ring on the current stream (x contiguous [M, K], y [M, O]): the status."""
    return _fn["mma12_aux_linear"](aux, cus, data.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), _words(sym, 4, "the mma12 pack's parameters"), O, K,
                                    x.data_ptr(), x.size(0), bias.data_ptr() if bias is not None else None, y.data_ptr(), ctypes.byref(blas), _stream(_device()))


def error_string(r):
    return _error_string(r).decode()


def attn_decode(q, kd, kb, kbb, kt, vd, vb, vbb, vt, tk, tv, tlen, pairs, G, P, scale, out):
    d = q.get_device()
    if d != _device():
        return _there(attn_decode, d, q, kd, kb, kbb, kt, vd, vb, vbb, vt, tk, tv, tlen, pairs, G, P, scale, out)
    k3, v3 = _words(kt, 3, "the mma pack's parameters"), _words(vt, 3, "the mma pack's parameters")
    D = q.size(-1)
    _check(D in (64, 128) and 1 <= G <= 16 and q.is_contiguous() and tlen < 64 and P + (tlen > 0) > 0, "head_dim 64 or 128, up to 16 queries a KV head")
    s = _stream(d)
    ws = _workspace("attn_decode", d, s, D, tlen, pairs, P)
    r = _fn["attn_decode"](q.data_ptr(), D, kd.data_ptr(), kb.data_ptr(), kbb.data_ptr(), k3, vd.data_ptr(), vb.data_ptr(), vbb.data_ptr(), v3, tk.data_ptr(), tv.data_ptr(), tlen, pairs, G, P, scale, out.data_ptr(), ws[1], ws[2], _counters("attn_decode", d, s, pairs, 1 << 12), s)
    if r:
        _fail("attn_decode", r)


def step(data, a, b, words, n_words, shape, bias, ways, big=False, x1_max=0):
    """A generation step's product over one pack in the mma layouts as one C call, glyd_gpu_*_linear by the route the
    Linear takes: what does not change between calls made once (the pack's addresses and words, O and K, the bias;
    each M's workspace bytes at its first call, and its stream's done counters), the checks that hold by the pack's
    making left out. words: its n_words tiers (3) or the 12-bit layout's words (4, its base). routes[M]: the route for
    M tokens (the library's: a step's kernel), or None; big: the prompt kernel's route past them to big_max tokens.
    run(x): Y [..., O] for X contiguous [..., K] of M rows on the pack's device (made current for the call where it is
    not: a layer on another GPU), where a route takes M; else None (the checked path)."""
    O, K = shape
    d, dev = data.get_device(), data.device
    name = "mma12_linear" if n_words == 4 else "mma_linear"
    fn = _fn[name]
    head = (data.data_ptr(), a.data_ptr(), b.data_ptr(), _words(words, n_words, "the pack's mma or mma12 parameters"), O, K)
    bias = bias.data_ptr() if bias is not None else None
    _counters(name, d, None, 0, _UNITS)  # the device's, made now: never in a CUDA graph's memory pool
    plans = [None] * len(ways)
    bf16 = torch.bfloat16

    def run(x):
        if x.shape[-1] != K or not x.is_contiguous() or x.get_device() != d:
            return None
        if _device() != d:  # accelerate's device map leaves the first GPU current
            with torch.cuda.device(d):
                return run(x)
        M = x.numel() // K
        if big and M >= len(plans) and M < x1_max:  # a prompt (big: none on Hopper, nor where K is not a multiple of 64)
            s = _stream(d)
            w = _workspace(name, d, s, O, K, M, X1)
            y = torch.empty(*x.shape[:-1], O, dtype=bf16, device=dev)
            r = fn(*head, x.data_ptr(), M, bias, y.data_ptr(), X1, w[1], w[2], _counters(name, d, s, _units(O, M), _UNITS), s)
            if r:
                _fail(name, r)
            return y
        plan = plans[M] if M < len(plans) else False
        if plan is None:
            way = ways[M]
            plan = plans[M] = way is not None and [way, _need(name, d, (O, K, M, way)), None, None]
        if not plan:
            return None
        way, need, cs, done = plan
        s = _stream(d)
        if s != cs:  # the stream's counters (kept for the next call but in a graph's node)
            done = _counters(name, d, s, _units(O, M), _UNITS)
            if not local.fresh:
                plan[2], plan[3] = s, done
        w = _kept.get((d, s))
        if local.fresh or w is None or w[2] < need:
            w = _workspace(name, d, s, O, K, M, way)
        y = torch.empty(*x.shape[:-1], O, dtype=bf16, device=dev)
        r = fn(*head, x.data_ptr(), M, bias, y.data_ptr(), way, w[1], w[2], done, s)
        if r:
            _fail(name, r)
        return y

    return run


def lookup(core, planes, exc, exc_base, top, K):
    """An embedding's lookup in the fast format as one C call (fast_decode): run(ids) -> its rows, bf16
    [*ids.shape, K], for ids int64 and contiguous on the pack's device (made current for the call where it is
    not); else None."""
    d, dev, i64 = core.get_device(), core.device, torch.int64
    fn, head = _fn["fast_decode"], (core.data_ptr(), planes.data_ptr(), exc.data_ptr(), exc_base.data_ptr(), top, 0, 0)

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


def moe_way(ids, E, plan):
    d = ids.get_device()
    if d != _device():
        return _there(moe_way, d, ids, E, plan)
    _check(ids.dtype == torch.int64 and ids.is_contiguous() and plan.dtype == torch.int32 and plan.numel() >= 2 + 2 * E + ids.numel() and plan.get_device() == d, "ids int64, the plan int32 on the same GPU")
    r = _fn["moe_way"](ids.data_ptr(), ids.numel(), E, plan.data_ptr(), _stream(d))
    if r:
        _fail("moe_way", r)


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
    done = _counters(name, d, s, (O // 128 if act else O // 64) * min(E, T * k), 1 << 16)
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, E, O, K, x.data_ptr(), T, k, gather, plan.data_ptr(), act, bias.data_ptr() if bias.numel() else None, w.data_ptr() if weighted else None, int(w.dtype == torch.float32), ids.data_ptr() if weighted else None, y.data_ptr(), ws[1], ws[2], done, s)
    if r:
        _fail(name, r)


def mma_moe(data, blocks, block_base, tiers, E, O, K, x, k, gather, plan, act, bias, w, ids, y):
    _moe("mma_moe", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), E, O, K, x, k, gather, plan, act, bias, w, ids, y)


def mma12_moe(data, exc, exc_base, sym, E, O, K, x, k, gather, plan, act, bias, w, ids, y):
    _moe("mma12_moe", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), E, O, K, x, k, gather, plan, act, bias, w, ids, y)


def _moe_unpack(name, data, a, b, words, E, O, K, P, plan, out):
    """mma_moe_unpack, mma12_moe_unpack."""
    d = data.get_device()
    if d != _device():
        return _there(_moe_unpack, d, name, data, a, b, words, E, O, K, P, plan, out)
    _check(plan.dtype == torch.int32 and out.numel() >= E * O * K, "the plan int32, out [E O, K]")
    _check(plan.get_device() == d and out.get_device() == d, "every tensor on the pack's GPU")
    r = _fn[name](data.data_ptr(), a.data_ptr(), b.data_ptr(), words, E, O, K, P, plan.data_ptr(), out.data_ptr(), _stream(d))
    if r:
        _fail(name, r)


def mma_moe_unpack(data, blocks, block_base, tiers, E, O, K, P, plan, out):
    _moe_unpack("mma_moe_unpack", data, blocks, block_base, _words(tiers, 3, "the mma pack's parameters"), E, O, K, P, plan, out)


def mma12_moe_unpack(data, exc, exc_base, sym, E, O, K, P, plan, out):
    _moe_unpack("mma12_moe_unpack", data, exc, exc_base, _words(sym, 4, "the mma12 pack's parameters"), E, O, K, P, plan, out)


def kv_leaf_bytes(H, D, A):
    """A lossless KV cache page's bytes for H kv heads of D channels and an arena of A entries (the library's rule)."""
    n = ctypes.c_int64()
    r = _loaded("kv_leaf_bytes")(H, D, A, ctypes.byref(n))
    if r:
        _fail("kv_leaf_bytes", r)
    return n.value


def kv_extra_bytes(nchunks):
    """The bytes of a lossless KV cache's escape pool of nchunks chunks (the library's rule)."""
    n = ctypes.c_int64()
    r = _loaded("kv_extra_bytes")(nchunks, ctypes.byref(n))
    if r:
        _fail("kv_extra_bytes", r)
    return n.value


def kv_free_bay(H, D=128):
    """The arena's entries a lossless KV cache page of H kv heads of D channels holds without growing (the library's rule): what its header leaves up to a multiple of 128 bytes (D = 256: the page's 2 H heads of 128)."""
    base = kv_leaf_bytes(H, D, 0)
    return max(a for a in range(0, 1024) if kv_leaf_bytes(H, D, a) == base)


def kv_second_bytes(nlumps, H):
    """The bytes of a lossless KV cache's raw pool of nlumps slabs for pages of H heads of 128 channels (the library's rule; a head of 256 is two)."""
    n = ctypes.c_int64()
    r = _loaded("kv_second_bytes")(nlumps, H, ctypes.byref(n))
    if r:
        _fail("kv_second_bytes", r)
    return n.value


class Extra:
    """A lossless KV cache's escape pool (`chunks` chunks, buffer `buf`) and raw pool (`slabs` slabs for pages of `heads` heads of 128 channels, buffer `raw`), zeros when new; views of them for tests and tools:
    `bits`, `owner`, `ent`, `rbits`, `rowner`, `plane`."""

    def __init__(self, chunks, device=None, lumps=0, heads=0):
        self.chunks = int(chunks)
        self.lumps = int(lumps)
        self.heads = int(heads)
        dev = device if device is not None else "cuda"
        self.buf = torch.zeros(kv_extra_bytes(self.chunks), dtype=torch.uint8, device=dev)
        nb = ((self.chunks + 31) // 32 + 3) // 4 * 16
        no = (self.chunks + 3) // 4 * 16
        assert self.buf.numel() == nb + no + self.chunks * 256, "the extra memory's size is not the library's"
        self.bits = self.buf[:nb].view(torch.int32)
        self.owner = self.buf[nb : nb + no].view(torch.int32)[: self.chunks]
        self.ent = self.buf[nb + no :].view(torch.int32).view(self.chunks, 64)
        if self.lumps:
            assert self.heads >= 1, "the extra memory is made for at least one KV head"
            self.raw = torch.zeros(kv_second_bytes(self.lumps, self.heads), dtype=torch.uint8, device=dev)
            rb = ((self.lumps + 31) // 32 + 3) // 4 * 16
            ro = (self.lumps + 3) // 4 * 16
            assert self.raw.numel() == rb + ro + self.lumps * self.heads * 2048, "the second extra memory's size is not the library's"
            self.rbits = self.raw[:rb].view(torch.int32)
            self.rowner = self.raw[rb : rb + ro].view(torch.int32)[: self.lumps]
            self.plane = self.raw[rb + ro :].view(self.lumps, self.heads * 2048)
        else:
            self.raw = None

    @property
    def ptr(self):
        return self.buf.data_ptr()

    @property
    def second_ptr(self):
        return self.raw.data_ptr() if self.raw is not None else 0


def _esc(addon):
    """(the escape pool's address, its chunks, the raw pool's address, its slabs) of an Escape (zeros where there is none)."""
    if addon is None:
        return (0, 0, 0, 0)
    return (addon.ptr if addon.chunks else 0, addon.chunks, addon.second_ptr, addon.lumps)


def kv_write(key, value, cells, store, leaf_stride, kbase, vbase, A, addon, layer, status, lost=None):
    """One step's tokens into the pool's pages (glyd_gpu_kv_write): key, value [>= len(slots), H, D] bf16 (rows of any
    stride, a head's channels contiguous), slots int64 [N] (page * 16 + offset, negative: none), pool uint8 (page p at
    p * page_stride bytes), kbase, vbase uint8 [H, D], escape the Escape pool (None: none), layer this layer's number
    (0 to 254), status int32 [kv_status_words(ring)] on the GPU (the places of each of its rings are what its size says), lost: an int32 bitmap, two bits a block, that makes the rings name a block once (None: none)."""
    d = store.get_device()
    if d != _device():
        return _there(kv_write, key, value, cells, store, leaf_stride, kbase, vbase, A, addon, layer, status, lost)
    N, H, D = cells.numel(), key.size(1), key.size(2)
    _check(key.dim() == 3 and value.shape[1:] == key.shape[1:] and key.size(0) >= N and value.size(0) >= N and key.stride(2) == 1 and key.stride(1) == D and value.stride(2) == 1 and value.stride(1) == D, "key, value [tokens, H, D] with contiguous heads")
    _check(cells.dtype == torch.int64 and cells.is_contiguous() and kbase.dtype == torch.uint8 and kbase.is_contiguous() and vbase.dtype == torch.uint8 and vbase.is_contiguous(), "slot_mapping int64, the tables uint8 [H, D]")
    _check(status.dtype == torch.int32 and status.is_contiguous() and kv_aux_of(status) >= 1, "status int32 as kv_status_words makes it")
    ep, nc, rp, ns = _esc(addon)
    r = _fn["kv_write"](key.data_ptr(), key.stride(0), value.data_ptr(), value.stride(0), cells.data_ptr(), N, store.data_ptr(), leaf_stride, kbase.data_ptr(), vbase.data_ptr(), H, D, A, ep, nc, rp, ns, layer, status.data_ptr(),
                        lost.data_ptr() if lost is not None else 0, lost.numel() * 32 if lost is not None else 0, kv_aux_of(status), _stream(d))
    if r:
        _fail("kv_write", r)


def kv_release(store, leaf_stride, nblocks, blocks, H, A, addon, layer, status, lost=None):
    """The pages of the blocks (int32 [n] on the GPU, each below nblocks) cleared as a page's first token clears them (glyd_gpu_kv_release): the chunks or slab a page holds back to the pools, its count and
    directory zero, the block's bits in `lost` cleared. For blocks that no request holds. H: the page's heads of 128 channels (a head of 256 is two)."""
    d = store.get_device()
    if d != _device():
        return _there(kv_release, store, leaf_stride, nblocks, blocks, H, A, addon, layer, status, lost)
    _check(blocks.dtype == torch.int32 and blocks.is_contiguous() and status.dtype == torch.int32 and status.numel() >= KV_AUX_AT, "blocks int32, status int32 as kv_status_words makes it")
    ep, nc, rp, ns = _esc(addon)
    r = _fn["kv_release"](store.data_ptr(), leaf_stride, nblocks, blocks.data_ptr(), blocks.numel(), H, 128, A, ep, nc, rp, ns, layer, status.data_ptr(), lost.data_ptr() if lost is not None else 0,
                          lost.numel() * 32 if lost is not None else 0, _stream(d))
    if r:
        _fail("kv_release", r)


def kv_unpack(store, leaf_stride, leaves, kbase, vbase, A, addon, out_k, out_v):
    """The pages (int32 [n]) back to bf16 into out_k, out_v [n, 16, H, D] (glyd_gpu_kv_unpack)."""
    d = store.get_device()
    if d != _device():
        return _there(kv_unpack, store, leaf_stride, leaves, kbase, vbase, A, addon, out_k, out_v)
    n, H, D = out_k.size(0), out_k.size(2), out_k.size(3)
    _check(leaves.dtype == torch.int32 and leaves.numel() >= n and out_k.is_contiguous() and out_v.is_contiguous() and out_v.shape == out_k.shape and out_k.size(1) == 16, "blocks int32 [n], out [n, 16, H, D]")
    ep, nc, rp, ns = _esc(addon)
    r = _fn["kv_unpack"](store.data_ptr(), leaf_stride, leaves.data_ptr(), n, kbase.data_ptr(), vbase.data_ptr(), H, D, A, ep, nc, rp, ns, out_k.data_ptr(), out_v.data_ptr(), _stream(d))
    if r:
        _fail("kv_unpack", r)


def kv_gather(store, leaf_stride, leaves, dst, kbase, vbase, A, addon, out):
    """The pages (int32 [n]) decoded into the blocks dst (int32 [n]) of the bf16 cache `out` [blocks, 16, H, 2 D] (glyd_gpu_kv_gather): a token's keys then its values, a head's 2 D channels."""
    d = store.get_device()
    if d != _device():
        return _there(kv_gather, store, leaf_stride, leaves, dst, kbase, vbase, A, addon, out)
    n = leaves.numel()
    H, D = out.size(2), out.size(3) // 2
    _check(leaves.dtype == torch.int32 and dst.dtype == torch.int32 and dst.numel() >= n and out.dtype == torch.bfloat16 and out.is_contiguous() and out.dim() == 4 and out.size(1) == 16, "blocks, dst int32 [n], out bf16 [blocks, 16, H, 2 D]")
    ep, nc, rp, ns = _esc(addon)
    r = _fn["kv_gather"](store.data_ptr(), leaf_stride, leaves.data_ptr(), dst.data_ptr(), n, kbase.data_ptr(), vbase.data_ptr(), H, D, A, ep, nc, rp, ns, out.data_ptr(), _stream(d))
    if r:
        _fail("kv_gather", r)


def kv_work_put(key, value, cells, out):
    """The tokens' keys and values [n, H, D] bf16 into the staging cache out [blocks, 16, H, 2 D] at slots int64 [n] (glyd_gpu_kv_stage_put): a negative slot is skipped. Nothing here waits for the device."""
    d = out.get_device()
    if d != _device():
        return _there(kv_work_put, key, value, cells, out)
    n, H, D = cells.numel(), key.size(1), key.size(2)
    _check(key.dim() == 3 and value.shape[1:] == key.shape[1:] and key.size(0) >= n and value.size(0) >= n and key.stride(2) == 1 and key.stride(1) == D and value.stride(2) == 1 and value.stride(1) == D, "key, value [tokens, H, D] with contiguous heads")
    _check(cells.dtype == torch.int64 and cells.is_contiguous() and out.dtype == torch.bfloat16 and out.is_contiguous() and out.dim() == 4 and out.size(1) == 16 and out.size(2) == H and out.size(3) == 2 * D, "slot_mapping int64, out bf16 [blocks, 16, H, 2 D]")
    r = _fn["kv_work_put"](key.data_ptr(), key.stride(0), value.data_ptr(), value.stride(0), cells.data_ptr(), n, H, D, out.data_ptr(), out.size(0), _stream(d))
    if r:
        _fail("kv_work_put", r)


def kv_decode_shares(B, H, max_leaves, device=None):
    """The most CTAs the KV decode splits a (sequence, kv head) over for a batch of B sequences (rounded up to a power of two) of a model whose longest has max_pages pages on this
    device (glyd_gpu_kv_decode_splits): a sequence's output's bits depend on its length and this number."""
    d = _device() if device is None else device
    if d != _device():
        return _there(kv_decode_shares, B, H, max_leaves, d)
    out = ctypes.c_int64()
    r = _loaded("kv_decode_shares")(B, H, max_leaves, ctypes.byref(out))
    if r:
        _fail("kv_decode_shares", r)
    return out.value


def kv_decode_workspace_bytes(B, H, G, shares, D=128):
    """The bytes of the KV decode's partial sums for a grid of B sequences x H kv heads x `splits` CTAs (glyd_gpu_kv_decode_workspace), on the current device."""
    _loaded("kv_decode_shares")  # (the library loaded: the first call may be this one)
    return _need("kv_decode", _device(), (B, H, G, D, shares))


KV_COUNTER_BYTES = 1 << 20  # the first bytes of a decode workspace: counters that are zero between calls (glyd_gpu_kv_decode_workspace)
_WS_SEEN = set()  # (workspaces whose counters were read as zero before a first call outside a CUDA graph)


def kv_decode_workspace(B, H, G, shares, device=None, D=128):
    """A decode workspace for B sequences x H kv heads x `splits` CTAs (kv_decode_workspace_bytes), zero-filled as kv_decode needs it (a call leaves it so); the same memory every call."""
    return torch.zeros(kv_decode_workspace_bytes(B, H, G, shares, D), dtype=torch.uint8, device=torch.device("cuda", _device() if device is None else device))


def kv_decode(q, store, leaf_stride, kbase, vbase, block_table, seq_lens, shares, H, G, A, addon, scale, out, workspace):
    """Attention of one query token a sequence over its pages (glyd_gpu_kv_decode): q [B.., H G, D] (rows of any stride),
    block_table int32 [B, pages], seq_lens int32 [B] (B = seq_lens.numel(): the grid's, fixed in a CUDA graph; a length 0: a zero output), out like q, `splits` the most
    CTAs a (sequence, kv head) is split over (kv_decode_splits), workspace a zero-filled uint8 tensor of at least kv_decode_workspace_bytes(B, H, G, splits) (kv_decode_workspace) that is the same
    memory every call (a CUDA graph keeps its address). Nothing here waits for the device, allocates or reads the lengths on the host: it can be captured in a CUDA graph."""
    d = store.get_device()
    if d != _device():
        return _there(kv_decode, q, store, leaf_stride, kbase, vbase, block_table, seq_lens, shares, H, G, A, addon, scale, out, workspace)
    B, D = seq_lens.numel(), q.size(-1)
    _check(q.dim() == 3 and q.size(0) >= B and q.size(1) == H * G and q.stride(2) == 1 and q.stride(1) == D and out.dim() == 3 and out.size(0) >= B and out.shape[1:] == q.shape[1:] and out.stride(2) == 1 and out.stride(1) == D, "q and out [B, H G, D] with contiguous heads")
    _check(block_table.dtype == torch.int32 and block_table.stride(1) == 1 and block_table.size(0) >= B and seq_lens.dtype == torch.int32 and seq_lens.is_contiguous(), "block_table int32 [B, blocks], seq_lens int32 [B]")
    _check(workspace.dtype == torch.uint8 and workspace.numel() >= kv_decode_workspace_bytes(B, H, G, shares, D), "workspace uint8 of kv_decode_workspace_bytes")
    if workspace.data_ptr() not in _WS_SEEN and not torch.cuda.is_current_stream_capturing():  # (a workspace's first call: its counters are zero, as kv_decode_workspace makes it)
        _check(not bool(workspace[: B * H * 4].any()), "workspace zero-filled (kv_decode_workspace)")
        _WS_SEEN.add(workspace.data_ptr())
    ep, nc, rp, ns = _esc(addon)
    r = _fn["kv_decode"](q.data_ptr(), q.stride(0), store.data_ptr(), leaf_stride, kbase.data_ptr(), vbase.data_ptr(), block_table.data_ptr(), block_table.stride(0), seq_lens.data_ptr(), B, shares, H, G, D, A, ep, nc, rp, ns, scale, out.data_ptr(), out.stride(0), workspace.data_ptr(), workspace.numel(), _stream(d))
    if r:
        _fail("kv_decode", r)


def f8_size(w, ld, O, K, heads):
    """The lossless FP8 pack's size (glyd_gpu_f8_size): heads [patches x 8] uint32 written from w [O, K] (uint8 bytes, rows ld apart), the words its var needs back
    (the stream waited for)."""
    d = w.get_device()
    if d != _device():
        return _there(f8_size, d, w, ld, O, K, heads)
    s = _stream(d)
    ws = _workspace("f8_size", d, s, O, K)
    n = ctypes.c_int64()
    r = _fn["f8_size"](w.data_ptr(), ld, O, K, heads.data_ptr(), ws[1], ws[2], ctypes.byref(n), s)
    if r:
        _fail("f8_size", r)
    return n.value


def f8_pack(w, ld, O, K, heads, core, var):
    """w's patches into core [patches x 6144] and var (heads from f8_size, var of the words it said)."""
    d = w.get_device()
    if d != _device():
        return _there(f8_pack, d, w, ld, O, K, heads, core, var)
    r = _fn["f8_pack"](w.data_ptr(), ld, O, K, heads.data_ptr(), core.data_ptr(), var.data_ptr(), var.numel(), _stream(d))
    if r:
        _fail("f8_pack", r)


def f8_unpack(core, heads, var, O, K, out, ld):
    """The matrix back into out [O, K] (rows ld bytes apart): one launch, nothing allocated or waited for."""
    d = core.get_device()
    if d != _device():
        return _there(f8_unpack, d, core, heads, var, O, K, out, ld)
    r = _fn["f8_unpack"](core.data_ptr(), heads.data_ptr(), var.data_ptr(), O, K, out.data_ptr(), ld, _stream(d))
    if r:
        _fail("f8_unpack", r)


def f8_unpack_experts(core, heads, var, E, O, K, ids, out, ld):
    """The matrices ids (int32 on the GPU) of a stack of E [O, K] into out [len(ids), O, K] (rows ld apart): one launch for any ids."""
    d = core.get_device()
    if d != _device():
        return _there(f8_unpack_experts, d, core, heads, var, E, O, K, ids, out, ld)
    r = _fn["f8_unpack_experts"](core.data_ptr(), heads.data_ptr(), var.data_ptr(), E, O, K, ids.data_ptr(), ids.numel(), out.data_ptr(), ld, _stream(d))
    if r:
        _fail("f8_unpack_experts", r)


def f8_linear(core, heads, var, N, K, kind, scale, lds, x, y):
    """y [M, N] = x [M, K] W^T (M of 1 to 64; x and y bf16, rows contiguous) from W [N, K] held as core, heads and var, and its scales (kind: the layout of
    scale, its rows lds apart): one launch, nothing waited for; its workspace and counters the stream's kept ones (made outside a CUDA graph's capture)."""
    d = core.get_device()
    if d != _device():
        return _there(f8_linear, d, core, heads, var, N, K, kind, scale, lds, x, y)
    M, s = x.size(0), _stream(d)
    ws = _workspace("f8_linear", d, s, N, K, M)
    r = _fn["f8_linear"](core.data_ptr(), heads.data_ptr(), var.data_ptr(), N, K, kind, scale.data_ptr(), lds, x.data_ptr(), x.stride(0), M, y.data_ptr(), y.stride(0), ws[1], ws[2],
                         _counters("f8_linear", d, s, -(-N // 128), 1024), s)
    if r:
        _fail("f8_linear", r)
