"""Glyd for Python: compression for the data that fills object storage.

    import glyd
    c = glyd.compress(data)                    # --max; level="ultra"/"cold"/"default"/"fast"
    c = glyd.compress(log_bytes, records=True) # logs, dumps, CSV, JSON lines as typed columns
    data = glyd.decompress(c)                  # any Glyd stream

    p = glyd.pack([obj1, obj2, ...])           # many small objects as one stream
    obj2 = glyd.unpack(p, 1)

    with glyd.Store("bucket/") as s:           # objects compressed across each other
        i = s.put("wed.tar", data)             # a delta against the object it most resembles
        data = s.get(i)

Binds the C ABI (include/glyd.h) through ctypes. The shared library is
found next to this file (libglyd.dylib / .so / glyd.dll), at $GLYD_LIB,
or on the system path; bindings/python/build.sh builds and places it.
"""
import ctypes
import os
import sys

__version__ = "0.14.4"

_LEVELS = {"default": 0, "fast": 1, "turbo": 2, "max": 3, "ultra": 4, "cold": 5, "max-long": 6}


def _load():
    """libglyd_store (the codec and the store, BUSL-1.1) when present,
    else libglyd (the codec alone, BSD-3-Clause OR GPL-2.0; Store then raises)."""
    ext = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    pre = "" if sys.platform == "win32" else "lib"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("GLYD_LIB")]
    for stem in ("glyd_store", "glyd"):
        candidates += [os.path.join(here, pre + stem + ext), pre + stem + ext]
    last = None
    for c in candidates:
        if not c:
            continue
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            last = e
    raise OSError(f"libglyd not found (set GLYD_LIB or run bindings/python/build.sh): {last}")


_lib = _load()
_u8p = ctypes.POINTER(ctypes.c_uint8)
_lib.glyd_version.restype = ctypes.c_char_p
_lib.glyd_free.argtypes = [_u8p, ctypes.c_size_t]
_lib.glyd_compress2.argtypes = [_u8p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_decompress2.argtypes = [_u8p, ctypes.c_size_t, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_decompressed_len.argtypes = [_u8p, ctypes.c_size_t]
_lib.glyd_decompressed_len.restype = ctypes.c_int64
_lib.glyd_compress_with_base.argtypes = [_u8p, ctypes.c_size_t, _u8p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_decompress_with_base.argtypes = [_u8p, ctypes.c_size_t, _u8p, ctypes.c_size_t, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_pack.argtypes = [ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_unpack_object.argtypes = [_u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_pack_len.argtypes = [_u8p, ctypes.c_size_t]
_lib.glyd_pack_len.restype = ctypes.c_int64
_HAS_STORE = hasattr(_lib, "glyd_store_open")
if _HAS_STORE:
    _lib.glyd_store_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.glyd_store_open.restype = ctypes.c_void_p
_lib.glyd_store_close.argtypes = [ctypes.c_void_p]
_lib.glyd_store_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _u8p, ctypes.c_size_t]
_lib.glyd_store_put.restype = ctypes.c_int64
_lib.glyd_store_get.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_u8p), ctypes.POINTER(ctypes.c_size_t)]
_lib.glyd_store_id_of.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.glyd_store_id_of.restype = ctypes.c_int64
_lib.glyd_store_delete.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.glyd_store_compact.argtypes = [ctypes.c_void_p]
_lib.glyd_store_compact.restype = ctypes.c_int64
_lib.glyd_store_flush.argtypes = [ctypes.c_void_p]
_lib.glyd_store_rebase.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.glyd_store_verify.argtypes = [ctypes.c_void_p]
_lib.glyd_store_verify.restype = ctypes.c_int64
_lib.glyd_store_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
_lib.glyd_store_set_level.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.glyd_store_count.argtypes = [ctypes.c_void_p]
_lib.glyd_store_count.restype = ctypes.c_int64


def version():
    return _lib.glyd_version().decode()


def _buf(data):
    b = bytes(data)
    return (ctypes.c_uint8 * len(b)).from_buffer_copy(b) if b else None, len(b)


def _take(out, out_len):
    """Bytes from a library buffer, which is then freed."""
    n = out_len.value
    b = ctypes.string_at(out, n) if n else b""
    _lib.glyd_free(out, n)
    return b


def compress(data, level="max", records=False, threads=0):
    """Compress bytes at a level ("default", "fast", "turbo", "max",
    "ultra", "cold"); records=True for logs, dumps, CSV and JSON lines;
    threads=1 for one core."""
    src, n = _buf(data)
    out, out_len = _u8p(), ctypes.c_size_t()
    if _lib.glyd_compress2(src, n, _LEVELS[level], int(bool(records)), threads, ctypes.byref(out), ctypes.byref(out_len)):
        raise ValueError("glyd: bad argument")
    return _take(out, out_len)


def decompress(data):
    """Any Glyd stream back (a base-mode stream needs decompress_with_base)."""
    src, n = _buf(data)
    out, out_len = _u8p(), ctypes.c_size_t()
    r = _lib.glyd_decompress2(src, n, ctypes.byref(out), ctypes.byref(out_len))
    if r:
        raise ValueError("glyd: not a Glyd stream, or corrupt" if r == -2 else "glyd: bad argument")
    return _take(out, out_len)


def decompressed_len(data):
    src, n = _buf(data)
    r = _lib.glyd_decompressed_len(src, n)
    if r < 0:
        raise ValueError("glyd: not a Glyd stream")
    return r


def compress_with_base(base, data, ultra=False):
    """A new version against an old one; decoding needs the same base."""
    b, bn = _buf(base)
    s, sn = _buf(data)
    out, out_len = _u8p(), ctypes.c_size_t()
    if _lib.glyd_compress_with_base(b, bn, s, sn, int(bool(ultra)), ctypes.byref(out), ctypes.byref(out_len)):
        raise ValueError("glyd: bad argument")
    return _take(out, out_len)


def decompress_with_base(base, data):
    b, bn = _buf(base)
    s, sn = _buf(data)
    out, out_len = _u8p(), ctypes.c_size_t()
    r = _lib.glyd_decompress_with_base(b, bn, s, sn, ctypes.byref(out), ctypes.byref(out_len))
    if r:
        raise ValueError("glyd: corrupt stream or wrong base" if r == -2 else "glyd: bad argument")
    return _take(out, out_len)


def pack(objects, level="max"):
    """Many small objects as one stream with an index (2-4x fewer bytes
    than a dictionary per object on events and logs)."""
    objs = [bytes(o) for o in objects]
    keep = [(ctypes.c_uint8 * len(o)).from_buffer_copy(o) if o else None for o in objs]
    ptrs = (_u8p * max(len(objs), 1))(*[ctypes.cast(k, _u8p) if k is not None else _u8p() for k in keep])
    lens = (ctypes.c_size_t * max(len(objs), 1))(*[len(o) for o in objs])
    out, out_len = _u8p(), ctypes.c_size_t()
    if _lib.glyd_pack(ptrs, lens, len(objs), _LEVELS[level], ctypes.byref(out), ctypes.byref(out_len)):
        raise ValueError("glyd: bad argument")
    return _take(out, out_len)


def unpack(packed, index):
    """Object `index` of a pack."""
    src, n = _buf(packed)
    out, out_len = _u8p(), ctypes.c_size_t()
    r = _lib.glyd_unpack_object(src, n, index, ctypes.byref(out), ctypes.byref(out_len))
    if r:
        raise ValueError("glyd: not a pack, corrupt, or no such object" if r == -2 else "glyd: bad argument")
    return _take(out, out_len)


def pack_len(packed):
    src, n = _buf(packed)
    r = _lib.glyd_pack_len(src, n)
    if r < 0:
        raise ValueError("glyd: not a pack")
    return r


class Store:
    """Objects compressed across each other: put() keeps an object as a
    delta against the stored object it most resembles when that pays,
    small objects in packs; get() rebuilds it. Metadata lives in `path`;
    the objects there too, or in `s3` (s3://bucket/prefix, or any
    S3-compatible service through AWS_ENDPOINT_URL)."""

    def __init__(self, path, s3=None):
        if not _HAS_STORE:
            raise OSError("glyd: the store needs libglyd_store (the glyd-store crate); libglyd carries the codec only")
        self._h = _lib.glyd_store_open(os.fsencode(path), s3.encode() if s3 else None)
        if not self._h:
            raise OSError(f"glyd: cannot open the store at {path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._h:
            _lib.glyd_store_close(self._h)
            self._h = None

    def put(self, name, data):
        src, n = _buf(data)
        i = _lib.glyd_store_put(self._h, name.encode(), src, n)
        if i < 0:
            raise OSError("glyd: put failed")
        return i

    def get(self, id):
        out, out_len = _u8p(), ctypes.c_size_t()
        r = _lib.glyd_store_get(self._h, id, ctypes.byref(out), ctypes.byref(out_len))
        if r:
            raise KeyError(id)
        return _take(out, out_len)

    def id_of(self, name):
        i = _lib.glyd_store_id_of(self._h, name.encode())
        return None if i < 0 else i

    def delete(self, id):
        if _lib.glyd_store_delete(self._h, id):
            raise KeyError(id)

    def compact(self):
        return _lib.glyd_store_compact(self._h)

    def flush(self):
        _lib.glyd_store_flush(self._h)

    def rebase(self, id):
        if _lib.glyd_store_rebase(self._h, id):
            raise KeyError(id)

    def verify(self):
        """Objects that failed to read back."""
        return _lib.glyd_store_verify(self._h)

    def stats(self):
        raw, stored = ctypes.c_uint64(), ctypes.c_uint64()
        _lib.glyd_store_stats(self._h, ctypes.byref(raw), ctypes.byref(stored))
        return raw.value, stored.value

    def set_level(self, level):
        if _lib.glyd_store_set_level(self._h, _LEVELS[level]):
            raise ValueError(level)

    def __len__(self):
        return _lib.glyd_store_count(self._h)
