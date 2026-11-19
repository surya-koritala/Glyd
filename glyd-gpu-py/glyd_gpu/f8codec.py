"""FP8 (e4m3) weight codecs behind one interface: what vllm_fp8.py packs a Linear's weight with, and decodes it from before
every product. A codec holds the bytes of one weight matrix and writes them back, exactly, into a buffer made beforehand:

    p = f8codec.pack(w)          # w: a [rows, cols] contiguous byte matrix on the GPU (uint8, or float8_e4m3fn: its bytes); a Packed
    p.nbytes()                   # the bytes p holds on the GPU
    f8codec.decode(p, out)       # out: a contiguous byte matrix of p.shape, made beforehand; the bytes of w, written into it

decode is the one call a CUDA graph captures, so it takes nothing but p's tensors and out: no host synchronization, no allocation
(its workspace, if it has any, is made in pack), pointers that stay the same from one call to the next. Bytes, not values: the
matrix is whatever vLLM's own FP8 kernel is fed (the checkpoint's [N, K] row-major bytes, a transposed view's storage, Marlin's
int32 tiles read as bytes), and the codec does not know which.

A mixture of experts' experts are one pack of the stacked matrix in experts_form(E, O, K) (the [E * O, K] stack, or its bytes 128 at a time, so
that each matrix is whole patches of 64 rows), as the library's f8_pack takes it, and decode_experts(p, E, ids, out) decodes the experts ids (int32 on the GPU, one outside [0, E) skipped) into out [len(ids), O, K], the matrices
one after the other, in one call that never reads the ids on the host: the step's routed experts are decoded inside a captured graph.

The codec in effect: DEFAULT, or the "f8_codec" option of --additional-config's "glyd". "f8" (DEFAULT) is the lossless FP8 format (the library's
glyd_gpu_f8_pack, _unpack and _unpack_experts through kernels.py; fp8ref.py's docstring is the layout): about 6.7 bits a value where the rows' bytes
are a multiple of 128 (a matrix whose rows are not is packed as its bytes 128 at a time, so that no patch is mostly padding). "identity" (the bytes as
they are, no gain) stays for the checks. Under the Apache License 2.0 (LICENSE), as the rest of Glyd."""
import os

import torch

DEFAULT = "f8"  # the codec packs use
_CODECS = {}


class Packed:
    """One packed matrix: its codec's name, shape (rows, cols: the byte matrix decode writes), the GPU tensors decode reads
    (and nothing else), and the codec's own integers."""

    def __init__(self, codec, shape, tensors, meta=(), obj=None):
        self.codec, self.shape, self.tensors, self.meta, self.obj = codec, tuple(shape), tuple(tensors), tuple(meta), obj  # (obj: the codec's own object of the pack, if it has one)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)


def _bytes(w):
    """w's bytes as a [rows, cols] uint8 matrix, on w's own memory (a view): a contiguous 2-D tensor of one byte a value."""
    if w.dim() != 2 or not w.is_contiguous() or w.element_size() != 1:
        raise ValueError(f"glyd: an FP8 codec packs a contiguous [rows, cols] matrix of one byte a value, not {w.dtype} {tuple(w.shape)} with strides {tuple(w.stride())}")
    return w.view(torch.uint8)


class Identity:
    """The bytes as they are: pack keeps w's own memory (no copy: w is not written to after it), decode copies them into out."""

    name, version = "identity", 1

    @staticmethod
    def pack(w):
        b = _bytes(w)
        return Packed("identity", b.shape, (b,))

    @staticmethod
    def decode(p, out):
        out.view(torch.uint8).copy_(p.tensors[0])  # (a device-to-device copy: a CUDA graph captures it)

    @staticmethod
    def decode_experts(p, E, ids, out):
        """The stack's E matrices [O, K] (p: [E * O, K]) ids selects, into out [len(ids), O, K]; an id outside [0, E) leaves its matrix of out as it is."""
        n, O = ids.numel(), p.shape[0] // E
        stack, o = p.tensors[0].view(E, O, p.shape[1]), out.view(torch.uint8).view(n, O, p.shape[1])
        o.copy_(torch.where(((ids >= 0) & (ids < E))[:, None, None], stack[ids.clamp(0, E - 1).long()], o))  # (no read of ids on the host)


WIDTH = 128  # (the columns of a patch of the lossless format)


def form(rows, cols):
    """The shape a [rows, cols] byte matrix is packed as: itself where cols is a multiple of WIDTH (a row-major weight [N, K]'s own patches: what
    f8_linear reads too) or its bytes do not divide by WIDTH, else its bytes WIDTH at a time (no patch mostly padding)."""
    n = rows * cols
    return (rows, cols) if cols % WIDTH == 0 or n % WIDTH else (n // WIDTH, WIDTH)


def experts_form(E, O, K):
    """The shape a stack of E matrices [O, K] (bytes) is packed as, each matrix whole patches of 64 rows (f8_unpack_experts' unit): form()'s, else its bytes
    WIDTH at a time (Marlin's tiles: few rows of many bytes); None where neither is, or where form() would shape the whole stack otherwise (Lossless.pack
    packs the stack as form() shapes it: a K whose bytes are not a multiple of WIDTH while the stack's are)."""
    rows, cols = form(O, K)
    if rows % 64 and (O * K) % (64 * WIDTH) == 0:
        rows, cols = O * K // WIDTH, WIDTH
    f = (E * rows, cols)
    return f if rows % 64 == 0 and form(*f) == f else None


class Lossless:
    """The lossless FP8 format (kernels.f8_pack and its decoders, on the GPU), of the matrix as form() shapes it (meta: that shape). The weight's own
    memory is not kept: its caller frees it."""

    name, version = "f8", 2  # (version: the layout's)

    @staticmethod
    def pack(w):
        from . import kernels

        b = _bytes(w)
        f = form(*b.shape)
        q = kernels.f8_pack(b.reshape(f))
        return Packed("f8", b.shape, (q.core, q.heads, q.var), (f,), q)

    @staticmethod
    def decode(p, out):
        from . import kernels

        kernels.f8_unpack_into(p.obj, out.view(torch.uint8).view(p.meta[0]))

    @staticmethod
    def decode_experts(p, E, ids, out):
        """As Identity's, p the pack of the stack in experts_form."""
        from . import kernels

        if p.meta[0] != p.shape:  # (a stack packed as another shape would be decoded across the experts' bounds)
            raise ValueError(f"glyd: experts packed as {p.meta[0]}, not as their stack {p.shape}")
        kernels.f8_unpack_experts(p.obj, E, ids, out.view(torch.uint8).view(out.shape[0], p.shape[0] // E, p.shape[1]))


def register(codec):
    """Make codec (name, version, pack(w) -> Packed, decode(p, out), decode_experts(p, E, ids, out) where it holds experts) available by its name; returns it."""
    _CODECS[codec.name] = codec
    return codec


def get(name=None):
    """The codec called name, else DEFAULT's; refused where there is none by that name."""
    # dev: GLYD_F8_CODEC
    name = name or os.environ.get("GLYD_F8_CODEC", "").strip()
    # end dev
    name = name or DEFAULT
    if name not in _CODECS:
        raise ValueError(f"glyd: FP8 codec {name!r}: one of {', '.join(sorted(_CODECS))}")
    return _CODECS[name]


def pack(w, codec=None):
    return get(codec).pack(w)


def decode(p, out):
    get(p.codec).decode(p, out)


def decode_experts(p, E, ids, out):
    get(p.codec).decode_experts(p, E, ids, out)


register(Identity)
register(Lossless)
