"""Main's 12-bit layout beside this tree's split byte in one process, for xcheck.py and layer.py: main's library
through a second copy of glyd.gpu._lib (the C API is version 4 in both), main's pack_mma12 taken from its kernels.py,
and pair(), a call through each with its own pack.

    PYTHONPATH=TREE/bindings/python GLYD_GPU_LIB=TREE_LIB python SCRIPT MAIN_TREE MAIN_LIB ..."""
import ast, importlib.util, os
import torch
import torch.nn.functional as F
from glyd.gpu import _lib as new, kernels as g


def _function(src, name):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


class Old12(g.Mma12):
    """A pack in the 12-bit layout before split byte: its four words of symbols (sym), no base."""

    def __init__(self, shape, data, exc, exc_base, sym):
        self.shape, self.data, self.exc, self.exc_base, self.sym = shape, data, exc, exc_base, sym
        self.sm, self.n = data, shape[0] * shape[1]


def load(main_tree, main_lib):
    """(old, old_pack): main's library as a module of _lib's functions, and main's pack_mma12 (making Old12s)."""
    new.load(os.environ["GLYD_GPU_LIB"])
    g._ext = new
    spec = importlib.util.spec_from_file_location("main_lib", new.__file__)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.load(main_lib)
    src, here = open(os.path.join(main_tree, "bindings/python/glyd/gpu/kernels.py")).read(), open(g.__file__).read()
    hist = [ast.get_source_segment(s, _function(s, "_hist")) for s in (src, here)]
    assert hist[0] == hist[1], "_hist differs between the trees"
    ns = {"torch": torch, "F": F, "_hist": g._hist, "Mma12": Old12}
    exec(compile(ast.Module(body=[_function(src, "pack_mma12")], type_ignores=[]), "main:kernels.py", "exec"), ns)
    return old, ns["pack_mma12"]


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def pair(old, f, po, pn):
    """f(pack) through main's library with main's pack po, then through this tree's with split byte's pn: (a, b), each
    a copy of its output or the RuntimeError it raised."""
    got = []
    for lib, p in ((old, po), (new, pn)):
        g._ext = lib
        try:
            got.append(f(p).clone())
        except RuntimeError as e:
            got.append(e)
    g._ext = new
    return got
