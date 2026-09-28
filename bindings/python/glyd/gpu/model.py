"""A model's weights held packed on the GPU: GLinear and GEmbedding in place
of nn.Linear and nn.Embedding, Merged and Part for the Linears that take
the same input run as one product (as serving engines run them), and
compress() for a model already loaded (hf.py packs one as it loads).

    model = compress(model)              # a bf16 transformers model, in place

A Linear whose matrix the mma layouts take (rows a multiple of 64,
columns of 16) goes in the tiered layout (10.80 bits a weight) or the
12-bit one (12.04, a lighter decode), best_layout's choice for the GPU;
an embedding (rows a multiple of 128 long) in the fast format, its rows
decoded as they are looked up; anything else stays as it is.

Eager, a generation step's product through the prebuilt library is one C
call (_lib.step: what does not change between calls made once); under
torch.compile each GLinear and GEmbedding is one node of the graph
(glyd::linear, glyd::embedding), run as eager, and CUDA graphs capture
its kernels.
"""
import hashlib
import itertools
import os
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import _lib, kernels as g

SCRATCH = 128 << 20  # weights; bigger matrices are decoded in row blocks
WG_MIN, WG_MAX = int(os.environ.get("GLYD_WG_MIN", 17)), int(os.environ.get("GLYD_WG_MAX", 128))  # Hopper: steps of this many tokens multiply by wgmma
MID_MIN = int(os.environ.get("GLYD_MID_MIN", 17))  # Ampere and Ada: steps of this many tokens to 64 by mma_gemm_mid


class Scratch:
    buf = {}  # one a device: {device: tensor}
    graphed = set()  # the devices whose buffer a torch.compile graph's node has decoded into (a CUDA graph keeps its address)
    replaced = []  # those buffers once a bigger one took their place: kept


_modules = weakref.WeakValueDictionary()  # handle: its GLinear or GEmbedding, for the ops below
_handles = itertools.count()


class _Node:
    """A module the ops below run: its handle, and its eager call (step: _lib's, over its own pack), made again for a
    copy (copy.deepcopy, pickle)."""

    def _node(self):
        self.handle = next(_handles)
        _modules[self.handle] = self
        self.step = self._step()

    def __getstate__(self):
        return {**self.__dict__, "step": None}

    def __setstate__(self, state):
        super().__setstate__(state)
        self._node()


class _Weight:
    """A packed module's weight as a model's own code reads it (Llama 4: the embedding's device; HunYuan V4, DeepSeek
    V3.2: a Linear's dtype; Gemma 4: the pad token's row): its device, dtype and shape as they are; its rows by index
    (an embedding's: those rows alone decoded), anything else and a torch function on the matrix decoded."""

    dtype, requires_grad = torch.bfloat16, False

    def __init__(self, m):
        self.m = m

    device = property(lambda self: self.m.p.sm.device)
    shape = property(lambda self: torch.Size(self.m.p.shape))

    def tensor(self):
        return unpack(self.m.p)

    def __getitem__(self, i):
        i = i if isinstance(i, tuple) else (i,)
        if isinstance(self.m, GEmbedding) and isinstance(i[0], (int, torch.Tensor)):
            ids = torch.as_tensor(i[0], device=self.device).remainder(self.shape[0])
            return self.m.rows(ids.reshape(-1)).view(*ids.shape, -1)[(slice(None),) * ids.dim() + i[1:]]
        return self.tensor()[i]

    def __getattr__(self, name):
        return getattr(self.tensor(), name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        f = lambda a: a.tensor() if isinstance(a, cls) else a
        return func(*map(f, args), **{k: f(v) for k, v in (kwargs or {}).items()})


class GLinear(_Node, nn.Module):
    """nn.Linear over a packed matrix p (bias: bf16, or None). fused: products
    straight from the packed weights where a kernel takes the step (in the
    mma layouts up to 64 tokens, and prompts but on Hopper; one-token steps
    in the others); else the matrix decoded into the scratch buffer, then
    PyTorch's matmul. exact: every product the matrix decoded whole, then
    F.linear on the input as it came, as nn.Linear does: its outputs bit for
    bit (over fused). gemm_max: the fast format's fused steps, in tokens."""

    weight = property(_Weight)  # as a model's own code reads it

    def __init__(self, p, bias, fused=True, exact=False, gemm_max=64):
        super().__init__()
        self.p, self.bias = p, bias
        self.fused, self.exact, self.gemm_max = fused, exact, gemm_max
        O, K = p.shape
        self.in_features, self.out_features = K, O
        rows = 64 if isinstance(p, g.Mma) else getattr(p, "rows_per_tile", 1) or 1
        # Whole when it fits the scratch (a split matmul sums in another order), and always for exact.
        self.block = O if exact or O * K <= SCRATCH else max(rows, SCRATCH // K // rows * rows)
        cc = torch.cuda.get_device_capability(p.sm.device)
        self.hopper = cc == (9, 0)  # the TMA and wgmma kernel is sm_90a code: Hopper alone
        self.mid = cc in ((8, 0), (8, 6), (8, 7), (8, 9))  # (an A100 its own kernel: producer and consumer warps)
        self._node()

    def kernel(self, M):
        """The fused product for M tokens in the mma layouts (kernels.py's), or None: decoded, then PyTorch's matmul."""
        K, twelve = self.in_features, isinstance(self.p, g.Mma12)
        if self.hopper and WG_MIN <= M <= WG_MAX and K % 64 == 0 and twelve:  # TMA and wgmma
            return g.mma_gemm_wg
        if self.mid and MID_MIN <= M <= 64 and K % 64 == 0 and twelve:  # cp.async and mma.sync
            return g.mma_gemm_mid
        if M <= 64:
            return g.mma_gemm
        if K % 64 == 0 and not self.hopper:  # past WG_MAX tokens on Hopper (a prompt) the tensor cores outrun our decode
            return g.mma_gemm_big
        return None

    def _step(self):
        """A generation step's product (1-64 tokens) as one C call, where it is a fused one through the prebuilt
        library: _lib.step over the pack, the function for each M; else None."""
        p = self.p
        if not self.fused or self.exact or not isinstance(p, g.Mma) or g.lib() is None:
            return None
        twelve = isinstance(p, g.Mma12)
        name = {g.mma_gemm: "mma12_gemm" if twelve else "mma_gemm", g.mma_gemm_mid: "mma12_gemm_mid", g.mma_gemm_wg: "mma12_gemm_wg"}
        names = [None] + [name.get(self.kernel(M)) for M in range(1, 65)]
        return _lib.step(p.data, *((p.exc, p.exc_base, p.sym, 4) if twelve else (p.blocks, p.block_base, p.tiers, 3)), p.shape, self.bias, names)

    def decode_rows(self, r0, r1):
        p, K = self.p, self.p.shape[1]
        buf = Scratch.buf[p.sm.device]
        if _lib.local.fresh:
            Scratch.graphed.add(p.sm.device)
        out = buf[: (r1 - r0) * K]
        if isinstance(p, g.Mma):
            g.mma_unpack(p, out, r0, r1 - r0)
        elif isinstance(p, g.Fast):
            g._ext.fast_decode(p.sm, p.planes, p.exc, p.exc_base, p.top, r0, r1 - r0, g._none(out.device), K, out.view(torch.int16))
        elif p.split:  # tiles split its rows: decoded whole (it fits the scratch)
            assert r0 == 0 and r1 == p.shape[0]
            g.unpack(p, buf)
        else:
            T = p.rows_per_tile
            tiles = torch.arange(r0 // T, (r1 + T - 1) // T, device=out.device)
            full = buf[: tiles.numel() * p.tw]
            g.decode_tiles(p, tiles, full)
            out = full[: (r1 - r0) * K]
        return out.view(r1 - r0, K)

    def forward(self, x):
        if torch.compiler.is_compiling():  # one node of the graph (glyd::linear), which runs what follows (no gradient, as eager)
            return torch.ops.glyd.linear(x.detach() if x.requires_grad else x, self.handle, self.out_features)
        if self.step is not None:  # a generation step's product: one C call
            y = self.step(x)
            if y is not None:
                return y
        O, K = self.out_features, self.in_features
        if self.exact:  # as nn.Linear: F.linear on the input as it came, the matrix decoded whole
            return F.linear(x, self.decode_rows(0, O), self.bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if self.fused and isinstance(self.p, g.Mma):
            f = self.kernel(x2.shape[0])
            if f is not None:
                return f(self.p, x2, self.bias).view(*lead, O)
        if self.fused and x2.shape[0] == 1:
            f = g.fast_gemv if isinstance(self.p, g.Fast) else g.gemv
            return f(self.p, x2[0], self.bias).view(*lead, O)
        n = x2.shape[0]
        if self.fused and isinstance(self.p, g.Fast) and 1 < n <= 16 and K % 512 == 0:
            # A few tokens: the batched product, the tokens padded to 2, 4, 8 or 16.
            m = 1 << (n - 1).bit_length()
            xp = x2 if m == n else torch.cat([x2, x2.new_zeros(m - n, K)])
            return g.fast_bgemv(self.p, xp, self.bias)[:n].view(*lead, O)
        if self.fused and isinstance(self.p, g.Fast) and n <= self.gemm_max and K % 64 == 0:
            return g.fast_gemm(self.p, x2, self.bias).view(*lead, O)
        if self.block >= O:
            return F.linear(x2, self.decode_rows(0, O), self.bias).view(*lead, O)
        y = torch.empty(x2.shape[0], O, dtype=x.dtype, device=x.device)
        for r0 in range(0, O, self.block):
            r1 = min(O, r0 + self.block)
            y[:, r0:r1] = F.linear(x2, self.decode_rows(r0, r1), None if self.bias is None else self.bias[r0:r1])
        return y.view(*lead, O)


class Merged(nn.Module):
    """Linears that take the same input (q, k, v; gate, up) as one product, as
    serving engines run them: lin computes them all (their rows stacked,
    `sizes` each); the first member called computes it, each member returns
    its slice (the input kept until all have: in a list, whose items cost
    less to set than a module's attributes, and which torch.compile traces
    as the list it is)."""

    def __init__(self, lin, sizes):
        super().__init__()
        self.lin, self.sizes = lin, list(sizes)
        self.held = [None, None, 0]  # the input, its product's slices, the members yet to take theirs

    def part(self, i, x):
        h = self.held
        if h[0] is not x:
            h[0], h[1], h[2] = x, self.lin(x).split(self.sizes, -1), len(self.sizes)
        h[2] -= 1
        y = h[1][i]
        if h[2] == 0:
            h[0] = h[1] = None
        return y


class Part(nn.Module):
    def __init__(self, group, i):
        super().__init__()
        self.i, self.group = i, [group]  # the group is its module's child, not this one's

    def forward(self, x):
        return self.group[0].part(self.i, x)


class GEmbedding(_Node, nn.Module):
    weight = property(_Weight)  # as a model's own code reads it

    def __init__(self, p, scale=None):
        super().__init__()
        self.p = p
        self.num_embeddings, self.embedding_dim = p.shape
        # Gemma's embedding multiplies its rows by sqrt(hidden size) in the weights' dtype: the same product here
        self.register_buffer("scale", scale, persistent=False)
        self._node()

    def _step(self):
        """The fast format's lookup as one C call through the prebuilt library (_lib.lookup); else None."""
        p = self.p
        return _lib.lookup(p.sm, p.planes, p.exc, p.exc_base, p.top, self.embedding_dim) if isinstance(p, g.Fast) and g.lib() is not None else None

    def rows(self, ids):
        """Rows ids of the table, decoded."""
        rows = self.step(ids) if self.step is not None else None
        if rows is None:
            rows = (g.fast_rows(self.p, ids) if isinstance(self.p, g.Fast) else g.rows(self.p, ids)).view(*ids.shape, -1)
        return rows

    def forward(self, ids):
        if torch.compiler.is_compiling():  # one node of the graph (glyd::embedding), which runs what follows
            return torch.ops.glyd.embedding(ids, self.handle, self.embedding_dim)
        rows = self.rows(ids)
        return rows if self.scale is None else rows * self.scale.to(rows.dtype)


# torch.compile: a GLinear or GEmbedding is one node of the graph, an op that runs the module as eager, where a kernel's
# workspace is made for the call alone (a CUDA graph keeps the addresses it captured; the kept ones are eager calls').
def _run(handle, x):
    _lib.local.fresh = True
    try:
        return _modules[handle].forward(x)
    finally:
        _lib.local.fresh = False


@torch.library.custom_op("glyd::linear", mutates_args=())
def _linear(x: torch.Tensor, handle: int, out_features: int) -> torch.Tensor:
    return _run(handle, x)


@_linear.register_fake
def _(x, handle, out_features):
    return x.new_empty((*x.shape[:-1], out_features))


@torch.library.custom_op("glyd::embedding", mutates_args=())
def _embedding(ids: torch.Tensor, handle: int, embedding_dim: int) -> torch.Tensor:
    return _run(handle, ids)


@_embedding.register_fake
def _(ids, handle, embedding_dim):
    return ids.new_empty((*ids.shape, embedding_dim), dtype=torch.bfloat16)


def decoder(model):
    """The decoder stack: model.model, or its language_model when the checkpoint also carries a vision tower (Gemma 3, Gemma 4)."""
    return getattr(model.model, "language_model", model.model)


def plain(m):
    """An nn.Linear that runs as one (its class's forward nn.Linear's), which a GLinear takes the place of; not a
    module of a model's own that is one by class (Llama 4's and Phi-3.5-MoE's routers, DeepSeek V4's grouped output
    projection)."""
    return isinstance(m, nn.Linear) and type(m).forward is nn.Linear.forward


def groups(model):
    """The Linears to run as one product, as (module, names): each decoder layer's q, k, v and gate, up where all are
    nn.Linear (a linear-attention layer, Qwen3-Next, Qwen3.5, has no self_attn: only its MLP merges)."""
    out = []
    try:
        layers = decoder(model).layers
    except AttributeError:  # not a decoder stack we know: nothing merged
        return out
    for layer in layers:
        for mod, names in ((getattr(layer, "self_attn", None), ("q_proj", "k_proj", "v_proj")), (getattr(layer, "mlp", None), ("gate_proj", "up_proj"))):
            if mod is not None and all(plain(getattr(mod, c, None)) for c in names):
                out.append((mod, names))
    return out


def stack(linears):
    """One nn.Linear computing all of linears: their weights (and biases) stacked by rows."""
    w = torch.cat([l.weight.data for l in linears])
    assert len({l.bias is None for l in linears}) == 1
    lin = torch.nn.utils.skip_init(nn.Linear, w.shape[1], w.shape[0], bias=linears[0].bias is not None, dtype=w.dtype)
    lin.weight = nn.Parameter(w, requires_grad=False)
    if lin.bias is not None:
        lin.bias = nn.Parameter(torch.cat([l.bias.data for l in linears]), requires_grad=False)
    return lin


@torch.no_grad()
def merge_linears(model):
    """Every group of groups(model) as one product (Merged over the stacked nn.Linear); the number of groups."""
    gs = groups(model)
    for mod, names in gs:
        mod.merged = Merged(stack([getattr(mod, c) for c in names]), [getattr(mod, c).weight.shape[0] for c in names])
        for i, c in enumerate(names):
            setattr(mod, c, Part(mod.merged, i))
    return len(gs)


def pack(w, linear, layout):
    """w (bf16, on its GPU) packed: for a Linear in `layout` ("mma" tiered, "mma12") where its rows are a multiple of 64
    and its columns of 16, for an embedding in the fast format where its rows are a multiple of 128 long; else None."""
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        return None
    if linear and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0:
        return (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
    if not linear and w.shape[1] % 128 == 0:
        return g.pack_fast(w)
    return None


def unpack(p):
    """A pack's matrix, bf16."""
    if isinstance(p, g.Fast):
        return g.fast_unpack(p)
    return g.mma_unpack(p) if isinstance(p, g.Mma) else g.unpack(p)


def check(p, w, name):
    """p decodes to w bit for bit, else ValueError."""
    if not torch.equal(unpack(p).view(torch.int16), w.view(torch.int16)):
        raise ValueError(f"glyd: {name} decoded to other bits than its weights")


def sha256(w):
    """The sha256 of a tensor's bytes, row-major as safetensors holds them, copied to the host 256 MB at a time."""
    h, b = hashlib.sha256(), w.contiguous().view(-1).view(torch.uint8)
    for a in range(0, b.numel(), 1 << 28):
        h.update(b[a : a + (1 << 28)].cpu().numpy())
    return h.hexdigest()


def auto_layout(model, gpus=1, device=0):
    """best_layout for model's weights: its Linears the mma layouts take, against the rest (a tied weight once, as
    once tied); (layout, why)."""
    from . import moe

    tied = getattr(model, "all_tied_weights_keys", None) or {}
    lin = sum(m.weight.numel() * 2 for m in model.modules() if plain(m) and m.weight.shape[0] % 64 == 0 and m.weight.shape[1] % 16 == 0)
    experts = moe.packable_bytes(model)  # a mixture of experts' layers, packed as the Linears
    lin += experts
    return g.best_layout(lin, sum(p.numel() * p.element_size() for n, p in model.named_parameters() if n not in tied) - lin, gpus, device, moe=experts > 0)


@torch.no_grad()
def pack_modules(model, pack_fn, device_of, **mode):
    """Every nn.Linear and nn.Embedding of model replaced by a GLinear (mode: its options) or GEmbedding over its weight
    packed by pack_fn(w, linear) on device_of(module), the bias moved there; kept as it is where pack_fn gives None. A
    weight several modules share (an embedding tied to the output layer) is packed once for each kind. The packs, by
    (weight, device, linear)."""
    packed = {}
    for m in list(model.modules()):
        for cname, child in list(m.named_children()):
            if plain(child) or isinstance(child, nn.Embedding):
                linear = isinstance(child, nn.Linear)
                dev = device_of(child)
                key = (child.weight.data_ptr(), dev, linear)  # a weight tied to embedding and output: a pack for each
                if key not in packed:
                    packed[key] = pack_fn(child.weight.data.to(dev), linear)
                p = packed[key]
                if p is None:
                    continue
                bias = child.bias.data.to(dev) if linear and child.bias is not None else None
                setattr(m, cname, GLinear(p, bias, **mode) if linear else GEmbedding(p, getattr(child, "embed_scale", None)))
    return packed


def set_scratch(model, exact):
    """The buffer matrices are decoded into, one on each GPU holding packs: the largest pack's weights (exact: whole;
    else up to SCRATCH, past which they are decoded in row blocks), never smaller than it was (another model's). One
    a torch.compile graph's node has decoded into is kept when a bigger one takes its place: a CUDA graph captured
    with it writes there when it is replayed."""
    from . import moe

    need = moe.scratch(model, exact)  # exact: an Experts module's matrices, decoded at once
    for m in model.modules():
        if isinstance(m, (GLinear, GEmbedding)):
            d = m.p.sm.device
            need[d] = max(need.get(d, 0), m.p.n if exact else min(m.p.n, SCRATCH))
    for d, n in need.items():
        old = Scratch.buf.get(d)
        if old is None or old.numel() < n + 16384 * 8:
            Scratch.buf[d] = torch.empty(n + 16384 * 8, dtype=torch.bfloat16, device=d)
            if old is not None and d in Scratch.graphed:
                Scratch.replaced.append(old)
                Scratch.graphed.discard(d)


@torch.no_grad()
def compress(model, *, layout="auto", exact=False, merge=True):
    """Packs an already-loaded bf16 model in place on the GPU and returns it:
    its Linears in `layout` ("auto": best_layout's pick for the GPU; "mma":
    tiered; "mma12": 12-bit) and its embeddings in the fast format, each on
    the GPU its weight is on (a weight on the CPU: the current GPU, and the
    rest of the model with it unless it is spread over several). exact:
    every product decodes its matrix whole and multiplies by F.linear, as
    nn.Linear does: outputs bit for bit bf16's (and no merging). merge: q, k,
    v and gate, up as one product each (not with exact). A mixture of
    experts' Experts modules packed too (moe.py)."""
    cuda = {p.device for p in model.parameters() if p.is_cuda}
    home = min(cuda, key=lambda d: d.index) if cuda else torch.device("cuda", torch.cuda.current_device())
    if merge and not exact:
        merge_linears(model)
    if layout == "auto":
        layout = auto_layout(model, max(1, len(cuda)), home)[0]
    pack_modules(model, lambda w, linear: pack(w, linear, layout), lambda m: m.weight.device if m.weight.is_cuda else home, exact=exact)
    from . import moe

    moe.compress(model, layout, lambda m: next(m.parameters()).device if next(m.parameters()).is_cuda else home, exact)
    if len(cuda) < 2:
        model.to(home)
    set_scratch(model, exact)
    torch.cuda.empty_cache()
    return model
