"""A model's weights held packed on the GPU: GLinear and GEmbedding in place
of nn.Linear and nn.Embedding, Merged and Part for the Linears that take
the same input run as one product (as serving engines run them), and
pack_modules() to put them in a model's place, the packs made by a
function given.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import kernels as g

SCRATCH = 128 << 20  # weights; bigger matrices are decoded in row blocks
WG_MIN, WG_MAX = int(os.environ.get("GLYD_WG_MIN", 17)), int(os.environ.get("GLYD_WG_MAX", 128))  # Hopper: steps of this many tokens multiply by wgmma
MID_MIN = int(os.environ.get("GLYD_MID_MIN", 17))  # GDDR Ampere and Ada: steps of this many tokens to 64 by mma_gemm_mid


class Scratch:
    buf = {}  # one a device: {device: tensor}


class GLinear(nn.Module):
    """nn.Linear over a packed matrix p (bias: bf16, or None). fused: products
    straight from the packed weights where a kernel takes the step (in the
    mma layouts up to 64 tokens, and prompts but on Hopper; one-token steps
    in the others); else the matrix decoded into the scratch buffer, then
    PyTorch's matmul. exact: every product the matrix decoded whole, then
    F.linear on the input as it came, as nn.Linear does: its outputs bit for
    bit (over fused). gemm_max: the fast format's fused steps, in tokens."""

    def __init__(self, p, bias, fused=True, exact=False, gemm_max=64):
        super().__init__()
        self.p, self.bias = p, bias
        self.fused, self.exact, self.gemm_max = fused, exact, gemm_max
        O, K = p.shape
        step = 64 if isinstance(p, g.Mma) else getattr(p, "rows_per_tile", 1) or 1
        # Whole when it fits the scratch (a split matmul sums in another order), and always for exact.
        self.block = O if exact or O * K <= SCRATCH else max(step, SCRATCH // K // step * step)
        cc = torch.cuda.get_device_capability(p.sm.device)
        self.hopper = cc == (9, 0)  # the TMA and wgmma kernel is sm_90a code: Hopper alone
        self.mid = cc in ((8, 6), (8, 7), (8, 9))  # (on an A100 mma_gemm is the faster, measured)

    def decode_rows(self, r0, r1):
        p, K = self.p, self.p.shape[1]
        buf = Scratch.buf[p.sm.device]
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
        O, K = self.p.shape
        if self.exact:  # as nn.Linear: F.linear on the input as it came, the matrix decoded whole
            return F.linear(x, self.decode_rows(0, O), self.bias)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        # Past WG_MAX tokens on Hopper (a prompt) the tensor cores outrun our decode: decode the matrix, cuBLAS multiplies.
        if self.fused and isinstance(self.p, g.Mma):
            M = x2.shape[0]
            if self.hopper and WG_MIN <= M <= WG_MAX and K % 64 == 0 and isinstance(self.p, g.Mma12):  # TMA and wgmma
                return g.mma_gemm_wg(self.p, x2, self.bias).view(*lead, O)
            if self.mid and MID_MIN <= M <= 64 and K % 64 == 0 and isinstance(self.p, g.Mma12):  # cp.async and mma.sync, the same plan
                return g.mma_gemm_mid(self.p, x2, self.bias).view(*lead, O)
            if M <= 64:
                return g.mma_gemm(self.p, x2, self.bias).view(*lead, O)
            if K % 64 == 0 and not self.hopper:
                return g.mma_gemm_big(self.p, x2, self.bias).view(*lead, O)
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
    its slice (the input kept until all have)."""

    def __init__(self, lin, sizes):
        super().__init__()
        self.lin, self.sizes = lin, list(sizes)
        self.x = self.parts = None
        self.left = 0

    def part(self, i, x):
        if self.x is not x:
            self.x, self.parts, self.left = x, self.lin(x).split(self.sizes, -1), len(self.sizes)
        y = self.parts[i]
        self.left -= 1
        if self.left == 0:
            self.x = self.parts = None
        return y


class Part(nn.Module):
    def __init__(self, group, i):
        super().__init__()
        self.i, self.group = i, [group]  # the group is its module's child, not this one's

    def forward(self, x):
        return self.group[0].part(self.i, x)


class GEmbedding(nn.Module):
    def __init__(self, p, scale=None):
        super().__init__()
        self.p = p
        # Gemma's embedding multiplies its rows by sqrt(hidden size) in the weights' dtype: the same product here
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, ids):
        rows = g.fast_rows(self.p, ids) if isinstance(self.p, g.Fast) else g.rows(self.p, ids)
        rows = rows.view(*ids.shape, -1)
        return rows if self.scale is None else rows * self.scale.to(rows.dtype)


def decoder(model):
    """The decoder stack: model.model, or its language_model when the checkpoint also carries a vision tower (Gemma 3, Gemma 4)."""
    return getattr(model.model, "language_model", model.model)


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
            if mod is not None and all(isinstance(getattr(mod, c, None), nn.Linear) for c in names):
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


@torch.no_grad()
def pack_modules(model, pack_fn, device_of, **mode):
    """Every nn.Linear and nn.Embedding of model replaced by a GLinear (mode: its options) or GEmbedding over its weight
    packed by pack_fn(w, linear) on device_of(module), the bias moved there; kept as it is where pack_fn gives None. A
    weight several modules share (an embedding tied to the output layer) is packed once for each kind. The packs, by
    (weight, device, linear)."""
    packed = {}
    for m in list(model.modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, (nn.Linear, nn.Embedding)):
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
    else up to SCRATCH, past which they are decoded in row blocks), never smaller than it was (another model's)."""
    need = {}
    for m in model.modules():
        if isinstance(m, (GLinear, GEmbedding)):
            d = m.p.sm.device
            need[d] = max(need.get(d, 0), m.p.n if exact else min(m.p.n, SCRATCH))
    for d, n in need.items():
        if d not in Scratch.buf or Scratch.buf[d].numel() < n + 16384 * 8:
            Scratch.buf[d] = torch.empty(n + 16384 * 8, dtype=torch.bfloat16, device=d)
