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

Eager, a fused product through the prebuilt library is one C call (a
generation step's, and a prompt's to the decode ahead: _lib.step, what
does not change between calls made once); under torch.compile each
GLinear and GEmbedding is one node of the graph (glyd::linear,
glyd::embedding), run as eager, and CUDA graphs capture its kernels.
"""
import hashlib
import itertools
import os
import re
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import _lib, format as fmt, kernels as g

SCRATCH = 128 << 20  # weights; bigger matrices are decoded in row blocks
# Which kernel a product for M tokens takes on a GPU is the library's route (glyd_gpu.cu's route_for, as measured:
# GLYD_WG_MIN, GLYD_WG_MAX, GLYD_MID_MIN and GLYD_DEC_MIN move it there): Hopper's steps and prompts of 17 to 512
# tokens by wgmma, past that decoded for cuBLAS (Qwen3-8B's layer on an H100 PCIe: 638 us against 864 at 512 tokens,
# 1065 against 964 at 640; Qwen3-32B's 1622 against 2253, then within 6% either way to 1024); Ampere's and Ada's 17 to
# 64 (an A100's to 128) by mma_gemm_mid; an A100's 12-bit prompts from 769 tokens decoded for cuBLAS; a prompt of a
# matrix whose K is not a multiple of 64 decoded (the prompt kernel's blocks).
# A prompt's products from this many tokens: each matrix decoded for cuBLAS, the next ones meanwhile (Ahead). On
# GeForce Ada (measured on an RTX 4080 SUPER) past 512 tokens in the tiered layout, past 1792 in the 12-bit one where
# its fused kernel takes the prompt (the library's route AHEAD), else past 640 (exact, or not fused: each matrix
# decoded first); elsewhere, until measured, the fused kernel or the decode as before (the L4, L40S and RTX 6000 Ada
# sum in fp32 at twice the rate: a product's time decodes half as much beside it). The 12-bit layout's length loses
# least across Qwen3-1.7B, 4B and 8B (one pass, fused against decoded ahead, 1024-4096 tokens): to 1792 Qwen3-1.7B's
# fused pass is the faster but at 1280, Qwen3-4B-Instruct-2507's but at 1664, Qwen3-8B's at 1024, 1408 and 1536 alone
# (1.0-4.6% slower at the other six); at 1793-2047 Qwen3-1.7B's is 3.5-4.0% faster, Qwen3-4B's 1.8-3.1% and
# Qwen3-8B's 0.7-4.3% slower.
AHEAD_MIN = int(os.environ.get("GLYD_AHEAD_MIN", 0)) or None
AHEAD_WARPS = int(os.environ.get("GLYD_AHEAD_WARPS", 0))  # a decode ahead's warps an SM (0: 3 tiered, 2 12-bit), few enough to sit beside a cuBLAS block
AHEAD_RATE = float(os.environ.get("GLYD_AHEAD_RATE", 2.2e-3))  # weights decoded ahead beside a product, for each of its weights and tokens
AHEAD_FLOPS = float(os.environ.get("GLYD_AHEAD_FLOPS", 30e9))  # and only beside products of this many flops (cuBLAS's kernels for fewer leave them no room, or slow down beside them)
AHEAD_HOLD = 5000  # ns a product's decodes ahead wait for it to place its blocks


class Scratch:
    buf = {}  # one a device: {device: tensor}
    graphed = set()  # the devices whose buffer a torch.compile graph's node has decoded into (a CUDA graph keeps its address)
    replaced = []  # those buffers once a bigger one took their place: kept


class Ahead:
    """A prompt's products on a GPU with its matrices decoded ahead, on a
    second stream, beside the products before theirs. The order is recorded
    from a prompt (the GLinears it calls whole, until the first comes again)
    and each matrix given a place in the scratch buffer, used as a ring
    (plan). A decode ahead runs beside one of the order's largest products
    (hosts: cuBLAS's kernels for the small ones leave its blocks no room, or
    slow down beside them), started as the product starts (the stream waits
    for an event recorded before it, then AHEAD_HOLD ns, so that the
    product's blocks are placed first), by a few warps an SM (so that an SM
    holds them beside a cuBLAS block), and as much as the product's time
    allows (AHEAD_RATE; schedule), taking the order's rows as their places
    are free (the products that read them done); a product waits for its
    matrix's decodes ahead (events between the streams, never the host) and
    decodes the rest on the current stream (the order's first: all of it,
    nothing to hide behind). A call off the order, or another decode into the
    buffer, waits for the decodes ahead and ends the prompt's run of them
    (the order's first call starts it again); the calls off the order are
    recorded, and one that comes again makes them the order (another model,
    another path through this one). A prompt that ends before its order
    does leaves decodes ahead queued: the next call below the threshold
    waits for them (settle); where that call is the order's next product
    (the output layer at one token), the next prompt ends the order there,
    and a prompt that stops short otherwise (an error: out of memory, an
    interrupt) leaves the order whole. The calls a prompt makes past its
    order's end (an order recorded from a prompt that stopped short, or
    ended there before) are added to it as the next starts. Waits are the current
    stream's on the side stream's (the GPU's, never the host's), none while
    a CUDA graph is captured (it must not wait on work queued before it);
    the packs and the buffer are the side stream's too (record_stream), so
    that none is given out again before its decodes are done."""

    of = {}  # {device: Ahead}
    queued = False  # decodes ahead on some device not yet waited for

    def __init__(self, d):
        self.d, self.side = d, torch.cuda.Stream(d, priority=-1)  # high: its blocks placed as soon as launched
        self.sms = torch.cuda.get_device_properties(d).multi_processor_count
        self.rec, self.chain, self.pos, self.live, self.off = [], None, 0, False, False
        self.tail, self.end = False, None  # rec: the calls past the order's end; the order to end at `end`

    @staticmethod
    def get(d):
        """The device's, where a product may decode ahead: not in a torch.compile graph's node (its CUDA graph
        keeps the addresses it captured) nor while a CUDA graph is captured; else None."""
        if _lib.local.fresh or torch.cuda.is_current_stream_capturing():
            return None
        a = Ahead.of.get(d)
        if a is None:
            a = Ahead.of[d] = Ahead(d)
        return a

    def join(self):
        """The current stream waits for the side stream's decodes ahead (on the GPU), but while a CUDA graph is
        captured."""
        if self.live and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(self.d).wait_stream(self.side)
            self.live = False

    @staticmethod
    def settle(h):
        """Every device's decodes ahead waited for: a call below the threshold (module h's), after a prompt that
        ended before its order did; where h is the order's next product, the order to end there, else the prompt's
        run of them ended (a call off the order midway: what it decodes may write their places)."""
        for a in Ahead.of.values():
            if a.live and a.pos:
                if a.chain[a.pos] == h:
                    a.end = a.pos
                else:
                    a.off = True
            a.join()
        Ahead.queued = any(a.live for a in Ahead.of.values())

    @staticmethod
    def stop(d):
        """Before a decode into the device's buffer on the current stream that is not the order's: the decodes ahead
        done first, and the prompt's run of them ended (their places may be written)."""
        a = Ahead.of.get(d)
        if a is not None and a.live:
            a.join()
            a.off = True

    @staticmethod
    def reset(d):
        """The device's buffer about to be replaced: its decodes ahead done first (the old buffer is given out again
        on the current stream), the order recorded again."""
        a = Ahead.of.pop(d, None)
        if a is not None:
            torch.cuda.current_stream(d).wait_stream(a.side)

    def plan(self, chain, room):
        """The order's places in a ring of `room` weights, each where the last product before it that reads the
        place (conf) is earliest; places 256 weights apart, as PyTorch's allocator's are."""
        ns = [_modules[h].p.n for h in chain]
        offs, conf = [], []
        for j, n in enumerate(ns):
            best = None
            for o in sorted({0} | {(offs[i] + ns[i] + 255) // 256 * 256 for i in range(max(0, j - 32), j)}):
                if o + n <= room:
                    c = next((i for i in range(j - 1, -1, -1) if offs[i] < o + n and o < offs[i] + ns[i]), -1)
                    if best is None or c < best[0]:
                        best = (c, o)
            conf.append(best[0])
            offs.append(best[1])
        self.chain, self.offs, self.conf, self.plans = chain, offs, conf, {}
        self.gate = [torch.cuda.Event() for _ in chain]
        self.ready = [torch.cuda.Event() for _ in chain]
        self.pos, self.off, self.tail, self.end = 0, False, False, None  # (live kept: the last order's decodes ahead are waited for as this one starts)
        for h in chain:  # read and written by the side stream: none given out again before its work is done
            p = _modules[h].p
            for t in (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base):
                t.record_stream(self.side)
        Scratch.buf[self.d].record_stream(self.side)

    def schedule(self, M):
        """For prompts of M tokens: the rows each host starts decoding ahead, [(k, r0, r1)], as many as AHEAD_RATE
        M times its weights, the order's matrices' in turn as their places are free; and each matrix's rows decoded
        ahead (from its first: the rest are decoded before its product). Hosts: the order's first product, and
        those of AHEAD_FLOPS and of 0.4 times the weights of the largest tenth's smallest at least."""
        s = self.plans.get(M)
        if s is None:
            shapes = [_modules[h].p.shape for h in self.chain]
            N = len(shapes)
            beside, done = [[] for _ in range(N)], [0] * N
            big = sorted(O * K for O, K in shapes)[(N - 1) * 9 // 10]  # (not the largest: an output layer would be)
            for j in range(N):
                n = shapes[j][0] * shapes[j][1]
                budget = AHEAD_RATE * M * n if j == 0 or (n >= 0.4 * big and 2 * M * n >= AHEAD_FLOPS) else 0
                for k in range(j + 1, N):
                    O, K = shapes[k]
                    if self.conf[k] >= j or done[k] == O:
                        continue
                    take = min(O - done[k], int(budget // (64 * K)) * 64)
                    if take <= 0:
                        break
                    beside[j].append((k, done[k], done[k] + take))
                    done[k] += take
                    budget -= take * K
            s = self.plans[M] = (beside, done)
        return s

    def find(self, lin):
        """lin's place in the order where the prompt follows it, else None: a matrix the buffer does not hold twice,
        or a call off the order (or with none yet), recorded: a call that comes again ends the recording, the calls
        from it on the new order (the order's own first call starts it again instead: the order ended where the
        last prompt went on below the threshold, or with the calls it made past its end added)."""
        room = Scratch.buf[self.d].numel() - 16384 * 8
        if 2 * lin.p.n > room:
            return None
        h, c = lin.handle, self.chain
        if c is not None and (c[0] not in _modules or c[-1] not in _modules):  # a model let go
            c = self.chain = None
        if c is not None:
            if not self.off and c[self.pos] == h:
                return self.pos
            if c[0] == h:
                if self.tail and self.rec:  # the last prompt went on past the order's end: the order to there
                    self.plan(c + self.rec, room)
                elif self.end == self.pos and not self.off:  # it went on below the threshold from here: the order to here
                    self.plan(c[: self.pos], room)
                self.rec, self.tail, self.end = [], False, None
                return 0
            if not self.off:
                self.tail = self.pos == 0  # past the order's end (else off its path)
            self.join()  # off the order: none of its decodes ahead left queued past here
            self.off = True
        if h in self.rec:
            self.plan(self.rec[self.rec.index(h) :], room)
            self.rec = []
            return 0
        self.rec.append(h)
        return None

    def at(self, k, r0, r1):
        """Rows [r0, r1) of the order's matrix k in its place."""
        K = _modules[self.chain[k]].in_features
        return Scratch.buf[self.d][self.offs[k] + r0 * K : self.offs[k] + r1 * K]

    def product(self, lin, j, product, M):
        """product(W) for lin's matrix W, place j of the order (M: the prompt's tokens)."""
        main = torch.cuda.current_stream(self.d)
        if j == 0:
            self.join()  # a prompt ended off the order: its decodes ahead done first
            self.off, self.m = False, M
        beside, done = self.schedule(self.m // 128 * 128)  # (as for fewer tokens than it has: less time beside the products)
        O = lin.out_features
        if done[j]:
            main.wait_event(self.ready[j])
        if done[j] < O:
            g.mma_unpack(lin.p, self.at(j, done[j], O), done[j], O - done[j])
        if beside[j]:
            self.gate[j].record(main)
        y = product(self.at(j, 0, O).view(O, lin.in_features))
        if beside[j]:
            self.side.wait_event(self.gate[j])
            with torch.cuda.stream(self.side):
                g.hold(AHEAD_HOLD)
                for k, r0, r1 in beside[j]:
                    p = _modules[self.chain[k]].p
                    g.mma_unpack(p, self.at(k, r0, r1), r0, r1 - r0, (AHEAD_WARPS or (2 if isinstance(p, g.Mma12) else 3)) * self.sms)
                    if r1 == done[k]:
                        self.ready[k].record(self.side)
        self.pos = j + 1 if j + 1 < len(self.chain) else 0
        self.live = self.pos > 0  # decodes ahead not yet waited for
        Ahead.queued |= self.live
        return y


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
        args, kwargs = torch.utils._pytree.tree_map(lambda a: a.tensor() if isinstance(a, cls) else a, (args, kwargs or {}))  # (nested: torch.cat's list)
        return func(*args, **kwargs)


class GLinear(_Node, nn.Module):
    """nn.Linear over a packed matrix p (bias: bf16, or None). fused: products
    straight from the packed weights where a kernel takes the step (in the
    mma layouts by the library's route: up to 64 tokens, an A100's 12-bit to
    128, and prompts: on Hopper to 512 tokens, on an A100 in the 12-bit
    layout to 768; one-token steps in the others); else the matrix decoded
    into the scratch buffer, then
    PyTorch's matmul (on GeForce Ada a prompt past 512 tokens tiered, past
    1792 12-bit fused and past 640 not, decoded ahead of its product where
    Ahead takes it). exact: every
    product the matrix decoded whole, then F.linear on the input as it came,
    as nn.Linear does: its outputs bit for bit (over fused). gemm_max: the
    fast format's fused steps, in tokens."""

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
        self.gpu = gpu_code(cc, torch.cuda.get_device_name(p.sm.device))  # the GPU as the library's routes take it
        self.a100 = cc == (8, 0)  # its mid kernel takes steps to 128 tokens; its prompts past 768 are decoded for cuBLAS
        self.step_max = 128 if self.a100 else 64  # tokens to which a step's kernel (not a prompt's) is one C call (_step)
        self.hopper = cc == (9, 0)  # the TMA and wgmma kernel is sm_90a code: Hopper alone
        # prompts decoded ahead, then cuBLAS, whatever K (the library's route AHEAD where fused and not exact)
        twelve = 1793 if fused and not exact else 641
        self.ahead = AHEAD_MIN or ((twelve if isinstance(p, g.Mma12) else 513) if self.gpu == g.GEFORCE + 89 else 1 << 62)
        self._node()

    def route(self, M):
        """The library's route for M tokens on this GPU (kernels.py's DECODE, GEMM, MID, WG, BIG, AHEAD), and the last
        token count from M on that takes it (the mma layouts)."""
        return g.route(self.p, self.gpu, M)

    def after(self, M, routes):
        """The first token count from M on whose route is one of routes, else 1 << 62."""
        while M < 1 << 62:
            r, last = self.route(M)
            if r in routes:
                return M
            M = last + 1
        return 1 << 62

    def kernel(self, M):
        """The fused product for M tokens in the mma layouts (kernels.py's), or None: decoded, then PyTorch's matmul
        (a prompt's matrices decoded ahead, Ahead). The library's route, a prompt's by the prompt kernel (K a multiple
        of 64) but from self.ahead tokens where the matrix fits the scratch: decoded ahead (one past it never is)."""
        r = self.route(M)[0]
        if r in (g.BIG, g.AHEAD):
            return g.mma_gemm_big if self.in_features % 64 == 0 and (M < self.ahead or self.block < self.out_features) else None
        return {g.GEMM: g.mma_gemm, g.MID: g.mma_gemm_mid, g.WG: g.mma_gemm_wg}.get(r)

    def decoded(self, M):
        """Whether a prompt of M tokens is decoded for cuBLAS, never fused (kernel(M) None, and no fused fallback in
        whole()): the library's route DECODE (an A100's in the 12-bit layout from 769 tokens, where cuBLAS on the
        decoded matrix outruns the fused kernel; Hopper's past its wgmma kernel)."""
        return self.route(M)[0] == g.DECODE

    def _step(self):
        """A product as one C call where it is a fused one through the prebuilt library (glyd_gpu_*_linear, by
        kernel(M)'s route): a generation step's to step_max tokens (64, an A100's 128: mma_gemm_mid's), and a
        prompt's past the last of them to self.ahead tokens by the prompt kernel (an A100's in the 12-bit layout to
        768, from which it is decoded for cuBLAS); _lib.step over the pack; else None."""
        p = self.p
        if not self.fused or self.exact or not isinstance(p, g.Mma) or g.lib() is None:
            return None
        small = {g.mma_gemm: g.GEMM, g.mma_gemm_mid: g.MID, g.mma_gemm_wg: g.WG}
        routes = [None] + [small.get(self.kernel(M)) for M in range(1, self.step_max + 1)]
        while len(routes) > 1 and routes[-1] is None:  # past the steps' kernels: a prompt's (big) or none
            routes.pop()
        big = self.kernel(len(routes)) is g.mma_gemm_big
        top = min(self.ahead, self.after(len(routes), (g.DECODE,)))
        twelve = isinstance(p, g.Mma12)
        return _lib.step(p.data, *((p.exc, p.exc_base, p.sym, 4) if twelve else (p.blocks, p.block_base, p.tiers, 3)), p.shape, self.bias, routes, big, top)

    def decode_rows(self, r0, r1):
        p, K = self.p, self.p.shape[1]
        Ahead.stop(p.sm.device)
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

    def whole(self, product, x):
        """product(W), this matrix decoded whole into the scratch buffer, for input x: ahead, for a prompt of
        self.ahead tokens or more where its order says this product comes next (Ahead); where Ahead does not take
        it (the order not known yet, recorded now, or not followed; under torch.compile; during a capture), fused
        where a kernel takes the prompt; else decoded now."""
        M = x.numel() // self.in_features
        a = Ahead.get(self.p.sm.device) if M >= self.ahead and isinstance(self.p, g.Mma) else None
        j = a.find(self) if a is not None else None
        if j is not None:
            return a.product(self, j, product, M)
        if isinstance(self.p, g.Mma) and self.fused and not self.exact and not self.hopper and self.in_features % 64 == 0 and not self.decoded(M):
            return g.mma_gemm_big(self.p, x, self.bias)
        return product(self.decode_rows(0, self.out_features))

    def forward(self, x):
        if torch.compiler.is_compiling():  # one node of the graph (glyd::linear), which runs what follows (no gradient, as eager)
            return torch.ops.glyd.linear(x.detach() if x.requires_grad else x, self.handle, self.out_features)
        if Ahead.queued and x.numel() < self.ahead * self.in_features:  # a prompt ended before its order did
            Ahead.settle(self.handle)
        if self.step is not None:  # a fused product: one C call
            y = self.step(x)
            if y is not None:
                return y
        O, K = self.out_features, self.in_features
        if self.exact:  # as nn.Linear: F.linear on the input as it came, the matrix decoded whole (a prompt's ahead)
            if x.numel() < self.ahead * K:
                return F.linear(x, self.decode_rows(0, O), self.bias)
            return self.whole(lambda w: F.linear(x, w, self.bias), x)
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
            return self.whole(lambda w: F.linear(x2, w, self.bias), x2).view(*lead, O)
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
        # Gemma's embedding multiplies its rows by sqrt(hidden size) in the weights' dtype, NLLB-MoE's by a float
        # (a scalar kept in fp32 as the product runs): the same product here
        if isinstance(scale, torch.Tensor) or scale is None:
            self.register_buffer("scale", scale, persistent=False)
        else:
            self.scale = scale
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
        s = self.scale
        return rows if s is None else rows * (s.to(rows.dtype) if isinstance(s, torch.Tensor) else s)


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


def gpu_code(cc, name):
    """A GPU as the library's routes take it (glyd_gpu.h, glyd_gpu_gpu): its compute capability cc, major * 10 +
    minor, plus its class by name: GEFORCE with "GeForce" in it, A10 with "A10" in it as a word (an A10, not an A10G,
    A100 or A40), else none."""
    cls = g.GEFORCE if "GeForce" in name else g.A10 if re.search(r"\bA10\b", name, re.ASCII) else 0
    return cc[0] * 10 + cc[1] + cls


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
        for mod, names in zip((getattr(layer, "self_attn", None), getattr(layer, "mlp", None)), fmt.GROUPS):
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
            n = m.p.n if exact else min(m.p.n, SCRATCH)
            if isinstance(m, GLinear) and isinstance(m.p, g.Mma) and m.p.n <= SCRATCH and m.ahead < 1 << 62:
                n = max(n, 2 * m.p.n)  # a prompt's matrices decoded ahead (Ahead): two at once at least
            need[d] = max(need.get(d, 0), n)
    for d, n in need.items():
        old = Scratch.buf.get(d)
        if old is None or old.numel() < n + 16384 * 8:
            Ahead.reset(d)  # its places were in the old buffer
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
