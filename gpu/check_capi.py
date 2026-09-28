"""glyd_gpu.cu's entry points through both of its hosts on the same random
inputs: the pybind module (PyTorch's JIT build) and the prebuilt library's C
API (the glyd package's glyd/gpu/_lib.py over libglyd_gpu_cudaN.so). Every
output compared bit for bit, and against the weights or an fp32 product so
they are not both wrong: odd shapes, split rows, escapes and exceptions few
and many, 1 to 600 tokens, bias; the errors alike; the package's one-call
paths (GLinear.step, GEmbedding.step) against the checked calls, and a
prompt's matrices decoded ahead (model.Ahead) against decoded on the
current stream; the calls' host time.

    python check_capi.py [LIBRARY]      (default: $GLYD_GPU_LIB, else the one next to glyd_gpu.py)"""
import os, sys, time, torch
import torch.nn.functional as F

if len(sys.argv) > 1:
    os.environ["GLYD_GPU_LIB"] = sys.argv[1]
os.environ.setdefault("GLYD_GPU_MOE_SLOTS", "512")  # the experts' products split K where their units are few, as on an H100
import glyd_gpu as g
from glyd.gpu import _lib as glyd_gpu_lib

assert g._ext is glyd_gpu_lib, "no library found: run build_lib.sh, or give its path"
lib, jit = glyd_gpu_lib, g._jit()
dev = "cuda"
bf = torch.bfloat16
none = torch.empty(0, dtype=bf, device=dev)  # no bias, as glyd_gpu.py passes it
no_ids = torch.empty(0, dtype=torch.int64, device=dev)
counts = {}


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def exact(a, b):
    return torch.equal(bits(a), bits(b))


def both(name, *args, out=()):
    """name through each host, the arguments at positions `out` (its outputs) fresh copies each time; its
    outputs, or what it returns, compared bit for bit. The JIT host's back."""
    got = []
    for host in (jit, lib):
        a = list(args)
        for i in out:
            a[i] = args[i].clone()
        r = getattr(host, name)(*a)
        got.append([r] if r is not None else [a[i] for i in out])
    for x, y in zip(*got):
        assert x.dtype == y.dtype and x.shape == y.shape and exact(x, y), name
    counts[name] = counts.get(name, 0) + 1
    return got[0][0]


errors = []


def both_fail(name, *args):
    """name refused by each host."""
    for host in (jit, lib):
        try:
            getattr(host, name)(*args)
        except RuntimeError as e:
            errors.append(f"{name} ({'jit' if host is jit else 'lib'}): {str(e).splitlines()[0][:100]}")
            continue
        raise AssertionError(f"{name}: no error from {host.__name__}")
    counts[name + " (refused)"] = counts.get(name + " (refused)", 0) + 1


def nan(*shape):
    return torch.full(shape, float("nan"), dtype=bf, device=dev)


def near(y, ref, tol=1e-2):
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < tol, err


def weights(n, wild=0.0):
    """n bf16 weights of a trained matrix's spread, `wild` of them at exponents far from the common ones."""
    w = torch.randn(n, device=dev) * 0.02
    m = torch.rand(n, device=dev) < wild
    w[m] = torch.randn(int(m.sum()), device=dev) * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device=dev).float())
    return w.to(bf)


torch.manual_seed(0)

# The dense format: lane_bits and write_codes (the packer's passes), decode (all tiles, some, staged or in
# place), gemv (tiles of whole rows or splitting them, V 4 and 16, bias).
for shape, wild in [((100_003,), 0.01), ((512, 1024), 0.0), ((300, 384), 0.02), ((64, 16384), 0.001), ((96, 8192), 0.05)]:
    w = weights(shape[0] * (shape[1] if len(shape) > 1 else 1), wild).view(shape)
    p = g.pack(w)
    u = w.contiguous().view(torch.int16).flatten()
    lengths, codes, _ = g.code_tables(g._hist(u).cpu().numpy())
    len_t, code_t = torch.tensor(lengths, dtype=torch.uint8, device=dev), torch.tensor(codes, dtype=torch.int32, device=dev)
    lanes = both("lane_bits", u, len_t, p.tw, p.V).to(torch.int64)
    offs = (torch.cumsum(lanes, 0) - lanes).to(torch.int32)
    assert exact(both("write_codes", u, len_t, code_t, offs, torch.zeros_like(p.stream), p.tw, p.V, out=(4,)), p.stream)
    for tile_words in (p.tile_words, 0):
        d = both("decode", p.sm, p.stream, p.offs, p.tables, p.n, p.tw, p.V, tile_words, no_ids, nan(p.n), out=(9,))
        assert exact(d, u)
    if p.rows_per_tile:
        tiles = torch.tensor([3, 0, 1, 3], device=dev) % ((shape[0] + p.rows_per_tile - 1) // p.rows_per_tile)
        d = both("decode", p.sm, p.stream, p.offs, p.tables, p.n, p.tw, p.V, p.tile_words, tiles, nan(tiles.numel() * p.tw), out=(9,))
        assert exact(d.view(-1, p.rows_per_tile, shape[1])[1], w[: p.rows_per_tile])
    if len(shape) == 2:
        O, K = shape
        x = torch.randn(K, dtype=bf, device=dev)
        bias = torch.randn(O, dtype=bf, device=dev)
        for b in (none, bias):
            for tile_words in (p.tile_words, 0):
                y = both("gemv", p.sm, p.stream, p.offs, p.tables, O, K, p.tw, p.V, tile_words, x, b, nan(O), p.sum, p.count, out=(11,))
                near(y, F.linear(x.float(), w.float(), b.float() if b.numel() else None))
    print(f"dense {shape}, V {p.V}, tiles of {p.tw}{', split rows' if p.split else ''}: {p.bits_per_weight():.2f} bits, the same through both")

# The fast format: fast_decode (a range, listed rows), fast_gemv (4, 8 and 16 weights a lane, a row's warps
# 1-4), fast_gemm (split K or not, 1-600 tokens; O a multiple of 16, refused otherwise), fast_bgemv (2-16
# tokens, X in 1-4 segments).
for (O, K), wild in [((1000, 512), 0.01), ((304, 2304), 0.02), ((17008, 384), 0.001), ((128, 4096), 0.05), ((80, 1536), 0.0)]:
    w = weights(O * K, wild).view(O, K)
    f = g.pack_fast(w)
    fa = (f.sm, f.planes, f.exc, f.exc_base, f.top)
    assert exact(both("fast_decode", *fa, 0, O, no_ids, K, nan(O * K), out=(9,)), w)
    assert exact(both("fast_decode", *fa, O // 3, 7, no_ids, K, nan(7 * K), out=(9,)), w[O // 3 : O // 3 + 7])
    ids = torch.randint(0, O, (9,), device=dev)
    assert exact(both("fast_decode", *fa, 0, 0, ids, K, nan(9 * K), out=(9,)), w[ids])
    bias = torch.randn(O, dtype=bf, device=dev)
    for b in (none, bias):
        bb = b.float() if b.numel() else None
        x = torch.randn(K, dtype=bf, device=dev)
        near(both("fast_gemv", *fa, O, K, x, b, nan(O), out=(9,)), F.linear(x.float(), w.float(), bb))
        for M in [1, 5, 64, 65, 130, 600]:
            x = torch.randn(M, K, dtype=bf, device=dev)
            if O % 16:
                if M == 1:
                    both_fail("fast_gemm", *fa, O, K, x, b, nan(M, O))
                continue
            near(both("fast_gemm", *fa, O, K, x, b, nan(M, O), out=(9,)), F.linear(x.float(), w.float(), bb))
        if K % 512 == 0:
            for M in [2, 4, 8, 16]:
                x = torch.randn(M, K, dtype=bf, device=dev)
                near(both("fast_bgemv", *fa, O, K, x, b, nan(M, O), out=(9,)), F.linear(x.float(), w.float(), bb))
    print(f"fast {(O, K)}: {int(f.exc.numel())} escapes, the same through both")

# The mma layouts, tiered and 12-bit: unpack (all rows, a row block on; a warp a step, and a few warps taking every so
# many), mma_gemm (1-64 tokens), mma_gemm_big (65-600, both variants), mma12_gemm_mid (1-600), mma12_gemm_wg (Hopper:
# refused elsewhere), as the self-test's matrices: odd row blocks, units shared by blocks, escapes and exceptions few
# and many.
hopper = torch.cuda.get_device_capability() == (9, 0)
for O, K, wild in [(64, 64, 0), (192, 128, 0), (128, 4096, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02)]:
    w = weights(O * K, wild).view(O, K)
    bias = torch.randn(O, dtype=bf, device=dev)
    for q in (g.pack_mma(w), g.pack_mma12(w)):
        twelve = isinstance(q, g.Mma12)
        s = "mma12" if twelve else "mma"
        pk = (q.data, q.exc, q.exc_base, q.sym) if twelve else (q.data, q.blocks, q.block_base, q.tiers)
        for warps in (0, 1, 7, 160):  # a warp a step; a few warps each taking every so many steps (a decode ahead)
            assert exact(both(f"{s}_unpack", *pk, K, 0, O, nan(O * K), warps, out=(7,)), w)
            if O >= 128:
                assert exact(both(f"{s}_unpack", *pk, K, 64, 64, nan(64 * K), warps, out=(7,)), w[64:128])
        for b in (none, bias):
            bb = b.float() if b.numel() else None
            for M in [1, 7, 16, 17, 32, 33, 64, 65, 100, 128, 129, 256, 257, 600]:
                x = torch.randn(M, K, dtype=bf, device=dev)
                ref = F.linear(x.float(), w.float(), bb)
                if M <= 64:
                    near(both("mma12_gemm" if twelve else "mma_gemm", *pk, O, K, x, b, nan(M, O), out=(8,)), ref)
                else:
                    for variant in ([0, 1, 2] if M in (65, 600) else [0]):
                        near(both(f"{s}_gemm_big", *pk, O, K, x, b, nan(M, O), variant, out=(8,)), ref)
                if twelve:
                    near(both("mma12_gemm_mid", *pk, O, K, x, b, nan(M, O), out=(8,)), ref)
                    if hopper:
                        near(both("mma12_gemm_wg", *pk, O, K, x, b, nan(M, O), out=(8,)), ref)
                    elif M in (1, 600) and b is none:
                        both_fail("mma12_gemm_wg", *pk, O, K, x, b, nan(M, O))
    print(f"mma {O}x{K}: {int(q.exc_base[-1])} exceptions (12-bit), the same through both")

# A mixture of experts' layer, its E matrices [O, K] stacked: moe_route (each token's k experts sorted by expert, a
# few routed nowhere), mma_moe_unpack and mma12_moe_unpack (the experts hit), mma_moe and mma12_moe (the rows by
# expert, + bias; the gate's SiLU and GELU fused; weighted and each token's rows added, the weights bf16 and fp32),
# 1-300 tokens (passes of 16, 32 and 64, K split over blocks or not; from 48 pairs an expert, mma_gemm_big_kernel's
# tiles), against fp32; and (twice) tokens listing an expert twice, not as a top-k would: more pairs an expert than tokens.
for E, O, K, k, T, wild, *twice in [(8, 256, 192, 2, 1, 0.01), (8, 256, 192, 2, 70, 0.02), (64, 128, 2048, 8, 8, 0.001), (40, 1024, 1536, 8, 3, 0.0), (16, 192, 64, 4, 33, 0.1), (4, 128, 256, 1, 17, 0.0), (4, 128, 208, 2, 150, 0.0), (4, 256, 256, 2, 300, 0.01), (6, 128, 320, 3, 200, 0.02), (5, 192, 128, 2, 160, 0.01), (2, 128, 528, 1, 150, 0.01), (4, 256, 256, 2, 300, 0.01, 1), (4, 128, 208, 2, 300, 0.0, 1)]:
    w = weights(E * O * K, wild).view(E, O, K)
    bias = torch.randn(E, O, dtype=bf, device=dev)
    ids = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
    if twice:
        ids[: 2 * T // 3] = 0  # expert 0 twice: some 450 pairs for 300 tokens
    if T > 2:
        ids[1, 0] = E  # routed nowhere, as an expert-parallel sentinel
    P, flat = T * k, ids.view(-1)
    plan = both("moe_route", flat, E, torch.full((2 + 2 * E + P,), -7, dtype=torch.int32, device=dev), out=(2,))
    valid = (flat < E).nonzero().view(-1)
    n, order = int(plan[0]), plan[2 + 2 * E : 2 + 2 * E + len(valid)].long()
    assert torch.equal(plan[1 : 1 + n].long(), flat[valid].unique()) and int(plan[1 + E + n]) == len(valid)
    assert torch.equal(order, valid[torch.sort(flat[valid], stable=True).indices]), "the pairs sorted by expert, in order within each"
    expert = flat[order]
    x = torch.randn(T, K, dtype=bf, device=dev)
    xs = torch.randn(P, K, dtype=bf, device=dev)  # the down product's input: the pairs in the plan's order
    rows = torch.einsum("pok,pk->po", w[expert].float(), x[order // k].float())  # [valid pairs, O], the plan's order
    rows_s = torch.einsum("pok,pk->po", w[expert].float(), xs[: len(valid)].float())
    for q in (g.pack_mma(w.view(E * O, K)), g.pack_mma12(w.view(E * O, K))):
        twelve = isinstance(q, g.Mma12)
        s = "mma12_moe" if twelve else "mma_moe"
        pk = (q.data, q.exc, q.exc_base, q.sym) if twelve else (q.data, q.blocks, q.block_base, q.tiers)
        # exact: the experts hit decoded into their rows, the rest left as they were
        u = both(f"{s}_unpack", *pk, E, O, K, P, plan, nan(E * O * K), out=(9,)).view(E, O, K)
        hit = torch.zeros(E, dtype=torch.bool, device=dev)
        hit[flat[valid]] = True
        assert exact(u[hit], w[hit]) and torch.isnan(u[~hit].float()).all()
        for b in (none, bias):
            bb = b[expert].float() if b.numel() else 0
            y = both(s, *pk, E, O, K, x, k, 1, plan, 0, b, none, none, nan(P, O), out=(15,))
            near(y[: len(valid)], rows + bb)
            if O % 128 == 0:
                gate, up = (rows + bb).chunk(2, -1)
                for act, f in ((1, F.silu), (2, lambda v: F.gelu(v, approximate="tanh"))):
                    y = both(s, *pk, E, O, K, x, k, 1, plan, act, b, none, none, nan(P, O // 2), out=(15,))
                    near(y[: len(valid)], f(gate) * up)
            for wd in (torch.bfloat16, torch.float32):
                wt = torch.rand(T, k, device=dev).to(wd)
                y = both(s, *pk, E, O, K, xs, k, 0, plan, 0, b, wt, flat, nan(T, O), out=(15,))
                ref = torch.zeros(P, O, device=dev)
                ref[order] = (rows_s + bb) * wt.view(-1)[order, None].float()
                near(y, ref.view(T, k, O).sum(1))
    print(f"moe: {E} experts of {O}x{K}, {T} tokens by {k}{', one expert twice' if twice else ''}, {n} experts hit: the same through both")
both_fail("mma12_moe", *pk, E, O, K, x, k, 1, plan, 1, none, wt, flat, nan(T, O))  # a gate and weights at once

# Attention over packed KV pages: head_dim 64 and 128, 1-16 queries a KV head, pages and tails of 0-63 tokens.
for D, G, pairs, P, tlen in [(128, 7, 8, 3, 5), (128, 1, 2, 1, 0), (64, 7, 4, 2, 63), (128, 16, 2, 2, 17), (128, 8, 16, 16, 40), (64, 2, 3, 5, 1)]:
    k = (torch.randn(P * pairs * 64, D, device=dev) * torch.randn(1, D, device=dev).exp()).to(bf)
    v = torch.randn(P * pairs * D, 64, device=dev).to(bf)
    tk, tv = torch.randn(pairs, tlen, D, device=dev).to(bf), torch.randn(pairs, tlen, D, device=dev).to(bf)
    kp, vp = g.pack_mma(k), g.pack_mma(v)
    qq = torch.randn(pairs * G, D, device=dev).to(bf)
    o = both("attn_decode", qq, kp.data, kp.blocks, kp.block_base, kp.tiers, vp.data, vp.blocks, vp.block_base, vp.tiers, tk, tv, tlen, pairs, G, P, D**-0.5, nan(pairs * G, D), out=(16,))
    keys = torch.cat([k.view(P, pairs, 64, D).transpose(0, 1).reshape(pairs, P * 64, D), tk], 1).float()
    vals = torch.cat([v.view(P, pairs, D, 64).permute(1, 0, 3, 2).reshape(pairs, P * 64, D), tv], 1).float()
    att = torch.softmax(qq.float().view(pairs, G, D) @ keys.transpose(1, 2) * D**-0.5, -1) @ vals
    near(o.view(pairs, G, D), att, 2e-2)
print("attn_decode: head_dim 64 and 128, 1-16 queries a head, pages and tails, the same through both")

# On a stream of its own: that stream handed to the library, the product ordered with the work around it.
w = weights(128 * 256).view(128, 256)
q = g.pack_mma12(w)
pk = (q.data, q.exc, q.exc_base, q.sym)
with torch.cuda.stream(torch.cuda.Stream()):
    assert glyd_gpu_lib._stream(torch.cuda.current_device()) == torch.cuda.current_stream().cuda_stream != 0
    x = torch.randn(33, 256, dtype=bf, device=dev)
    near(both("mma12_gemm_mid", *pk, 128, 256, x, none, nan(33, 128), out=(8,)), F.linear(x.float(), w.float()))
    near(both("mma12_gemm", *pk, 128, 256, x[:7], none, nan(7, 128), out=(8,)), F.linear(x[:7].float(), w.float()))
torch.cuda.synchronize()

# The package's one-call paths (_lib.step and _lib.lookup, as GLinear and GEmbedding call them) against the checked
# calls: 1-64 tokens, both layouts, bias, and an embedding's rows.
from glyd.gpu import model as gm

w = weights(512 * 1024, 0.01).view(512, 1024)
packs = (g.pack_mma(w), g.pack_mma12(w))
for q in packs:
    for b in (None, torch.randn(512, dtype=bf, device=dev)):
        lin = gm.GLinear(q, b)
        for M in range(1, 65):
            x = torch.randn(M, 1, 1024, dtype=bf, device=dev)
            assert exact(lin.step(x), lin.kernel(M)(q, x.view(M, 1024), b)), ("GLinear.step", type(q).__name__, M)
        counts["GLinear.step"] = counts.get("GLinear.step", 0) + 64
# GLinear's routing on an A100 whatever this GPU is (compute capability 8.0 read while it is made): the 12-bit
# layout's 17-64 tokens by mma_gemm_mid, the rest as elsewhere; the one-call path the same functions.
cc = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda device=None: (8, 0)
try:
    a100 = [gm.GLinear(q, None) for q in packs]
finally:
    torch.cuda.get_device_capability = cc
for q, lin in zip(packs, a100):
    twelve = isinstance(q, g.Mma12)
    for M in (1, 16, 17, 32, 33, 64, 65, 128):
        assert lin.kernel(M) is (g.mma_gemm_mid if twelve and 17 <= M <= 64 else g.mma_gemm if M <= 64 else g.mma_gemm_big), ("A100 routing", type(q).__name__, M)
        if M <= 64:
            x = torch.randn(M, 1024, dtype=bf, device=dev)
            assert exact(lin.step(x), lin.kernel(M)(q, x, None)), ("A100 GLinear.step", type(q).__name__, M)
            counts["GLinear.step"] += 1
# A prompt's matrices decoded ahead (model.Ahead; made to on any GPU, beside products of any size): GLinears of odd
# shapes, both layouts, called in turn as a prompt calls them, the first time recorded, then followed; a decode on
# the current stream midway, a prompt that ends before its order does (the next ends the order there), another
# order between (recorded, then followed), then the first again; at 600 tokens, then 2100 (the order kept). The
# products on the order (as many as said, from the first) bit for bit as with their matrices decoded on the current
# stream, the rest the fused kernel's (on Hopper, whose prompts take no fused kernel, decoded on the current stream
# too).
shapes = [(1024, 512), (512, 1024), (3072, 512), (512, 1536), (192, 512), (2048, 1024)]
lins = [gm.GLinear((g.pack_mma12 if i % 2 else g.pack_mma)(weights(O * K, 0.01).view(O, K)), None) for i, (O, K) in enumerate(shapes)]
other = [gm.GLinear(g.pack_mma(weights(O * K).view(O, K)), None) for O, K in shapes[:3]]
for lin in lins + other:
    lin.ahead = 513
gm.set_scratch(torch.nn.ModuleList(lins + other), False)
flops, gm.AHEAD_FLOPS = gm.AHEAD_FLOPS, 0
product, placed = gm.Ahead.product, []
gm.Ahead.product = lambda a, lin, j, f, M: (placed.append(j), product(a, lin, j, f, M))[1]
dev_ = torch.device(dev, torch.cuda.current_device())
for M, first in ((600, 0), (2100, 6)):
    for ls, stop, on in [(lins, None, first), (lins, None, 6), (lins, None, 6), (lins, 4, 4), (lins[:3], None, 3), (lins, None, 3), (other, None, 0), (other, None, 3), (lins, None, 0), (lins, None, 6)]:
        gen = torch.Generator(device=dev).manual_seed(M)
        placed.clear()
        for i, lin in enumerate(ls):
            if i == stop:
                gm.Ahead.stop(dev_)
                lins[0].decode_rows(0, 64)
            x = torch.randn(M, lin.in_features, dtype=bf, device=dev, generator=gen)
            y = lin(x)
            assert exact(y, F.linear(x, g.mma_unpack(lin.p)) if i < on or lin.hopper else g.mma_gemm_big(lin.p, x)), ("a prompt decoded ahead", M, i, on)
        assert placed == list(range(on)), ("the order followed", M, placed, on)
        counts["GLinear decoded ahead"] = counts.get("GLinear decoded ahead", 0) + on
        counts["GLinear off the order (fused)"] = counts.get("GLinear off the order (fused)", 0) + len(ls) - on
    assert any(gm.Ahead.of[dev_].schedule(M // 128 * 128)[0]), "decodes ahead in the order"
# Where Ahead does not take a prompt's product, the fused kernel, as below the decode ahead (but on Hopper): under
# torch.compile (a graph's node: _lib.local.fresh), and a matrix past the scratch (decoded in row blocks, never ahead).
if not lins[0].hopper:
    placed.clear()
    glyd_gpu_lib.local.fresh = True
    for lin in lins:
        x = torch.randn(600, lin.in_features, dtype=bf, device=dev)
        assert exact(lin(x), g.mma_gemm_big(lin.p, x)), ("a compiled prompt: fused", lin.p.shape)
    glyd_gpu_lib.local.fresh = False
    big = gm.GLinear(g.pack_mma(weights(1024 * 512).view(1024, 512)), None)
    big.ahead, big.block = 513, 128  # as a matrix past the scratch
    x = torch.randn(600, 512, dtype=bf, device=dev)
    assert exact(big(x), g.mma_gemm_big(big.p, x)) and not placed, "past the scratch: fused"
    counts["GLinear where Ahead does not take it (fused)"] = len(lins) + 1
# A prompt that ends before its order does leaves decodes ahead queued; here they wait 50 ms (the hold before each
# host's), and meanwhile its modules are let go and memory of their sizes given out and written: none of it where
# the queued decodes read (record_stream), no illegal address. A call below the threshold waits for what is queued.
hold, gm.AHEAD_HOLD = gm.AHEAD_HOLD, 50_000_000
ls = [gm.GLinear((g.pack_mma12 if i % 2 else g.pack_mma)(weights(O * K, 0.01).view(O, K)), None) for i, (O, K) in enumerate(shapes)]
xs = [torch.randn(600, lin.in_features, dtype=bf, device=dev) for lin in ls]
for lin in ls:
    lin.ahead = 513
for n in (6, 6, 3):  # recorded, followed, then a prompt that ends before its order does
    for lin, x in zip(ls[:n], xs):
        lin(x)
a = gm.Ahead.of[dev_]
assert a.live and gm.Ahead.queued and any(k >= 3 for k, _, _ in a.schedule(512)[0][2]), "decodes ahead queued past the prompt"
held = {t.data_ptr(): t.numel() * t.element_size() for lin in ls for t in vars(lin.p).values() if isinstance(t, torch.Tensor)}
del ls, lin
junk = [torch.full((n,), 0x7F, dtype=torch.uint8, device=dev) for n in held.values()]  # offsets of 2^31 or so, where read
assert not held.keys() & {j.data_ptr() for j in junk}, "memory the queued decodes read given out again"
torch.cuda.synchronize()  # the queued decodes done: no illegal address
gm.AHEAD_HOLD = hold
gm.GLinear(g.pack_mma(weights(512 * 512).view(512, 512)), None)(torch.randn(64, 512, dtype=bf, device=dev))
assert not a.live and not gm.Ahead.queued, "a call below the threshold waits for what is queued"
del junk, xs
counts["GLinear decodes ahead past a prompt's end"] = 1
gm.Ahead.product, gm.AHEAD_FLOPS = product, flops
e = weights(1000 * 256).view(1000, 256)
ids = torch.randint(0, 1000, (4, 3), device=dev)
emb = gm.GEmbedding(g.pack_fast(e))  # held: its step keeps the pack's addresses, not the pack
assert exact(emb.step(ids), e[ids])
counts["GEmbedding.step"] = 1

# Refused alike: too many tokens, X not 16-byte aligned, rows not a multiple of 64.
both_fail("mma12_gemm", *pk, 128, 256, torch.randn(65, 256, dtype=bf, device=dev), none, nan(65, 128))
both_fail("mma12_gemm_mid", *pk, 128, 256, torch.randn(4 * 256 + 1, dtype=bf, device=dev)[1:].view(4, 256), none, nan(4, 128))
both_fail("mma12_unpack", *pk, 256, 0, 100, nan(100 * 256), 0)
for host in (jit, lib):  # a stream held, through each host
    host.hold(2000)
counts["hold"] = 1
both_fail("hold", -1)

# A small product's time a call through each host: the host's time where it is the longer (ctypes'), else the kernel's.
x = torch.randn(1, 256, dtype=bf, device=dev)
for host in (jit, lib):
    for name, args in [("mma12_gemm", (*pk, 128, 256, x, none, nan(1, 128))), ("mma12_gemm_mid", (*pk, 128, 256, x, none, nan(1, 128)))]:
        f = getattr(host, name)
        for _ in range(100):
            f(*args)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(2000):
            f(*args)
        torch.cuda.synchronize()
        print(f"{'jit' if host is jit else 'lib'} {name}: {(time.perf_counter() - t) / 2000 * 1e6:.1f} us a call")

for e in errors:
    print("refused:", e)
print(f"library {g._prebuilt()} (CUDA {lib.cuda_version()}), {torch.cuda.get_device_name()}: {sum(counts.values())} calls compared bit for bit, all identical")
for name in sorted(counts):
    print(f"  {name}: {counts[name]}")
