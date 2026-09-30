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


def both_fail(name, *args, says=""):
    """name refused by each host (its message saying says)."""
    for host in (jit, lib):
        try:
            getattr(host, name)(*args)
        except RuntimeError as e:
            assert says in str(e), (name, says, str(e)[:300])
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
# many), mma_gemm (1-64 tokens), mma_gemm_big (65-600, every variant), mma12_gemm_mid (1-600), mma12_gemm_wg (Hopper:
# refused elsewhere), as the self-test's matrices: odd row blocks, units shared by blocks, escapes and exceptions few
# and many.
hopper = torch.cuda.get_device_capability() == (9, 0)
for O, K, wild in [(64, 64, 0), (192, 128, 0), (128, 4096, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (17408, 1024, 0.01)]:
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
                    for variant in ([0, 1, 2, 3] if M in (65, 600) else [0]):
                        near(both(f"{s}_gemm_big", *pk, O, K, x, b, nan(M, O), variant, out=(8,)), ref)
                if twelve:
                    near(both("mma12_gemm_mid", *pk, O, K, x, b, nan(M, O), out=(8,)), ref)
                    if hopper:
                        near(both("mma12_gemm_wg", *pk, O, K, x, b, nan(M, O), out=(8,)), ref)
                    elif M in (1, 600) and b is none:
                        both_fail("mma12_gemm_wg", *pk, O, K, x, b, nan(M, O))
    print(f"mma {O}x{K}: {int(q.exc_base[-1])} exceptions (12-bit), the same through both")

# The 12-bit layout's words: its base hb (0-120) in each byte of sym[0], sym[1-3] zero; any other words refused by
# its entry points, through both hosts (the 12-bit layout before split byte held its 15 commonest exponents there).
q = g.pack_mma12(weights(128 * 256).view(128, 256))
x = torch.randn(8, 256, dtype=bf, device=dev)
for sym in ([0x7B7A7978, 0x7F7E7D7C, 0x83828180, 0x868584], [121 * 0x01010101, 0, 0, 0], [q.sym[0], 1, 0, 0], [q.sym[0] ^ 1, 0, 0, 0]):
    pk = (q.data, q.exc, q.exc_base, sym)
    both_fail("mma12_gemm", *pk, 128, 256, x, none, nan(8, 128))
    both_fail("mma12_gemm_mid", *pk, 128, 256, x, none, nan(8, 128))
    both_fail("mma12_gemm_big", *pk, 128, 256, x, none, nan(8, 128), 0)
    both_fail("mma12_linear", *pk, 128, 256, x, none, nan(8, 128), -1)
    both_fail("mma12_unpack", *pk, 256, 0, 128, nan(128 * 256), 0)
print("the 12-bit layout's words other than its base refused alike")

# mma_gemm_big's own choice of blocks: of 128 tokens on GeForce Ada where the last of 256 would be half empty or
# less, to 1024 tokens tiered and 4224 12-bit (as measured), and on an A100 12-bit to 640 (its others of 256 by two
# row blocks, variant 3); of 256 past 128 tokens elsewhere. On an A100 two candidates give the same bits where they
# split K alike, so a matrix too whose two split K differently at every length GLinear fuses there (256 x 16384:
# on 40 to 220 SMs at each of these lengths, by the grid's rule), and the other candidate's bits checked to differ.
ada = torch.cuda.get_device_capability() == (8, 9) and "GeForce" in torch.cuda.get_device_name()
a100 = torch.cuda.get_device_capability() == (8, 0)
for (O, K), Ms in (((256, 512), (300, 600, 800, 1025, 1100, 4200, 4353)), ((256, 16384), (129, 256, 257, 384, 385, 512, 513, 640, 641, 768))):
    w = weights(O * K).view(O, K)
    for q, most in ((g.pack_mma(w), 1024), (g.pack_mma12(w), 4224)):
        here = a100 and isinstance(q, g.Mma12)
        for M in Ms:
            x = torch.randn(M, K, dtype=bf, device=dev)
            v = 1 if (ada or here) and 1 <= M % 256 <= 128 and M <= (640 if here else most) else 3 if here else 2
            y = g.mma_gemm_big(q, x)
            assert exact(y, g.mma_gemm_big(q, x, variant=v)), ("blocks of 128 or 256", type(q).__name__, (O, K), M, v)
            if here and K == 16384:
                assert not exact(y, g.mma_gemm_big(q, x, variant=4 - v)), ("an A100's candidates told apart", M, v)
            counts["mma_gemm_big's blocks"] = counts.get("mma_gemm_big's blocks", 0) + 1

# Prompt products on two streams at once on one GPU, through each host: each stream's done counters its own (the C
# API's: one stream's products at a time on a set), so each product the same as alone. A small one: its few blocks
# (a unit's stages shared) beside the other stream's (with a set a device, 4% of them wrong).
q = g.pack_mma(weights(192 * 1024).view(192, 1024))
pk, x = (q.data, q.blocks, q.block_base, q.tiers), torch.randn(100, 1024, dtype=bf, device=dev)
streams = [torch.cuda.Stream() for _ in range(2)]
assert len({lib._counters("mma_gemm_big", torch.cuda.current_device(), t.cuda_stream, 1, 1) for t in streams}) == 2, "a set a stream"
for host in (jit, lib):
    alone = nan(100, 192)
    host.mma_gemm_big(*pk, 192, 1024, x, none, alone, 0)
    ys = [[nan(100, 192) for _ in range(1000)] for _ in streams]
    torch.cuda.synchronize()
    for i in range(1000):
        for t, y in zip(streams, ys):
            with torch.cuda.stream(t):
                host.mma_gemm_big(*pk, 192, 1024, x, none, y[i], 0)
    torch.cuda.synchronize()
    assert all(exact(y, alone) for y in ys[0] + ys[1]), ("two streams' prompt products at once", host.__name__)
counts["mma_gemm_big on two streams at once"] = 4000

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

# The routes: the library's (glyd_gpu_*_route, through both hosts) against GLinear's rule on main before the routes
# moved into the library, as main last had it (model.py at 44393a9, v0.24.0), written out here, with the L4's decode
# for cuBLAS since (its class 3089: from 896 tokens tiered and 2560 12-bit) and the L40S's decode ahead (its class 4089:
# from 1024 tiered and 2048 12-bit): its kernel(M), else whole()'s decode, ahead from its decode-ahead threshold (which
# never looked at K) and now below it; on each GPU (a compute capability, plus its class by name: GeForce, A10, L4,
# L40S), both layouts, K a multiple of 64 or not, 0-5000 tokens;
# each run's last token count the last that takes its route. Then glyd_gpu_*_linear by this GPU's route (-1) and by each route given: its kernel's bits,
# through both hosts (DECODE and AHEAD the prompt kernel's, on every GPU), or refused alike where the route's kernel
# does not take the product: DECODE and AHEAD where K is not a multiple of 64 with cudaErrorNotSupported, the one
# refusal of a route the library gives (glyd_gpu.h), every such route past 64 tokens refused so.
MID_MIN, DEC_MIN = int(os.environ.get("GLYD_MID_MIN", 17)), int(os.environ.get("GLYD_DEC_MIN", 0)) or None
WG_MIN, WG_MAX = int(os.environ.get("GLYD_WG_MIN", 17)), int(os.environ.get("GLYD_WG_MAX", 1024))


def main_ahead(gpu, twelve, fused=True, exact_=False):
    """main's GLinear.ahead: a prompt's matrices decoded ahead from this many tokens, whatever K: GeForce Ada's (513
    tiered; 12-bit 1793 fused and not exact, else 641), an A10's but exact (512 tiered, 640 12-bit); and since, an
    L40S's but exact (its class, 4089: 1024 tiered, 2048 12-bit)."""
    if gpu == 1089:
        return (1793 if fused and not exact_ else 641) if twelve else 513
    if gpu == 4089:
        return (2048 if twelve else 1024) if not exact_ else 1 << 62
    return (640 if twelve else 512) if gpu == 2086 and not exact_ else 1 << 62


def main_dec(gpu, twelve):
    """main's GLinear.dec: a 12-bit prompt decoded for cuBLAS, never fused, from GLYD_DEC_MIN tokens where it is set
    (any GPU), else an A100's from 769; and since, an L4's (its class, 3089) from 2560 tokens 12-bit and 896
    tiered."""
    if twelve and DEC_MIN:
        return DEC_MIN
    if gpu == 3089:
        return 2560 if twelve else 896
    return 769 if twelve and gpu % 1000 == 80 else 1 << 62


def main_route(gpu, twelve, K, M):
    """main's GLinear, fused and not exact, its matrix within the scratch, as a route: kernel(M) (WG, MID, GEMM, BIG;
    decoded(M) DECODE), else whole()'s decode: ahead (AHEAD) from main_ahead, now (DECODE) below it (Hopper's past
    WG_MAX, K not a multiple of 64)."""
    cc = gpu % 1000
    a100, hopper, mid = cc == 80, cc == 90, cc in (80, 86, 87, 89)
    ahead = main_ahead(gpu, twelve)
    if hopper and WG_MIN <= M <= WG_MAX and K % 64 == 0 and twelve:
        return g.WG
    if mid and MID_MIN <= M <= (128 if a100 else 64) and K % 64 == 0 and twelve:
        return g.MID
    if M >= main_dec(gpu, twelve):  # decoded(M)
        return g.DECODE
    if M <= 64:
        return g.GEMM
    if K % 64 == 0 and not hopper and M < ahead:
        return g.BIG
    return g.AHEAD if M >= ahead else g.DECODE


looked_up = 0  # the library's routes compared with main_route (lookups, apart from the calls compared through both hosts)
for gpu in (80, 86, 87, 89, 1086, 1089, 2086, 3089, 4089, 90, 100, 120, 1120):
    for twelve in (False, True):
        for K in (1024, 1040):
            name = "mma12_route" if twelve else "mma_route"
            got = [lib._route(name, gpu, 512, K, M) for M in range(5001)]  # (without GLYD_GPU_WITH_SPLIT: main's rule; SPLIT's, asked, below)
            assert [r for r, _ in got] == [main_route(gpu, twelve, K, M) for M in range(5001)], ("routes", gpu, twelve, K)
            for M, (r, last) in enumerate(got):
                assert all(got[i][0] == r for i in range(M, min(last, 5000) + 1)) and (last >= 5000 or got[last + 1][0] != r), ("a route's last", gpu, twelve, K, M)
            assert jit.mma12_route(gpu, 512, K, 700) == got[700] if twelve else jit.mma_route(gpu, 512, K, 700) == got[700]
            looked_up += 5001
from glyd.gpu import model as gm

gm.Split.of[torch.device(dev, torch.cuda.current_device())] = False  # (the GLinears below take today's routes; the route SPLIT's own checks at the end)
# a GPU's code: its compute capability and its class by name, alike in the library (C) and the package (Python)
assert jit.gpu() == lib.gpu() == gm.gpu_code(torch.cuda.get_device_capability(), torch.cuda.get_device_name())
for name, cls in (("NVIDIA A10", g.A10), ("NVIDIA A10-24GB", g.A10), ("NVIDIA A10G", 0), ("NVIDIA A100-SXM4-80GB", 0), ("NVIDIA A40", 0), ("NVIDIA RTX A6000", 0), ("NVIDIA GeForce RTX 4080 SUPER", g.GEFORCE), ("A10", g.A10), ("NVIDIA A10_X", 0),
                  ("NVIDIA L4", g.L4), ("L4", g.L4), ("NVIDIA L40S", g.L40S), ("L40S", g.L40S), ("NVIDIA L40", 0), ("NVIDIA RTX 6000 Ada Generation", 0),
                  ("NVIDIA H100 PCIe", g.PCIE), ("NVIDIA A100-PCIE-40GB", g.PCIE), ("NVIDIA A10 PCIe", g.A10), ("NVIDIA H100 80GB HBM3", 0), ("NVIDIA GH200 480GB", g.GH200), ("NVIDIA H200", 0)):
    assert gm.gpu_code((8, 6), name) == 86 + cls, (name, cls)
here = lib.gpu()
for O, K, wild in [(192, 128, 0), (1024, 2048, 0.001), (192, 4096, 0.1), (192, 1040, 0.01)]:
    w = weights(O * K, wild).view(O, K)
    bias = torch.randn(O, dtype=bf, device=dev)
    for q in (g.pack_mma(w), g.pack_mma12(w)):
        twelve = isinstance(q, g.Mma12)
        s = "mma12" if twelve else "mma"
        pk = (q.data, q.exc, q.exc_base, q.sym) if twelve else (q.data, q.blocks, q.block_base, q.tiers)
        for b in (none, bias):
            for M in [1, 16, 17, 33, 64, 65, 128, 129, 600, 1100, 2000]:
                x = torch.randn(M, K, dtype=bf, device=dev)
                routes = {g.route(q, here, M)[0], g.GEMM if M <= 64 else g.BIG} | ({g.MID} if twelve else set())
                for r in [-1] + sorted(routes):
                    k = g.route(q, here, M)[0] if r < 0 else r
                    # refused: a prompt to be decoded (DECODE, AHEAD) where K is not a multiple of 64 (no kernel takes
                    # it: cudaErrorNotSupported, the caller decodes W), and a kernel given for such a K
                    if k in (g.DECODE, g.AHEAD) and K % 64:
                        assert r >= 0 or M > 64 or (twelve and DEC_MIN and M >= DEC_MIN), ("refused by the route", s, K, M, k)
                        both_fail(f"{s}_linear", *pk, O, K, x, b, nan(M, O), r, says="operation not supported")
                        counts["linear refused, not supported (K not a multiple of 64)"] = counts.get("linear refused, not supported (K not a multiple of 64)", 0) + 1
                        continue
                    if k in (g.MID, g.WG, g.BIG) and K % 64:
                        both_fail(f"{s}_linear", *pk, O, K, x, b, nan(M, O), r)
                        continue
                    y = both(f"{s}_linear", *pk, O, K, x, b, nan(M, O), r, out=(8,))
                    if k == g.GEMM:
                        ref = getattr(lib, f"{s}_gemm")
                        want = nan(M, O)
                        ref(*pk, O, K, x, b, want)
                    elif k in (g.MID, g.WG):
                        want = nan(M, O)
                        getattr(lib, f"mma12_gemm_{'mid' if k == g.MID else 'wg'}")(*pk, O, K, x, b, want)
                    else:  # BIG; DECODE and AHEAD by the prompt kernel
                        want = nan(M, O)
                        getattr(lib, f"{s}_gemm_big")(*pk, O, K, x, b, want, 0)
                    assert exact(y, want), ("linear", s, O, K, M, r)
                    counts["linear, its route's kernel"] = counts.get("linear, its route's kernel", 0) + 1
print(f"routes on 13 GPUs as main's GLinear took them (the L4's decode and the L40S's decode ahead since); linear by this GPU's ({here}) and by each route, as the route's kernel")

# The route SPLIT (option 2): its rule pinned, as measured end to end (benchmarks/gpu/option2-2026-09-29): a 12-bit
# prompt, K a multiple of 64, on an A100 SXM (80) from 769 to 4096 tokens, on a GH200 (6090) from 2048 to 8192 for a
# matrix of O and K at least 5120, on an H100 SXM or H200 (90) and a PCIe card
# (5080, 5090) never (not measured); its decode's
# SMs by the GPU and M, asked for (GLYD_GPU_WITH_SPLIT); every other route main's, through both hosts; without the flag
# none of it, every route main's (the C API's other callers: v0.25.1's routes). The route's 'last'
# as every route's.
SPLIT_MIN, SPLIT_MAX, SPLIT_SMS = (int(os.environ.get(v, 0)) for v in ("GLYD_SPLIT_MIN", "GLYD_SPLIT_MAX", "GLYD_SPLIT_SMS"))


def split_rule(gpu, twelve, O, K, M):
    """The route SPLIT's decode SMs by the rule (0: another route)."""
    cc = gpu % 1000
    a100, hopper = gpu == 80, gpu == g.GH200 + 90
    if not twelve or K % 64 or SPLIT_MIN < 0 or cc < 80:
        return 0
    lo = SPLIT_MIN or (769 if a100 else 2048 if hopper and O >= 5120 and K >= 5120 else 1 << 62)
    hi = SPLIT_MAX or (4096 if a100 else 8192)
    if not lo <= M <= hi:
        return 0
    if SPLIT_SMS:
        return SPLIT_SMS
    if a100:
        return 12 if M < 1536 else 8 if M < 3072 else 4
    if cc == 90:
        return 12 if M < 6144 else 4
    return 12


pinned = 0
for gpu in (80, 5080, 90, 5090, 6090, 86, 89, 1089, 2086, 3089, 4089, 100, 120):
    for twelve in (False, True):
        for O, K in ((512, 1024), (512, 1040), (131072, 1024), (4096, 4096), (5120, 5120), (5056, 8192), (8192, 5056)):  # (8B's class on Hopper; one large there; each side under 5120)
            name = "mma12_route" if twelve else "mma_route"
            Ms = sorted({*range(0, 2100, 7), 767, 768, 769, 1023, 1024, 1025, 1535, 1536, 2047, 2048, 3071, 3072, 4096, 4097, 6143, 6144, 8192, 8193})
            for M in Ms:
                asked = gpu | g.WITH_SPLIT
                r, last = lib._route(name, asked, O, K, M)
                sms = split_rule(gpu, twelve, O, K, M)
                assert (r == g.SPLIT) == (sms > 0) and (sms or r == main_route(gpu, twelve, K, M)), ("the route SPLIT's rule", gpu, twelve, O, K, M, r)
                assert (lib.mma12_split_sms(asked, O, K, M) == jit.mma12_split_sms(asked, O, K, M) == sms) if twelve else True, ("the route SPLIT's SMs", gpu, O, K, M)
                assert lib._route(name, gpu, O, K, M)[0] == main_route(gpu, twelve, K, M) and (not twelve or lib.mma12_split_sms(gpu, O, K, M) == 0), ("not asked: main's route", gpu, O, K, M)
                if last < 1 << 62:
                    assert lib._route(name, asked, O, K, last)[0] == r and lib._route(name, asked, O, K, last + 1)[0] != r, ("the route SPLIT's last", gpu, O, K, M, last)
                pinned += 1
print(f"the route SPLIT's rule: {pinned} routes pinned, asked (an A100 SXM's from 769 to 4096; a GH200's from 2048 to 8192, O and K at least 5120; no H100 SXM, H200 or PCIe card) and not (main's), its SMs alike through both hosts")

# Its decode, built for few SMs: every row bit for bit as the pack's (rows from 0 and a row block on; grids for 1, 3,
# 16 and 200 SMs), through both hosts; K not a multiple of 64 refused alike.
for O, K, wild in [(64, 64, 0), (192, 128, 0), (1024, 2048, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (17408, 1024, 0.01)]:
    w = weights(O * K, wild).view(O, K)
    q = g.pack_mma12(w)
    pk = (q.data, q.exc, q.exc_base, q.sym)
    for sms in (1, 3, 16, 200):
        assert exact(both("mma12_unpack_split", *pk, K, 0, O, nan(O * K), sms, out=(7,)), w), ("the route SPLIT's decode", O, K, sms)
        if O >= 128:
            assert exact(both("mma12_unpack_split", *pk, K, 64, O - 64, nan((O - 64) * K), sms, out=(7,)), w[64:]), ("the route SPLIT's decode, from row 64", O, K, sms)
    print(f"the route SPLIT's decode {O}x{K}: {int(q.exc_base[-1])} exceptions, bit for bit on 1-200 SMs, the same through both")
q = g.pack_mma12(weights(128 * 1040).view(128, 1040))
both_fail("mma12_unpack_split", q.data, q.exc, q.exc_base, q.sym, 1040, 0, 128, nan(128 * 1040), 16, says="")

# The package's one-call paths (_lib.step and _lib.lookup, as GLinear and GEmbedding call them) against the checked
# calls: a step's (to step_max tokens), a prompt's to the decode ahead (past it: none), both layouts, bias, and an
# embedding's rows.

w = weights(512 * 1024, 0.01).view(512, 1024)
packs = (g.pack_mma(w), g.pack_mma12(w))
for q in packs:
    for b in (None, torch.randn(512, dtype=bf, device=dev)):
        lin = gm.GLinear(q, b)
        for M in list(range(1, 65)) + [65, 100, 128, 129, 256, 300, 512, 600, 700]:
            x = torch.randn(M, 1, 1024, dtype=bf, device=dev)
            f = lin.kernel(M)
            if f is None or (M > lin.step_max and f is not g.mma_gemm_big):  # decoded (ahead), then cuBLAS; a step's kernel past step_max (Hopper's mma_gemm_wg to GLYD_WG_MAX): the checked call's
                assert lin.step(x) is None, ("GLinear.step past the fused kernels", type(q).__name__, M)
                continue
            assert exact(lin.step(x), f(q, x.view(M, 1024), b)), ("GLinear.step", type(q).__name__, M)
            counts["GLinear.step"] = counts.get("GLinear.step", 0) + 1
# GLinear's routing on an A100 whatever this GPU is (compute capability 8.0 read while it is made: the library's route
# for an A100): the 12-bit layout's GLYD_MID_MIN (17) to 128 tokens by mma_gemm_mid, from its decode threshold (769;
# GLYD_DEC_MIN where set) decoded for cuBLAS (None), the rest as elsewhere; the one-call path the same functions.
cc, gpu_name = torch.cuda.get_device_capability, torch.cuda.get_device_name


def made_as(cap, name, make):
    """make() with this GPU read as another while GLinears are made: its compute capability and name."""
    torch.cuda.get_device_capability, torch.cuda.get_device_name = lambda device=None: cap, lambda device=None: name
    try:
        return make()
    finally:
        torch.cuda.get_device_capability, torch.cuda.get_device_name = cc, gpu_name


a100 = made_as((8, 0), "NVIDIA A100-SXM4-40GB", lambda: [gm.GLinear(q, None) for q in packs])
for q, lin in zip(packs, a100):
    twelve = isinstance(q, g.Mma12)
    for M in (1, 16, 17, 32, 33, 64, 65, 128, 129, 768, 769, 4096):
        want = g.mma_gemm_mid if twelve and MID_MIN <= M <= 128 else None if twelve and M >= main_dec(80, twelve) else g.mma_gemm if M <= 64 else g.mma_gemm_big
        assert lin.kernel(M) is want, ("A100 routing", type(q).__name__, M)
        if M <= 64 or want is g.mma_gemm_mid:
            x = torch.randn(M, 1024, dtype=bf, device=dev)
            assert exact(lin.step(x), lin.kernel(M)(q, x, None)), ("A100 GLinear.step", type(q).__name__, M)
            counts["GLinear.step"] += 1
    if twelve:  # a prompt from its decode threshold through forward: decoded for cuBLAS, not the fused kernel (nor whole()'s fallback to it)
        gm.set_scratch(torch.nn.ModuleList([lin]), False)
        x = torch.randn(main_dec(80, True), 1024, dtype=bf, device=dev)
        assert lin.step(x) is None and exact(lin(x), F.linear(x, g.mma_unpack(q))), "an A100's prompt from its decode threshold, decoded"
        counts["A100 prompt decoded"] = 1
# And on Hopper whatever this GPU is (9.0 while made): no prompt kernel for the one-call path, so a prompt's product is
# the checked call's (mma_gemm_wg to WG_MAX tokens in the 12-bit layout, else decoded, then cuBLAS).
hopper = made_as((9, 0), "NVIDIA H100 PCIe", lambda: [gm.GLinear(q, None) for q in packs])
for q, lin in zip(packs, hopper):
    for M in (65, 128, 600, 2100):
        assert lin.step(torch.randn(M, 1024, dtype=bf, device=dev)) is None, ("Hopper GLinear.step, a prompt", type(q).__name__, M)
    # Its routes: the 12-bit layout's GLYD_WG_MIN (17) to GLYD_WG_MAX tokens by mma_gemm_wg (past 128 in
    # mma12_wgp_kernel), past WG_MAX decoded for cuBLAS (None); the tiered one's steps to 64 tokens by mma_gemm, past
    # them decoded.
    if isinstance(q, g.Mma12):
        want = [(M, g.mma_gemm_wg) for M in (17, 128, 129, WG_MAX)] + [(WG_MAX + 1, None)]
    else:
        want = [(M, g.mma_gemm) for M in (1, 17, 64)] + [(M, None) for M in (65, 129, WG_MAX, WG_MAX + 1)]
    for M, f in want:
        assert lin.kernel(M) is f, ("Hopper routing", type(q).__name__, M)
# And on an A10 whatever this GPU is (8.6 and its name while made; the library's routes by its class, 2086): a prompt
# decoded ahead from 640 tokens in the 12-bit layout and 512 tiered, fused below (the one-call path to there), but exact
# (as elsewhere: decoded on the current stream); on an A10G (half-rate tensor cores, no class) fused throughout.
for name, want in (("NVIDIA A10", (512, 640)), ("NVIDIA A10G", (1 << 62, 1 << 62))):
    a10 = made_as((8, 6), name, lambda: [gm.GLinear(q, None) for q in packs])
    assert all(made_as((8, 6), name, lambda: gm.GLinear(q, None, exact=True)).ahead == 1 << 62 for q in packs), (name, "exact: no decode ahead")
    for q, lin in zip(packs, a10):
        twelve = isinstance(q, g.Mma12)
        a = want[twelve]
        assert lin.gpu == (g.A10 if name == "NVIDIA A10" else 0) + 86 and lin.ahead == a == main_ahead(lin.gpu, twelve), (name, type(q).__name__, lin.ahead)
        b = min(a, main_dec(lin.gpu, twelve))  # fused below (a 12-bit prompt decoded from GLYD_DEC_MIN where it is set)
        for M in (a - 1, a) if a < 1 << 62 else (1024,):
            x = torch.randn(M, 1024, dtype=bf, device=dev)
            assert (lin.kernel(M) is g.mma_gemm_big) == (M < b) and (lin.step(x) is None) == (M >= b), (name, type(q).__name__, M)
            counts["GLinear's routes on an A10 / A10G"] = counts.get("GLinear's routes on an A10 / A10G", 0) + 1
# And on an L4 whatever this GPU is (8.9 and its name while made; the library's routes by its class, 3089): a prompt
# decoded for cuBLAS on the current stream from 896 tokens tiered and 2560 12-bit (GLYD_DEC_MIN where set), fused
# below (the one-call path to there), never ahead, exact as elsewhere (decoded on the current stream); on an L40 (the
# same compute capability, no class) fused throughout, but a 12-bit prompt from GLYD_DEC_MIN where it is set. (An
# L40S's, decoded ahead from its thresholds: below.)
for name, want in (("NVIDIA L4", (main_dec(3089, False), main_dec(3089, True))), ("NVIDIA L40", (main_dec(89, False), main_dec(89, True)))):
    l4 = made_as((8, 9), name, lambda: [gm.GLinear(q, None) for q in packs])
    assert all(made_as((8, 9), name, lambda: gm.GLinear(q, None, exact=True)).ahead == 1 << 62 for q in packs), (name, "exact: no decode ahead")
    for q, lin in zip(packs, l4):
        twelve = isinstance(q, g.Mma12)
        d = want[twelve]
        assert lin.gpu == (g.L4 if name == "NVIDIA L4" else 0) + 89 and lin.ahead == 1 << 62, (name, type(q).__name__, lin.gpu, lin.ahead)
        for M in (d - 1, d) if d < 1 << 62 else (896, 2560, 5000):
            x = torch.randn(M, 1024, dtype=bf, device=dev)
            assert (lin.kernel(M) is g.mma_gemm_big) == (M < d) and lin.decoded(M) == (M >= d) and (lin.step(x) is None) == (M >= d), (name, type(q).__name__, M)
            counts["GLinear's routes on an L4 / L40"] = counts.get("GLinear's routes on an L4 / L40", 0) + 1
        if d < 1 << 62:  # a prompt from its decode threshold through forward: decoded for cuBLAS, not the fused kernel
            gm.set_scratch(torch.nn.ModuleList([lin]), False)
            x = torch.randn(d, 1024, dtype=bf, device=dev)
            assert exact(lin(x), F.linear(x, g.mma_unpack(q))), (name, type(q).__name__, "a prompt from its decode threshold, decoded")
            counts["L4 prompt decoded"] = counts.get("L4 prompt decoded", 0) + 1
# And on an L40S whatever this GPU is (8.9 and its name while made; the library's routes by its class, 4089): a prompt
# decoded ahead from 1024 tokens tiered and 2048 12-bit (the route AHEAD), fused below (the one-call path to there),
# but exact (decoded on the current stream, as elsewhere); its scratch buffer holding two of its matrices. A 12-bit
# prompt from GLYD_DEC_MIN tokens where it is set: the route DECODE (decoded, as the ahead does; main's semantics).
l40s = made_as((8, 9), "NVIDIA L40S", lambda: [gm.GLinear(q, None) for q in packs])
assert all(made_as((8, 9), "NVIDIA L40S", lambda: gm.GLinear(q, None, exact=True)).ahead == 1 << 62 for q in packs), ("L40S", "exact: no decode ahead")
for q, lin in zip(packs, l40s):
    twelve = isinstance(q, g.Mma12)
    a, d = main_ahead(4089, twelve), main_dec(4089, twelve)
    assert lin.gpu == g.L40S + 89 and lin.ahead == a == (2048 if twelve else 1024), ("L40S", type(q).__name__, lin.gpu, lin.ahead)
    for M in (a - 1, a):
        x = torch.randn(M, 1024, dtype=bf, device=dev)
        route = g.DECODE if M >= d else g.AHEAD if M >= a else g.BIG
        assert (lin.kernel(M) is g.mma_gemm_big) == (route == g.BIG) and lin.decoded(M) == (M >= d) and (lin.step(x) is None) == (route != g.BIG) and lin.route(M)[0] == route, ("L40S", type(q).__name__, M)
        counts["GLinear's routes on an L40S"] = counts.get("GLinear's routes on an L40S", 0) + 1
    gm.set_scratch(torch.nn.ModuleList([lin]), False)
    assert gm.Scratch.buf[q.sm.device].numel() >= 2 * q.n, ("L40S", "the scratch buffer holds two of its matrices")
# On GeForce Ada whatever this GPU is: GLinear's decode-ahead threshold main's whatever K (513 tiered; 12-bit 1793 fused
# and not exact, 641 exact or not fused), and its kernel(M) main's, K a multiple of 64 or not (1024, 1040).
for K in (1024, 1040):
    wk = weights(512 * K, 0.01).view(512, K)
    for q in (g.pack_mma(wk), g.pack_mma12(wk)):
        twelve = isinstance(q, g.Mma12)
        for fused, exact_ in ((True, False), (False, False), (True, True)):
            lin = made_as((8, 9), "NVIDIA GeForce RTX 4090", lambda: gm.GLinear(q, None, fused=fused, exact=exact_))
            assert lin.gpu == 1089 and lin.ahead == main_ahead(1089, twelve, fused, exact_), ("GeForce Ada's decode ahead", K, twelve, fused, exact_, lin.ahead)
            counts["GLinear's decode ahead, as main's"] = counts.get("GLinear's decode ahead, as main's", 0) + 1
            if fused and not exact_:
                for M in (1, 16, 17, 64, 65, 512, 513, 1792, 1793, 5000):
                    want = {g.WG: g.mma_gemm_wg, g.MID: g.mma_gemm_mid, g.GEMM: g.mma_gemm, g.BIG: g.mma_gemm_big}.get(main_route(1089, twelve, K, M))
                    assert lin.kernel(M) is want, ("GeForce Ada's kernel(M), as main's", K, twelve, M)
# A prompt's matrices decoded ahead (model.Ahead; made to on any GPU, beside products of any size): GLinears of odd
# shapes, both layouts, called in turn as a prompt calls them: the first prompt stopped short (an error), recorded,
# the order from it then made whole by the calls past its end; followed; a decode on the current stream midway; a
# prompt stopped short (the order kept whole); one gone on below the threshold at its next product (the next ends
# the order there), then one past the order's end (the order to there again); another order between (recorded, then
# followed), then the first again; at 600 tokens (on Hopper WG_MAX + 1, past its wgmma kernel), then 2100 (the order
# kept). The products on the order (as many as said, from the first) bit for bit as with their matrices decoded on
# the current stream, the rest the fused kernel's (on Hopper, whose prompts past WG_MAX take no fused kernel, and an
# A100's 12-bit prompts from 769 tokens, which take none, and a matrix whose K is not a multiple of 64, which none
# takes: decoded on the current stream too), below the threshold the step's kernel's. Then all of it again with GLinears made as on an A100 (its
# compute capability and name read while they are made), where this GPU is not one: its routes on this GPU's kernels.
shapes = [(1024, 512), (512, 1024), (3072, 512), (512, 1536), (192, 512), (2048, 1024), (768, 1040)]  # (K 1040: never fused)
flops, gm.AHEAD_FLOPS = gm.AHEAD_FLOPS, 0
product, placed = gm.Ahead.product, []
gm.Ahead.product = lambda a, lin, j, f, M: (placed.append(j), product(a, lin, j, f, M))[1]
dev_ = torch.device(dev, torch.cuda.current_device())
for route in [None] + ([] if torch.cuda.get_device_capability() == (8, 0) else [(8, 0)]):
    make = lambda: ([gm.GLinear((g.pack_mma12 if i % 2 else g.pack_mma)(weights(O * K, 0.01).view(O, K)), None) for i, (O, K) in enumerate(shapes)], [gm.GLinear(g.pack_mma(weights(O * K).view(O, K)), None) for O, K in shapes[:3]])
    lins, other = made_as(route, "NVIDIA A100-SXM4-40GB", make) if route else make()
    for lin in lins + other:
        lin.ahead = 513
        lin.step = lin._step()  # its one-call path to the new threshold
    gm.Ahead.reset(dev_)  # (the route before's order done with)
    gm.set_scratch(torch.nn.ModuleList(lins + other), False)
    tag = " (as on an A100)" if route else ""
    n = len(lins)
    common = [(lins, n, None, None), (lins, 4, 4, None), (lins[:3], 3, None, None), (lins, n, None, None), (lins[:4], 3, None, 3), (lins, 3, None, None),
              (lins, n, None, None), (other, 0, None, None), (other, 3, None, None), (lins, 0, None, None), (lins, n, None, None)]
    first = WG_MAX + 1 if route is None and cc() == (9, 0) else 600  # (Hopper's 12-bit prompts to WG_MAX: mma_gemm_wg's, never on the order)
    for M, seq in ((first, [(lins[:3], 0, None, None), (lins, 3, None, None)] + common), (2100, common)):
        for ls, on, stop, short in seq:  # ls in turn at M tokens (from index short on at 64), a decode before index stop
            gen = torch.Generator(device=dev).manual_seed(M)
            placed.clear()
            for i, lin in enumerate(ls):
                if i == stop:
                    gm.Ahead.stop(dev_)
                    lins[0].decode_rows(0, 64)
                x = torch.randn(64 if short is not None and i >= short else M, lin.in_features, dtype=bf, device=dev, generator=gen)
                y = lin(x)
                if short is not None and i >= short:
                    assert exact(y, lin.kernel(64)(lin.p, x)), ("below the threshold", route, M, i)
                    continue
                decoded = lin.hopper or lin.decoded(M) or lin.in_features % 64  # (off the order: no fused kernel)
                assert exact(y, F.linear(x, g.mma_unpack(lin.p)) if i < on or decoded else g.mma_gemm_big(lin.p, x)), ("a prompt decoded ahead", route, M, i, on)
                if i >= on and decoded:
                    counts["GLinear off the order, decoded" + tag] = counts.get("GLinear off the order, decoded" + tag, 0) + 1
            assert placed == list(range(on)), ("the order followed", route, M, placed, on)
            counts["GLinear decoded ahead" + tag] = counts.get("GLinear decoded ahead" + tag, 0) + on
            counts["GLinear off the order" + tag] = counts.get("GLinear off the order" + tag, 0) + len(ls) - on
        assert any(gm.Ahead.of[dev_].schedule(M // 128 * 128)[0]), ("decodes ahead in the order", route)
# A call below the threshold midway through a prompt, to a module that is not the order's next (Ahead.settle): the
# prompt's run of decodes ahead ended there, as by a decode midway, the rest of the prompt off the order.
gen = torch.Generator(device=dev).manual_seed(7)
placed.clear()
for i, lin in enumerate(lins):
    if i == 2:
        other[0](torch.randn(64, other[0].in_features, dtype=bf, device=dev, generator=gen))
        assert gm.Ahead.of[dev_].off and not gm.Ahead.queued, "settle: a call off the order midway ends the prompt's run"
    x = torch.randn(600, lin.in_features, dtype=bf, device=dev, generator=gen)
    decoded = lin.hopper or lin.decoded(600) or lin.in_features % 64
    assert exact(lin(x), F.linear(x, g.mma_unpack(lin.p)) if i < 2 or decoded else g.mma_gemm_big(lin.p, x)), ("off the order after settle", i)
assert placed == [0, 1], ("decoded ahead before a call off the order", placed)
counts["GLinear off the order after a call below the threshold"] = len(lins) - 2
# Where Ahead does not take a prompt's product, the fused kernel, as below the decode ahead (but on Hopper): under
# torch.compile (a graph's node: _lib.local.fresh), and a matrix past the scratch (decoded in row blocks, never ahead).
if cc() != (9, 0):  # this GPU (lins were last made as on an A100 where it is not one)
    placed.clear()
    glyd_gpu_lib.local.fresh = True
    for lin in lins:
        x = torch.randn(600, lin.in_features, dtype=bf, device=dev)
        assert exact(lin(x), F.linear(x, g.mma_unpack(lin.p)) if lin.in_features % 64 else g.mma_gemm_big(lin.p, x)), ("a compiled prompt: fused", lin.p.shape)
    glyd_gpu_lib.local.fresh = False
    big = gm.GLinear(g.pack_mma(weights(1024 * 512).view(1024, 512)), None)
    big.ahead, big.block = 513, 128  # as a matrix past the scratch
    big.step = big._step()
    x = torch.randn(600, 512, dtype=bf, device=dev)
    assert exact(big(x), g.mma_gemm_big(big.p, x)) and not placed, "past the scratch: fused"
    counts["GLinear where Ahead does not take it (fused)"] = len(lins) + 1
# A prompt that ends before its order does leaves decodes ahead queued; here they wait 50 ms (the hold before each
# host's), and meanwhile its modules are let go and memory of their sizes given out and written: none of it where
# the queued decodes read (record_stream), no illegal address. A call below the threshold waits for what is queued.
# (On Hopper the prompts are WG_MAX + 1 tokens: to WG_MAX its 12-bit ones are mma_gemm_wg's, never on the order.)
hold, gm.AHEAD_HOLD = gm.AHEAD_HOLD, 50_000_000
ls = [gm.GLinear((g.pack_mma12 if i % 2 else g.pack_mma)(weights(O * K, 0.01).view(O, K)), None) for i, (O, K) in enumerate(shapes)]
P = WG_MAX + 1 if cc() == (9, 0) else 600
xs = [torch.randn(P, lin.in_features, dtype=bf, device=dev) for lin in ls]
for lin in ls:
    lin.ahead = 513
    lin.step = lin._step()
for n in (len(ls), len(ls), 3):  # recorded, followed, then a prompt that ends before its order does
    for lin, x in zip(ls[:n], xs):
        lin(x)
a = gm.Ahead.of[dev_]
assert a.live and gm.Ahead.queued and any(k >= 3 for k, _, _ in a.schedule(P // 128 * 128)[0][2]), "decodes ahead queued past the prompt"
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

# The route SPLIT's products (the library alone: the ring and PyTorch's cuBLAS; the JIT host has no ring), on this GPU's
# SMs set apart by green contexts where they can be: each within 1e-2 of fp32 and the same bits run to run, with and
# without a bias; W in one chunk and in several (a small slot); matrices queued and multiplied in order, one off the
# queue, on a stream of its own; the queue's front a pack whole, not its data alone; refused during a CUDA graph's
# capture. Then GLinear by it (made as on an A100 SXM, from 769 tokens): its products as the ring's (Split.run called),
# never exact's (F.linear's bits, Split.run not called), and today's route where it cannot run.
dev_ = torch.device(dev, torch.cuda.current_device())
del gm.Split.of[dev_]
fns = gm.Split.blas_fns()
ring_buf = torch.empty(3 * (64 << 20), dtype=torch.uint8, device=dev)
ring = lib.ring_create(ring_buf, 64 << 20)
r, dsms, gsms = lib.ring_split(ring, 12)
if r == 0 and fns:
    ws = torch.empty(32 << 20, dtype=torch.uint8, device=dev)
    blas = lib.Blas(None, fns["cublasGemmEx"], fns["cublasSetStream_v2"], fns["cublasGetStream_v2"], fns["cublasSetWorkspace_v2"], fns["cublasSetSmCountTarget"], fns["cublasGetSmCountTarget"], ws.data_ptr(), 32 << 20)
    small = torch.empty(3 * (1 << 20), dtype=torch.uint8, device=dev)
    ring_small = lib.ring_create(small, 1 << 20)  # a slot of 1 MiB: W in several chunks
    cases = [((1024, 2048), 0.001), ((3072, 1024), 0.02), ((192, 4096), 0.1), ((4096, 512), 0.0)]
    ws_ = [weights(O * K, wild).view(O, K) for (O, K), wild in cases]
    qs = [g.pack_mma12(w) for w in ws_]
    for rg, tag in ((ring, "a chunk a matrix"), (ring_small, "several chunks")):
        for M in (65, 769, 2000):
            xs = [torch.randn(M, w.shape[1], dtype=bf, device=dev) for w in ws_]
            bias = [torch.randn(w.shape[0], dtype=bf, device=dev) for w in ws_]
            for use_bias in (False, True):
                assert lib.ring_reset(rg) == 0
                for q in qs:  # queued in order, then multiplied in order
                    assert lib.mma12_ring_queue(rg, 12, q.data, q.exc, q.exc_base, q.sym, *q.shape) == 0
                outs = []
                for rep in range(2):
                    ys = []
                    for q, w, x, b in zip(qs, ws_, xs, bias):
                        y = nan(M, w.shape[0])
                        blas.handle = torch.cuda.current_blas_handle()
                        assert lib.mma12_ring_linear(rg, 12, q.data, q.exc, q.exc_base, q.sym, *q.shape, x, b if use_bias else None, y, blas) == 0
                        ys.append(y)
                    outs.append(ys)
                for y, w, x, b in zip(outs[0], ws_, xs, bias):
                    near(y, F.linear(x.float(), w.float(), b.float() if use_bias else None))
                assert all(exact(a, b) for a, b in zip(*outs)), ("the route SPLIT: run to run", tag, M, use_bias)
                counts["the route SPLIT's products (" + tag + ")"] = counts.get("the route SPLIT's products (" + tag + ")", 0) + 2 * len(qs)
    # two layers of the same matrices queued: each chunk's decode gated by the product of the last chunk of its shape
    # (the layer before: 5 chunks back in 4 MiB slots) in a ring of 16 slots, the same bits as a 3-slot ring's of the
    # same chunks, whose slots gate it before that; the second layer's products the first's
    narrow, wide = torch.empty(3 * (4 << 20), dtype=torch.uint8, device=dev), torch.empty(16 * (4 << 20), dtype=torch.uint8, device=dev)
    ring_narrow, ring_wide = lib.ring_create(narrow, 4 << 20), lib.ring_create(wide, 4 << 20)
    for M in (769, 2000):
        xs = [torch.randn(M, w.shape[1], dtype=bf, device=dev) for w in ws_]
        outs = []
        for rg in (ring_narrow, ring_wide):
            assert lib.ring_reset(rg) == 0
            for q in qs + qs:
                assert lib.mma12_ring_queue(rg, 12, q.data, q.exc, q.exc_base, q.sym, *q.shape) == 0
            ys = []
            for q, w, x in zip(qs + qs, ws_ + ws_, xs + xs):
                y = nan(M, w.shape[0])
                blas.handle = torch.cuda.current_blas_handle()
                assert lib.mma12_ring_linear(rg, 12, q.data, q.exc, q.exc_base, q.sym, *q.shape, x, None, y, blas) == 0
                ys.append(y)
            outs.append(ys)
        assert all(exact(a, b) for a, b in zip(*outs)), ("the route SPLIT: gated decodes, the 3-slot ring's bits", M)
        assert all(exact(a, b) for a, b in zip(outs[1][:len(qs)], outs[1][len(qs):])), ("the route SPLIT: the second layer's products the first's", M)
        for y, w, x in zip(outs[1], ws_, xs):
            near(y, F.linear(x.float(), w.float()))
        counts["the route SPLIT's gated decodes (two layers)"] = counts.get("the route SPLIT's gated decodes (two layers)", 0) + 2 * len(qs)
    assert lib.ring_destroy(ring_wide) == 0 and lib.ring_destroy(ring_narrow) == 0
    with torch.cuda.stream(torch.cuda.Stream()):  # on a stream of its own, off the queue
        x = torch.randn(1000, 2048, dtype=bf, device=dev)
        y = nan(1000, 1024)
        blas.handle = torch.cuda.current_blas_handle()
        assert lib.mma12_ring_linear(ring, 12, qs[0].data, qs[0].exc, qs[0].exc_base, qs[0].sym, 1024, 2048, x, None, y, blas) == 0
        near(y, F.linear(x.float(), ws_[0].float()))
    # W queued, then a pack of W's data and exceptions but another base multiplied (a pack freed and its addresses taken
    # again): decoded from its own, not W's slot taken as its
    q0 = qs[0]
    other = g.Mma12(q0.shape, q0.data, q0.exc, q0.exc_base, q0.hb + 1 if q0.hb < 120 else q0.hb - 1)
    assert lib.ring_reset(ring) == 0 and lib.mma12_ring_queue(ring, 12, q0.data, q0.exc, q0.exc_base, q0.sym, *q0.shape) == 0
    x, y = torch.randn(1000, 2048, dtype=bf, device=dev), nan(1000, 1024)
    blas.handle = torch.cuda.current_blas_handle()
    assert lib.mma12_ring_linear(ring, 12, other.data, other.exc, other.exc_base, other.sym, *other.shape, x, None, y, blas) == 0
    near(y, F.linear(x.float(), g.mma_unpack(other).float()))
    counts["the route SPLIT: the queue's front a pack whole"] = 1
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    x, y = torch.randn(1000, 2048, dtype=bf, device=dev), nan(1000, 1024)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph.capture_begin()
        refused = lib.mma12_ring_linear(ring, 12, qs[0].data, qs[0].exc, qs[0].exc_base, qs[0].sym, 1024, 2048, x, None, y, blas)
        graph.capture_end()
    assert refused == 801, ("the route SPLIT refused in a capture: cudaErrorNotSupported", refused)
    # GLinear by the route, made as on an A100 (its route from 769 tokens)
    ran, run, measured = [], gm.Split.run, gm.Split.measured
    gm.Split.run = lambda s, lin, x: (ran.append(lin.handle), run(s, lin, x))[1]
    gm.Split.measured = staticmethod(lambda d, gpu: True)  # (as an A100 SXM's: its 108 SMs)
    lins = made_as((8, 0), "NVIDIA A100-SXM4-40GB", lambda: [gm.GLinear(q, None) for q in qs])
    exact_lins = made_as((8, 0), "NVIDIA A100-SXM4-40GB", lambda: [gm.GLinear(q, None, exact=True) for q in qs])
    gm.set_scratch(torch.nn.ModuleList(lins + exact_lins), False)
    for M in (768, 769, 1024, 3000):
        xs = [torch.randn(M, w.shape[1], dtype=bf, device=dev) for w in ws_]
        for rep in range(3):  # recorded, then followed twice: the same bits
            ran.clear()
            ys = [lin(x) for lin, x in zip(lins, xs)]
            assert (len(ran) == len(lins)) == (M >= 769), ("GLinear by the route SPLIT from 769 tokens (an A100)", M, len(ran))
            if rep == 0:
                first = ys
            assert all(exact(a, b) for a, b in zip(first, ys)), ("GLinear by the route SPLIT: run to run", M, rep)
        for y, w, x in zip(ys, ws_, xs):
            near(y, F.linear(x.float(), w.float()))
        for lin, x, w in zip(exact_lins, xs, ws_):
            ran.clear()
            assert exact(lin(x), F.linear(x, w)) and not ran, ("exact never takes the route SPLIT", M)
        counts["GLinear by the route SPLIT (as on an A100)"] = counts.get("GLinear by the route SPLIT (as on an A100)", 0) + 3 * len(lins)
    # a CUDA graph capturing a prompt's product: today's route (Split.off while the stream is captured), replayed
    x = torch.randn(1024, lins[0].in_features, dtype=bf, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        lins[0](x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    ran.clear()
    with torch.cuda.graph(graph):
        yg = lins[0](x)
    graph.replay()
    torch.cuda.synchronize()
    assert not ran, "a captured prompt: today's route"
    near(yg, F.linear(x.float(), ws_[0].float()))
    counts["GLinear captured in a CUDA graph: today's route"] = 1
    del graph
    gm.Split.stop(dev_)  # where it cannot run: today's route (decoded, then cuBLAS), bit for bit
    for lin, q, w in zip(lins, qs, ws_):
        x = torch.randn(1024, w.shape[1], dtype=bf, device=dev)
        ran.clear()
        assert exact(lin(x), F.linear(x, g.mma_unpack(q))) and not ran, "the route SPLIT off: today's route"
    counts["GLinear, the route SPLIT off: today's route"] = len(lins)
    gm.Split.run, gm.Split.measured = run, measured
    print(f"the route SPLIT: a split of {dsms} + {gsms} SMs; products within 1e-2 of fp32, the same bits run to run; GLinear by it as on an A100")
else:
    print(f"the route SPLIT cannot run on this GPU ({lib.error_string(r) if r else 'no cuBLAS found'}): GLinear takes today's route")
    gm.Split.of[dev_] = False
# On this GPU as it is (its own code): a 12-bit GLinear's prompt takes the route SPLIT where the rule gives it, else
# today's route (the library's without SPLIT), within 1e-2 of fp32, no ring made for it (on Ada, an A10 or a PCIe card:
# every prompt so)
gm.Split.of.pop(dev_, None)
own = [(1024, 4096), (4096, 4096)]
wts = [weights(O * K, 0.01).view(O, K) for O, K in own]
lins = [gm.GLinear(g.pack_mma12(w), None) for w in wts]
gm.set_scratch(torch.nn.ModuleList(lins), False)
for M in (769, 1024, 2048, 4096, 8192):
    for lin, w, (O, K) in zip(lins, wts, own):
        takes = split_rule(here, True, O, K, M) > 0
        assert (lin.route(M)[0] == g.SPLIT) == takes, ("this GPU's route SPLIT by the rule", here, O, K, M)
        if not takes:
            assert lin.route(M)[0] == g.route(lin.p, here, M)[0], ("today's route", here, O, K, M)
            x = torch.randn(M, K, dtype=bf, device=dev)
            near(lin(x), F.linear(x.float(), w.float()))
            assert dev_ not in gm.Split.of, ("no ring made", here, O, K, M)
            counts["GLinear on this GPU's own code, not SPLIT: today's route, no ring"] = counts.get("GLinear on this GPU's own code, not SPLIT: today's route, no ring", 0) + 1
gm.Split.stop(dev_)
del lins, wts

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
print(f"library {g._prebuilt()} (CUDA {lib.cuda_version()}), {torch.cuda.get_device_name()}: {sum(counts.values())} calls compared bit for bit, all identical; {looked_up} routes as main's rule (0.24.0's GLinear, with the L4's decode since)")
for name in sorted(counts):
    print(f"  {name}: {counts[name]}")
