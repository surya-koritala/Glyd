"""Split byte against the 12-bit layout before it, in the same kernels (both.py: main's library and packer beside this
tree's, one process, the same matrices and inputs). Every 12-bit entry point this GPU takes: each output bit for bit
the same as main's (a call refused by one refused by the other), and this tree's against the weights or an fp32
product (within 1e-2 of the largest output) so they are not both wrong.

- The self-test's matrices and check_capi's (odd row blocks, units shared by blocks, exceptions few and many; K not a
  multiple of 64), and every bf16 bit pattern: mma_unpack (all rows, a row block on; a warp a step and a few warps
  each taking every so many), mma_gemm (1-64 tokens), mma_gemm_mid (1-600), mma_gemm_big (every variant, 65-2100),
  mma_gemm_wg (Hopper, 1-2100), mma_linear by each route and by the library's (1-2100), with and without bias (on
  Hopper a prompt's DECODE and AHEAD, which main's refused there, this tree's prompt kernel's bits).
- A mixture of experts' layer: mma_moe_unpack, mma_moe (the rows by expert, SiLU and GELU fused, weighted sums).
- MODELs (their directories or Hub names in the cache): every Linear weight, lm_head and experts' packed by both and
  decoded bit for bit, their sizes; layers 0, 1, 2, the middle one and the last: their products as above at 1-1100
  tokens (experts': 1-300 tokens by their top-k).

    PYTHONPATH=TREE/bindings/python GLYD_GPU_LIB=TREE_LIB [SYNTHETIC=0] python xcheck.py MAIN_TREE MAIN_LIB [MODEL ...]"""
import glob, json, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
import both
from both import bits, g, new

old, old_pack = both.load(sys.argv[1], sys.argv[2])
dev, bf = "cuda", torch.bfloat16
cc = torch.cuda.get_device_capability()
hopper = cc == (9, 0)
gpu = new.gpu()
counts, refused = {}, {}


def pair(name, f, po, pn):
    """f through both (both.pair): the same bits, or refused by both; this tree's output back (None: refused)."""
    a, b = both.pair(old, f, po, pn)
    if isinstance(a, Exception) or isinstance(b, Exception):
        assert isinstance(a, Exception) and isinstance(b, Exception), (name, a, b)
        refused[name] = refused.get(name, 0) + 1
        return None
    assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b)), name
    counts[name] = counts.get(name, 0) + 1
    return b


def near(y, ref, what):
    if y is not None:
        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, (what, err)


def unpacks(w, po, pn):
    O, K = w.shape
    for warps in (0, 1, 7, 160):
        assert torch.equal(bits(pair("mma_unpack", lambda p: g.mma_unpack(p, warps=warps), po, pn)), bits(w))
        if O >= 128:
            assert torch.equal(bits(pair("mma_unpack", lambda p: g.mma_unpack(p, row0=64, rows=64, warps=warps), po, pn)), bits(w[64:128]))


MS = [1, 2, 7, 8, 16, 17, 32, 33, 63, 64, 65, 100, 128, 129, 256, 257, 300, 512, 513, 600, 1100, 2100]
ROUTES = [-1, g.DECODE, g.GEMM, g.MID, g.WG, g.BIG, g.AHEAD]


def products(w, po, pn, Ms=MS, scale=1.0):
    """w's products through each 12-bit entry point that takes M tokens (mma_gemm to 64, mma_gemm_mid to 600,
    mma_gemm_big's variants past 64, mma_gemm_wg on Hopper; mma_linear by each route), with and without bias."""
    O, K = w.shape
    bias = torch.randn(O, dtype=bf, device=dev)
    for M in Ms:
        x = (torch.randn(M, K, device=dev) * scale).to(bf)
        for b in (None, bias):
            ref = F.linear(x.float(), w.float(), None if b is None else b.float())
            if M <= 64:
                near(pair("mma_gemm", lambda p: g.mma_gemm(p, x, b), po, pn), ref, ("mma_gemm", M))
            if K % 64 == 0 and M <= 600:
                near(pair("mma_gemm_mid", lambda p: g.mma_gemm_mid(p, x, b), po, pn), ref, ("mma_gemm_mid", M))
            if K % 64 == 0 and M > 64:
                for v in (0, 1, 2, 3):
                    near(pair(f"mma_gemm_big variant {v}", lambda p: g.mma_gemm_big(p, x, b, v), po, pn), ref, ("mma_gemm_big", v, M))
            if hopper and K % 64 == 0:
                near(pair("mma_gemm_wg", lambda p: g.mma_gemm_wg(p, x, b), po, pn), ref, ("mma_gemm_wg", M))
            for r in ROUTES if b is None else [-1]:
                if hopper and K % 64 == 0 and (g.route(pn, gpu, M)[0] if r < 0 else r) in (g.DECODE, g.AHEAD):
                    y = g.mma_linear(pn, x, b, r)  # (main refused these on Hopper: this tree's linear against its prompt kernel)
                    assert torch.equal(bits(y), bits(g.mma_gemm_big(pn, x, b, 0))), ("mma_linear", r, M)
                    counts[f"mma_linear route {r}, Hopper's prompt kernel"] = counts.get(f"mma_linear route {r}, Hopper's prompt kernel", 0) + 1
                    near(y, ref, ("mma_linear", r, M))
                    continue
                near(pair(f"mma_linear route {r}", lambda p: g.mma_linear(p, x, b, r), po, pn), ref, ("mma_linear", r, M))


def moe(w, po, pn, k, Ts, scale=1.0):
    """w [E, O, K] packed [E O, K]: a layer's products for T tokens by k experts each (a few routed nowhere). The rows
    compared: those the kernel writes, one a pair of the plan; past them (a pair routed nowhere) mma_moe's output is
    torch.empty's, never written, whatever memory the allocator hands out."""
    E, O, K = w.shape
    for T in Ts:
        ids = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
        if T > 2:
            ids[1, 0] = E  # routed nowhere, as an expert-parallel sentinel
        P, flat = T * k, ids.view(-1)
        plan = g.moe_route(ids, E)
        valid = (flat < E).nonzero().view(-1)
        order = plan[2 + 2 * E : 2 + 2 * E + len(valid)].long()
        expert = flat[order]
        x, xs = (torch.randn(T, K, device=dev) * scale).to(bf), (torch.randn(P, K, device=dev) * scale).to(bf)
        rows, rows_s = (torch.empty(len(valid), O, device=dev) for _ in range(2))  # the pairs' rows in the plan's order, by expert
        for e in expert.unique().tolist():
            j = (expert == e).nonzero().view(-1)
            rows[j], rows_s[j] = x[order[j] // k].float() @ w[e].float().t(), xs[j].float() @ w[e].float().t()
        hit = torch.zeros(E, dtype=torch.bool, device=dev)
        hit[flat[valid]] = True
        u = pair("mma_moe_unpack", lambda p: g.mma_moe_unpack(p, E, plan, P, torch.full((E * O * K,), float("nan"), dtype=bf, device=dev)), po, pn).view(E, O, K)
        assert torch.equal(bits(u[hit]), bits(w[hit])) and torch.isnan(u[~hit].float()).all()
        for b in (None, torch.randn(E, O, dtype=bf, device=dev)):
            bb = b[expert].float() if b is not None else 0
            y = pair("mma_moe", lambda p: g.mma_moe(p, E, x, plan, ids, 0, b)[: len(valid)], po, pn)
            near(y, rows + bb, ("mma_moe", T))
            if O % 128 == 0:
                gate, up = (rows + bb).chunk(2, -1)
                for act, f in ((1, F.silu), (2, lambda v: F.gelu(v, approximate="tanh"))):
                    y = pair(f"mma_moe act {act}", lambda p: g.mma_moe(p, E, x, plan, ids, act, b)[: len(valid)], po, pn)
                    near(y, f(gate) * up, ("mma_moe", act, T))
            wt = torch.rand(T, k, device=dev).to(bf)
            y = pair("mma_moe weighted", lambda p: g.mma_moe(p, E, xs, plan, ids, 0, b, wt, gather=False), po, pn)
            ref = torch.zeros(P, O, device=dev)
            ref[order] = (rows_s + bb) * wt.view(-1)[order, None].float()
            near(y, ref.view(T, k, O).sum(1), ("mma_moe weighted", T))


def weights(n, wild=0.0):
    w = torch.randn(n, device=dev) * 0.02
    m = torch.rand(n, device=dev) < wild
    w[m] = torch.randn(int(m.sum()), device=dev) * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device=dev).float())
    return w.to(bf)


def synthetic():
    torch.manual_seed(0)
    t0 = time.time()
    for O, K, wild in [(64, 64, 0), (192, 128, 0), (128, 4096, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (17408, 1024, 0.01), (8960, 128, 0), (256, 1040, 0.01), (256, 16384, 0)]:
        w = weights(O * K, wild).view(O, K)
        po, pn = old_pack(w), g.pack_mma12(w)
        unpacks(w, po, pn)
        products(w, po, pn)
        print(f"{O}x{K}: exceptions {int(po.exc_base[-1])} (12-bit) / {int(pn.exc_base[-1])} (split byte, hb {pn.hb}): the same bits", flush=True)
    every = (torch.arange(65536, device=dev) - 32768).to(torch.int16)
    most = torch.rand(65536, device=dev) < 0.9
    for name, u in (("shuffled", every[torch.randperm(65536, device=dev)]), ("in order", every), ("hb 0", torch.where(most, every & -32641, every)), ("hb 120", torch.where(most, every | 0x7F00, every))):
        w = u.view(bf).view(256, 256)
        po, pn = old_pack(w), g.pack_mma12(w)
        unpacks(w, po, pn)
        print(f"every bf16 bit pattern, {name}: exceptions {int(po.exc_base[-1])} / {int(pn.exc_base[-1])} (hb {pn.hb}), decoded the same bits", flush=True)
    for E, O, K, k, Ts, wild in [(8, 256, 192, 2, (1, 70), 0.01), (64, 128, 2048, 8, (8,), 0.001), (16, 192, 64, 4, (33,), 0.1), (4, 256, 256, 2, (300,), 0.01), (40, 1024, 1536, 8, (3, 64), 0.0)]:
        w = weights(E * O * K, wild).view(E, O, K)
        po, pn = old_pack(w.view(E * O, K)), g.pack_mma12(w.view(E * O, K))
        moe(w, po, pn, k, Ts)
        print(f"moe: {E} experts of {O}x{K} by {k}, {Ts} tokens: the same bits", flush=True)
    print(f"synthetic: {sum(counts.values())} calls the same bits, {sum(refused.values())} refused by both ({time.time() - t0:.0f} s)", flush=True)


if os.environ.get("SYNTHETIC", "1") != "0":  # SYNTHETIC=0: the models alone
    synthetic()


def model_dir(m):
    if os.path.exists(os.path.join(m, "config.json")):
        return m
    return glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", f"models--{m.replace('/', '--')}", "snapshots", "*"))[0]


def tensors(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else [f for f in os.listdir(d) if f.endswith(".safetensors")]
    for fn in files:
        with safe_open(os.path.join(d, fn), "pt", device=dev) as f:
            for key in f.keys():
                parts = key.split(".")
                lin = parts[-1] == "weight" and len(parts) > 1 and ("proj" in parts[-2] or parts[-2] in ("input_linear", "output_linear", "lm_head"))
                if lin or "experts" in parts[:-1] or parts[-1].endswith("_proj"):
                    yield key, f.get_tensor(key)


for m in sys.argv[3:]:
    d, t0 = model_dir(m), time.time()
    cfg = json.load(open(os.path.join(d, "config.json")))
    L, k = cfg["num_hidden_layers"], cfg.get("num_experts_per_tok", 0)
    layers = {0, 1, 2, L // 2, L - 1}
    n = n_t = skipped = b_old = b_new = x_old = x_new = 0
    before = dict(counts)
    for key, t in tensors(d):
        if t.dtype != bf or t.dim() not in (2, 3) or t.shape[-2] % 64 or t.shape[-1] % 16:
            skipped += 1
            continue
        w = t.reshape(-1, t.shape[-1])
        po, pn = old_pack(w), g.pack_mma12(w)
        assert torch.equal(bits(pair("mma_unpack", lambda p: g.mma_unpack(p), po, pn)), bits(w)), key
        n_t, n = n_t + 1, n + w.numel()
        b_old, b_new, x_old, x_new = b_old + po.nbytes(), b_new + pn.nbytes(), x_old + int(po.exc_base[-1]), x_new + int(pn.exc_base[-1])
        layer = next((int(p) for p in key.split(".") if p.isdigit()), None)
        if layer in layers:
            if t.dim() == 3:
                moe(t, po, pn, k or 2, (1, 8, 64, 300), 0.5)
            else:
                unpacks(w, po, pn)
                products(w, po, pn, [1, 8, 17, 33, 64, 65, 128, 300, 1100], 0.5)
        del po, pn, w, t
    torch.cuda.empty_cache()
    print(f"{m}: {n_t} tensors ({n / 1e9:.3f} B weights; {skipped} others left out) decoded bit for bit in both layouts; "
          f"12-bit {8 * b_old / n:.3f} bits a weight, {1e6 * x_old / n:.0f} exceptions a million; split byte {8 * b_new / n:.3f}, "
          f"{1e6 * x_new / n:.0f}; layers {sorted(layers)}: {sum(counts.values()) - sum(before.values())} calls the same bits ({time.time() - t0:.0f} s)", flush=True)
print("calls:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
print("refused by both:", ", ".join(f"{k} {v}" for k, v in sorted(refused.items())) or "none")
print(f"all the same bits: {sum(counts.values())} calls ({torch.cuda.get_device_name()}, compute capability {cc[0]}.{cc[1]})")
