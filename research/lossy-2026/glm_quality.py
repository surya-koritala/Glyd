"""Output quality of GLM-5.3-Flash when its routed experts are rounded to 2-4 bits (Linear GLY-127): the KL divergence of each next-token distribution from
the FP8 model's, the top token's agreement and the perplexity, over the held-out text of glm_skew.py (six domains, ~49k tokens), for six ways of rounding.
Only the 288 routed experts of the 42 MoE layers change; attention, shared experts, dense layers, routers, embedding and head stay as released.
  plan   [--text]      shell lines for the launcher; with --text the variant table and its time budget
  run    --weights DIR --data DIR --out DIR   the pod job (GPUs): load, reference, routing counts, then one layer-by-layer pass per variant
  report DIR           results.jsonl + meta.json -> quality.md and quality.json (numpy only; the job runs it after every variant)
  selftest             a tiny random Glm5Next model on the CPU: every variant end to end, the block codes checked against calib.py's; with GPUs also the codes on each of them
                       against the CPU's and the tiny model split over them as the real one is (SELFTEST_GPU=0: CPU only)
How it fits 4 x 96 GB. The FP8 model (328 GB) stays on the GPUs for the whole job, so every variant starts from the released weights without a reload, and the
rounded experts never exist all at once (packed codes next to the FP8 weights would be 328 + 171 GB > 384 GB). A variant is not one sequence after another through
all layers but all held-out sequences through one layer after another (their residual streams, 67 MB each, wait on the CPU between layers). At a MoE layer the
experts are read from the FP8 weights a chunk at a time (~8 of 288), rounded, applied to every token routed to them, and dropped: the chunk's float32 and bf16 copies
are all the memory a variant needs (<= 5 GB). Each layer sees the layers before it already rounded, which is exactly the compressed model's forward, and the order GPTQ
needs for its sequential calibration. The model's own layer code runs everything but the routed experts: a stand-in for the layer's MoE block collects its input, the
experts below compute its output, and a second call of the layer (the stand-in now returning it) finishes the layer.
Specs: int<B> = symmetric codes, one bf16 scale per run of 32 chosen by mse among 16 clips; hlm<B> = the sub-3-bit spike's best format, a randomized Hadamard rotation of
the inputs, a per-row 2^B-entry Lloyd-Max codebook, the run's absmax (bf16) as scale, rounded to nearest (RTN); hlm<B>g = the same with GPTQ act-order (error feedback)
on the Hessians X^T X of the tokens routed to each expert; ref = the FP8 weights dequantised to bf16, not rounded (the control); hbr<B>[g] and hbp<B>[g] = nemotron_hbr.py's neuron-row recipe (ROWS holds its rounding: GLM does not use them).
Hummingbird, at most 3.0 bits/weight averaged over the experts (the selftests assert it on both models' expert shapes): hlm2g, hb40 (the 40% most-used experts of each layer hlm3g, the rest hlm2g)
and hb20 (the 20% most-used int4, the rest hlm2g). They are not in ORDER, the default run: name them in --variants.
Every variant is scored against the FP8 model and, once ref has run, against ref too: the FP8 kernels round each activation to FP8 and a Glyd engine would not, so the first
carries that floor (ref's own KL) and the second is what rounding the weights alone costs.
The block codes are calib.py's (branch sub3bit, ac0ef626), ported to run on many experts at once and checked against verbatim copies in selftest.
No token is used anywhere: models and datasets are ungated.
nemotron_quality.py measures Nemotron 3 Super the same way: it brings its own Run (what a pass needs of the model; the class below is GLM's), experts' forward, plan and selftest, and runs everything else from here."""
import argparse, collections, contextlib, functools, gc, itertools, json, math, os, re, sys, time, traceback, types

import glm_skew as gs  # the pins, the six text domains and their prep

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:  # (plan and report need neither)
    np = torch = F = None

BLOCK = 32       # weights one bf16 scale covers: a run of a row (the Glyd block code)
TOPK = 128       # tokens of the FP8 distribution kept at each position; the rest is one tail bucket, so every KL here is a lower bound
HEAD_ROWS = 512  # positions a time through the head (512 x 154,880 float32 = 317 MB)
BYTES = {"ref": 6, "int": 8, "hlm": 10, "gptq": 5, "hbr": 10, "hbp": 10, "hbr_gptq": 7, "hbp_gptq": 7}  # float32 copies of an expert's weights alive at once while it is rounded (sizes the chunk of experts)
ROWS = {}        # spec kind -> the function that rounds a group of experts of that kind (hbr and hbp, neuron rows: nemotron_hbr.py registers them); see rounded
# name -> the plan. share: % of a layer's most-used experts that get `hot` (None: every expert gets `rest`); min: minutes expected on the pod (a guess until a first run)
VARIANTS = {
    "ref":    dict(item=0, what="control: the FP8 experts dequantised to bf16, not rounded (what this measurement cannot beat)", share=None, hot=None, rest="ref", min=3),
    "mix13":  dict(item=5, what="mixed by usage: the 13% most-used experts of each layer int4, the rest hlm3 (the GLM on 24 GB + 128 GB plan)", share=13, hot="int4", rest="hlm3", min=10),
    "int4":   dict(item=1, what="every expert int4 (symmetric codes, mse-fit bf16 scale per run of 32)", share=None, hot=None, rest="int4", min=7),
    "hlm3":   dict(item=3, what="every expert hlm3: rotation + per-row codebook at 3 bits, rounded to nearest", share=None, hot=None, rest="hlm3", min=8),
    "int3":   dict(item=2, what="every expert int3 (as int4, 3 bits)", share=None, hot=None, rest="int3", min=7),
    "cold50": dict(item=6, what="the coldest 50% of each layer's experts hlm2, the other 50% int4 (the aggressive plan)", share=50, hot="int4", rest="hlm2", min=8),
    "hlm3g":  dict(item=4, what="every expert hlm3 with GPTQ act-order on routed-token Hessians (experts with few calibration tokens fall back to RTN)", share=None, hot=None, rest="hlm3g", min=30),
    # Hummingbird: at most 3.0 bits/weight averaged over the experts (both selftests assert it, on their model's expert shapes). Minutes from the first run's hlm3g (1,477 s) and routing counts (91 s): hlm2g and hb20 one
    # GPTQ call a chunk, hb40 two (GPTQ's part of the pass twice)
    "hlm2g":  dict(item=7, what="every expert hlm2 with GPTQ act-order (hlm3g at 2 bits)", share=None, hot=None, rest="hlm2g", min=25),
    "hb40":   dict(item=8, what="Hummingbird, mixed by usage: the 40% most-used experts of each layer hlm3g, the rest hlm2g", share=40, hot="hlm3g", rest="hlm2g", min=40),
    "hb20":   dict(item=9, what="Hummingbird, mixed by usage: the 20% most-used experts of each layer int4, the rest hlm2g", share=20, hot="int4", rest="hlm2g", min=25),
}
ORDER = ["ref", "mix13", "int4", "hlm3", "int3", "cold50", "hlm3g"]  # the default run, most important first (5, then 1, 3); the pass that may not fit, GPTQ, last
HUMMINGBIRD = ["hlm2g", "hb40", "hb20"]  # not in the default run: named in --variants / VARIANTS
SETUP_MIN = 12   # minutes of the job before the first pass: glm_skew.py's run on this pod class took pyenv 62 s, download 147 s (in the background), load 107 s, 70 s of kernel warm-up in the first forward; plus the selftest, fla, prep, check, reference
GLM_MATS = [(4096, 4096), (4096, 2048)]  # rows x columns of an expert's matrices: gate_up [2I, H] and down [H, I] (I 2048, H 4096)


class Deadline(Exception):
    pass


class Abort(Exception):
    pass


T_LOG = time.time()


def log(*a):  # (minutes into this step: the pod's clock is UTC, and the job's own steps.txt says when the step began)
    print(f"+{(time.time() - T_LOG) / 60:5.1f}m", *a, flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------ plan
def parse_spec(s):
    """'ref' | 'int4' | 'hlm3' | 'hlm3g' | 'hbr3g' -> (kind, bits, gptq). hbr and hbp (nemotron_hbr.py): the neuron-row recipe with one rotation of the latent a layer / one an expert."""
    m = re.fullmatch(r"(ref|int|hlm|hbr|hbp)(\d?)(g?)", s)
    if not m or (m.group(1) == "ref") == bool(m.group(2)) or (m.group(3) and m.group(1) in ("int", "ref")):
        raise ValueError(f"spec {s}: ref, int<B>, hlm<B>, hlm<B>g, hbr<B>, hbr<B>g, hbp<B> or hbp<B>g")
    return m.group(1), int(m.group(2) or 0), bool(m.group(3))


def spec_bpw(spec, mats, ref=8 + 32 / (128 * 128)):
    """Stored bits a weight of an expert under spec, mats being the (rows, cols) of its matrices (GLM_MATS for GLM): the codes, the scale of each run of 32, a row's codebook and the
    rotation's signs; ref: what the unrounded reference takes (default: FP8 and a float32 scale per 128 x 128 tile)."""
    kind, b, _ = parse_spec(spec)
    if kind == "ref":
        return ref
    if kind in ("hbr", "hbp"):  # neuron rows as long as the narrower side of a matrix (the latent width): codes, a bf16 scale a run of 32, 2^b bf16 codebook entries a row; a layer's shared rotation costs 1,024 sign bits a layer (not counted), a per-expert one the narrow side's width of bits an expert
        n_w = sum(r * c for r, c in mats)
        return sum((b + 16 / BLOCK + 16 * (1 << b) / min(r, c)) * r * c for r, c in mats) / n_w + (min(min(r, c) for r, c in mats) / n_w if kind == "hbp" else 0)
    f = lambda rows, cols: b + 16 / BLOCK + ((16 * (1 << b) / cols + 1 / rows) if kind == "hlm" else 0)
    return sum(f(r, c) * r * c for r, c in mats) / sum(r * c for r, c in mats)


def layer_specs(v, hot_row, E):
    """The spec of each of a layer's E experts under variant v; hot_row: bool (E,), the most-used ones."""
    return [v["rest"]] * E if v["share"] is None else [v["hot"] if hot_row[e] else v["rest"] for e in range(E)]


def variant_bpw(v, mats, ref=8 + 32 / (128 * 128)):
    if v["share"] is None:
        return spec_bpw(v["rest"], mats, ref)
    return (v["share"] * spec_bpw(v["hot"], mats, ref) + (100 - v["share"]) * spec_bpw(v["rest"], mats, ref)) / 100


def plan_rows(run, mats, L, E, jmm, order=None):
    """The variant lines of plan --text for run (a Run class), in the order they run (order: their names; default run.ORDER): minutes expected, the job minute each would end at, bits/weight all in, size of the experts."""
    t, n_w, order = run.SETUP_MIN, sum(r * c for r, c in mats), order or run.ORDER
    for n in order:
        v = run.VARIANTS[n]
        bw, t = variant_bpw(v, mats, run.REF_BPW), t + v["min"]
        print(f"  {n:7s} item {v['item']}  ~{v['min']:2d} min, ends ~job minute {t:3d}{'' if t + 2 <= jmm else ' (too late: skipped unless faster than guessed)'}  {bw:5.2f} bits/weight, experts {L * E * n_w * bw / 8e9:6.1f} GB  {v['what']}")
    if len(order) < len(run.VARIANTS):
        print(f"  not in this run (name them in VARIANTS): {' '.join(n for n in run.VARIANTS if n not in order)}")
    return sum(run.VARIANTS[n]["min"] for n in order)


def plan(a):
    m = gs.MODELS["glm"]
    I, H, E, L = 2048, 4096, m["ne"], m["n_moe"]
    if a.text:
        jmm = int(os.environ.get("JOB_MAX_MIN", 128))
        print(f"{L} MoE layers x {E} experts, top-{m['k']}; an expert is gate_up [{2 * I}, {H}] + down [{H}, {I}] = {3 * I * H:,} weights; {L * E * 3 * I * H / 1e9:.1f} B routed-expert weights")
        print("variants in the order they run (item = the GLY-127 list; bits/weight all in). The minutes are guesses, not measurements: the skew run's 2,300 tokens/s through the model and an op count of the rounding.")
        print(f"The job measures its own speed and starts no pass that would not end 2 min before JOB_MAX_MIN {jmm}. The setup before the first (venv, download, pyenv, selftest, fla, prep, load, check, FP8 reference) is ~{SETUP_MIN} min:")
        total = plan_rows(Run, GLM_MATS, L, E, jmm)
        print(f"  total ~{total} min of passes after the setup (the routing counts for mix13 and cold50, ~2 min, are inside mix13's)")
        return
    gs.plan(argparse.Namespace(model="glm"))  # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB
    print(f'ORDER="{" ".join(ORDER)}"')


# ------------------------------------------------------------------------------------------------------------------------------------- block codes
def fit_scale(w, q):
    """The bf16 scale of each run of w (..., BLOCK): the clip of 16 (0.70..1.0 of absmax / q) with the least squared error (kld_study.py's int:B:32:mse:bf16)."""
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    best_err = best = None
    for r in torch.linspace(0.70, 1.0, 16, device=w.device):
        s = (amax * r / q).bfloat16().float()
        err = ((torch.clamp(torch.round(w / s), -q, q) * s - w) ** 2).sum(-1, keepdim=True)
        if best is None:
            best_err, best = err, s
        else:
            better = err < best_err
            best_err, best = torch.where(better, err, best_err), torch.where(better, s, best)
    return best


def quant_int(W, bits):
    """W (n, rows, cols) float32 as int<bits> codes times the bf16 scales, back to bf16."""
    q = (1 << (bits - 1)) - 1
    w = W.view(*W.shape[:-1], W.shape[-1] // BLOCK, BLOCK)
    s = fit_scale(w, q)
    return (torch.clamp(torch.round(w / s), -q, q) * s).view(W.shape).bfloat16()


@functools.lru_cache(maxsize=None)
def _hadamard(R, dev):
    H = torch.ones(1, 1)
    while H.shape[0] < R:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / R ** 0.5).to(dev)


def fwht(x):
    """Orthonormal Walsh-Hadamard transform over blocks of R = gcd(n, 128) values of the last dim (calib.py's, as one product with the R x R matrix); symmetric, so its own inverse."""
    R = math.gcd(x.shape[-1], 128)
    return (x.reshape(-1, R) @ _hadamard(R, x.device)).reshape(x.shape)


@functools.lru_cache(maxsize=None)
def signs(n, seed, dev):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)


def lloyd_max(x, wt, k, per=1, iters=25, elems=1 << 26):
    """A k-entry codebook for each row of x (rows, n): the weighted 1-D k-means (Lloyd-Max) of the row from a uniform start on -1..1, sorted, bf16 (calib.py's rule). Row r is
    weighted by wt[r // per] (wt (W, n); None: equally). Sorting each row once turns an iteration into k cumulative-sum lookups, where calib.py's passes over the row k times."""
    out, n = [], x.shape[1]
    step = max(1, elems // n)
    for r0 in range(0, x.shape[0], step):
        xr = x[r0:r0 + step]
        r = xr.shape[0]
        xs, perm = xr.sort(-1)
        if wt is None:
            cx = F.pad(xs.cumsum(-1), (1, 0))
        else:
            ws = wt[torch.arange(r0, r0 + r, device=x.device) // per].gather(1, perm)
            cx, cw = F.pad((xs * ws).cumsum(-1), (1, 0)), F.pad(ws.cumsum(-1), (1, 0))
        del perm
        c = torch.linspace(-1, 1, k, device=x.device).expand(r, k).clone()
        lo, hi = torch.zeros(r, 1, dtype=torch.long, device=x.device), torch.full((r, 1), n, dtype=torch.long, device=x.device)
        for _ in range(iters):
            b = torch.cat([lo, torch.searchsorted(xs, (c[:, 1:] + c[:, :-1]) / 2, right=True), hi], 1)  # the sorted values up to each midpoint: cluster j is b[j]..b[j+1]
            num = cx.gather(1, b[:, 1:]) - cx.gather(1, b[:, :-1])
            den = (b[:, 1:] - b[:, :-1]).float() if wt is None else cw.gather(1, b[:, 1:]) - cw.gather(1, b[:, :-1])
            c = torch.where(den > 0, num / den.clamp_min(1e-30), c).sort(-1).values
        out.append(c)
    return torch.cat(out).bfloat16().float()


def assign(v, cb):
    """Each value of v (rows, n) as the nearest entry of its row's codebook cb (rows, k), sorted."""
    return cb.gather(1, torch.searchsorted(((cb[:, 1:] + cb[:, :-1]) / 2).contiguous(), v.contiguous()))


def quant_hlm(W, bits, seed):
    """W (n, rows, cols) float32 rounded to nearest in the rotated basis (calib.py's hlmgptq without the error feedback): signs and Hadamard on the inputs, per row a 2^bits
    codebook fitted on the run-normalised values, run absmax (bf16) as the scale; rotated back, bf16."""
    n, rows, cols = W.shape
    s = signs(cols, seed, W.device)
    runs = fwht(W * s).view(n * rows, cols // BLOCK, BLOCK)
    amax = runs.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    cb = lloyd_max((runs / amax).view(n * rows, cols), None, 1 << bits)
    sc = amax.bfloat16().float()
    v = assign((runs / sc).view(n * rows, cols), cb).view(n * rows, cols // BLOCK, BLOCK) * sc
    return (fwht(v.view(n, rows, cols)) * s).bfloat16()


def gptq_hlm(W, Hs, bits, seed, damp=0.01, step=128):
    """calib.py's hlmgptq:B:32:bf16:act for n matrices at once: W (n, rows, cols) float32 and Hs, a list of n Hessians (cols, cols) float32 (X^T X / tokens); both are consumed. The columns
    go in order of falling Hessian diagonal (whole runs), each rounded to its row's codebook with the error fed to the columns after (GPTQ, 1% damping). Returns the rounded W (bf16)
    and ok (n,), False where the damped Hessian was still not positive definite (that matrix is garbage: the caller rounds it to nearest). Each Hessian is turned into its
    Cholesky factor on its own, so only the factors and the weights are n matrices big while the columns are rounded (those n at once: the loop is bound by kernel launches)."""
    n, rows, cols = W.shape
    runs, dev = cols // BLOCK, W.device
    s = signs(cols, seed, dev)
    U, wt, perm, ok = torch.empty(n, cols, cols, device=dev), torch.empty(n, cols, device=dev), torch.empty(n, cols, dtype=torch.long, device=dev), [True] * n
    for j in range(n):
        H, Hs[j] = Hs[j], None
        dead = H.diagonal() == 0
        w = fwht(W[j].masked_fill(dead[None, :], 0) * s)
        H = fwht(fwht(H * s).T * s).T  # T H T^T
        wt[j] = H.diagonal()
        dead = wt[j] == 0
        H.diagonal().masked_fill_(dead, 1)
        W[j] = w.masked_fill(dead[None, :], 0)
        order = H.diagonal().view(runs, BLOCK).sum(1).argsort(descending=True)
        perm[j] = (order[:, None] * BLOCK + torch.arange(BLOCK, device=dev)).reshape(-1)
        H = H[perm[j]][:, perm[j]]
        wt[j] = wt[j][perm[j]]
        W[j] = W[j][:, perm[j]]
        H.diagonal().add_(damp * H.diagonal().mean())
        L, info = torch.linalg.cholesky_ex(H)
        del H
        if info.item():
            ok[j] = False
            U[j] = torch.eye(cols, device=dev)
            continue
        Uj, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
        ok[j] = not info.item()
        U[j] = Uj if ok[j] else torch.eye(cols, device=dev)
    w4 = W.view(n, rows, runs, BLOCK)
    cb = lloyd_max((w4 / w4.abs().amax(-1, keepdim=True).clamp_min(1e-30)).view(n * rows, cols), wt / wt.mean(-1, keepdim=True).clamp_min(1e-30) + 1e-6, 1 << bits, per=rows)
    Q, mids = torch.empty_like(W), ((cb[:, 1:] + cb[:, :-1]) / 2).contiguous()
    for i1 in range(0, cols, step):
        i2 = min(i1 + step, cols)
        W1, E1, U1 = W[:, :, i1:i2].clone(), torch.empty(n, rows, i2 - i1, device=dev), U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % BLOCK == 0:
                sc = W1[:, :, i:i + BLOCK].abs().amax(-1).clamp_min(1e-30).bfloat16().float()  # (n, rows)
            col, v, e = W1[:, :, i], Q[:, :, i1 + i], E1[:, :, i]
            torch.mul(cb.gather(1, torch.searchsorted(mids, (col / sc).reshape(-1, 1))).view(n, rows), sc, out=v)
            torch.sub(col, v, out=e)
            e.div_(U1[:, i, i][:, None])
            W1[:, :, i:].addcmul_(e[:, :, None], U1[:, i, None, i:], value=-1)
        if i2 < cols:
            W[:, :, i2:] -= torch.bmm(E1, U[:, i1:i2, i2:])
    return (fwht(Q.gather(2, perm.argsort(-1)[:, None, :].expand(n, rows, cols))) * s).bfloat16(), torch.tensor(ok, device=dev)


def free_all():
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def on(dev):
    """dev as the current CUDA device for a block: kernels launched without a device guard (Triton's, the FP8 ones) run on the current device, and accelerate's hooks do not set it."""
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


def host_hops():
    """From here on accelerate's hooks move a tensor from one GPU to another through the host. On some hosts a direct GPU-to-GPU copy returns wrong data
    (2026-10-10: an A6000 pair and a 4 x RTX PRO 6000 pod; the model's own forward was NaN or garbage from the first layer on the second card, while the
    layer-by-layer pass, which always goes through the host, was exact). Returns the GPU pairs whose direct copy of a test tensor came back wrong."""
    import accelerate.hooks as ah
    from accelerate.utils.operations import recursively_apply
    send = getattr(ah.send_to_device, "direct", ah.send_to_device)

    def staged(obj, device, non_blocking=False, skip_keys=None):
        d = torch.device("cuda", device) if isinstance(device, int) else torch.device(device) if isinstance(device, str) and device.startswith("cuda") else device
        if isinstance(d, torch.device) and d.type == "cuda":
            obj = recursively_apply(lambda t: t.cpu() if t.is_cuda and t.device != d else t, obj)
        return send(obj, device, non_blocking=non_blocking, skip_keys=skip_keys)
    staged.direct = send
    ah.send_to_device = staged
    n, x, bad = torch.cuda.device_count(), torch.randn(1 << 20, generator=torch.Generator().manual_seed(5)), []
    for i in range(n):
        xi = x.to(f"cuda:{i}")
        bad += [(i, j) for j in range(n) if j != i and not torch.equal(xi.to(f"cuda:{j}").cpu(), x)]
    return bad


@contextlib.contextmanager
def tf32():
    """(TF32 for the Hessian products only: restored after, as calib.py's run does.)"""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


# ------------------------------------------------------------------------------------------------------------------------------ the routed experts
def deq(w, s):
    """float32 weights of an expert stack: FP8 (n, rows, cols) times its block scales (n, rows / b, cols / b) as transformers' Fp8Dequantize, or a bf16 stack as it is."""
    if s is None:
        return w.float()
    n, r, c = w.shape
    sr, sc = s.shape[-2:]
    return (w.float().view(n, sr, r // sr, sc, c // sc) * s.float().view(n, sr, 1, sc, 1)).view(n, r, c)


def expert_names(e):
    """The names of the two weight tensors of the experts e (n, rows, cols): gate_up_proj and down_proj, or up_proj and down_proj where the experts have no gate (transformers' has_gate, true if unsaid)."""
    return ("gate_up_proj" if getattr(e, "has_gate", True) else "up_proj", "down_proj")


def activate(e, h):
    """What the experts do between their two projections to h, the first one's output: the model's own gate (swiglu), or for experts without one its activation function (relu2)."""
    return e._apply_gate(h) if getattr(e, "has_gate", True) else e.act_fn(h)


def fetch(e, a, b):
    """The first projection (gate_up (n, 2I, H); up for experts without a gate) and down (n, H, I) of experts a..b-1, float32, from the weights the model holds (FP8 + block scales, or bf16)."""
    def part(name):
        s = getattr(e, name + "_scale_inv", None)
        return deq(getattr(e, name)[a:b], None if s is None else s[a:b])
    first, last = expert_names(e)
    return part(first), part(last)


class Route:
    """Which tokens go to which expert, in the order the model's own loop visits them (expert, top-k slot, token), so the sums match it to the bit."""

    def __init__(self, idx, E):
        N, K = idx.shape
        dev = idx.device
        order = (idx * (K * N) + torch.arange(K, device=dev)[None, :] * N + torch.arange(N, device=dev)[:, None]).reshape(-1).argsort()
        self.tok, self.pos = order // K, order % K
        self.cnt = torch.bincount(idx.reshape(-1), minlength=E).tolist()
        self.start = [0] + list(itertools.accumulate(self.cnt))

    def of(self, e):
        s = self.start[e]
        return self.tok[s:s + self.cnt[e]], self.pos[s:s + self.cnt[e]]


class Err:
    """Weight error by spec for one layer: sums of squared error and of squares and the count, kept on the device until read."""

    def __init__(self):
        self.d = {}

    def add(self, spec, w, q):
        t = self.d.setdefault(spec, torch.zeros(3, dtype=torch.float64, device=w.device))
        t += torch.stack([((w - q.float()) ** 2).sum().double(), (w ** 2).sum().double(), torch.tensor(float(w.numel()), dtype=torch.float64, device=w.device)])

    def read(self):
        return {k: v.tolist() for k, v in self.d.items()}


def gptq_chunk(e, gu, dn, ids, X, rt, ctx, li):
    """Experts `ids` (weights gu, dn), hlm<B>g: the Hessians of the calibration tokens routed to each (the first ctx.ncal tokens of X), GPTQ for those with enough, nearest for the rest."""
    toks = [rt.of(i)[0] for i in ids]
    toks = [t[t < ctx.ncal] for t in toks]
    use = [j for j in range(len(ids)) if len(toks[j]) >= ctx.min_tok]
    qgu, qdn = torch.empty(gu.shape, dtype=torch.bfloat16, device=gu.device), torch.empty(dn.shape, dtype=torch.bfloat16, device=gu.device)
    fall = [j for j in range(len(ids)) if j not in use]
    if use:
        Hi, Hd = [], []
        with tf32():
            for j in use:
                x = X[toks[j]]
                xf = x.float()
                Hi.append(xf.T @ xf / len(x))
                act = activate(e, F.linear(x, gu[j].bfloat16())).float()  # (the down projection's input, from the released gate_up as the layer's forward has it)
                Hd.append(act.T @ act / len(x))
        sel = torch.tensor(use, device=gu.device)
        q1, ok1 = gptq_hlm(gu[sel], Hi, ctx.bits, li * 8)
        q2, ok2 = gptq_hlm(dn[sel], Hd, ctx.bits, li * 8 + 1)
        qgu[sel], qdn[sel] = q1, q2
        bad = (~(ok1 & ok2)).tolist()
        fall += [j for j, b in zip(use, bad) if b]
        ctx.done += len(use) - sum(bad)
    if fall:
        sel = torch.tensor(fall, device=gu.device)
        qgu[sel], qdn[sel] = quant_hlm(gu[sel], ctx.bits, li * 8), quant_hlm(dn[sel], ctx.bits, li * 8 + 1)
        ctx.fell += len(fall)
    return qgu, qdn


def rounded(e, a, gu, dn, specs, li, err, rt, X, ctx):
    """Experts a.. (float32 gu, dn) rounded as specs say: bf16."""
    groups = collections.defaultdict(list)
    for j in range(gu.shape[0]):
        groups[specs[a + j]].append(j)
    qgu = qdn = None
    for spec, js in groups.items():
        kind, bits, g = parse_spec(spec)
        whole = len(js) == gu.shape[0]
        sel = None if whole else torch.tensor(js, device=gu.device)
        wgu, wdn = (gu, dn) if whole else (gu[sel], dn[sel])
        if kind == "ref":
            q1, q2 = wgu.bfloat16(), wdn.bfloat16()
        elif kind == "int":
            q1, q2 = quant_int(wgu, bits), quant_int(wdn, bits)
        elif kind in ROWS:
            q1, q2 = ROWS[kind](e, wgu, wdn, [a + j for j in js], X, rt, ctx, li, bits, g, kind)
        elif g:
            ctx.bits = bits
            q1, q2 = gptq_chunk(e, wgu, wdn, [a + j for j in js], X, rt, ctx, li)
        else:
            q1, q2 = quant_hlm(wgu, bits, li * 8), quant_hlm(wdn, bits, li * 8 + 1)
        err.add(spec, wgu, q1)
        err.add(spec, wdn, q2)
        if whole:
            return q1, q2
        if qgu is None:
            qgu, qdn = torch.empty(gu.shape, dtype=torch.bfloat16, device=gu.device), torch.empty(dn.shape, dtype=torch.bfloat16, device=gu.device)
        qgu[sel], qdn[sel] = q1, q2
    return qgu, qdn


def experts_forward(e, X, idx, wgt, specs, li, err, chunk, ctx):
    """The routed experts' output for the tokens X (N, H) with routing idx, wgt (N, K), the experts rounded by specs: what the layer's Glm5NextTextExperts / FP8Experts forward does,
    with the weights read from the model's FP8 (or bf16) tensors a chunk of experts at a time, and the same arithmetic (FP8Experts scales the product in bf16 and sums in float32)."""
    N, E = X.shape[0], e.gate_up_proj.shape[0]
    fp8 = hasattr(e, "gate_up_proj_scale_inv")
    rt = Route(idx, E)
    acc = torch.zeros(N, X.shape[1], dtype=torch.float32 if fp8 else X.dtype, device=X.device)
    for a in range(0, E, chunk):
        b = min(E, a + chunk)
        if not any(rt.cnt[a:b]):
            continue
        gu, dn = fetch(e, a, b)
        qgu, qdn = rounded(e, a, gu, dn, specs, li, err, rt, X, ctx)
        del gu, dn
        for j in range(b - a):
            tk, ps = rt.of(a + j)
            if not len(tk):
                continue
            cur = F.linear(e._apply_gate(F.linear(X[tk], qgu[j])), qdn[j])
            cur = cur * (wgt[tk, ps, None].to(cur.dtype) if fp8 else wgt[tk, ps, None])
            acc.index_add_(0, tk, cur.to(acc.dtype))
        del qgu, qdn
    return acc.to(X.dtype)


# --------------------------------------------------------------------------------------------------------------------------------------- the pass
class Mlp(torch.nn.Module if torch else object):
    """Stands where a layer's MoE block stands: first collects its input (and returns zeros), then returns the outputs it was given."""

    def __init__(self):
        super().__init__()
        self.x, self.y, self.i = [], None, 0

    def forward(self, x):
        if self.y is None:
            self.x.append(x)
            return torch.zeros_like(x)
        self.i += 1
        return self.y[self.i - 1]


class Run:
    """The model and what a pass needs of it. This is GLM-5.3-Flash's: hyper-connection streams, layers that hand the sparse attention indexer's top-k on to the next, gated FP8 experts and a
    shared expert on the block's input. Another model brings a subclass (nemotron_quality.py's) that sets the class attributes and replaces __init__ and the methods from embed to finish."""
    NAME = "glm"       # glm_skew.py's MODELS key: repo, pinned revision, model class
    SLOT = "mlp"       # the attribute of a layer that holds its MoE block (a stand-in takes its place during a pass)
    REF = "FP8"        # the reference model as the reports call it
    REF_BPW = 8 + 32 / (128 * 128)  # stored bits a weight of the reference experts takes (FP8 and a float32 scale per 128 x 128 tile)
    CONTROL = True     # whether ref (the experts unrounded, through this code) can differ from the reference, so that each variant is also scored against it (FP8 kernels round each activation, the control does not)
    REF_MAX = 0.5      # the KL of ref from the reference above which the job stops: the unrounded experts do not reproduce the model, so nothing after it means anything
    KEPT = "attention, shared experts, dense layers, routers, embedding and head"  # what the rounding leaves as released, for the report
    CHUNK_CAP = 16     # experts rounded at a time, at most
    VARIANTS = VARIANTS
    ORDER = ORDER
    SETUP_MIN = SETUP_MIN

    def __init__(self, model):
        self.m, self.lm = model, model.model.language_model
        self.hc, self.layers = self.lm.config.hc_mult, self.lm.layers
        self.moe = [i for i, l in enumerate(self.layers) if hasattr(l.mlp, "experts")]
        self.E = self.layers[self.moe[0]].mlp.experts.gate_up_proj.shape[0]
        self.K = self.lm.config.num_experts_per_tok
        self.eager = False  # (whether the model's own experts were switched to the eager loop: see robust)

    def dev(self, li):
        return next(self.layers[li].parameters()).device

    def block(self, li):
        """Layer li's MoE block."""
        return getattr(self.layers[li], self.SLOT)

    def mats(self):
        """The (rows, cols) of the matrices of one expert."""
        e = self.block(self.moe[0]).experts
        return [tuple(getattr(e, n).shape[1:]) for n in expert_names(e)]

    @property
    def width(self):
        """Values a position of a residual stream takes."""
        return self.hc * self.lm.config.hidden_size

    @property
    def dev0(self):
        return self.lm.embed_tokens.weight.device

    @property
    def head_dev(self):
        return self.lm.norm.weight.device

    def embed(self, x):
        """The residual stream the token ids x (S,) start with: a CPU tensor, [1, S, hc, H]."""
        return self.lm.embed_tokens(torch.from_numpy(x).long()[None].to(self.dev0)).unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous().cpu()

    def step(self, layer, h, carry, S, dev):
        """The layer on the stream h (on dev, S positions): its output stream and what the next layer takes along (here the indexer's top-k of the sparse attention layers)."""
        mask, pos = torch.ones(1, S, dtype=torch.bool, device=dev), torch.arange(S, device=dev)[None]
        return layer(h, attention_mask=mask, position_ids=pos, position_embeddings=None, input_ids=None, past_key_values=None, prev_topk_indices=carry)

    def normed(self, h):
        """What the head takes of the final streams h (CPU): after the final norm, on the head's device."""
        return self.lm.norm(self.lm.hc_head(h.to(self.head_dev)))

    def expert_input(self, moe, X, S):
        """What the routed experts take of the block's input X (all sequences' positions, [N * S, H]): the router's choice is made on X itself; here the experts take it too."""
        return X

    def routed(self, e, Xe, idx, w, specs, li, err, chunk, ctx):
        """The routed experts' weighted sums for the tokens Xe with routing idx, w, the experts rounded by specs."""
        return experts_forward(e, Xe, idx, w, specs, li, err, chunk, ctx)

    def finish(self, moe, routed, X, S):
        """The block's outputs, a [1, S, H] tensor a sequence, from the routed experts' sums: the shared expert (the model's own, on the block's input) added."""
        seq = lambda i: X[i * S:(i + 1) * S].view(1, S, -1)
        for i in range(X.shape[0] // S):
            routed[i * S:(i + 1) * S].add_(moe.shared_experts(seq(i)).view(S, -1))
        return [routed[i * S:(i + 1) * S].view(1, S, -1) for i in range(X.shape[0] // S)]


def robust(run, fn, *a, **kw):
    """fn(...), and once more with the eager experts if the model's own experts fail (glm_skew.py's fallback: the grouped FP8 kernels are the likeliest thing to fail on a new card)."""
    try:
        return fn(*a, **kw)
    except (torch.cuda.OutOfMemoryError, Deadline, Abort):
        raise
    except Exception as ex:
        if run.eager:
            raise
        log(f"forward failed ({type(ex).__name__}: {str(ex)[:300]}): trying experts_implementation=eager")
        run.eager = True
        run.m.set_experts_implementation("eager")
        return fn(*a, **kw)


def pick_chunk(args, run, li, moe, specs, X):
    """Experts a time: --chunk if given, else what half the memory still free on the layer's device holds (at most run.CHUNK_CAP)."""
    if getattr(args, "chunk", 0):
        return args.chunk
    d = run.dev(li)
    if d.type != "cuda":
        return 16
    torch.cuda.empty_cache()
    room = torch.cuda.mem_get_info(d)[0] - X.shape[0] * X.shape[1] * 4  # (the float32 sum of the experts' outputs)
    e = moe.experts
    per = sum(getattr(e, n)[0].numel() for n in expert_names(e)) * 4 * max(BYTES[k + "_gptq" if k in ROWS and g else "gptq" if g else k] for k, _, g in map(parse_spec, set(specs)))
    return int(max(1, min(run.CHUNK_CAP, 0.5 * room / per)))


def routed_outputs(run, li, moe, Xe, idx, w, specs, err, args, ctx):
    """The routed experts' weighted sums for the tokens Xe (what the experts take) with routing idx, w, by run.routed; an out-of-memory starts the layer again with half as many experts a time."""
    snap, chunk = (ctx.done, ctx.fell), pick_chunk(args, run, li, moe, specs, Xe)
    while True:
        oom = None
        try:
            routed = run.routed(moe.experts, Xe, idx, w, specs, li, err, chunk, ctx)
        except torch.cuda.OutOfMemoryError as ex:
            oom = ex
        if oom is None:
            return routed
        if chunk == 1:
            raise oom
        oom, chunk = None, chunk // 2  # (the failed frames' tensors go with the exception: free them, then try again with fewer experts)
        err.d.clear()
        ctx.done, ctx.fell = snap
        gc.collect()
        torch.cuda.empty_cache()
        log(f"layer {li}: out of memory, {chunk} experts a time")


def moe_outputs(run, li, moe, xs, specs, err, args, ctx):
    """Layer li's MoE block on the inputs xs (a [1, S, H] tensor a sequence; emptied), the routed experts rounded by specs: [1, S, H] outputs. The router and everything around the
    routed experts (the shared expert, a latent projection) are the model's own."""
    S = xs[0].shape[1]
    X = torch.cat([x.reshape(-1, x.shape[-1]) for x in xs])
    xs.clear()
    ws, idxs = [], []
    for i in range(X.shape[0] // S):  # (a sequence at a time, as the model calls it)
        _, w, idx = moe.gate(X[i * S:(i + 1) * S].view(1, S, -1))
        ws.append(w)
        idxs.append(idx)
    routed = routed_outputs(run, li, moe, run.expert_input(moe, X, S), torch.cat(idxs), torch.cat(ws), specs, err, args, ctx)
    return run.finish(moe, routed, X, S)


def layerwise(run, ids, specs_of=None, ncal=0, args=None, deadline=0, label="", tick=None):
    """The model's forward over the sequences ids (N, S) one layer after another: the residual streams after the last layer (a list of CPU tensors), the weight error
    by spec and the GPTQ counts. specs_of(li) -> the specs of MoE layer li's experts (None: the model's own experts); the first ncal sequences are GPTQ's calibration."""
    S, N = ids.shape[1], len(ids)
    hs = [run.embed(x) for x in ids]
    tks, werr, t0 = [None] * N, {}, time.time()
    t_moe, n_moe = None, 0  # (when the first MoE layer began and how many are done: the rate of the rest)
    ctx = types.SimpleNamespace(ncal=ncal * S, min_tok=getattr(args, "gptq_min_tok", 2048), bits=3, done=0, fell=0, label=label, export=getattr(args, "exporter", None))  # (label: the variant; export: nemotron_hbr.py's Exporter, or None)
    for li, layer in enumerate(run.layers):
        if deadline and time.time() > deadline:
            raise Deadline(f"{label}: deadline before layer {li}")
        dev = run.dev(li)

        def call(i):
            return run.step(layer, hs[i].to(dev), tks[i], S, dev)
        specs = specs_of(li) if specs_of is not None and li in run.moe else None
        if specs is None:
            with on(dev):
                for i in range(N):
                    h, tk = call(i)
                    hs[i], tks[i] = h.cpu(), None if tk is None else tk.cpu()
        else:
            t_moe = t_moe or time.time()
            moe, st, err = run.block(li), Mlp(), Err()
            setattr(layer, run.SLOT, st)
            try:
                with on(dev):
                    for i in range(N):
                        call(i)
                    st.y = moe_outputs(run, li, moe, st.x, specs, err, args, ctx)
                    for i in range(N):
                        h, tk = call(i)
                        hs[i], tks[i] = h.cpu(), None if tk is None else tk.cpu()
            finally:
                setattr(layer, run.SLOT, moe)
            for k, v in err.read().items():
                w = werr.setdefault(k, [0.0, 0.0, 0.0])
                for j in range(3):
                    w[j] += v[j]
            n_moe += 1
            late = time.time() + (time.time() - t_moe) / n_moe * sum(1 for m in run.moe if m > li) + 90  # (the rest of the MoE layers at the rate so far, and scoring)
            if deadline and n_moe >= min(4, len(run.moe) - 1) and late > deadline:
                raise Deadline(f"{label}: at layer {li} it would end {(late - deadline) / 60:.0f} min after the deadline")
        if tick:
            tick(li, time.time() - t0)
    return hs, werr, ctx


# ------------------------------------------------------------------------------------------------------------------------------------- scoring
def scores(run, h, tgt, ref=None, ref2=None, keep=False):
    """The next-token distribution after each position of one sequence's final residual streams h ([1, S, ...], CPU), against tgt (S-1,), the tokens that follow: the top-TOPK ids and
    log-probs (ref None: the reference model's; keep: also when there is one, for the control that becomes ref2), the KL divergence from ref ({'ids', 'lp'} of the same positions) and from
    ref2 (the control: kl2) and whether the top token agrees with each; the target's NLL always."""
    with on(run.head_dev):
        return _scores(run, run.normed(h[:, :-1]), tgt, ref, ref2, keep)


def _scores(run, x, tgt, ref, ref2, keep):
    out = collections.defaultdict(list)
    for c0 in range(0, x.shape[1], HEAD_ROWS):
        lp = run.m.lm_head(x[:, c0:c0 + HEAD_ROWS]).float().log_softmax(-1)[0]
        out["nll"].append(-lp.gather(1, torch.from_numpy(tgt[c0:c0 + HEAD_ROWS]).long().to(lp.device)[:, None])[:, 0].cpu())
        if ref is None or keep:
            v, i = lp.topk(TOPK, -1)
            out["ids"].append(i.int().cpu())
            out["lp"].append(v.cpu())
        for sfx, r in (("", ref), ("2", ref2)):
            if r is None:
                continue
            ids, rlp = r["ids"][c0:c0 + HEAD_ROWS].to(lp.device).long(), r["lp"][c0:c0 + HEAD_ROWS].to(lp.device)
            qk = lp.gather(1, ids)
            p = rlp.exp()
            pt, qt = (1 - p.sum(-1)).clamp_min(1e-12), (1 - qk.exp().sum(-1)).clamp_min(1e-12)
            out["kl" + sfx].append(((p * (rlp - qk)).sum(-1) + pt * (pt.log() - qt.log())).cpu())
            hit = ids == lp.argmax(-1)[:, None]  # where the variant's top token sits in the reference's list: it agrees if it is a best one there (bf16 logits tie: a tied token counts)
            out["agree" + sfx].append((hit.any(-1) & ((rlp * hit).sum(-1) == rlp[:, 0])).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(parts, dom, doms):
    """Per-position results of all sequences -> the numbers of one variant, overall and by domain (+ the per-position arrays). kl, top1: against the FP8 model; kl_ctl, top1_ctl:
    against the control (the FP8 experts dequantised to bf16), which has no FP8 activation rounding in it."""
    cat = {k: torch.cat([p[k] for p in parts]).numpy() for k in parts[0] if k not in ("ids", "lp")}
    dom = np.asarray(dom)
    names = (("", ""), ("2", "_ctl"))

    def block(m):
        r = {"n": int(m.sum()), "ppl": float(math.exp(cat["nll"][m].astype(np.float64).mean()))}
        for sfx, nm in names:
            if "kl" + sfx in cat:
                r.update({"kl" + nm: float(cat["kl" + sfx][m].astype(np.float64).mean()), "top1" + nm: float(cat["agree" + sfx][m].mean())})
        return r
    out = {"positions": len(dom), **block(np.ones(len(dom), bool)), "by_domain": {d: block(dom == i) for i, d in enumerate(doms)}}
    for sfx, nm in names:
        if "kl" + sfx in cat:
            per_seq = np.array([p["kl" + sfx].double().mean().item() for p in parts])
            out["kl" + nm + "_se"] = float(per_seq.std(ddof=1) / len(per_seq) ** 0.5)  # (over the sequences: the positions of one are not independent)
    if "kl" in cat:
        k = cat["kl"].astype(np.float64)
        out.update(kl_median=float(np.median(k)), kl_p99=float(np.quantile(k, 0.99)), kl_p999=float(np.quantile(k, 0.999)), kl_max=float(k.max()))
    return out, cat


# ----------------------------------------------------------------------------------------------------------------------------------------- the job
def ram_gb():
    """GB of RAM this process may use: the container's limit if there is one, else what the host has free (None: unknown)."""
    free = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                free = int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 1 << 50:
                free = min(free, int(v) / 2 ** 30) if free else int(v) / 2 ** 30
        except OSError:
            pass
    return free


def load_data(d, cal_n, held_n):
    ld = lambda split, n: {dom: np.load(f"{d}/{split}_{dom}.npy")[:n] for dom in gs.DOMAINS}
    return ld("held", held_n), ld("calib", cal_n)


def load_model(path, name="glm"):
    import transformers
    n = torch.cuda.device_count()
    kw = {}
    if n:  # (glm_skew.py's 0.9 of each card: the layout its run measured, 71-77 GiB on each of four 95.6 GiB cards, so 18-24 GiB stay free for the rounded experts, the Hessians and the streams)
        frac = float(os.environ.get("MEM_FRAC", 0.9))
        kw = dict(device_map="auto", max_memory={i: int(torch.cuda.get_device_properties(i).total_memory * frac) for i in range(n)})
    if n > 1:
        bad = host_hops()
        log(f"GPU-to-GPU copies go through the host; direct copies that came back wrong on this pod: {bad or 'none'}")
    return getattr(transformers, gs.MODELS[name]["cls"]).from_pretrained(path, dtype=torch.bfloat16, **kw).eval()


def count_routing(run, cal, held, out, nseq):
    """Routing counts per layer and expert over the first nseq calibration sequences of each domain and all held-out ones, by the model's own forward with a hook on every router;
    written as glm_skew.py's counts/<split>_<domain>.npz (so its analyze works on them). Returns the pooled calibration counts and the pooled held-out ones (layers, experts)."""
    E = run.E
    cnt = {li: torch.zeros(E, dtype=torch.int64, device=run.dev(li)) for li in run.moe}
    mass = {li: torch.zeros(E, dtype=torch.float32, device=run.dev(li)) for li in run.moe}
    uniq = {li: torch.zeros((), dtype=torch.int64, device=run.dev(li)) for li in run.moe}

    def hook(mo, i, o, li):
        c = torch.bincount(o[2].reshape(-1), minlength=E)
        cnt[li] += c
        mass[li].index_add_(0, o[2].reshape(-1), o[1].reshape(-1).float())
        uniq[li] += (c > 0).sum()
    hooks = [run.block(li).gate.register_forward_hook(lambda mo, i, o, li=li: hook(mo, i, o, li)) for li in run.moe]
    try:
        return _count(run, cal, held, out, nseq, cnt, mass, uniq)
    finally:
        for h in hooks:
            h.remove()


def _count(run, cal, held, out, nseq, cnt, mass, uniq):
    os.makedirs(f"{out}/counts", exist_ok=True)
    pooled = {}
    dev0 = run.dev0
    for split, data, n in (("calib", cal, nseq), ("held", held, None)):
        for dom in gs.DOMAINS:
            seqs = data[dom][:n]
            for t in (cnt, mass, uniq):
                for v in t.values():
                    v.zero_()
            t1 = time.time()
            for x in seqs:
                before = [{k: v.clone() for k, v in t.items()} for t in (cnt, mass, uniq)]
                try:
                    run.m(input_ids=torch.from_numpy(x).long()[None].to(dev0), use_cache=False, logits_to_keep=1)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as ex:  # (as glm_skew.py: once, try the eager experts; the layers that counted before the failure are given back)
                    if run.eager:
                        raise
                    log(f"forward failed ({type(ex).__name__}: {str(ex)[:300]}): trying experts_implementation=eager")
                    run.eager = True
                    run.m.set_experts_implementation("eager")
                    for t, saved in zip((cnt, mass, uniq), before):
                        for k, v in t.items():
                            v.copy_(saved[k])
                    run.m(input_ids=torch.from_numpy(x).long()[None].to(dev0), use_cache=False, logits_to_keep=1)
            tokens = seqs.shape[0] * seqs.shape[1]
            c = torch.stack([cnt[li].cpu() for li in run.moe]).numpy()
            assert (c.sum(1) == tokens * run.K).all(), f"{split}_{dom}: counts per layer {c.sum(1)[:3]}... are not tokens x {run.K}"
            np.savez_compressed(f"{out}/counts/{split}_{dom}.npz", counts=c, mass=torch.stack([mass[li].cpu() for li in run.moe]).numpy().astype(np.float64), ntok=tokens,
                                uniq=torch.stack([uniq[li].cpu() for li in run.moe]).numpy() / len(seqs), layers=np.array(run.moe))
            pooled[split] = pooled.get(split, 0) + c.astype(np.float64)
            log(f"counts {split}_{dom}: {tokens} tokens in {time.time() - t1:.0f} s")
    return pooled["calib"], pooled["held"]


def emit(out, row):
    with open(f"{out}/results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    render(out)


def run_all(model, a, run_cls=Run):
    """The job, given a loaded model and its Run class; a: data (prep's .npy files), out, variants, held_seqs, count_seqs, gptq_seqs, gptq_min_tok, chunk, deadline."""
    t_start = time.time()
    run = run_cls(model)
    table = run.VARIANTS
    held, cal = load_data(a.data, max(a.count_seqs, a.gptq_seqs), a.held_seqs)
    doms = list(gs.DOMAINS)
    hids = np.concatenate([held[d] for d in doms])
    S = hids.shape[1]
    dom = np.concatenate([np.full(len(held[d]) * (S - 1), i) for i, d in enumerate(doms)])
    tgts = [x[1:] for x in hids]
    mcfg = gs.MODELS[run.NAME]
    L, E = len(run.moe), run.E
    mats = run.mats()
    n_w = sum(r * c for r, c in mats)
    meta = {"repo": mcfg["repo"], "rev": mcfg["rev"], "seq": S, "topk": TOPK, "domains": doms, "held_sequences": len(hids), "held_tokens": int(hids.size), "positions": len(dom),
            "moe_layers": L, "experts": E, "expert_weights": n_w, "top_k": run.K, "count_seqs": a.count_seqs, "gptq_seqs": a.gptq_seqs, "gptq_min_tok": a.gptq_min_tok,
            "order": a.variants, "hot": {}, "ref": run.REF, "ref_bpw": run.REF_BPW, "kept": run.KEPT, "control": run.CONTROL}
    os.makedirs(f"{a.out}/perpos", exist_ok=True)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    left = lambda: a.deadline - time.time() if a.deadline else 1e9
    tick = lambda name: (lambda li, s: log(json.dumps({"variant": name, "layer": li, "seconds": round(s, 1)})) if li % 4 == 3 or li == len(run.layers) - 1 else None)

    try:
        # the model's own forward against the layer-by-layer pass with the model's own experts (the code the variants run)
        t0 = time.time()
        two = hids[[0, -1]]

        def compare():
            hs, _, _ = layerwise(run, two, None, args=a, label="check")
            chk = []
            for h, x in zip(hs, two):  # (in slices of positions: a sequence's float32 log-probs are 1.3 GB)
                mine = run.m.lm_head(run.normed(h))[0]
                hf = run.m(input_ids=torch.from_numpy(x).long()[None].to(run.dev0), use_cache=False).logits[0]
                kls, same = [], []
                for c0 in range(0, S, HEAD_ROWS):
                    p, q = hf[c0:c0 + HEAD_ROWS].float().log_softmax(-1), mine[c0:c0 + HEAD_ROWS].to(hf.device).float().log_softmax(-1)
                    kls.append((p.exp() * (p - q)).sum(-1))
                    same.append(p.argmax(-1) == q.argmax(-1))
                chk.append((float(torch.cat(kls).mean()), float(torch.cat(same).float().mean())))
            return chk
        chk = robust(run, compare)
        check = {"kl": max(c[0] for c in chk), "top1": min(c[1] for c in chk), "seconds": round(time.time() - t0, 1)}
        log(f"layer-by-layer pass vs the model's own forward on 2 sequences: KL {check['kl']:.2e}, top-1 agreement {check['top1']:.4f}")
        if not (check["kl"] < 1e-3 and check["top1"] > 0.995):
            raise Abort(f"the layer-by-layer pass does not reproduce the model's forward: {check}")

        # the reference model: the top TOPK next-token log-probs of every held-out position
        t0 = time.time()
        hs, _, _ = robust(run, layerwise, run, hids, None, args=a, label=run.REF.lower(), deadline=a.deadline, tick=tick(run.REF.lower()))
        parts = [scores(run, h, t) for h, t in zip(hs, tgts)]
        ref = {k: torch.cat([p[k] for p in parts]) for k in ("ids", "lp")}
        base, _ = summarize(parts, dom, doms)
        base["tail_mass"] = float((1 - ref["lp"].double().exp().sum(-1)).clamp_min(0).mean())  # (what the top-TOPK leaves out: the KLs here are lower bounds)
        del hs, parts
        if base["ppl"] > 200:  # (a language model on this text: ~3-30)
            base["warning"] = f"perplexity {base['ppl']:.0f} on ordinary text: the {run.REF} model or the text is broken, whatever follows is measured against that"
            log("WARNING: " + base["warning"])
        emit(a.out, {"variant": "base", "what": f"the {run.REF} model as released", "seconds": round(time.time() - t0, 1), "driver_check": check, **base})
        log(f"reference: perplexity {base['ppl']:.3f} over {base['positions']} positions in {time.time() - t0:.0f} s")
    except Exception as ex:  # (nothing else can run: leave a row for quality.md and stop)
        emit(a.out, {"variant": "base", "what": f"the {run.REF} model as released", "error": f"{type(ex).__name__}: {str(ex)[:300]}"})
        raise
    refslice = lambda i: {k: v[i * (S - 1):(i + 1) * (S - 1)] for k, v in ref.items()}

    ref2slice = lambda i: {k: v[i * (S - 1):(i + 1) * (S - 1)] for k, v in ref2.items()}
    pooled = held_counts = ref2 = None
    hot, factor = {}, 1.0
    for name in a.variants:
        v = table[name]
        row = {"variant": name, "item": v["item"], "what": v["what"]}
        need = v["min"] * 60 * factor
        if left() < need + 120:
            row["skipped"] = f"{left() / 60:.0f} min left, {need / 60:.0f} expected"
            log(f"{name}: skipped, {row['skipped']}")
            emit(a.out, row)
            continue
        try:
            if v["share"] is not None:
                if pooled is None:
                    t1 = time.time()
                    pooled, held_counts = count_routing(run, cal, held, a.out, a.count_seqs)
                    log(f"routing counts in {time.time() - t1:.0f} s")
                k = max(1, round(v["share"] * E / 100))
                hot[name] = gs.top_mask(pooled, k)
                meta["hot"][str(v["share"])] = {"k": k, "calib": gs.hit(hot[name], pooled), "held": gs.hit(hot[name], held_counts)}
                json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
            specs_of = lambda li: layer_specs(v, hot[name][run.moe.index(li)] if name in hot else None, E)
            gptq = any(parse_spec(s)[2] for s in (v["rest"], v["hot"] or v["rest"]))
            ids, ncal = hids, 0
            if gptq:  # the calibration sequences go through first and are rounded layer by layer like the held-out ones after them
                n = a.gptq_seqs
                gb = lambda n: (len(doms) * n + len(hids)) * S * run.width * 2 / 2 ** 30
                avail = ram_gb()
                while avail and gb(n) > 0.45 * avail and n > 1:
                    n -= 1
                ncal = len(doms) * n
                ids = np.concatenate([cal[d][:n] for d in doms] + [hids])
                log(f"{name}: {ncal} calibration sequences ({ncal * S} tokens a layer), streams {gb(n):.1f} GB of {avail or 0:.0f} GB RAM")
            t1 = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            hs, werr, ctx = layerwise(run, ids, specs_of, ncal=ncal, args=a, deadline=a.deadline, label=name, tick=tick(name))
            t_pass, t2 = time.time() - t1, time.time()
            parts = [scores(run, h, t, refslice(i), ref2slice(i) if ref2 else None, keep=name == "ref" and run.CONTROL) for i, (h, t) in enumerate(zip(hs[ncal:], tgts))]
            res, cat = summarize(parts, dom, doms)
            np.savez_compressed(f"{a.out}/perpos/{name}.npz", nll=cat["nll"], dom=dom.astype(np.int8), **{k: cat[k] for k in ("kl", "agree", "kl2", "agree2") if k in cat})
            ctl = {k: torch.cat([p[k] for p in parts]) for k in ("ids", "lp")} if name == "ref" and run.CONTROL else None
            del hs, parts
            bw = sum(spec_bpw(s, mats, run.REF_BPW) for li in run.moe for s in specs_of(li)) / (L * E)
            peak = [round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 1) for i in range(torch.cuda.device_count())]
            row.update(res, bits_per_weight=bw, experts_gb=L * E * n_w * bw / 8e9, seconds={"pass": round(t_pass, 1), "score": round(time.time() - t2, 1)}, peak_gib=peak,
                       weight_error={k: {"rel_rmse": (x[0] / x[1]) ** 0.5, "weights": int(x[2])} for k, x in werr.items()})
            if gptq:
                row["gptq"] = {"experts_gptq": ctx.done, "experts_rtn_fallback": ctx.fell, "min_tokens": a.gptq_min_tok, "calibration_tokens": ncal * S}
            if name != "ref":  # (the control rounds nothing: its speed says little of the others')
                factor = min(3.0, max(0.5, (time.time() - t1) / (v["min"] * 60)))
            log(f"{name}: KL {res['kl']:.4f}, top-1 {res['top1']:.4f}, perplexity {res['ppl']:.3f} ({run.REF} {base['ppl']:.3f}), {bw:.3f} bits/weight, {time.time() - t1:.0f} s, peak GiB per GPU {peak}")
            if name == "ref" and res["kl"] > run.REF_MAX:
                raise Abort(f"the control (ref) does not reproduce the {run.REF} model (KL {res['kl']:.3f}): every variant would be measured against a broken path")
            if name == "ref" and run.CONTROL:
                ref2 = ctl  # (from here on every variant is also scored against the control: what the rounding alone costs, without the FP8 kernels' activation rounding)
        except Deadline as ex:
            row["aborted"] = str(ex)
            log(f"{name}: {ex}")
        except Abort as ex:
            emit(a.out, {**row, "error": str(ex)})
            raise
        except Exception as ex:  # (one variant failing does not end the job: the next one may not)
            row["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            log(f"{name}: {row['error']}")
            traceback.print_exc()
        ex = None
        free_all()
        emit(a.out, row)
    log(f"done in {(time.time() - t_start) / 60:.1f} min")


def cmd_run(a, run_cls=Run):
    a.variants = [v.strip() for v in (a.variants or ",".join(run_cls.ORDER)).split(",") if v.strip()]
    assert not (getattr(a, "export", None) and run_cls is Run), "--export is nemotron_quality.py's"
    bad = [v for v in a.variants if v not in run_cls.VARIANTS]
    assert not bad, f"variants {bad}: {list(run_cls.VARIANTS)}"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = load_model(a.weights, run_cls.NAME)
    n = torch.cuda.device_count()
    log(f"loaded in {time.time() - t0:.0f} s on {n} GPUs, GiB per GPU {[round(torch.cuda.memory_allocated(i) / 2 ** 30, 1) for i in range(n)]}")
    with torch.inference_mode():
        run_all(model, a, run_cls)


# ---------------------------------------------------------------------------------------------------------------------------------------- report
def render(out):
    meta = json.load(open(f"{out}/meta.json"))
    rows = [json.loads(l) for l in open(f"{out}/results.jsonl")]
    for name, text in (("quality.json", json.dumps({"meta": meta, "rows": rows}, indent=1)), ("quality.md", markdown(meta, rows))):
        open(f"{out}/{name}.tmp", "w").write(text)
        os.replace(f"{out}/{name}.tmp", f"{out}/{name}")  # (whole or not at all: the launcher pulls ~/results every minute)


def markdown(meta, rows):
    fk = lambda x: "" if x is None else (f"{x:.4f}" if x >= 1e-4 or x == 0 else f"{x:.1e}")
    pc = lambda x: "" if x is None else f"{100 * x:.2f}"
    rn, cc = meta.get("ref", "FP8"), meta.get("control", True)  # (the reference model's name; whether the control gets columns of its own: only where it can differ from the reference)
    ref_bpw = meta.get("ref_bpw", 8 + 32 / 16384)
    line = lambda cells: "|" + "|".join(" " if c is None else f" {c} " for i, c in enumerate(cells) if cc or i not in (6, 9)) + "|"  # (None: a cell left empty)
    base = next((r for r in rows if r["variant"] == "base" and "ppl" in r), None)
    doms = meta["domains"]
    done = [r for r in rows if "kl" in r and not r.get("error")]
    ctl = any("kl_ctl" in r for r in done)
    o = [f"## Routed experts at 2-4 bits: {meta['repo']} @ {meta['rev'][:8]}", "",
         f"Held-out text: {meta['held_sequences']} sequences of {meta['seq']} tokens ({meta['held_tokens']:,} tokens, {len(doms)} domains), {meta['positions']:,} next-token positions scored. "
         f"Reference: the {rn} model as released. KL divergence is taken over its top {meta['topk']} tokens and the rest of the vocabulary as one bucket (a lower bound on the full divergence). "
         f"Only the routed experts of the {meta['moe_layers']} MoE layers are rounded ({meta['experts']} each, {meta['expert_weights']:,} weights an expert); "
         f"{meta.get('kept', 'attention, shared experts, dense layers, routers, embedding and head')} stay as released. Every layer of a variant sees the layers before it already rounded.", ""]
    if base:
        o += [f"{rn} model: perplexity {base['ppl']:.3f}; its top {meta['topk']} tokens leave out {pc(base.get('tail_mass'))}% of the probability on average. The layer-by-layer pass with the model's own experts reproduces its forward "
              f"to KL {base['driver_check']['kl']:.1e} (top-1 agreement {base['driver_check']['top1']:.4f}).", ""]
        if base.get("warning"):
            o += [f"**WARNING: {base['warning']}**", ""]
    o += [line(["item", "variant", "what", "bits/weight", "experts GB", f"KL from {rn}, mean (s.e. over sequences)", "KL from control", "KL p99", f"top-1 = {rn} %", "top-1 = control %", "perplexity", f"vs {rn}", "minutes"]),
          "|" + "|".join(a for i, a in enumerate(["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]) if cc or i not in (6, 9)) + "|"]
    for r in rows:
        if r["variant"] == "base" and base is r:
            o.append(line([None, rn, "the model as released", f"{ref_bpw:.2f}", f"{meta['moe_layers'] * meta['experts'] * meta['expert_weights'] * ref_bpw / 8e9:.1f}", "0", None, "0", "100", None, f"{r['ppl']:.3f}", None,
                           f"{r['seconds'] / 60:.1f}"]))
        elif "kl" in r and not r.get("error"):
            s = r["seconds"]
            o.append(line([r["item"], r["variant"], r["what"], f"{r['bits_per_weight']:.3f}", f"{r['experts_gb']:.1f}", f"{fk(r['kl'])} ({fk(r['kl_se'])})", fk(r.get("kl_ctl")), fk(r["kl_p99"]), pc(r["top1"]),
                           pc(r.get("top1_ctl")), f"{r['ppl']:.3f}", f"{100 * (r['ppl'] / base['ppl'] - 1):+.2f}%", f"{(s['pass'] + s['score']) / 60:.1f}"]))
        else:
            o.append(line([r.get("item", ""), r["variant"], r.get("what", ""), None, None, None, None, None, None, None, None, r.get("error") or r.get("skipped") or r.get("aborted") or "", None]))
    ref = next((r for r in done if r["variant"] == "ref"), None)
    if ref and cc and rn != "FP8":
        o += ["", f"The control (ref) is the experts as released, run through the same path unrounded: its KL {fk(ref['kl'])} from the {rn} model is the floor of that column (the path sums the experts in "
              f"float32, the model's own kernels in {rn}). The columns 'from control' score every other variant against the control instead: what rounding the weights alone costs."]
    elif ref and cc:
        o += ["", f"The control (ref) is the FP8 experts dequantised to bf16 and run through the same path: its KL {fk(ref['kl'])} from the FP8 model is the floor of that column (the FP8 model's kernels also round each "
              "activation to FP8; a Glyd engine would not). The columns 'from control' score every other variant against the control instead: what rounding the weights alone costs."]
    elif ref:
        o += ["", f"The control (ref) is the experts as released, run through the same path unrounded: its KL {fk(ref['kl'])} from the {rn} model is the check that the path reproduces the model (0 up to the order of "
              "float additions), so the KL of every other variant is what rounding the weights alone costs."]
    for key, title, f in (("kl", f"KL divergence from {rn}, mean", fk), ("kl_ctl", "KL divergence from the control (ref), mean", fk), ("top1", f"top-1 agreement with {rn}, %", pc),
                          ("top1_ctl", "top-1 agreement with the control, %", pc), ("ppl", "perplexity", lambda x: f"{x:.3f}")):
        if key.endswith("_ctl") and not ctl:
            continue
        o += ["", f"### {title}, by domain", "", "| variant | all | " + " | ".join(doms) + " |", "|---|" + "---:|" * (len(doms) + 1)]
        if key == "ppl" and base:
            o.append(f"| {rn} | " + " | ".join(f(x) for x in [base["ppl"]] + [base["by_domain"][d]["ppl"] for d in doms]) + " |")
        for r in done:
            if key in r:
                o.append(f"| {r['variant']} | " + " | ".join(f(x) for x in [r[key]] + [r["by_domain"][d][key] for d in doms]) + " |")
    if meta.get("hot"):
        o += ["", "### The most-used experts (profile from the calibration text)", "", "| share of a layer's experts | per layer | calibration routed tokens covered % | held-out routed tokens covered % |", "|---:|---:|---:|---:|"]
        for p, h in sorted(meta["hot"].items(), key=lambda kv: int(kv[0])):
            o.append(f"| {p}% | {h['k']} | {pc(h['calib'])} | {pc(h['held'])} |")
    errs = [r for r in done if r.get("weight_error")]
    if errs:
        o += ["", f"### Weight error (relative RMSE of the rounded experts against the {rn} original, all layers)", "", "| variant | spec | weights | relative RMSE |", "|---|---|---:|---:|"]
        for r in errs:
            for k, x in sorted(r["weight_error"].items()):
                o.append(f"| {r['variant']} | {k} | {x['weights']:,} | {x['rel_rmse']:.4f} |")
    for r in done:
        if "gptq" in r:
            g = r["gptq"]
            o += ["", f"GPTQ ({r['variant']}): {g['experts_gptq']} expert-layers rounded with GPTQ, {g['experts_rtn_fallback']} fell back to nearest (fewer than {g['min_tokens']} of the {g['calibration_tokens']:,} calibration tokens routed to them)."]
    skipped = [r for r in rows if not (r["variant"] == "base" and "ppl" in r) and ("kl" not in r or r.get("error"))]
    if skipped:
        o += ["", "Not measured: " + "; ".join(f"{r['variant']} ({r.get('error') or r.get('skipped') or r.get('aborted')})" for r in skipped) + "."]
    return "\n".join(o) + "\n"


def cmd_report(a):
    render(a.dir)
    print(open(f"{a.dir}/quality.md").read())


# -------------------------------------------------------------------------------------------------------------------------------------- selftest
def tiny_model(seed=0):
    """A random Glm5Next of 6 layers (3 dense, then 2 KDA + 1 sparse-MLA... as the real one's pattern, MoE with 16 experts, top-4), weights scaled so that rounding an expert is felt."""
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
    torch.manual_seed(seed)
    cfg = Glm5NextConfig(
        text_config=dict(vocab_size=256, hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
                         n_routed_experts=16, num_experts_per_tok=4, q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32, v_head_dim=32, index_n_heads=4, index_head_dim=32,
                         index_topk=64, index_kpool=4, linear_head_dim=32, linear_num_heads=4, pad_token_id=0, eos_token_id=[1]),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=128))
    m = Glm5NextForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    m.set_experts_implementation("eager")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("experts.gate_up_proj", "experts.down_proj")):
                p.copy_(torch.randn_like(p) * 0.06)
            elif n.endswith("gate.weight"):
                p.copy_(torch.randn_like(p) * 0.05)
            elif n.endswith("embed_tokens.weight"):
                p.copy_(torch.randn_like(p))
            elif n == "lm_head.weight":
                p.copy_(torch.randn_like(p) * 0.3)
            elif "shared_experts" in n:
                p.mul_(0.5)
    return m


def cuda_check(devices):
    """The block codes and the FP8 dequantisation on each of the devices against the CPU's: the CPU selftest cannot see a CUDA op that behaves differently, and wrong rounding would not raise an
    error, it would only make the numbers wrong. The same weights to a few codes that fall either side of a boundary, GPTQ's error under H the same to a few percent."""
    g = torch.Generator().manual_seed(1)
    W = (torch.randn(3, 128, 256, generator=g) * 0.05).bfloat16().float()
    X = torch.randn(2048, 256, generator=g) * (1 + 4.0 * (torch.arange(256) % 17 == 0))
    H = X.T @ X / 2048
    w8, s8 = (torch.randn(2, 256, 256, generator=g) * 40).to(torch.float8_e4m3fn), torch.rand(2, 2, 2, generator=g) + 0.01
    herr = lambda q: float((((q.float() - W[0]) @ H) * (q.float() - W[0])).sum())

    def run(d):
        with on(d):
            w, h = W.to(d), H.to(d)
            return {"int4": quant_int(w.clone(), 4), "hlm3": quant_hlm(w.clone(), 3, 5), "hlm2": quant_hlm(w.clone(), 2, 5), "gptq": gptq_hlm(w.clone(), [h.clone() for _ in range(3)], 3, 7)[0],
                    "lloyd": lloyd_max(torch.rand(64, 256, generator=torch.Generator().manual_seed(2)).to(d) * 2 - 1, None, 8), "deq": deq(w8.to(d), s8.to(d))}
    want = run(torch.device("cpu"))
    for d in devices:
        got = {k: v.cpu() for k, v in run(torch.device(d)).items()}
        for k in want:
            if k != "gptq":  # (GPTQ feeds each column's error to the next: a flipped code moves the rest, so only its error under H is compared)
                diff = float(((got[k].float() - want[k].float()).abs() > 1e-6 * want[k].float().abs().max()).float().mean())
                assert diff == 0 if k == "deq" else diff < 0.02, (d, k, diff)
        assert abs(herr(got["gptq"][0]) / herr(want["gptq"][0]) - 1) < 0.1, (d, "gptq error under H", herr(got["gptq"][0]), herr(want["gptq"][0]))
    return len(devices)


def codes_selftest():
    """The block codes (CPU) against verbatim copies of calib.py's and kld_study.py's functions: nothing here is a model's, so the selftest of every model that rounds with them runs it."""
    # ---- verbatim copies of calib.py's and kld_study.py's functions (branch sub3bit, ac0ef626), the reference the ported ones are checked against
    def o_fwht(x):
        n = x.shape[-1]
        R = math.gcd(n, 128)
        y, h = x.reshape(-1, n // R, R), 1
        while h < R:
            y = y.reshape(-1, n // R, R // (2 * h), 2, h)
            y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
            h *= 2
        return y.reshape(x.shape) / R ** 0.5

    def o_lloyd_max(x, wt, k, iters=25, chunk=2048):
        out = []
        for r0 in range(0, x.shape[0], chunk):
            xr = x[r0:r0 + chunk]
            c = torch.linspace(-1, 1, k, device=x.device).expand(xr.shape[0], k).clone()
            wx, wr = xr * wt, wt.expand_as(xr)
            for _ in range(iters):
                idx = (xr[:, :, None] - c[:, None, :]).abs().argmin(-1)
                num, den = torch.zeros_like(c).scatter_add_(1, idx, wx), torch.zeros_like(c).scatter_add_(1, idx, wr)
                c = torch.where(den > 0, num / den.clamp_min(1e-30), c)
            out.append(c.sort(-1).values)
        return torch.cat(out).bfloat16().float()

    def o_gptq(w, H, bits, block, act_order=False, damp=0.01, step=128, lm=False, rot_seed=None):  # (returns the dequantized matrix only)
        rows, cols = w.shape
        q = (1 << (bits - 1)) - 1
        runs = cols // block
        W, H = w.float().clone(), H.float().clone()
        dead = H.diagonal() == 0
        W[:, dead] = 0
        wt = H.diagonal().clone()
        if rot_seed is not None:
            s = signs(cols, rot_seed, w.device)
            W = o_fwht(W * s)
            H = o_fwht(o_fwht(H * s).T * s).T
            wt = H.diagonal().clone()
            dead = H.diagonal() == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        perm = torch.arange(cols, device=w.device)
        if act_order:
            order = torch.argsort(H.diagonal().view(runs, block).sum(1), descending=True)
            perm = (order[:, None] * block + torch.arange(block, device=w.device)).reshape(-1)
            W, H, wt = W[:, perm], H[perm][:, perm], wt[perm]
        if lm:
            norm = W.view(rows, runs, block) / W.view(rows, runs, block).abs().amax(-1, keepdim=True).clamp_min(1e-30)
            cb = o_lloyd_max(norm.reshape(rows, cols), (wt / wt.mean().clamp_min(1e-30) + 1e-6)[None, :], 1 << bits)
        H.diagonal().add_(damp * H.diagonal().mean())
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        Q = torch.zeros_like(W)
        for i1 in range(0, cols, step):
            i2 = min(i1 + step, cols)
            W1, E1, U1 = W[:, i1:i2].clone(), torch.zeros(rows, i2 - i1, device=w.device), U[i1:i2, i1:i2]
            for i in range(i2 - i1):
                if i % block == 0:
                    if lm:
                        sc = W1[:, i:i + block].abs().amax(-1).clamp_min(1e-30).bfloat16().float()
                    else:
                        sc = fit_scale(W1[:, i:i + block].reshape(rows, 1, block), q).view(rows)
                if lm:
                    c = (W1[:, i, None] / sc[:, None] - cb).abs().argmin(-1)
                    v = cb.gather(1, c[:, None])[:, 0] * sc
                else:
                    c = torch.clamp(torch.round(W1[:, i] / sc), -q, q)
                    v = c * sc
                Q[:, i1 + i] = v
                e = (W1[:, i] - v) / U1[i, i]
                W1[:, i:] -= e[:, None] * U1[i, i:][None, :]
                E1[:, i] = e
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
        Q = Q[:, torch.argsort(perm)]
        if rot_seed is not None:
            Q = o_fwht(Q) * s
        return Q.bfloat16()

    def o_int(t, bits, block):  # kld_study.int_tensor(t, bits, block, 'mse', scale_bf16=True)
        q = (1 << (bits - 1)) - 1
        w = t.float().view(-1, t.shape[1] // block, block)
        s = fit_scale(w, q)
        return (torch.clamp(torch.round(w / s), -q, q) * s).view(-1, t.shape[1]).to(t.dtype)

    def oracle_lloyd(x, wt, k, per=1, **kw):  # the same call as lloyd_max's, by calib.py's
        if wt is None:
            return o_lloyd_max(x, torch.ones(1, x.shape[1]), k)
        return torch.cat([o_lloyd_max(x[j * per:(j + 1) * per], wt[j][None, :], k) for j in range(len(wt))])
    herr = lambda dw, H: float(((dw.float() @ H) * dw.float()).sum())

    # ---- the block codes against calib.py's
    for shape in ((5, 96), (3, 128), (2, 4, 256), (7, 64)):
        t = torch.randn(*shape)
        assert torch.allclose(fwht(t), o_fwht(t), atol=1e-5) and torch.allclose(fwht(fwht(t)), t, atol=1e-5) and torch.allclose(fwht(t).norm(dim=-1), t.norm(dim=-1), atol=1e-4), shape
    Wt = (torch.randn(3, 64, 128) * 0.05).bfloat16()
    for bits in (2, 3, 4, 5):
        assert torch.equal(quant_int(Wt.float(), bits), o_int(Wt.view(-1, 128), bits, BLOCK).view(3, 64, 128)), bits
    gen = torch.Generator().manual_seed(5)
    for k in (4, 8, 16):  # the sorted Lloyd-Max finds the codebook of calib.py's passes (to bf16), weighted or not, in one chunk or many; Lloyd's iteration can end in another state when a value
        # sits within rounding of a boundary (the two sum in another order): here at most 2% of the rows, and never a worse codebook overall
        xl = torch.cat([torch.randn(60, 256, generator=gen) * 0.5, torch.rand(60, 256, generator=gen) * 2 - 1, (torch.rand(60, 256, generator=gen) - 0.5).pow(3) * 8]).clamp(-1, 1)[:120]
        wl = torch.rand(2, 256, generator=gen).pow(4) * 3 + 1e-6
        for wt, per in ((None, 1), (wl, 60)):
            a_, b_ = lloyd_max(xl, wt, k, per=per), oracle_lloyd(xl, wt, k, per=per)
            assert float((((a_ - b_).abs().max(-1).values) > 0.02).float().mean()) <= 0.02, (k, (a_ - b_).abs().max())
            mse = lambda cb: float(((xl - assign(xl, cb)) ** 2).sum())
            assert mse(a_) <= mse(b_) * 1.003, (k, mse(a_), mse(b_))
            assert torch.equal(a_, lloyd_max(xl, wt, k, per=per, elems=256 * 7)), "chunked rows differ"
    Wo = (torch.randn(48, 96) * 0.05 * torch.where(torch.arange(96) % 17 == 0, 4.0, 1.0)).bfloat16().float()  # (bf16 values: calib.py rounds its input to bf16)
    saved = globals()["lloyd_max"]
    try:
        globals()["lloyd_max"] = oracle_lloyd  # with calib.py's own codebook fit the ported rounding and GPTQ must agree with calib.py's gptq_matrix to rounding noise
        rtn = quant_hlm(Wo[None].repeat(2, 1, 1), 3, 7)
        ref = o_gptq(Wo.bfloat16(), torch.eye(96), 3, BLOCK, True, lm=True, rot_seed=7)  # (H = identity: no error to feed on, so nearest)
        diff = (rtn[0].float() - ref.float()).abs()
        assert torch.equal(rtn[0], rtn[1]) and float(diff.max()) < 1e-5 * float(ref.float().abs().max()) and float((diff > 0).float().mean()) < 0.01, ("hlm RTN", float(diff.max()), float((diff > 0).float().mean()))
        Xo = torch.randn(2048, 96) * torch.where(torch.arange(96) % 17 == 0, 12.0, 1.0)
        Xo = Xo + 0.5 * Xo.roll(1, 1)
        Xo[:, 5] = 0  # a dead input channel
        Hs = torch.stack([Xo.T @ Xo / 2048, (Xo.roll(3, 0) * 0.7).T @ (Xo.roll(3, 0) * 0.7) / 2048])
        Ws = torch.stack([Wo, Wo.flip(0) * 1.3]).bfloat16().float()
        got, ok = gptq_hlm(Ws.clone(), [h.clone() for h in Hs], 3, 11)
        assert ok.all()
        for j in range(2):
            want = o_gptq(Ws[j].bfloat16(), Hs[j], 3, BLOCK, True, lm=True, rot_seed=11)
            d = (got[j].float() - want.float()).abs()
            assert float(d.max()) < 1e-5 * float(want.float().abs().max()) and float((d > 0).float().mean()) < 0.01, ("gptq", j, float(d.max()), float((d > 0).float().mean()))
            assert herr(got[j].float() - Ws[j], Hs[j]) <= herr(quant_hlm(Ws[j][None], 3, 11)[0].float() - Ws[j], Hs[j]), "GPTQ is no better than nearest under H"
        one, _ = gptq_hlm(Ws[:1].clone(), [Hs[0].clone()], 3, 11)
        assert (one[0].float() - got[0].float()).abs().max() < 1e-6, "a matrix alone differs from the same matrix in a batch"
        # hlm2g's: the same code at 2 bits. Only the largest difference is compared: 1.4% and 1.5% of the two matrices' entries are exactly 0 under the oracle's butterfly (the codes cancel) and up to 1.3e-8 under the product with the Hadamard matrix
        got2b, ok2b = gptq_hlm(Ws.clone(), [h.clone() for h in Hs], 2, 11)
        assert ok2b.all()
        for j in range(2):
            want = o_gptq(Ws[j].bfloat16(), Hs[j], 2, BLOCK, True, lm=True, rot_seed=11)
            assert float((got2b[j].float() - want.float()).abs().max()) < 1e-5 * float(want.float().abs().max()), ("gptq, 2 bits", j, float((got2b[j].float() - want.float()).abs().max()))
        # more than one block of 128 columns (the real matrices have 16 and 32): the error fed on past a block goes through the block update
        Wm = (torch.randn(2, 40, 288) * 0.05).bfloat16().float()
        Xm = torch.randn(4096, 288) * torch.where(torch.arange(288) % 17 == 0, 8.0, 1.0)
        Hm = [Xm.T @ Xm / 4096, Xm.roll(5, 0).T @ Xm.roll(5, 0) / 4096]
        gm, okm = gptq_hlm(Wm.clone(), [h.clone() for h in Hm], 3, 3)
        assert okm.all()
        for j in range(2):
            wm = o_gptq(Wm[j].bfloat16(), Hm[j], 3, BLOCK, True, lm=True, rot_seed=3)
            dm = (gm[j].float() - wm.float()).abs()
            assert float(dm.max()) < 1e-3 * float(wm.float().abs().max()) and float((dm > 0).float().mean()) < 0.01, ("gptq, 288 columns", j, float(dm.max()), float((dm > 0).float().mean()))
    finally:
        globals()["lloyd_max"] = saved
    # the ported codebook fit in its own right: GPTQ beats nearest under H, and hlm beats the uniform grid, on a matrix with outlier channels at 2 bits
    got, ok = gptq_hlm(Ws.clone(), [h.clone() for h in Hs], 2, 13)
    e_g = [herr(got[j].float() - Ws[j], Hs[j]) for j in range(2)]
    e_n = [herr(quant_hlm(Ws[j][None], 2, 13)[0].float() - Ws[j], Hs[j]) for j in range(2)]
    e_u = [herr(quant_int(Ws[j][None], 2)[0].float() - Ws[j], Hs[j]) for j in range(2)]
    assert ok.all() and all(g < n < u for g, n, u in zip(e_g, e_n, e_u)), (e_g, e_n, e_u)
    bad_h = [Hs[0].clone(), -torch.eye(96)]  # not positive definite even damped: flagged, the other matrix untouched
    got2, ok2 = gptq_hlm(Ws.clone(), list(bad_h), 3, 11)
    assert ok2.tolist() == [True, False] and torch.equal(got2[0], gptq_hlm(Ws[:1].clone(), [Hs[0].clone()], 3, 11)[0][0])


def selftest(_=None):
    import shutil, tempfile
    for pkg in ("fla", "causal_conv1d"):
        sys.modules[pkg] = None  # (the tiny model is on the CPU: with these installed, transformers binds their Triton / CUDA kernels when the model code is imported, and they take no CPU tensors)
    torch.manual_seed(0)
    n_cuda = cuda_check([f"cuda:{i}" for i in range(torch.cuda.device_count())]) if torch.cuda.is_available() and os.environ.get("SELFTEST_GPU", "1") != "0" else 0
    assert cuda_check(["cpu"]) == 1  # (the check itself: the CPU against itself)
    # ---- specs, bits
    assert parse_spec("hlm3g") == ("hlm", 3, True) and parse_spec("hlm2g") == ("hlm", 2, True) and parse_spec("int4") == ("int", 4, False) and parse_spec("ref") == ("ref", 0, False)
    for bad in ("int", "ref3", "int4g", "hlm", "x3"):
        try:
            parse_spec(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert abs(spec_bpw("int4", GLM_MATS) - 4.5) < 1e-9 and abs(spec_bpw("int3", GLM_MATS) - 3.5) < 1e-9
    assert 3.54 < spec_bpw("hlm3", GLM_MATS) < 3.55 and abs(spec_bpw("hlm3g", GLM_MATS) - spec_bpw("hlm3", GLM_MATS)) < 1e-12
    assert abs(variant_bpw(VARIANTS["mix13"], GLM_MATS) - (0.13 * 4.5 + 0.87 * spec_bpw("hlm3", GLM_MATS))) < 1e-9 and layer_specs(VARIANTS["int4"], None, 3) == ["int4"] * 3
    assert layer_specs(VARIANTS["mix13"], [True, False, False], 3) == ["int4", "hlm3", "hlm3"] and set(ORDER) | set(HUMMINGBIRD) == set(VARIANTS) and not set(ORDER) & set(HUMMINGBIRD)
    # Hummingbird on GLM's matrices: hlm2 is 2 bits of codes + 1/2 of scale + 64/cols of codebook (4 entries of 16 bits a row) + 1/rows of the rotation's signs = 2.5159 (4096 x 4096) and 2.5315 (4096 x 2048), 2.5211 over an expert
    h2 = spec_bpw("hlm2g", GLM_MATS)
    assert abs(h2 - 2.5211) < 1e-4 and h2 == spec_bpw("hlm2", GLM_MATS)
    assert abs(variant_bpw(VARIANTS["hb40"], GLM_MATS) - (0.4 * spec_bpw("hlm3g", GLM_MATS) + 0.6 * h2)) < 1e-9 and abs(variant_bpw(VARIANTS["hb20"], GLM_MATS) - (0.2 * 4.5 + 0.8 * h2)) < 1e-9
    assert layer_specs(VARIANTS["hlm2g"], None, 3) == ["hlm2g"] * 3 and layer_specs(VARIANTS["hb40"], [True, False, False, True], 4) == ["hlm3g", "hlm2g", "hlm2g", "hlm3g"]
    assert layer_specs(VARIANTS["hb20"], [False, True, False], 3) == ["hlm2g", "int4", "hlm2g"]
    for n, k in (("hb40", 115), ("hb20", 58)):  # (the hot experts of a layer of 288: round(0.4 * 288), round(0.2 * 288))
        sp = layer_specs(VARIANTS[n], [e < k for e in range(288)], 288)
        assert sp.count(VARIANTS[n]["hot"]) == k and variant_bpw(VARIANTS[n], GLM_MATS) <= 3.0 and sum(spec_bpw(x, GLM_MATS) for x in sp) / 288 <= 3.0, n

    codes_selftest()

    # ---- FP8: the dequantisation is transformers', the experts' arithmetic is FP8Experts'
    from transformers.integrations.finegrained_fp8 import FP8Experts, Fp8Dequantize
    w8 = (torch.randn(5, 256, 384) * 40).to(torch.float8_e4m3fn)
    s8 = torch.rand(5, 2, 3) + 0.01
    assert torch.equal(deq(w8, s8), Fp8Dequantize(None)._dequantize_one(w8, s8, output_dtype=torch.float32)) and torch.equal(deq(w8[:, :, :128].bfloat16(), None), w8[:, :, :128].float())
    cfg8 = types.SimpleNamespace(hidden_size=256, moe_intermediate_size=128, num_local_experts=7, hidden_act="silu", swiglu_limit=10.0)
    e8 = FP8Experts(cfg8, block_size=(128, 128))
    with torch.no_grad():
        e8.gate_up_proj.data = (torch.randn(7, 256, 256) * 30).to(torch.float8_e4m3fn)
        e8.down_proj.data = (torch.randn(7, 256, 128) * 30).to(torch.float8_e4m3fn)
        e8.gate_up_proj_scale_inv.data = torch.rand(7, 2, 2) * 0.01 + 0.001
        e8.down_proj_scale_inv.data = torch.rand(7, 2, 1) * 0.01 + 0.001
        X8 = torch.randn(40, 256).bfloat16()
        w_, idx_ = torch.rand(40, 3), torch.stack([torch.randperm(7)[:3] for _ in range(40)])
        want = FP8Experts.__new__(FP8Experts)
        want.__dict__.update(e8.__dict__)
        want._parameters = dict(e8._parameters)
        for n_ in ("gate_up_proj", "down_proj"):
            want._parameters[n_] = torch.nn.Parameter(deq(getattr(e8, n_), getattr(e8, n_ + "_scale_inv")).bfloat16(), requires_grad=False)
        want_out = FP8Experts.forward(want, X8, idx_, w_)  # (bf16 weights: its F.linear path, the same arithmetic as with the FP8 kernel's)
        for chunk in (1, 3, 16):
            got = experts_forward(e8, X8, idx_, w_, ["ref"] * 7, 0, Err(), chunk, types.SimpleNamespace())
            assert torch.equal(got, want_out), ("FP8Experts arithmetic", chunk)
    rt = Route(idx_, 7)
    mask = torch.nn.functional.one_hot(idx_, 8).permute(2, 1, 0)
    for e_ in range(7):
        pos_, tok_ = torch.where(mask[e_])
        assert torch.equal(rt.of(e_)[0], tok_) and torch.equal(rt.of(e_)[1], pos_), e_

    # ---- the whole job on a tiny random model: every variant, and the checks around it
    q_extra = {f"t_{s}": dict(item=-1, what=f"test: every expert {s}", share=None, hot=None, rest=s, min=1) for s in ("int5", "int2", "hlm4", "hlm2")}
    VARIANTS.update(q_extra)
    m = tiny_model()
    S, d, out = 128, tempfile.mkdtemp(), tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    for dom in gs.DOMAINS:
        np.save(f"{d}/held_{dom}.npy", rng.integers(2, 256, (1, S)).astype(np.int32))
        np.save(f"{d}/calib_{dom}.npy", rng.integers(2, 256, (3, S)).astype(np.int32))
    names = ORDER + HUMMINGBIRD + list(q_extra)
    ns = lambda **kw: types.SimpleNamespace(data=d, out=out, variants=names, held_seqs=1, count_seqs=2, gptq_seqs=2, gptq_min_tok=300, chunk=0, deadline=0, **kw)
    with torch.inference_mode():
        # the model's own experts, the layer-by-layer pass and the head: logits to the bit
        ids = np.stack([np.load(f"{d}/held_{dom}.npy")[0] for dom in gs.DOMAINS[:2]])
        run = Run(m)
        hs, _, _ = layerwise(run, ids)
        bits = []
        for h, x in zip(hs, ids):  # (the same to the bit where it was written; the tolerance, a bf16 step of the logits, is for a BLAS that orders a product differently for another number of rows)
            a_, b_ = run.m.lm_head(run.lm.norm(run.lm.hc_head(h)))[0].float(), m(input_ids=torch.from_numpy(x).long()[None], use_cache=False).logits[0].float()
            assert (a_ - b_).abs().max() <= 0.07, "layer-by-layer differs from the forward"
            bits.append(torch.equal(a_, b_))
        try:
            layerwise(run, ids, deadline=1, label="late")
            raise AssertionError("no deadline")
        except Deadline:
            pass
        try:  # (a pass that cannot end before the deadline says so as soon as it knows, not at the deadline: here the 90 s kept for scoring alone are more than the 60 s left)
            layerwise(run, ids, lambda li: ["int4"] * run.E, deadline=time.time() + 60, label="slow", args=types.SimpleNamespace(chunk=0, gptq_min_tok=0))
            raise AssertionError("no projection")
        except Deadline as ex:
            assert "would end" in str(ex), ex
        # the model's own forward failing once is tried again with the eager experts, twice is raised
        tries = []

        def fails_once():
            tries.append(run.eager)
            if len(tries) == 1:
                raise RuntimeError("kernel")
            return "ok"
        assert robust(run, fails_once) == "ok" and tries == [False, True] and run.eager
        try:
            robust(run, lambda: 1 / 0)
            raise AssertionError("raised nothing")
        except ZeroDivisionError:
            pass
        run.eager = False
        # the routing counts are the same when the model's first forward fails and the eager experts take over
        held_d, cal_d = load_data(d, 3, 1)
        counts = count_routing(Run(m), cal_d, held_d, tempfile.mkdtemp(), 2)
        real_fwd, calls, failed = m.forward, [], []

        def bad_forward(*a_, **k_):
            calls.append(1)
            if len(calls) == 3:  # (the third sequence: counts of the first two are already in)
                failed.append(1)
                raise RuntimeError("grouped kernel")
            return real_fwd(*a_, **k_)
        m.forward = bad_forward
        try:
            run2 = Run(m)
            again = count_routing(run2, cal_d, held_d, tempfile.mkdtemp(), 2)
        finally:
            del m.forward
        assert failed and run2.eager and (counts[0] == again[0]).all() and (counts[1] == again[1]).all() and counts[0].sum() == 6 * 2 * S * run.K * len(run.moe), "counts changed by the retry"
        # an out-of-memory in the experts halves the chunk and changes nothing
        real, fired = globals()["experts_forward"], []

        def flaky(e, X, idx, w, specs, li, err, chunk, ctx):
            if not fired:
                fired.append(chunk)
                raise torch.cuda.OutOfMemoryError("simulated")
            fired.append(chunk)
            return real(e, X, idx, w, specs, li, err, chunk, ctx)
        a1 = ns()
        a1.variants = ["ref", "int4"]
        shutil.rmtree(out)
        os.makedirs(out)
        run_all(m, a1)
        kl_plain = {r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}["int4"]["kl"]
        globals()["experts_forward"] = flaky
        try:
            shutil.rmtree(out)
            os.makedirs(out)
            a1.chunk = 8
            run_all(m, a1)
        finally:
            globals()["experts_forward"] = real
        assert fired[0] == 8 and fired[1] == 4, fired
        assert abs({r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}["int4"]["kl"] - kl_plain) < 1e-5, "a smaller chunk changed the result"
        # the full job
        shutil.rmtree(out)
        os.makedirs(out)
        t0 = time.time()
        run_all(m, ns())
        secs = time.time() - t0
        # time left: a variant that would not end in time is skipped, one cut off by the deadline is recorded
        a2 = ns()
        a2.out, a2.variants, a2.deadline = tempfile.mkdtemp(), ["int4", "int3"], time.time() + 3600
        VARIANTS["int4"]["min"], old = 600, VARIANTS["int4"]["min"]
        try:
            run_all(m, a2)
        finally:
            VARIANTS["int4"]["min"] = old
        rows2 = {r["variant"]: r for r in map(json.loads, open(f"{a2.out}/results.jsonl"))}
        assert "skipped" in rows2["int4"] and "kl" in rows2["int3"], rows2
    # the pod's experts: FP8Experts with block scales (its Triton product, which needs a GPU, replaced by the dequantising one: the weights, the arithmetic around it and the accumulation are its own)
    out8 = tempfile.mkdtemp()
    kernel = FP8Experts.linear
    FP8Experts.linear = lambda self, input, weight, scale_inv, activation_scale=None: F.linear(input, deq(weight[None], scale_inv[None])[0].bfloat16())
    try:
        m8 = tiny_model()
        for li in Run(m8).moe:
            old_, new_ = m8.model.language_model.layers[li].mlp.experts, FP8Experts(m8.config.text_config, block_size=(128, 128))
            for name_ in ("gate_up_proj", "down_proj"):
                w_ = getattr(old_, name_).data.float()
                n_, r_, c_ = w_.shape
                sr_, sc_ = -(-r_ // 128), -(-c_ // 128)
                blk = w_.view(n_, sr_, r_ // sr_, sc_, c_ // sc_)
                sc8 = blk.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0
                getattr(new_, name_).data = (blk / sc8).to(torch.float8_e4m3fn).view(n_, r_, c_)
                getattr(new_, name_ + "_scale_inv").data = sc8.view(n_, sr_, sc_)
            m8.model.language_model.layers[li].mlp.experts = new_
        a8 = ns()
        a8.out, a8.variants = out8, ["ref", "int4", "hlm3g"]
        with torch.inference_mode():
            assert hasattr(Run(m8).layers[3].mlp.experts, "gate_up_proj_scale_inv") and Run(m8).layers[3].mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn
            run_all(m8, a8)
    finally:
        FP8Experts.linear = kernel
    rows8 = {r["variant"]: r for r in map(json.loads, open(f"{out8}/results.jsonl"))}
    assert rows8["base"]["driver_check"]["kl"] < 1e-6 and rows8["ref"]["kl"] < 1e-6 and rows8["ref"]["top1"] > 0.99 and 0 < rows8["int4"]["kl"] < 0.2 and "kl" in rows8["hlm3g"], rows8
    # the tiny model as the pod loads the real one: saved, then from_pretrained with a device_map over the GPUs (accelerate hooks, layers on different cards), through the same job
    from transformers import Glm5NextForConditionalGeneration
    ck, out_g = tempfile.mkdtemp(), tempfile.mkdtemp()
    tiny_model().save_pretrained(ck)
    nd = n_cuda  # (the six layers spread over the GPUs, the head on the last: the shape of the real model's map, which "auto" makes for 328 GB but would keep a 13 MB model on one card)
    if nd > 1:
        log(f"direct GPU-to-GPU copies that came back wrong: {host_hops() or 'none'} (the job's copies go through the host)")
    dm = {"model.visual": 0, "model.language_model.embed_tokens": 0, **{f"model.language_model.layers.{i}": i * nd // 6 for i in range(6)},
          "model.language_model.norm": nd - 1, "model.language_model.hc_head": nd - 1, "lm_head": nd - 1} if nd else None
    mg = Glm5NextForConditionalGeneration.from_pretrained(ck, dtype=torch.bfloat16, **({"device_map": dm} if dm else {})).eval()
    mg.set_experts_implementation("eager")
    ag = ns()
    ag.out, ag.variants = out_g, ["ref", "int4", "mix13", "hlm3g", "hb40"]
    with torch.inference_mode():
        run_all(mg, ag)
    rowsg = {r["variant"]: r for r in map(json.loads, open(f"{out_g}/results.jsonl"))}
    devs = {str(Run(mg).dev(li)) for li in range(6)}
    # (loose: on a GPU the same product for another number of rows may round differently, and this model is chaotic; what is caught is a path that does not work)
    assert rowsg["base"]["driver_check"]["kl"] < 5e-3 and rowsg["ref"]["kl"] < 0.5 * rowsg["int4"]["kl"] and 0 < rowsg["int4"]["kl"] and "kl" in rowsg["mix13"] and "kl" in rowsg["hlm3g"] and "kl" in rowsg["hb40"], (rowsg, devs)
    assert len(devs) >= min(2, n_cuda), f"the tiny model sits on {devs}: the device_map did not split it"
    rows = {r["variant"]: r for r in map(json.loads, open(f"{out}/results.jsonl"))}
    assert set(rows) == {"base"} | set(names), set(names) - set(rows)
    kl = {n: rows[n]["kl"] for n in names}
    # the identity variant: KL 0 (exactly, where run; the bound is for a BLAS that orders a product differently), the same top token and perplexity
    assert kl["ref"] < 1e-6 and rows["ref"]["top1"] > 0.99 and abs(rows["ref"]["ppl"] / rows["base"]["ppl"] - 1) < 1e-4 and np.load(f"{out}/perpos/ref.npz")["kl"].max() < 1e-4, "the identity variant is not the identity"
    assert rows["base"]["driver_check"]["kl"] < 1e-6 and rows["base"]["driver_check"]["top1"] > 0.99
    assert 0 < kl["t_int5"] < kl["int4"] < kl["int3"] < kl["t_int2"], kl
    assert 0 < kl["t_hlm4"] < kl["hlm3"] < kl["t_hlm2"], kl
    assert kl["hlm3"] < kl["int3"] and kl["int4"] < kl["hlm3"] and kl["mix13"] < kl["hlm3"] and kl["int4"] < kl["cold50"], kl
    assert all(abs(rows[n]["kl_ctl"] - rows[n]["kl"]) < 1e-6 and abs(rows[n]["top1_ctl"] - rows[n]["top1"]) < 1e-9 for n in names if n != "ref"), "here the control is the FP8 model: the same numbers"
    assert "kl_ctl" not in rows["ref"] and "KL divergence from the control" in open(f"{out}/quality.md").read()
    assert kl["hlm3g"] < kl["hlm3"] * 1.05, kl
    # Hummingbird: GPTQ at 2 bits beats nearest; the hot experts' extra bits help (hb40, hb20 < hlm2g) and the rest's missing ones cost (hlm3g < hb40, int4 < hb20)
    assert kl["hlm2g"] < kl["t_hlm2"] and kl["hb40"] < kl["hlm2g"] and kl["hb20"] < kl["hlm2g"] and kl["hlm3g"] < kl["hb40"] and kl["int4"] < kl["hb20"], kl
    g = rows["hlm3g"]["gptq"]
    assert g["experts_gptq"] > 0 and g["experts_rtn_fallback"] > 0 and g["experts_gptq"] + g["experts_rtn_fallback"] == 3 * 16, g
    gh = {n: rows[n]["gptq"] for n in HUMMINGBIRD}  # (GPTQ's experts over the 3 MoE layers: all 16 of hlm2g's and of hb40's (6 hlm3g, 10 hlm2g), the 13 hlm2g of hb20's)
    assert all(x["experts_gptq"] > 0 for x in gh.values()) and [x["experts_gptq"] + x["experts_rtn_fallback"] for x in gh.values()] == [3 * 16, 3 * 16, 3 * 13], gh
    we = {n: rows[n]["weight_error"] for n in ("int4", "int3", "hlm3", "mix13", "cold50", "hb40", "hb20")}
    assert 0.085 < we["int4"]["int4"]["rel_rmse"] < 0.10 and 0.19 < we["int3"]["int3"]["rel_rmse"] < 0.21 and we["hlm3"]["hlm3"]["rel_rmse"] < we["int3"]["int3"]["rel_rmse"], we  # (the ladder's: 0.0918 and 0.1965)
    assert we["mix13"]["int4"]["weights"] == 2 * 3 * 24_576 and we["mix13"]["hlm3"]["weights"] == 14 * 3 * 24_576 and we["cold50"]["hlm2"]["weights"] == 8 * 3 * 24_576, we
    assert we["hb40"]["hlm3g"]["weights"] == 6 * 3 * 24_576 and we["hb40"]["hlm2g"]["weights"] == 10 * 3 * 24_576 and we["hb20"]["int4"]["weights"] == 3 * 3 * 24_576 and we["hb20"]["hlm2g"]["weights"] == 13 * 3 * 24_576, we
    meta = json.load(open(f"{out}/meta.json"))
    # what quality.md says when things went wrong: a control that does not reproduce FP8, a base that never got made
    broken = [r for r in map(json.loads, open(f"{out}/results.jsonl")) if r["variant"] in ("base", "ref", "int4")]
    broken[1] = {**broken[1], "error": "the dequantised FP8 experts do not reproduce the FP8 model (KL 0.7)"}
    text = markdown(meta, broken)
    assert "reproduce the FP8 model (KL 0.7)" in text and "Not measured: ref" in text and "floor of that column" not in text and "| int4 |" in text, text
    text = markdown(meta, [{"variant": "base", "what": "the FP8 model as released", "error": "RuntimeError: boom"}])
    assert "RuntimeError: boom" in text and "Not measured: base" in text
    m_bad = tiny_model()
    m_bad.forward = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("boom"))
    ab = ns()
    ab.out, ab.variants = tempfile.mkdtemp(), ["ref"]
    try:
        with torch.inference_mode():
            run_all(m_bad, ab)
        raise AssertionError("a model that cannot run went on")
    except RuntimeError:
        pass
    assert "boom" in open(f"{ab.out}/quality.md").read() and not any("kl" in json.loads(l) for l in open(f"{ab.out}/results.jsonl")), "no row for the failed base"
    assert meta["hot"]["13"]["k"] == 2 and meta["hot"]["50"]["k"] == 8 and 0.1 < meta["hot"]["13"]["held"] < meta["hot"]["50"]["held"] <= 1, meta["hot"]
    assert meta["hot"]["20"]["k"] == 3 and meta["hot"]["40"]["k"] == 6, meta["hot"]
    md = open(f"{out}/quality.md").read()
    assert all(f"| {n} |" in md for n in names) and "### KL divergence from FP8, mean, by domain" in md and "fell back to nearest" in md and json.load(open(f"{out}/quality.json"))["meta"]["repo"] == meta["repo"]
    # the counts are glm_skew.py's: its analyze reads them
    gs.MODELS["tiny"] = dict(gs.MODELS["glm"], ne=16, k=4, n_moe=3, caches=(13, 25, 50))
    R = gs.analyze(argparse.Namespace(model="tiny", dir=out, json=f"{out}/skew.json", md=f"{out}/skew.md", quiet=True))
    assert R["layers"] == 3 and R["static"]["50"]["per_layer_alloc"]["heldout"] > 0
    print(f"selftest ok: rounding a tiny random Glm5Next ({secs:.0f} s for {len(names)} variants): KL {', '.join(f'{n} {kl[n]:.4f}' for n in ('ref', 't_int5', 'int4', 'int3', 't_int2', 'hlm3', 'mix13', 'cold50', 'hlm3g', 't_hlm2', 'hlm2g', 'hb40', 'hb20'))}; "
          f"weight error int4 {we['int4']['int4']['rel_rmse']:.4f} int3 {we['int3']['int3']['rel_rmse']:.4f} hlm3 {we['hlm3']['hlm3']['rel_rmse']:.4f}; GPTQ {g['experts_gptq']} experts, {g['experts_rtn_fallback']} fell back; "
          f"Hummingbird bits/weight on GLM's matrices {', '.join(f'{n} {variant_bpw(VARIANTS[n], GLM_MATS):.3f}' for n in HUMMINGBIRD)}; "
          f"with FP8Experts: ref KL {rows8['ref']['kl']:.1f}, int4 {rows8['int4']['kl']:.4f}; the layer-by-layer logits are the forward's {'to the bit' if all(bits) else 'within a bf16 step'}; block codes checked on {n_cuda} GPUs, the tiny model through run_all on {len(devs)} devices")
    shutil.rmtree(d)
    shutil.rmtree(out)
    shutil.rmtree(out8)
    shutil.rmtree(out_g)
    shutil.rmtree(ck)


def main(plan_fn=plan, run_fn=cmd_run, test_fn=selftest, variants=VARIANTS, doc=__doc__, order=ORDER):
    """The command line: plan, run, report, selftest; another model's module passes its own plan, run and selftest functions, variants, doc and default order."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--text", action="store_true")
    p.set_defaults(fn=plan_fn)
    p = sub.add_parser("run")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", help=f"comma list of {', '.join(variants)} (default: {', '.join(order)})")
    p.add_argument("--held-seqs", type=int, default=4, help="held-out sequences a domain")
    p.add_argument("--count-seqs", type=int, default=12, help="calibration sequences a domain for the routing counts (the hot and cold sets)")
    p.add_argument("--gptq-seqs", type=int, default=24, help="calibration sequences a domain for GPTQ's Hessians (fewer if the RAM for their streams is short)")
    p.add_argument("--gptq-min-tok", type=int, default=2048, help="calibration tokens an expert needs for GPTQ (fewer: rounded to nearest)")
    p.add_argument("--chunk", type=int, default=0, help="experts rounded at a time (0: by the free memory)")
    p.add_argument("--deadline", type=float, default=0, help="epoch seconds by which the job must be over")
    p.add_argument("--export", default=None, help="DIR: write the codes of the variant hbr40 there (nemotron_quality.py only)")
    p.set_defaults(fn=run_fn)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=test_fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
