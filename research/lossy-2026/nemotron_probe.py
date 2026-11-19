"""What the routed experts of Nemotron 3 Super leave to an expert-offload design (Linear GLY-130): three measurements on the bf16 model, nothing rounded, over the held-out text of glm_skew.py (24
sequences of 2,048 tokens, six domains) and over greedy decode steps.
  1. relu2 sparsity inside the experts. For every routed (token, expert) pair of every MoE layer: the share of the expert's 2,688 neurons that are exactly zero after relu2; the share of down's columns
     that must be kept, the largest activation first, for the expert's output to be within 1% and within 0.1% (relative L2) of the whole one; and how well the active neurons are named before up is
     computed, by the expert's most frequent ones in the calibration text (recall and output error at fixed budgets of neurons).
  2. Look-ahead routing. The router of MoE layer N (its pre-norm, gate, correction bias and top-22 selection: the model's own modules) applied to the residual stream at the input of block N - d, for
     d = 0..6: the share of the experts N really chose that it names (and of their routing weight), for as many experts as the model picks and for 1.5, 2 and 3 times as many.
  3. Per-expert importance. For every expert: how often it is chosen (calibration text), and when it is chosen its routing weight, the norm of its output and the norm of what it adds to the block's output.
  plan   [--text]      shell lines for the launcher; with --text what is measured and its time budget
  run    --weights DIR --data DIR --out DIR   the pod job (GPUs): load, one pass over the calibration and held-out sequences, greedy decodes, the arrays after every MoE layer
  report DIR           meta.json + prefill.npz + decode.npz -> probe.md and probe.json (numpy only; the job runs it at the end)
  selftest             a tiny random NemotronH on the CPU: every statistic on hand-made cases with known answers and against a naive reference on the model; with GPUs also on each of them
How a pass goes. nemotron_quality.py's layer-by-layer pass without the rounding: all sequences go through one layer after another with the residual streams on the host between layers, each block called
as the model calls it, and at a MoE layer the experts are read from the bf16 weights a chunk at a time (glm_quality.py's fetch), applied to the latent z = fc1_latent_proj(x) of the tokens routed to them
as NemotronHExperts does (bf16 up, relu2), and the statistics taken from their activations. The calibration sequences go through first in the same pass: their activations give each expert's neuron
frequencies and the routing counts that rank the experts, and only the held-out sequences are measured. A decode step is one token through the model's own generate() with a hook on every block, whose
inputs are kept; the same statistics are taken on them after the decodes, from the calibration profile.
Numbers are histograms and sums (exact for counts; 200 bins for ratios), so a layer, all layers and the decode steps are the same code and a quantile is a bin's edge. The error of dropping columns is
taken on float32 products of bf16 values (TF32 on the cards: bf16 values are exact there, and sums are float32). No token is used anywhere."""
import argparse, json, os, sys, tempfile, time, types

import glm_quality as gq      # the layer helpers: fetch, Route, on, tf32, log, load_model, robust
import glm_skew as gs         # the pins, the six text domains, the decode prompts
import nemotron_quality as nq # the model's Run: how its blocks are called

np, torch, F = gq.np, gq.torch, gq.F  # (None without numpy / torch: plan and report need neither)

HID, LAT, INTER = nq.HID, nq.LAT, nq.INTER
TOLS = (0.01, 0.001)                 # relative L2 error of an expert's output allowed when the columns of down of the smaller activations are dropped
BUDGETS = (0.0625, 0.125, 0.25, 0.5, 0.75)  # shares of an expert's neurons the active-set predictor may name
FETCH_X = (1, 1.5, 2, 3)             # experts the look-ahead router may name, as multiples of the number the model picks (22: 22, 33, 44, 66)
DMAX = 6                             # blocks back the look-ahead goes: the stream at the input of block N - d, d = 0..DMAX
BINS = 200                           # histogram bins of a ratio in [0, 1]
ROWS = 8192                          # pairs a time through the bisection (a block of rows is 88 MB of float32 at 2,688 neurons)
GROUPS = (13, 50)                    # the most used 13% and the least used 50% of a layer's experts by calibration counts: glm_quality.py's mix13 and cold50 sets
EX = ("n", "w", "y", "wy", "c", "z") + tuple(f"k{i}" for i in range(len(TOLS)))  # per-expert sums: pairs, routing weight, output norm, weight x output norm, contribution norm, zero neurons, columns kept
# minutes: guesses until a first run. glm_skew.py's run on this pod class (4 x A100 80 GB) put the model's forward at 1,237 tokens/s and the quality job's bf16 pass of 49k tokens took 38 s (4 x the
# sequences here: ~3 min); the statistics are an op count: per pair the up product, the down product, 24 bisection steps of the size of the down product (2 tolerances x 12), 10 for the predictor and its
# oracle (5 budgets x 2) and the contribution (1.5): ~38 products of 5.5 MFLOP, 1.08 M pairs a layer, 40 layers = 9 PFLOP, 2.5 min at 60 TFLOPS in TF32, about twice that with the sorts and the launches
SETUP_MIN, PASS_MIN, DECODE_MIN, REPORT_MIN = 8, 9, 4, 1


# ------------------------------------------------------------------------------------------------------------------------------------------------ plan
def plan(a):
    m = gs.MODELS["nemotron"]
    if a.text:
        jmm = int(os.environ.get("JOB_MAX_MIN", 45))
        print(f"{m['n_moe']} MoE layers x {m['ne']} experts, top-{m['k']}; an expert is up [{INTER}, {LAT}] + down [{LAT}, {INTER}] on the {LAT}-wide latent z = fc1_latent_proj(x) of the {HID}-wide x the router sees")
        print(f"measured on the 24 held-out sequences of glm_skew.py (6 domains x 4 x 2048 tokens = {6 * 4 * 2048:,} tokens, {6 * 4 * 2048 * m['k']:,} pairs a layer) and on greedy decodes of its 20 prompts (64 new tokens each)")
        print(f"  1. relu2 zeros, columns of down kept for {', '.join(f'{t:.1%}' for t in TOLS)} error, active-set prediction at {', '.join(f'{100 * b:g}%' for b in BUDGETS)} of the neurons (profile: 12 calibration sequences a domain)")
        print(f"  2. look-ahead: the router of layer N on the stream at the input of block N - d, d = 0..{DMAX}; recall of the {m['k']} chosen experts for {', '.join(str(round(m['k'] * x)) for x in FETCH_X)} named")
        print("  3. importance: routing weight, output norm and contribution norm of every expert against its calibration frequency")
        print(f"minutes (guesses, not measurements): setup ~{SETUP_MIN} (venv, download, pyenv, prep, selftest), pass ~{PASS_MIN} (load, 96 sequences through 88 layers ~3, statistics ~5), decodes ~{DECODE_MIN}, report ~{REPORT_MIN}: "
              f"~{SETUP_MIN + PASS_MIN + DECODE_MIN + REPORT_MIN} min of job; the job starts no phase that would not end 2 min before JOB_MAX_MIN {jmm}")
        return
    gs.plan(argparse.Namespace(model="nemotron"))  # REPO REV POD_CLASS DISK K NE N_MOE CACHE_PCTS WEIGHTS_GB
    print(f'SETUP_MIN="{SETUP_MIN}"')
    print(f'PASSES_MIN="{PASS_MIN + DECODE_MIN + REPORT_MIN}"')


# ------------------------------------------------------------------------------------------------------------------------------------- per-pair statistics
def order_rank(v):
    """Where each entry of a row of v (n, m) stands when the row is sorted largest first, ties by position: (n, m) int32."""
    order = v.float().argsort(dim=1, descending=True, stable=True)
    return torch.empty(order.shape, dtype=torch.int32, device=v.device).scatter_(1, order, torch.arange(order.shape[1], dtype=torch.int32, device=v.device).expand_as(order))


def cut_err(act, dn, keep):
    """The norm of the output the columns that are not kept carry, per row: act (n, I) float32 activations, dn (lat, I) down, keep (n, I) or (1, I) bool."""
    return torch.matmul(act * ~keep, dn.t()).norm(dim=1)


def keep_needed(act, dn, ynorm, rank, nnz, tols=TOLS):
    """For each tolerance, the fewest columns kept, largest activation first (rank: order_rank(act)), for ||dropped output|| <= tol * ||output|| (ynorm), per row: a bisection on the count between 0
    (all dropped: error 1) and nnz (only zeros dropped: error 0), each step one product of the size of the down projection. The count it ends on keeps the error within tol and drops it above tol one
    column fewer; it is the least such count if the error never rises as a column is added (first_crossing scans every count and says)."""
    out = []
    for tol in tols:
        lo, hi = torch.zeros_like(nnz), nnz.clone()
        for _ in range(act.shape[1].bit_length()):  # ponytail: assumes the error falls as columns are added; sampled against first_crossing in the job's check
            mid = (lo + hi) // 2
            ok = cut_err(act, dn, rank < mid[:, None]) <= tol * ynorm
            lo, hi = torch.where(ok, lo, mid), torch.where(ok, mid, hi)
        out.append(hi)
    return out


def first_crossing(act, dn, tols=TOLS):
    """keep_needed's counts the slow way, in float64: the error at every count k = 0..I for each row (what dropping all columns from rank k on leaves), and the least k within tol."""
    a, d = act.double(), dn.double()
    out = [[] for _ in tols]
    for i in range(a.shape[0]):
        o = a[i].argsort(descending=True, stable=True)
        r = (d[:, o] * a[i, o]).flip(1).cumsum(1).flip(1).norm(dim=0)
        r = torch.cat([r, r.new_zeros(1)])
        for t, tol in enumerate(tols):
            out[t].append(int((r <= tol * r[0]).nonzero()[0]))
    return [torch.tensor(o, dtype=torch.int32) for o in out]


def pair_stats(act, dn, prank=None, tols=TOLS, budgets=BUDGETS):
    """The statistics of n (token, expert) pairs of one expert: act (n, I) float32, the activations after relu2 (bf16 values), dn (lat, I) float32, its down matrix; prank (I,), the rank of each neuron by
    calibration frequency (order_rank of the counts), if the active-set predictor is wanted. Returns y (n, lat) the expert's output (before the routing weight), ynorm, zeros (neurons exactly zero),
    k (per tolerance: columns of down that must be kept) and, per budget B = share x I: recall (of the active neurons, by the B most frequent in the profile), e_pred (relative error of the output when
    only those B neurons are kept) and e_orc (the same for the B largest activations: what no predictor can beat)."""
    I = act.shape[1]
    y = act.matmul(dn.t())
    ynorm = y.norm(dim=1)
    pos = act > 0
    nnz = pos.sum(1, dtype=torch.int32)
    rank = order_rank(act)
    r = dict(y=y, ynorm=ynorm, zeros=I - nnz, k=keep_needed(act, dn, ynorm, rank, nnz, tols))
    if prank is not None:
        den, B = ynorm.clamp_min(1e-30), [round(f * I) for f in budgets]
        r["recall"] = [torch.where(nnz > 0, (pos & (prank < b)).sum(1) / nnz.clamp_min(1), torch.ones_like(ynorm)) for b in B]
        r["e_pred"] = [cut_err(act, dn, (prank < b)[None]) / den for b in B]
        r["e_orc"] = [cut_err(act, dn, rank < b) / den for b in B]
    return r


def hits_at(pidx, rank, tidx, ms, E):
    """Which of the experts the model chose (tidx (n, K)) a look-ahead router names, for each count m in ms (the first is K): the router's own top-K choice pidx (n, K) for m = K, the first m of its
    ranking rank (n, M >= max(ms)) for the larger ones. A list of bool (n, K)."""
    pos = torch.full((rank.shape[0], E), rank.shape[1], dtype=torch.int32, device=rank.device)
    pos.scatter_(1, rank, torch.arange(rank.shape[1], dtype=torch.int32, device=rank.device).expand_as(rank))
    p = pos.gather(1, tidx)
    return [(tidx[:, :, None] == pidx[:, None, :]).any(2) if i == 0 else p < m for i, m in enumerate(ms)]


def rhist(v):
    """Histogram of a ratio in [0, 1] over BINS bins (1.0 in the last)."""
    return torch.bincount((v.clamp(0, 1) * BINS).long().clamp_max(BINS - 1), minlength=BINS)


# ------------------------------------------------------------------------------------------------------------------------------------------- one layer
def new_acc(E, I, K, nf, dev):
    """The sums and histograms of one MoE layer, on its device: hz, hk (histograms over 0..I of the zero neurons and the columns kept per tolerance), ex (EX x experts sums), ssh (sum of the expert's share
    of the token's contribution), cal (calibration pairs), pr_h and pr_s (predictor: histogram of the recall, sums of recall, e_pred, e_orc and the pairs of each within TOLS[0]), la_h, la_w, la_wh, la_n
    (look-ahead by distance d and count named: histogram of the experts named right, sum of the weight share, histogram of it for the first count, tokens)."""
    z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=dev)
    return dict(hz=z(I + 1), hk=z(len(TOLS), I + 1), ex=z(len(EX), E, dt=torch.float64), ssh=z(E, dt=torch.float64), cal=z(E),
                pr_h=z(len(BUDGETS), BINS), pr_s=z(len(BUDGETS), 5, dt=torch.float64),
                la_h=z(DMAX + 1, nf, K + 1), la_w=z(DMAX + 1, nf, dt=torch.float64), la_wh=z(DMAX + 1, BINS), la_n=z(DMAX + 1))


def fetch_counts(K):
    return [round(K * x) for x in FETCH_X]


def lookahead(run, li, ring, idx, w, S, acc):
    """Look-ahead of MoE layer li into acc: for each distance d with the stream at the input of block li - d in ring (CPU, (T, 4096), the tokens in the order of idx (T, K), the experts the model
    chose, and w (T, K) their routing weights), the layer's own pre-norm and router on that stream S tokens a time (the shape the model calls them with), and which of idx the router names."""
    layer, moe = run.layers[li], run.block(li)
    gate, dev, (T, K) = moe.gate, run.dev(li), idx.shape
    ms = fetch_counts(K)
    assert gate.num_group == 1 and gate.top_k == K, "the ranking beyond the top-K is the router's scores only without expert groups"
    with gq.on(dev):
        for d in range(DMAX + 1):
            if li - d not in ring:
                continue
            for s0 in range(0, T, S):
                x = layer.norm(ring[li - d][s0:s0 + S].to(dev))
                logits, _, pidx = gate(x)
                rank = (logits.sigmoid() + gate.e_score_correction_bias).topk(ms[-1], 1).indices
                ti, tw = idx[s0:s0 + S], w[s0:s0 + S]
                for i, hit in enumerate(hits_at(pidx, rank, ti, ms, gate.num_experts)):
                    acc["la_h"][d, i] += torch.bincount(hit.sum(1), minlength=K + 1)
                    wr = (hit * tw).sum(1) / tw.sum(1)
                    acc["la_w"][d, i] += wr.sum().double()
                    if i == 0:
                        acc["la_wh"][d] += rhist(wr)
                acc["la_n"][d] += len(ti)


def layer_stats(run, li, X, S, nc, ring, prof=None, chunk=16, check=None):
    """Everything measured at MoE layer li. X: the block's input after its pre-norm for all tokens ((N * S, 4096) on the layer's device, S positions a sequence), the first nc of them calibration; ring:
    the streams the look-ahead reads (CPU, held-out tokens only); prof: the (experts, I) counts of calibration pairs each neuron was active in, or None to count them from the first nc tokens here;
    check: a dict (layers, entries, equal, max_dk) to fill with keep_needed's counts against first_crossing's on a few pairs. Returns the numpy arrays of new_acc and prof."""
    moe, dev, E, K = run.block(li), run.dev(li), run.E, run.K
    e, I = moe.experts, moe.experts.down_proj.shape[2]
    ws, ids = zip(*[moe.gate(X[i * S:(i + 1) * S].view(1, S, -1))[1:] for i in range(X.shape[0] // S)])  # (a sequence at a time, as the model calls the router)
    w, idx = torch.cat(ws), torch.cat(ids)
    Z = run.expert_input(moe, X, S)
    rt = gq.Route(idx, E)
    acc = new_acc(E, I, K, len(FETCH_X), dev)
    calibrate = prof is None
    prof = torch.zeros(E, I, dtype=torch.int32, device=dev) if calibrate else prof.to(dev)
    assert moe.fc2_latent_proj.bias is None
    fc2 = moe.fc2_latent_proj.weight.float()
    cmat = torch.zeros(X.shape[0] - nc, K, device=dev)  # what each held-out pair adds to the block's output, in norm
    ex = acc["ex"]
    with gq.tf32():
        for a0 in range(0, E, chunk):
            b0 = min(E, a0 + chunk)
            if not any(rt.cnt[a0:b0]):
                continue
            up, dn = gq.fetch(e, a0, b0)
            for j in range(b0 - a0):
                tk, ps = rt.of(a0 + j)
                if not len(tk):
                    continue
                act = e.act_fn(F.linear(Z[tk], up[j].bfloat16())).float()  # (NemotronHExperts: bf16 up, relu2 in bf16)
                hold = tk >= nc
                if calibrate:
                    prof[a0 + j] = (act[~hold] > 0).sum(0, dtype=torch.int32)
                    acc["cal"][a0 + j] = (~hold).sum()
                th, ph = tk[hold], ps[hold]
                if not len(th):
                    continue
                prank, ah, wj = order_rank(prof[a0 + j][None])[0], act[hold], w[th, ph]
                for r0 in range(0, len(th), ROWS):
                    sl = slice(r0, r0 + ROWS)
                    st = pair_stats(ah[sl], dn[j], prank)
                    if check is not None and r0 == 0 and len(th) >= 8 and not check["layers"].count(li):
                        want, got = first_crossing(ah[:16], dn[j]), [kk[:16].cpu() for kk in st["k"]]  # (the first 16 pairs of the first expert with 8 held-out pairs)
                        check["layers"].append(li)
                        check["entries"] += sum(g.numel() for g in got)
                        check["equal"] += sum(int((g == v).sum()) for g, v in zip(got, want))
                        check["max_dk"] = max([check["max_dk"]] + [int((g - v).abs().max()) for g, v in zip(got, want)])
                    wb = wj[sl]
                    cn = F.linear(st["y"] * wb[:, None], fc2).norm(dim=1)
                    cmat[th[sl] - nc, ph[sl]] = cn
                    acc["hz"] += torch.bincount(st["zeros"].long(), minlength=I + 1)
                    for t, kk in enumerate(st["k"]):
                        acc["hk"][t] += torch.bincount(kk.long(), minlength=I + 1)
                    ex[:, a0 + j] += torch.stack([x.double() for x in (torch.full((), float(len(wb)), device=dev), wb.sum(), st["ynorm"].sum(), (wb * st["ynorm"]).sum(), cn.sum(), st["zeros"].sum(),
                                                                       *[kk.sum() for kk in st["k"]])])
                    for b in range(len(BUDGETS)):
                        acc["pr_h"][b] += rhist(st["recall"][b])
                        acc["pr_s"][b] += torch.stack([st["recall"][b].sum(), st["e_pred"][b].sum(), st["e_orc"][b].sum(), (st["e_pred"][b] <= TOLS[0]).sum(), (st["e_orc"][b] <= TOLS[0]).sum()]).double()
            del up, dn
    share = cmat / cmat.sum(1, keepdim=True).clamp_min(1e-30)  # (the pair's part of its token's contributions, in norm: they sum to 1 over the K experts)
    acc["ssh"] += torch.zeros(E, dtype=torch.float64, device=dev).index_add_(0, idx[nc:].reshape(-1), share.reshape(-1).double())
    lookahead(run, li, ring, idx[nc:], w[nc:], S, acc)
    return {k: v.cpu().numpy() for k, v in acc.items()}, prof.cpu()


# ------------------------------------------------------------------------------------------------------------------------------------------ the passes
class Store:
    """The arrays of every MoE layer stacked (leading dimension: the MoE layers), written whole after each layer so a stopped job leaves what it measured; done = layers in."""

    def __init__(self, path, L):
        self.path, self.L, self.a, self.done = path, L, {}, 0

    def put(self, j, arrays):
        for k, v in arrays.items():
            self.a.setdefault(k, np.zeros((self.L,) + v.shape, v.dtype))[j] = v
        self.done = max(self.done, j + 1)
        np.savez_compressed(self.path + ".tmp.npz", done=self.done, **self.a)
        os.replace(self.path + ".tmp.npz", self.path)


def prefill(run, ids, ncal, deadline=0, chunk=16, tick=None, check=None):
    """The model's layers one after another over the sequences ids ((N, S) token ids, the ncal calibration sequences first, then the held-out ones), the residual streams on the host between layers, each
    block called as the model calls it, and at every MoE layer layer_stats on the block's input (captured by a hook on the block's MoE, so what is measured is what the model's own forward feeds it).
    tick(j, arrays, seconds) after the j-th MoE layer. Returns the final streams and the profile of each MoE layer (block index -> counts of calibration pairs each neuron was active in)."""
    S, N, nc = ids.shape[1], len(ids), ncal * ids.shape[1]
    hs = [run.embed(x) for x in ids]
    ring, prof = {}, {}
    for li, layer in enumerate(run.layers):
        if deadline and time.time() > deadline:
            raise gq.Deadline(f"deadline before block {li}")
        dev, cap = run.dev(li), []
        ring[li] = torch.cat([h.view(S, -1) for h in hs[ncal:]])  # (the stream at the input of block li, held-out tokens: what the look-ahead of the next layers reads)
        ring.pop(li - DMAX - 1, None)
        hook = layer.mixer.register_forward_pre_hook(lambda mod, args: cap.append(args[0])) if li in run.moe else None

        def layer_pass():  # (the model's own blocks; once more with the eager experts if its grouped kernels fail: from the first sequence, nothing of the failed try kept)
            cap.clear()
            with gq.on(dev):
                return [run.step(layer, hs[i].to(dev), None, S, dev)[0].cpu() for i in range(N)]
        try:
            outs = gq.robust(run, layer_pass)
        finally:
            if hook:
                hook.remove()
        hs[:] = outs  # (after the whole layer: the sequences' inputs are not replaced by outputs until all have run)
        if hook:
            X = torch.cat([c.reshape(-1, c.shape[-1]) for c in cap])
            cap.clear()
            t0 = time.time()
            arrays, prof[li] = layer_stats(run, li, X, S, nc, ring, chunk=chunk, check=check)
            del X
            gq.free_all()
            if tick:
                tick(run.moe.index(li), arrays, time.time() - t0)
    return hs, prof


def forward_check(run, hs, ids):
    """The final streams hs of the last two sequences of ids against the model's own forward on them: the KL divergence and top-token agreement of the two next-token distributions."""
    res = []
    for h, x in zip(hs[-2:], ids[-2:]):
        mine = run.m.lm_head(run.normed(h))[0]
        hf = run.m(input_ids=torch.from_numpy(x).long()[None].to(run.dev0), use_cache=False).logits[0]
        kl, same = [], []
        for c0 in range(0, len(x), gq.HEAD_ROWS):
            p, q = hf[c0:c0 + gq.HEAD_ROWS].float().log_softmax(-1), mine[c0:c0 + gq.HEAD_ROWS].to(hf.device).float().log_softmax(-1)
            kl.append((p.exp() * (p - q)).sum(-1))
            same.append(p.argmax(-1) == q.argmax(-1))
        res.append((float(torch.cat(kl).mean()), float(torch.cat(same).float().mean())))
    return {"kl": max(r[0] for r in res), "top1": min(r[1] for r in res)}


def decode_streams(run, prompts, new):
    """Greedy decodes of the prompts (lists of token ids), `new` tokens each, through the model's own generate() with a hook on every block that keeps the block's input at each one-token step, and one
    on every router that keeps the experts it chose: ({block: (steps, 4096) bf16 on the host}, {MoE block: (steps, K) the experts chosen}, the token ids of each decode: prompt and generated)."""
    cap, gens, chose = [[] for _ in run.layers], [], {b: [] for b in run.moe}

    def keep(b):
        def hook(mod, args):
            if args[0].shape[1] == 1:  # (the first call of a generate is the prompt)
                cap[b].append(args[0][0, 0].clone())
        return hook

    def keep_route(b):
        def hook(mod, args, out):
            if out[2].shape[0] == 1:
                chose[b].append(out[2][0].clone())
        return hook
    hooks = [l.register_forward_pre_hook(keep(b)) for b, l in enumerate(run.layers)] + [run.block(b).gate.register_forward_hook(keep_route(b)) for b in run.moe]
    pad = run.m.config.pad_token_id if run.m.config.pad_token_id is not None else run.m.config.eos_token_id
    try:
        for p in prompts:
            ids = torch.tensor(p)[None].to(run.dev0)
            gens.append(run.m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=new, do_sample=False, pad_token_id=pad)[0].tolist())
    finally:
        for h in hooks:
            h.remove()
    return {b: torch.stack(c).cpu() for b, c in enumerate(cap)}, {b: torch.stack(c).cpu() for b, c in chose.items()}, gens


def decode_stats(run, streams, chose, prof, store, chunk=16):
    """layer_stats on the decode steps (all steps one sequence, no calibration of their own: the profile of the prefill's calibration), each MoE layer's arrays into store. Returns, per MoE layer, the
    share of steps whose experts, chosen again by the router from the kept input, are the ones it chose while decoding (chose)."""
    agree = []
    for j, li in enumerate(run.moe):
        t0 = time.time()
        with gq.on(run.dev(li)):
            X = run.layers[li].norm(streams[li].to(run.dev(li)))
            idx = run.block(li).gate(X)[2]
            agree.append(float((idx.sort(1).values == chose[li].to(idx.device).sort(1).values).all(1).float().mean()))
            arrays, _ = layer_stats(run, li, X, X.shape[0], 0, streams, prof=prof[li], chunk=chunk)
        store.put(j, arrays)
        gq.free_all()
        gq.log(f"decode layer {li}: {time.time() - t0:.1f} s, routing as decoded for {agree[-1]:.4f} of the steps")
    return agree


def atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def run_probe(model, a, prompts):
    """The job, given a loaded model; a: data (prep's .npy files), out, cal_seqs, held_seqs, decode_new, chunk, deadline; prompts: token ids of the decode prompts. Writes meta.json, prefill.npz and
    decode.npz (after each MoE layer) and probe.md / probe.json (after each phase)."""
    t_start = time.time()
    run = nq.NemoRun(model)
    held, cal = gq.load_data(a.data, a.cal_seqs, a.held_seqs)
    doms = list(gs.DOMAINS)
    ids = np.concatenate([cal[d] for d in doms] + [held[d] for d in doms])
    ncal, S, L = sum(len(cal[d]) for d in doms), ids.shape[1], len(run.moe)
    mcfg, e0 = gs.MODELS["nemotron"], run.block(run.moe[0]).experts
    I, lat = e0.down_proj.shape[2], e0.up_proj.shape[2]
    meta = {"repo": mcfg["repo"], "rev": mcfg["rev"], "seq": S, "domains": doms, "calibration_sequences": ncal, "held_sequences": len(ids) - ncal, "held_tokens": (len(ids) - ncal) * S,
            "moe": run.moe, "blocks": len(run.layers), "experts": run.E, "top_k": run.K, "inter": I, "latent": lat, "hidden": run.width, "tols": TOLS, "budgets": BUDGETS,
            "fetch": fetch_counts(run.K), "dmax": DMAX, "bins": BINS, "ex": EX, "groups": GROUPS, "expert_bytes_4bit": 2 * lat * I // 2, "phases": {}, "errors": []}
    os.makedirs(a.out, exist_ok=True)
    atomic_json(f"{a.out}/meta.json", meta)
    left = lambda: a.deadline - time.time() if a.deadline else 1e9
    check = {"layers": [], "entries": 0, "equal": 0, "max_dk": 0}
    store, prof = Store(f"{a.out}/prefill.npz", L), None

    def tick(j, arrays, secs):
        store.put(j, arrays)
        n, kept = arrays["ex"][EX.index("n")].sum(), [arrays["ex"][EX.index(f"k{i}")].sum() / arrays["ex"][EX.index("n")].sum() / I for i in range(len(TOLS))]
        gq.log(f"layer {run.moe[j]}: {secs:.1f} s, {int(n)} pairs, zero share {arrays['ex'][EX.index('z')].sum() / n / I:.3f}, columns kept {kept[0]:.3f} / {kept[1]:.3f}")
    t0 = time.time()
    try:
        hs, prof = prefill(run, ids, ncal, deadline=a.deadline, chunk=a.chunk or 16, tick=tick, check=check)
        meta["walk_check"] = forward_check(run, hs, ids)
        gq.log(f"layer-by-layer pass vs the model's own forward on 2 sequences: KL {meta['walk_check']['kl']:.2e}, top-1 agreement {meta['walk_check']['top1']:.4f}")
        del hs
    except gq.Deadline as ex:
        meta["errors"].append(f"prefill: {ex}")
    except Exception as ex:  # (the arrays of the layers done stay; the decodes may still run)
        meta["errors"].append(f"prefill: {type(ex).__name__}: {str(ex)[:300]}")
        gq.log(meta["errors"][-1])
    meta["phases"]["prefill"] = {"seconds": round(time.time() - t0, 1), "layers": store.done}
    meta["bisect_check"] = check

    def finish():
        atomic_json(f"{a.out}/meta.json", meta)
        try:
            render(a.out)
        except Exception as ex:  # (the arrays and meta.json are written: the report step renders them again)
            gq.log(f"report: {type(ex).__name__}: {str(ex)[:300]}")
    finish()
    if prof is None or store.done < L:
        return
    if left() < DECODE_MIN * 60 + 120:
        meta["errors"].append(f"decode: skipped, {left() / 60:.0f} min left, {DECODE_MIN} expected")
        return finish()
    t0 = time.time()
    try:
        streams, chose, _ = decode_streams(run, prompts, a.decode_new)
        steps = len(streams[0])
        meta["decode"] = {"prompts": len(prompts), "new_tokens": a.decode_new, "steps": steps}
        gq.log(f"decode: {steps} steps of {len(prompts)} prompts in {time.time() - t0:.0f} s")
        agree = decode_stats(run, streams, chose, prof, Store(f"{a.out}/decode.npz", L), chunk=a.chunk or 16)
        meta["decode"]["routing_agreement"] = {"mean": float(np.mean(agree)), "min": float(np.min(agree))}
    except Exception as ex:
        meta["errors"].append(f"decode: {type(ex).__name__}: {str(ex)[:300]}")
        gq.log(meta["errors"][-1])
    meta["phases"]["decode"] = {"seconds": round(time.time() - t0, 1)}
    finish()
    gq.log(f"done in {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------------------------------------------------- report
def quantile_bin(h, p):
    """The first bin of the histogram h with at least the share p of its mass at or below it."""
    c = np.cumsum(h)
    return int(np.searchsorted(c, p * c[-1] * (1 - 1e-9)))  # (a hair under p * n: 0.1 * 30 is 3.0000000000000004 in floats)


def dist(h, scale, edge=0, mean=None):
    """n, mean, p10, p50 and p90 of the values bin / scale of a histogram (a ratio's bin counts as its upper edge, edge = 1); mean: the exact one where the sums have it."""
    h = np.asarray(h, np.float64)
    n = h.sum()
    if n == 0:
        return dict(n=0, mean=None, p10=None, p50=None, p90=None)
    return dict(n=int(n), mean=float((h * np.arange(len(h))).sum() / n / scale if mean is None else mean), **{f"p{q}": float((quantile_bin(h, q / 100) + edge) / scale) for q in (10, 50, 90)})


def ranks(x):
    """Ranks of x from 0, equal values sharing the mean of their ranks."""
    u, inv, cnt = np.unique(np.asarray(x, np.float64), return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - cnt + (cnt - 1) / 2)[inv]


def spearman(x, y):
    """Rank correlation of two vectors (None where one is constant)."""
    a, b = ranks(x) - (len(x) - 1) / 2, ranks(y) - (len(y) - 1) / 2
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else None


def groups(cal, E):
    """The experts of a layer as indices: the most used 13% by calibration counts, the least used 50%, the rest (glm_quality.py's mix13 and cold50 sets)."""
    order = np.argsort(-cal, kind="stable")
    k_hot, k_cold = max(1, round(GROUPS[0] * E / 100)), round(GROUPS[1] * E / 100)
    return {"hot": order[:k_hot], "mid": order[k_hot:E - k_cold], "cold": order[E - k_cold:]}


def la_cell(hit_h, w_sum, w_h, tokens, K):
    """One cell of the look-ahead tables: the share of the K experts the model chose that the router named (mean and quantiles over tokens, from the histogram of the number named right) and the mean
    share of their routing weight (with its quantiles where w_h, the histogram of it, is given)."""
    r = dict(tokens=int(tokens), recall=dist(hit_h, K, 0, (np.arange(K + 1) * hit_h).sum() / max(hit_h.sum(), 1) / K), weighted_mean=float(w_sum / max(tokens, 1)))
    if w_h is not None:
        r["weighted"] = dist(w_h, BINS, 1, r["weighted_mean"])
    return r


def summarize(z, meta, L=None, cal=None):
    """The numbers of one set of arrays z (prefill.npz or decode.npz, loaded) over its first L MoE layers: sparsity, the active-set predictor, the look-ahead and the importance, overall and, where
    the report has a table by layer, by layer. Pairs and tokens count the same in every layer, so a mean over layers is a mean over pairs. cal: the calibration pairs of each expert of each layer
    (z's own where None; the decode steps have none, they are ranked by the prefill's)."""
    L = int(z["done"]) if L is None else L
    I, E, K, ms, moe, bins = meta["inter"], meta["experts"], meta["top_k"], meta["fetch"], meta["moe"], meta["bins"]
    n, cal = z["ex"][:L, EX.index("n")], (z["cal"] if cal is None else cal)[:L]  # (layers, experts): held-out pairs and calibration pairs of each expert
    pairs = float(n.sum())
    out = {"layers": L, "pairs": int(pairs), "pairs_per_layer": int(pairs / max(L, 1))}
    if not L:
        return out
    sp = {"zero": dist(z["hz"][:L].sum(0), I), "zero_by_layer": [dist(z["hz"][l], I) for l in range(L)]}
    for t in range(len(meta["tols"])):
        sp[f"kept{t}"], sp[f"kept{t}_by_layer"] = dist(z["hk"][:L, t].sum(0), I), [dist(z["hk"][l, t], I) for l in range(L)]
        f, b = sp[f"kept{t}"]["mean"], meta["expert_bytes_4bit"]
        sp[f"bytes{t}"] = {"down_read": f * b / 2, "down_saved": (1 - f) * b / 2, "down_saved_share_of_expert": (1 - f) / 2, "both_read_if_named": f * b, "both_saved_if_named": (1 - f) * b}
    out["sparsity"] = sp
    out["predictor"] = [dict(budget=b, neurons=round(b * I), recall=dist(z["pr_h"][:L, i].sum(0), bins, 1, z["pr_s"][:L, i, 0].sum() / pairs), err_pred=float(z["pr_s"][:L, i, 1].sum() / pairs),
                             within_pred=float(z["pr_s"][:L, i, 3].sum() / pairs), err_orc=float(z["pr_s"][:L, i, 2].sum() / pairs), within_orc=float(z["pr_s"][:L, i, 4].sum() / pairs))
                        for i, b in enumerate(meta["budgets"])]
    top = lambda l, m: np.argsort(-cal[l], kind="stable")[:m]  # the m most used experts of layer l by calibration counts
    la = {"fetch": ms, "d0_all_named": float(z["la_h"][:L, 0, 0, K].sum() / max(z["la_n"][:L, 0].sum(), 1)), "random": [m / E for m in ms], "popular": [float(np.mean([n[l][top(l, m)].sum() / n[l].sum() for l in range(L)])) for m in ms]}
    la["by_d"] = [[la_cell(z["la_h"][:L, d, i].sum(0), z["la_w"][:L, d, i].sum(), z["la_wh"][:L, d].sum(0) if i == 0 else None, z["la_n"][:L, d].sum(), K) for i in range(len(ms))]
                  for d in range(meta["dmax"] + 1)]
    la["by_layer"] = []
    for l in range(L):
        dl = moe[l] - moe[l - 1] if l else None  # blocks from the previous MoE layer: the stream at its input, and the one two blocks before, are the look-ahead of a prefetch
        row = {"block": moe[l], "from_block": moe[l - 1] if l else None, "d_prev": dl, "popular": float(n[l][top(l, ms[0])].sum() / n[l].sum())}
        for name, d in (("at_prev", dl), ("two_earlier", None if dl is None else dl + 2)):
            ok = d is not None and d <= meta["dmax"] and moe[l] - d >= 0
            row[name] = dict(d=d, **la_cell(z["la_h"][l, d, 0], z["la_w"][l, d, 0], z["la_wh"][l, d], z["la_n"][l, d], K)) if ok else None
        la["by_layer"].append(row)
    for name in ("at_prev", "two_earlier"):  # (all the layers that have the stream, each at its own distance)
        sel = [(l, r[name]["d"]) for l, r in enumerate(la["by_layer"]) if r[name]]
        la[name] = la_cell(sum(z["la_h"][l, d, 0] for l, d in sel), sum(z["la_w"][l, d, 0] for l, d in sel), sum(z["la_wh"][l, d] for l, d in sel), sum(z["la_n"][l, d] for l, d in sel), K) if sel else None
    out["lookahead"] = la
    IMP = ("w", "y", "wy", "c")  # per-expert sums that become means over the pairs it got: routing weight, output norm, weight x output norm, contribution norm
    by_layer, pool, dec = [], {}, {}

    def accum(into, g, l, ix):  # (the experts ix of layer l into the sums of group g)
        p = into.setdefault(g, {})
        for k, v in (("n", n[l][ix]), ("cal", cal[l][ix]), ("share", z["ssh"][l][ix]), *[(k, z["ex"][l, EX.index(k)][ix]) for k in IMP]):
            p[k] = p.get(k, 0.0) + float(v.sum())
        p["experts"] = p.get("experts", 0) + len(ix)
    for l in range(L):
        use, mean = n[l] > 0, lambda name: z["ex"][l, EX.index(name)] / np.maximum(n[l], 1)
        gr = groups(cal[l], E)
        row = {"block": moe[l], **{f"spearman_{k}": spearman(cal[l][use], mean(k)[use]) for k in IMP}}
        for g, ix in gr.items():
            accum(pool, g, l, ix)
        for i, ix in enumerate(np.array_split(np.argsort(-cal[l], kind="stable"), 10)):
            accum(dec, i, l, ix)
        for k in ("wy", "c"):
            m_ = lambda g: z["ex"][l, EX.index(k)][gr[g]].sum() / max(n[l][gr[g]].sum(), 1)
            row[f"cold_over_hot_{k}"] = float(m_("cold") / m_("hot")) if m_("hot") else None
        by_layer.append(row)
    tot = lambda into, k: sum(v[k] for v in into.values())
    cell = lambda v, into: dict(experts_share=v["experts"] / (L * E), calibration_pairs_share=v["cal"] / max(tot(into, "cal"), 1), pairs_share=v["n"] / max(tot(into, "n"), 1),
                                **{f"mean_{k}": v[k] / max(v["n"], 1) for k in IMP}, mean_share_of_token=v["share"] / max(v["n"], 1), contribution_share=v["c"] / max(tot(into, "c"), 1))
    out["importance"] = {"by_layer": by_layer, "groups": {g: cell(v, pool) for g, v in pool.items()}, "deciles": [cell(dec[i], dec) for i in range(10)]}
    for k in [f"spearman_{x}" for x in IMP] + ["cold_over_hot_wy", "cold_over_hot_c"]:
        v = [r[k] for r in by_layer if r[k] is not None]
        out["importance"][k + "_mean"] = float(np.mean(v)) if v else None
    return out


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def markdown(meta, res):
    """probe.md: how to read it, a summary, then for the held-out text and for the decode steps the tables of every measurement, and the checks."""
    P = lambda x: "-" if x is None else f"{100 * x:.1f}"
    P2 = lambda x: "-" if x is None else f"{100 * x:.2f}"
    F2 = lambda x: "-" if x is None else f"{x:.2f}"
    F4 = lambda x: "-" if x is None else f"{x:.4f}"
    Q = lambda d, f=P: " | ".join(f(d[k]) for k in ("mean", "p10", "p50", "p90"))
    I, K, tols, ms, B4 = meta["inter"], meta["top_k"], meta["tols"], meta["fetch"], meta["expert_bytes_4bit"]
    pre = res.get("prefill", {})
    o = [f"## Expert-offload probe: {meta['repo']} @ {meta['rev'][:8]}", "",
         f"The bf16 model as released, nothing rounded. Held-out text: {meta['held_sequences']} sequences of {meta['seq']} tokens ({meta['held_tokens']:,} tokens, {len(meta['domains'])} domains) through {meta['blocks']} blocks; "
         f"{pre.get('pairs_per_layer', 0):,} routed (token, expert) pairs in each of the {pre.get('layers', 0)} MoE layers measured ({len(meta['moe'])} in the model). The neuron profile of each expert and the routing counts that rank the "
         f"experts come from {meta['calibration_sequences']} calibration sequences of the same domains, taken in the same pass. "
         + (f"Decode: {meta['decode']['steps']} one-token steps of {meta['decode']['prompts']} greedy decodes of {meta['decode']['new_tokens']} new tokens (thinking off)." if "decode" in meta else "No decode steps."), "",
         "How to read it:",
         f"- **Pair.** One token routed to one expert of one MoE layer ({K} a token a layer). Sparsity and predictor numbers are over pairs, look-ahead numbers over tokens. p10, p50, p90 are percentiles: exact for counts (neurons, columns, experts named), the upper edge of a bin 0.5% wide for a ratio (the predictor's recall, the weighted look-ahead).",
         f"- **Zero.** Neurons of the expert's {I:,} that are exactly 0 after relu2 for that pair, in the bf16 the model computes. A zero neuron needs no column of down.",
         f"- **Kept at {tols[0]:.0%} / {tols[1]:.1%}.** Columns of down kept, the largest activation first, for the expert's output to be within that relative L2 error of the whole one (float32 sums); zero neurons are not counted. The activations are "
         "known when it is counted: it is what the best choice of columns needs, not what a rule that finds them without computing up first keeps. 0.1% is the size of bf16's own rounding of an output (0.11% rms).",
         f"- **Bytes.** At 4 bits a weight an expert is {B4:,} B, down {B4 // 2:,} B. Down read = the mean kept share of down; 'both' = the same neurons' rows of up too, which needs the active set before up is read.",
         "- **Predicted.** The B most frequent active neurons of the expert in the calibration text. Recall = share of the pair's active neurons among them; error = relative L2 error of the output with only those B neurons "
         f"computed; oracle = the same for the pair's B largest activations (no predictor does better); within = share of pairs with error at most {tols[0]:.0%}.",
         "- **Look-ahead.** The router of MoE layer N (its pre-norm, gate, correction bias and top-K choice: the model's own modules) on the stream at the input of block N - d; d = 0 is the model's own choice and must name "
         "every expert. 'At previous' = the stream at the input of the previous MoE layer, 'two earlier' = the input of the block two before that. Recall = share of the K experts the model chose that it names, weighted = share "
         f"of their routing weight; x1.5, x2, x3 = the first {ms[1]}, {ms[2]}, {ms[3]} experts of its ranking. Random = m / {meta['experts']}; popular = the m most used experts of the layer in the calibration text, one fixed set.",
         f"- **Importance.** Weight = the routing weight (sigmoid scores renormalised over the {K}, times the routed scale). Output = L2 norm of the expert's {meta['latent']:,}-wide output before the weight. Weight x output = the mean "
         f"over the expert's pairs of the two. Contribution = L2 norm of what it adds to the block's {meta['hidden']:,}-wide output, fc2_latent_proj(weight x output). Share = contribution over the sum of the token's {K} contributions. "
         "Hot = a layer's 13% most used experts by calibration counts, cold = its 50% least used (the quality job's mix13 and cold50 sets). Spearman = rank correlation, across a layer's experts, of calibration frequency with the "
         "mean over the pairs the expert got.", ""]
    if pre.get("layers"):
        sp, pr, la, im = pre["sparsity"], pre["predictor"], pre["lookahead"], pre["importance"]
        cell = lambda c: "-" if not c else f"{P(c['recall']['mean'])}% (p10 {P(c['recall']['p10'])}%, weighted {P(c['weighted_mean'])}%)"
        o += ["### Summary, held-out text", "",
              f"- Zero after relu2: {P(sp['zero']['mean'])}% of an expert's neurons on average (p10 {P(sp['zero']['p10'])}%, p90 {P(sp['zero']['p90'])}%).",
              *[f"- Columns of down kept for {t:.1%} error: {P(sp[f'kept{i}']['mean'])}% on average (p10 {P(sp[f'kept{i}']['p10'])}%, p90 {P(sp[f'kept{i}']['p90'])}%); at 4 bits that reads {sp[f'bytes{i}']['down_read']:,.0f} B of an "
                f"expert's {B4 // 2:,} B of down and saves {P(sp[f'bytes{i}']['down_saved_share_of_expert'])}% of its {B4:,} B, or {sp[f'bytes{i}']['both_saved_if_named']:,.0f} B if the rows of up of the same neurons could be skipped too." for i, t in enumerate(tols)],
              "- Naming the active neurons from the calibration profile: " + "; ".join(f"{x['neurons']} neurons ({100 * x['neurons'] / I:g}%): recall {P(x['recall']['mean'])}%, output error {P2(x['err_pred'])}% (oracle {P2(x['err_orc'])}%)" for x in pr) + ".",
              f"- Look-ahead, recall of the {K} experts chosen: at the previous MoE layer's input {cell(la['at_prev'])}; two blocks earlier {cell(la['two_earlier'])}; the router at d = 0 names {P(la['d0_all_named'])}% of tokens whole.",
              f"- Importance: a cold expert's mean weight x output is {F2(im['cold_over_hot_wy_mean'])} of a hot one's, its mean contribution {F2(im['cold_over_hot_c_mean'])}; Spearman of calibration frequency with the mean weight x output "
              f"{F2(im['spearman_wy_mean'])}, with the mean contribution {F2(im['spearman_c_mean'])}.", ""]
    for key, title in (("prefill", "Held-out text"), ("decode", "Decode steps")):
        r = res.get(key)
        if not r or not r["layers"]:
            continue
        sp, la, im = r["sparsity"], r["lookahead"], r["importance"]
        o += [f"### {title}: relu2 sparsity inside the experts ({r['pairs']:,} pairs, {r['layers']} layers)", ""]
        o += table(["share of the expert's neurons, %", "mean", "p10", "p50", "p90"], [["zero (exactly 0 after relu2)", Q(sp["zero"])]] + [[f"columns of down kept for {t:.1%} error", Q(sp[f"kept{i}"])] for i, t in enumerate(tols)])
        o += ["", *table(["bytes of an expert read at 4 bits (mean over pairs)", *[f"kept for {t:.1%}" for t in tols]],
                         [["down, columns kept (B)", *[f"{sp[f'bytes{i}']['down_read']:,.0f}" for i in range(len(tols))]], ["down, saved (B)", *[f"{sp[f'bytes{i}']['down_saved']:,.0f}" for i in range(len(tols))]],
                          ["saved, share of the expert's bytes (%)", *[P(sp[f"bytes{i}"]["down_saved_share_of_expert"]) for i in range(len(tols))]],
                          ["up and down, if the neurons were named first, saved (B)", *[f"{sp[f'bytes{i}']['both_saved_if_named']:,.0f}" for i in range(len(tols))]]])]
        if key == "prefill":
            o += ["", "By layer (%):", "", *table(["MoE layer (block)", *[f"{n} {k}" for n in ("zero", f"kept {tols[0]:.0%}", f"kept {tols[1]:.1%}") for k in ("mean", "p10", "p50", "p90")]],
                                                 [[f"{i} ({b})", Q(sp["zero_by_layer"][i]), Q(sp["kept0_by_layer"][i]), Q(sp["kept1_by_layer"][i])] for i, b in enumerate(meta["moe"][:r["layers"]])])]
        o += ["", f"#### {title}: naming the active neurons before up is computed", ""]
        o += table(["neurons named", "of the expert's", "recall mean, %", "recall p10, %", "error mean, %", f"within {tols[0]:.0%}, %", "oracle error mean, %", f"oracle within {tols[0]:.0%}, %"],
                   [[x["neurons"], f"{100 * x['neurons'] / I:g}", P(x["recall"]["mean"]), P(x["recall"]["p10"]), P2(x["err_pred"]), P(x["within_pred"]), P2(x["err_orc"]), P(x["within_orc"])] for x in r["predictor"]])
        o += ["", f"#### {title}: look-ahead routing", ""]
        pooled = lambda name, c: [name, c["tokens"], P(c["recall"]["mean"]), "", "", "", P(c["recall"]["p10"]), P(c["weighted_mean"]), P(c["weighted"]["p10"])] if c else None
        o += table(["d (blocks)", "tokens", *[f"recall x{x}, mean %" for x in ("1", "1.5", "2", "3")], "recall x1, p10 %", "weighted x1, mean %", "weighted x1, p10 %"],
                   [[d, row[0]["tokens"], *[P(c["recall"]["mean"]) for c in row], P(row[0]["recall"]["p10"]), P(row[0]["weighted_mean"]), P(row[0].get("weighted", {}).get("p10"))] for d, row in enumerate(la["by_d"]) if row[0]["tokens"]]
                   + [x for x in (pooled(f"at previous MoE layer (d = {', '.join(sorted({str(b['at_prev']['d']) for b in la['by_layer'] if b['at_prev']}))})", la["at_prev"]),
                                  pooled(f"two blocks earlier (d = {', '.join(sorted({str(b['two_earlier']['d']) for b in la['by_layer'] if b['two_earlier']}))})", la["two_earlier"])) if x]
                   + [["random", "", *[P(x) for x in la["random"]], "", "", ""], ["popular", "", *[P(x) for x in la["popular"]], "", "", ""]])
        if key == "prefill":
            cell = lambda c: [c["d"], P(c["recall"]["mean"]), P(c["recall"]["p10"]), P(c["weighted_mean"])] if c else ["-"] * 4
            o += ["", f"By layer, for the {K} experts chosen (recall and weighted recall, %; L = the previous MoE layer, d = blocks back from the predicted layer N):", "",
                  *table(["predicted MoE layer N (block)", "L (block)", "d", "at previous: recall mean", "p10", "weighted", "d", "two earlier: recall mean", "p10", "weighted", f"popular {ms[0]}"],
                         [[f"{i} ({b['block']})", b["from_block"] if b["from_block"] is not None else "-", *cell(b["at_prev"]), *cell(b["two_earlier"]), P(b["popular"])] for i, b in enumerate(la["by_layer"])])]
        g = im["groups"]
        imp_head = ["experts", "of all experts, %", "of calibration pairs, %", "of these pairs, %", "mean weight", "mean output norm", "mean weight x output", "mean contribution",
                    "mean share of the token's, %", "of all contribution, %"]
        imp_row = lambda n, x: [n, P(x["experts_share"]), P(x["calibration_pairs_share"]), P(x["pairs_share"]), F4(x["mean_w"]), F4(x["mean_y"]), F4(x["mean_wy"]), F4(x["mean_c"]),
                                P(x["mean_share_of_token"]), P(x["contribution_share"])]
        o += ["", f"#### {title}: per-expert importance", ""]
        o += table(imp_head, [imp_row(n, g[n]) for n in ("hot", "mid", "cold")])
        o += ["", f"Mean over layers: Spearman of calibration frequency with the mean weight {F2(im['spearman_w_mean'])}, with the mean output norm {F2(im['spearman_y_mean'])}, with the mean weight x output "
                  f"{F2(im['spearman_wy_mean'])}, with the mean contribution {F2(im['spearman_c_mean'])}; a cold expert's mean weight x output over a hot one's {F2(im['cold_over_hot_wy_mean'])}, its mean contribution {F2(im['cold_over_hot_c_mean'])}.", ""]
        if key == "prefill":
            o += ["Experts of a layer by calibration frequency in tenths, the most used first (pooled over the layers):", "",
                  *table(["tenth", *imp_head[1:]], [imp_row(i + 1, x) for i, x in enumerate(im["deciles"])]), "",
                  *table(["MoE layer (block)", "Spearman weight", "output", "weight x output", "contribution", "cold / hot weight x output", "cold / hot contribution"],
                         [[f"{i} ({b['block']})", F2(b["spearman_w"]), F2(b["spearman_y"]), F2(b["spearman_wy"]), F2(b["spearman_c"]), F2(b["cold_over_hot_wy"]), F2(b["cold_over_hot_c"])] for i, b in enumerate(im["by_layer"])]), ""]
    chk = []
    if meta.get("phases"):
        chk.append("wall clock: " + ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in meta["phases"].items()))
    if "walk_check" in meta:
        w = meta["walk_check"]
        chk.append(f"the layer-by-layer pass against the model's own forward on 2 sequences: KL {w['kl']:.1e}, top-1 agreement {w['top1']:.4f}" + ("" if w["kl"] < 1e-3 and w["top1"] > 0.995 else " **(NOT the model's forward: read nothing above)**"))
    if meta.get("bisect_check", {}).get("entries"):
        b = meta["bisect_check"]
        chk.append(f"the bisection against scanning every count, on {b['entries']} counts of the first pairs of one expert in each of {len(b['layers'])} layers: {b['equal']} equal, largest difference {b['max_dk']} columns")
    for key in ("prefill", "decode"):
        if res.get(key, {}).get("layers"):
            chk.append(f"{key}: the look-ahead router at d = 0 (the same router on the same stream) names all {K} experts the model chose for {P(res[key]['lookahead']['d0_all_named'])}% of tokens")
    if "routing_agreement" in meta.get("decode", {}):
        g = meta["decode"]["routing_agreement"]
        chk.append(f"decode: the experts the router chooses again from the kept inputs are the ones it chose while decoding for {P(g['mean'])}% of the steps of a layer on average ({P(g['min'])}% in the worst layer)")
    chk += [f"**error:** {e}" for e in meta.get("errors", [])]
    if chk:
        o += ["### Checks", "", *[f"- {c}" for c in chk], ""]
    return "\n".join(o) + "\n"


def render(out):
    """meta.json + prefill.npz + decode.npz -> probe.md and probe.json, each whole or not at all (the launcher pulls the results every minute)."""
    meta = json.load(open(f"{out}/meta.json"))
    zs = {k: dict(np.load(f"{out}/{k}.npz")) for k in ("prefill", "decode") if os.path.exists(f"{out}/{k}.npz")}
    res = {k: summarize(z, meta, cal=zs["prefill"]["cal"] if k == "decode" and "prefill" in zs else None) for k, z in zs.items()}
    for name, text in (("probe.json", json.dumps({"meta": meta, **res}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))), ("probe.md", markdown(meta, res))):
        open(f"{out}/{name}.tmp", "w").write(text)
        os.replace(f"{out}/{name}.tmp", f"{out}/{name}")


def cmd_report(a):
    render(a.dir)
    print(open(f"{a.dir}/probe.md").read())


# ------------------------------------------------------------------------------------------------------------------------------------------- the job
def cmd_run(a):
    import transformers
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model = gq.load_model(a.weights, "nemotron")
    n = torch.cuda.device_count()
    gq.log(f"loaded in {time.time() - t0:.0f} s on {n} GPUs, GiB per GPU {[round(torch.cuda.memory_allocated(i) / 2 ** 30, 1) for i in range(n)]}")
    tok, m = transformers.AutoTokenizer.from_pretrained(a.weights), gs.MODELS["nemotron"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **m["tmpl"]) + m["suffix"] for _, p in gs.PROMPTS[:a.decode_prompts]]
    with torch.inference_mode():
        run_probe(model, a, [tok(t, add_special_tokens=False)["input_ids"] for t in prompts])


# -------------------------------------------------------------------------------------------------------------------------------------- selftest
def probe_model():
    """nemotron_quality's tiny random NemotronH with structure whose answers are known: the experts of block 1 have up[96:] = -up[:96], so exactly 96 of the 192 neurons are zero for every pair; blocks 5
    and 6 add nothing to the stream (block 5's routed output and shared expert and block 6's Mamba-2 out_proj are zero), so the stream at the input of blocks 5 and 6 is the one block 7 reads."""
    m = nq.tiny_model(seed=1)
    with torch.no_grad():
        up = m.model.layers[1].mixer.experts.up_proj
        up[:, 96:] = -up[:, :96]
        m.model.layers[5].mixer.fc2_latent_proj.weight.zero_()
        m.model.layers[5].mixer.shared_experts.down_proj.weight.zero_()
        m.model.layers[6].mixer.out_proj.weight.zero_()
    return m


def check_hand(dev):
    """Cases worked out on paper, on dev: zeros, the columns kept (orthonormal columns, so the error of dropping some is the root of the dropped squares over the total, and columns that cancel), the
    predictor's recall and errors, the look-ahead hits, the ranks, quantiles and correlations."""
    d, f = torch.device(dev), lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=d)
    rows = f([[4, 2, 1, .5, 0, 0, 0, 0], [0] * 8, [1] * 8, [0, 3, 0, 4, 0, 0, 0, 0], [4, 0, 0, 0, 0, 2, 0, 1]])
    prank = order_rank(f([[9, 8, 1, 0, 0, 0, 0, 0]]))[0]  # (neurons in the order of the calibration counts, ties by position: 0 1 2 3 ...)
    assert order_rank(f([[3, 1, 3, 0]])).tolist() == [[0, 2, 1, 3]] and prank.tolist() == list(range(8))
    st = pair_stats(rows, torch.eye(8, device=d), prank, tols=(0.51, 0.26, 0.11, 0.1, 0.0), budgets=(0.25, 0.75, 1.0))
    assert st["zeros"].tolist() == [4, 8, 0, 6, 5] and torch.allclose(st["ynorm"].cpu(), torch.tensor([21.25, 0, 8, 25, 21]).sqrt()) and torch.allclose(st["y"], rows)
    # errors of keeping the k largest: row 0 (4 2 1 .5) 1, .497, .243, .108, 0; row 2 sqrt((8 - k) / 8); row 3 (4 3) 1, .6, 0; row 4 (4 2 1) 1, .488, .218, 0
    assert torch.stack(st["k"], 1).tolist() == [[1, 2, 3, 4, 4], [0] * 5, [6, 8, 8, 8, 8], [2] * 5, [1, 2, 3, 3, 3]], torch.stack(st["k"], 1)
    r5 = np.sqrt(5 / 21)
    # budgets 2, 6, 8 neurons of the profile's order 0 1 2 ...: row 0 active {0,1,2,3}, row 2 all, row 3 {1,3}, row 4 {0,5,7}
    want = dict(recall=[[.5, 1, 1], [1, 1, 1], [.25, .75, 1], [.5, 1, 1], [1 / 3, 2 / 3, 1]],
                e_pred=[[np.sqrt(1.25 / 21.25), 0, 0], [0, 0, 0], [np.sqrt(.75), .5, 0], [.8, 0, 0], [r5, np.sqrt(1 / 21), 0]],
                e_orc=[[np.sqrt(1.25 / 21.25), 0, 0], [0, 0, 0], [np.sqrt(.75), .5, 0], [0, 0, 0], [np.sqrt(1 / 21), 0, 0]])
    for k, v in want.items():
        assert np.allclose(torch.stack(st[k], 1).cpu().numpy(), np.array(v), atol=1e-6), (k, torch.stack(st[k], 1))
    # columns that cancel: y = 3 d0 + 2 d1 = (1, 0) with d1 = -d0, so keeping only the larger column makes the error 2, and a bisection that assumed less still ends on the crossing
    dn = f([[1, -1, 0, 0], [0, 0, 1, 1]])
    st = pair_stats(f([[3, 2, 0, 0], [3, 2, 1, 0]]), dn, tols=(0.5, 0.8))
    assert torch.stack(st["k"], 1).tolist() == [[2, 2], [3, 2]] and first_crossing(f([[3, 2, 0, 0], [3, 2, 1, 0]]), dn, (0.5, 0.8))[0].tolist() == [2, 3] and not any(k in st for k in ("recall", "e_pred"))
    # the look-ahead: the model chose 0 1 2 (weights .5 .3 .2); the router's own top-3 is 2 1 5 and its ranking 2 1 5 0 7 3
    hit = hits_at(f([[2, 1, 5], [3, 6, 0]], torch.int64), f([[2, 1, 5, 0, 7, 3], [3, 6, 0, 4, 1, 2]], torch.int64), f([[0, 1, 2], [3, 4, 6]], torch.int64), (3, 4, 6), 8)
    assert [h.tolist() for h in hit] == [[[0, 1, 1], [1, 0, 1]], [[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]], hit
    assert abs(float((hit[0][0] * f([.5, .3, .2])).sum() / 1.0) - .5) < 1e-6 and rhist(f([0., .004, .006, .5, 1.])).tolist()[:2] + rhist(f([1.])).tolist()[-1:] == [2, 1, 1]
    # quantiles of a histogram, a rank correlation, the expert groups
    assert dist([0, 3, 0, 27], 1)["p10"] == 1 and dist([0, 3, 0, 27], 1)["p50"] == 3
    q = dist([1, 2, 3, 4, 0], 4)
    assert (q["mean"], q["p10"], q["p50"], q["p90"], q["n"]) == (.5, 0, .5, .75, 10) and dist([0, 0], 1)["mean"] is None and dist([0, 5], BINS, 1)["p10"] == 2 / BINS
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1 and spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1 and spearman([1, 2, 3], [5, 5, 5]) is None
    gr = groups(np.array([5, 9, 1, 7, 3, 8, 2, 6, 4, 0]), 10)
    assert gr["hot"].tolist() == [1] and sorted(gr["cold"].tolist()) == [2, 4, 6, 8, 9] and sorted(gr["mid"].tolist()) == [0, 3, 5, 7]


def check_device(dev):
    """pair_stats on dev against the CPU's on random bf16-valued activations (about 60% of them zero) and weights: the same zeros, recalls and (but for a pair on a boundary) columns kept; the same errors."""
    g = torch.Generator().manual_seed(3)
    act = (torch.relu(torch.randn(600, 160, generator=g) - 0.3) ** 2 * torch.rand(600, 1, generator=g) * 4).bfloat16().float()
    dn, prank = (torch.randn(48, 160, generator=g) * .1).bfloat16().float(), order_rank(torch.randint(0, 30, (1, 160), generator=g))[0]
    want = pair_stats(act, dn, prank)
    with gq.tf32():  # (the job's mode: bf16 values are exact in TF32, the sums float32)
        got = pair_stats(act.to(dev), dn.to(dev), prank.to(dev))
    assert torch.equal(want["zeros"], got["zeros"].cpu()) and all(torch.equal(a, b.cpu()) for a, b in zip(want["recall"], got["recall"]))
    assert all(float((a != b.cpu()).float().mean()) < .005 for a, b in zip(want["k"], got["k"])), "columns kept differ"
    assert all(torch.allclose(a, b.cpu(), rtol=1e-3, atol=1e-5) for k in ("e_pred", "e_orc") for a, b in zip(want[k], got[k])) and torch.allclose(want["y"], got["y"].cpu(), rtol=1e-3, atol=1e-4)
    # keep_needed against scanning every count, with columns that are not orthogonal: the count is a crossing for every row and the least one for all but the rare row
    sc = first_crossing(act[:60], dn)
    assert all(float((a != b.cpu()[:60]).float().mean()) < .03 for a, b in zip(sc, got["k"])), "keep_needed is not first_crossing"


def check_report():
    """summarize and markdown on arrays made by hand: 2 MoE layers (blocks 1 and 3) of 8 experts with 8 neurons, top-2, 100 pairs a layer; every number below is worked out on paper."""
    L, E, K, I, ms = 2, 8, 2, 8, [2, 3, 4, 6]
    meta = dict(repo="r", rev="0" * 40, seq=10, domains=["a"], calibration_sequences=3, held_sequences=5, held_tokens=50, moe=[1, 3], blocks=4, experts=E, top_k=K, inter=I, latent=8, hidden=16, tols=TOLS,
                budgets=BUDGETS, fetch=ms, dmax=DMAX, bins=BINS, ex=EX, groups=GROUPS, expert_bytes_4bit=64, errors=[])
    z = dict(done=np.array(L), hz=np.zeros((L, I + 1), np.int64), hk=np.zeros((L, 2, I + 1), np.int64), ex=np.zeros((L, len(EX), E)), ssh=np.zeros((L, E)), cal=np.zeros((L, E), np.int64),
             pr_h=np.zeros((L, len(BUDGETS), BINS), np.int64), pr_s=np.zeros((L, len(BUDGETS), 5)), la_h=np.zeros((L, DMAX + 1, 4, K + 1), np.int64), la_w=np.zeros((L, DMAX + 1, 4)), la_wh=np.zeros((L, DMAX + 1, BINS), np.int64),
             la_n=np.zeros((L, DMAX + 1), np.int64))
    n, cal, c, wy = np.array([20, 20, 15, 15, 10, 10, 5, 5.]), np.array([50, 40, 30, 20, 10, 5, 3, 2]), np.array([1, 1, 1, 1, .5, .5, .5, .5]), np.array([1, 1, 1, 1, .8, .8, .8, .8])
    for l in range(L):
        z["hz"][l, 4], z["hk"][l, 0, 2], z["hk"][l, 1, 3], z["cal"][l], z["ssh"][l] = 100, 100, 100, cal, n * .5
        for name, v in (("n", n), ("w", n * .5), ("y", n * 2), ("wy", n * wy), ("c", n * c)):
            z["ex"][l, EX.index(name)] = v
        z["pr_h"][l, :, BINS - 1], z["pr_s"][l] = 100, [100, 10, 5, 80, 90]
        for d, (hist, wsum, wbin) in {0: ([0, 0, 50], 50, BINS - 1), 1: ([0, 25, 25], 40, 159), 2: ([10, 20, 20], 35, 139)}.items():
            if d < 2 or l == 1:  # (the stream two blocks before block 3 is the one at the input of block 1: d = 2 exists for the second layer only)
                z["la_h"][l, d], z["la_w"][l, d], z["la_wh"][l, d, wbin], z["la_n"][l, d] = hist, wsum, 50, 50
    r = summarize(z, meta)
    sp = r["sparsity"]
    assert r["pairs"] == 200 and r["pairs_per_layer"] == 100 and (sp["zero"]["mean"], sp["zero"]["p10"], sp["zero"]["p90"]) == (.5, .5, .5) and sp["kept0"]["mean"] == .25 and sp["kept1"]["mean"] == .375
    assert sp["bytes0"] == {"down_read": 8.0, "down_saved": 24.0, "down_saved_share_of_expert": .375, "both_read_if_named": 16.0, "both_saved_if_named": 48.0}, sp["bytes0"]
    pr = r["predictor"][0]
    assert pr["neurons"] == round(BUDGETS[0] * I) and pr["recall"]["mean"] == 1 and pr["recall"]["p10"] == 1 and np.allclose([pr["err_pred"], pr["within_pred"], pr["err_orc"], pr["within_orc"]], [.1, .8, .05, .9])
    la = r["lookahead"]
    assert la["d0_all_named"] == 1 and la["by_d"][0][0]["recall"]["mean"] == 1 and la["by_d"][1][0]["recall"]["mean"] == .75 and la["by_d"][1][0]["recall"]["p10"] == .5
    assert la["by_d"][1][0]["weighted_mean"] == .8 and la["by_d"][1][0]["weighted"]["p10"] == .8 and la["by_d"][2][0]["tokens"] == 50 and abs(la["by_d"][2][0]["recall"]["mean"] - .6) < 1e-12
    assert np.allclose(la["popular"], [.4, .55, .7, .9]) and la["random"] == [.25, .375, .5, .75] and la["by_layer"][0]["at_prev"] is None and la["by_layer"][0]["two_earlier"] is None
    assert la["by_layer"][1]["d_prev"] == 2 and la["by_layer"][1]["at_prev"]["d"] == 2 and abs(la["by_layer"][1]["at_prev"]["weighted_mean"] - .7) < 1e-12 and la["by_layer"][1]["two_earlier"] is None
    assert la["at_prev"]["tokens"] == 50 and abs(la["at_prev"]["recall"]["mean"] - .6) < 1e-12 and la["two_earlier"] is None
    g = r["importance"]["groups"]
    assert (g["hot"]["pairs_share"], g["mid"]["pairs_share"], g["cold"]["pairs_share"]) == (.2, .5, .3) and (g["hot"]["experts_share"], g["cold"]["experts_share"]) == (.125, .5)
    assert np.allclose([g[k]["calibration_pairs_share"] for k in ("hot", "mid", "cold")], [50 / 160, 90 / 160, 20 / 160]) and g["cold"]["mean_c"] == .5 and g["hot"]["mean_y"] == 2 and g["cold"]["mean_wy"] == .8
    assert np.allclose([g[k]["contribution_share"] for k in ("hot", "mid", "cold")], [20 / 85, 50 / 85, 15 / 85]) and g["mid"]["mean_share_of_token"] == .5 and g["cold"]["mean_w"] == .5
    im, dc = r["importance"], r["importance"]["deciles"]
    assert im["cold_over_hot_wy_mean"] == .8 and im["cold_over_hot_c_mean"] == .5 and im["spearman_w_mean"] is None and im["spearman_y_mean"] is None
    assert abs(im["spearman_c_mean"] - 32 / np.sqrt(42 * 32)) < 1e-9 and abs(im["spearman_wy_mean"] - 32 / np.sqrt(42 * 32)) < 1e-9
    assert len(dc) == 10 and dc[0]["pairs_share"] == .2 and dc[0]["mean_c"] == 1 and dc[7]["pairs_share"] == .05 and dc[7]["mean_c"] == .5 and dc[9]["pairs_share"] == 0 and dc[0]["experts_share"] == .125
    dec = summarize(z, meta, cal=cal[::-1][None].repeat(L, 0))  # (the decode steps ranked by another calibration: the least used experts are now the popular ones)
    assert np.allclose(dec["lookahead"]["popular"], [.1, .2, .3, .6]) and dec["importance"]["groups"]["hot"]["pairs_share"] == .05
    md = markdown(meta, {"prefill": r, "decode": dec})
    assert "| zero (exactly 0 after relu2) | 50.0 | 50.0 | 50.0 | 50.0 |" in md and "| 1 | 100 | 75.0 | 75.0 | 75.0 | 75.0 | 50.0 | 80.0 | 80.0 |" in md and "| cold | 50.0 | 12.5 | 30.0 |" in md
    assert "| 1 (3) | 1 | 2 | 60.0 |" in md and "names all 2 experts the model chose for 100.0% of tokens" in md and "| at previous MoE layer (d = 2) | 50 | 60.0 |" in md and "### Summary, held-out text" in md and "### Decode steps: relu2" in md
    assert "| 1 | 12.5 | 31.2 | 20.0 |" in md and "a cold expert's mean weight x output is 0.80 of a hot one's, its mean contribution 0.50" in md
    one = summarize(z, meta, 1)
    assert one["layers"] == 1 and one["pairs"] == 100 and markdown(meta, {"prefill": one}) and summarize({**z, "done": np.array(0)}, meta) == {"layers": 0, "pairs": 0, "pairs_per_layer": 0}
    assert "### Held-out text" not in markdown(meta, {"prefill": summarize({**z, "done": np.array(0)}, meta)}) and "### Summary" not in markdown(meta, {})


def ref_k(a, D, tol):
    """The columns of D (lat, I) kept, the largest of a (I,) first, for ||dropped|| <= tol * ||D a||: every count scanned, in numpy float64 (nothing shared with keep_needed or first_crossing)."""
    o = np.argsort(-a, kind="stable")
    r = np.append(np.sqrt((np.cumsum((D[:, o] * a[o])[:, ::-1], axis=1)[:, ::-1] ** 2).sum(0)), 0.0)
    return int(np.argmax(r <= tol * r[0]))


def same_hist(a, b, share):
    """Whether two histograms of the same items differ by at most `share` of them in how far apart they put them (the largest gap between the cumulative counts)."""
    return np.abs(np.cumsum(a) - np.cumsum(b)).max() <= share * max(a.sum(), 1)


def reference(run, ids, ncal):
    """The arrays of layer_stats for every MoE layer by plain forwards and loops, sharing with the pass nothing but the model's modules and the constants: the model's own forward on each sequence with hooks
    (block inputs, MoE inputs, router outputs), then numpy float64 a pair at a time. A list, one dict a MoE layer, and the profile of each layer."""
    m, S, K, E = run.m, ids.shape[1], run.K, run.E
    blk, xin, route, hooks = [[] for _ in run.layers], {}, {}, []
    for b, l in enumerate(run.layers):
        hooks.append(l.register_forward_pre_hook(lambda mod, a_, b=b: blk[b].append(a_[0][0].cpu())))
        if b in run.moe:
            hooks.append(l.mixer.register_forward_pre_hook(lambda mod, a_, b=b: xin.setdefault(b, []).append(a_[0][0].cpu())))
            hooks.append(l.mixer.gate.register_forward_hook(lambda mod, a_, o, b=b: route.setdefault(b, []).append((o[1].cpu(), o[2].cpu()))))
    for x in ids:
        m(input_ids=torch.from_numpy(x).long()[None].to(run.dev0), use_cache=False)
    for h in hooks:
        h.remove()
    nc, nh, ms, out, profs = ncal * S, (len(ids) - ncal) * S, fetch_counts(K), [], []
    for j, b in enumerate(run.moe):
        layer, moe = run.layers[b], run.block(b)
        up, dn, fc2 = moe.experts.up_proj.detach().cpu(), moe.experts.down_proj.detach().cpu().double().numpy(), moe.fc2_latent_proj.weight.detach().cpu().double().numpy()
        I, W, IDX = dn.shape[2], torch.cat([r[0] for r in route[b]]), torch.cat([r[1] for r in route[b]])
        Z = torch.cat([moe.fc1_latent_proj(x.to(run.dev(b))).cpu() for x in xin[b]])
        act = lambda t, s: moe.experts.act_fn(F.linear(Z[t][None], up[IDX[t, s]]))[0].double().numpy()
        prof, cal = np.zeros((E, I), np.int64), np.zeros(E, np.int64)
        for t in range(nc):
            for s in range(K):
                prof[int(IDX[t, s])] += act(t, s) > 0
                cal[int(IDX[t, s])] += 1
        prank = [np.argsort(np.argsort(-prof[e], kind="stable"), kind="stable") for e in range(E)]  # (each neuron's place in the expert's order of calibration frequency)
        r = dict(hz=np.zeros(I + 1, np.int64), hk=np.zeros((len(TOLS), I + 1), np.int64), ex=np.zeros((len(EX), E)), ssh=np.zeros(E), cal=cal, pr_h=np.zeros((len(BUDGETS), BINS), np.int64),
                 pr_s=np.zeros((len(BUDGETS), 5)), la_h=np.zeros((DMAX + 1, len(ms), K + 1), np.int64), la_w=np.zeros((DMAX + 1, len(ms))), la_wh=np.zeros((DMAX + 1, BINS), np.int64), la_n=np.zeros(DMAX + 1, np.int64))
        cm = np.zeros((nh, K))
        for t in range(nc, nc + nh):
            for s in range(K):
                e, a, w = int(IDX[t, s]), act(t, s), float(W[t, s])
                D = dn[e]
                y = D @ a
                yn, ks, zs, nnz = np.linalg.norm(y), [ref_k(a, D, tol) for tol in TOLS], int((a == 0).sum()), int((a > 0).sum())
                cm[t - nc, s] = np.linalg.norm(fc2 @ (w * y))
                r["hz"][zs] += 1
                for ti, k in enumerate(ks):
                    r["hk"][ti, k] += 1
                r["ex"][:, e] += [1, w, yn, w * yn, cm[t - nc, s], zs, *ks]
                for bi, f in enumerate(BUDGETS):
                    B, keep_o = round(f * I), np.zeros(I, bool)
                    keep_o[np.argsort(-a, kind="stable")[:B]] = True
                    rec = 1.0 if not nnz else float(((a > 0) & (prank[e] < B)).sum()) / nnz
                    ep, eo = [np.linalg.norm(D @ (a * ~k_)) / yn if yn else 0.0 for k_ in (prank[e] < B, keep_o)]
                    r["pr_h"][bi, min(int(rec * BINS), BINS - 1)] += 1
                    r["pr_s"][bi] += [rec, ep, eo, ep <= TOLS[0], eo <= TOLS[0]]
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1e-30)  # (block 5's routed output is zeroed by probe_model: its contributions are all 0, and so are their shares)
        for t in range(nh):
            for s in range(K):
                r["ssh"][int(IDX[nc + t, s])] += share[t, s]
        for d in range(DMAX + 1):
            if b - d < 0:
                continue
            H = torch.cat(blk[b - d][ncal:])
            for s0 in range(0, nh, S):
                logits, _, pidx = moe.gate(layer.norm(H[s0:s0 + S].to(run.dev(b))))
                sc, pidx = (logits.sigmoid() + moe.gate.e_score_correction_bias).cpu().numpy(), pidx.cpu().numpy()
                for t in range(len(sc)):
                    rank, tw = np.argsort(-sc[t], kind="stable"), W[nc + s0 + t].tolist()
                    for i, mm in enumerate(ms):
                        pred = set(pidx[t].tolist()) if i == 0 else set(rank[:mm].tolist())
                        hit = [int(e) in pred for e in IDX[nc + s0 + t]]
                        wr = sum(w for h_, w in zip(hit, tw) if h_) / sum(tw)
                        r["la_h"][d, i, sum(hit)] += 1
                        r["la_w"][d, i] += wr
                        if i == 0:
                            r["la_wh"][d, min(int(wr * BINS), BINS - 1)] += 1
                r["la_n"][d] += len(sc)
        out.append(r)
        profs.append(prof)
    return out, profs

def test_data(d, S=128):
    """Random token files in prep's layout (calib_<domain>.npy, held_<domain>.npy; one sequence each) in d, and the sequences in the order run_probe reads them (calibration of each domain, then held-out)."""
    rng = np.random.default_rng(1)
    c, h = [rng.integers(2, 256, (1, S)).astype(np.int32) for _ in gs.DOMAINS], [rng.integers(2, 256, (1, S)).astype(np.int32) for _ in gs.DOMAINS]
    for dom, x, y in zip(gs.DOMAINS, c, h):
        np.save(f"{d}/calib_{dom}.npy", x)
        np.save(f"{d}/held_{dom}.npy", y)
    return np.concatenate(c + h), len(c)


def check_model(m, ids, ncal, exact):
    """prefill on the model m against reference(): the same arrays (the counts of pairs exactly; the rest to float rounding where exact, which the CPU is, to a card's other rounding of bf16 where not), the
    structure probe_model has (96 zeros of 192 in block 1; the look-ahead of block 7 exact from the inputs of blocks 7, 6 and 5, not from block 4's), the profile and the walk against the forward."""
    run = nq.NemoRun(m)
    got, again, chk, rows = {}, {}, {"layers": [], "entries": 0, "equal": 0, "max_dk": 0}, globals()["ROWS"]
    globals()["ROWS"] = 37  # (an expert has ~100 held-out pairs here: blocks of 37, not of 8,192)
    real, calls = run.step, []

    def flaky(layer, h, carry, S, dev):
        calls.append(1)
        if len(calls) == 20:  # (the 8th sequence of block 1, the first MoE layer: 12 sequences a block)
            raise RuntimeError("grouped kernel")
        return real(layer, h, carry, S, dev)
    try:
        hs, prof = prefill(run, ids, ncal, tick=lambda j, a_, s_: got.__setitem__(j, a_), check=chk, chunk=5)  # (5 experts a time: chunks of 5 and one of 1)
        # the model's own blocks failing once in the middle of a layer are run again from its first sequence, with nothing of the failed try kept
        run.step = flaky
        prefill(run, ids, ncal, tick=lambda j, a_, s_: again.__setitem__(j, a_), chunk=5)
    finally:
        globals()["ROWS"] = rows
        run.__dict__.pop("step", None)
    assert run.eager and len(calls) > 20 and all(np.array_equal(got[j][k], again[j][k]) for j in got for k in got[j]), "a layer run again is another layer"
    fc = forward_check(run, hs, ids)
    assert fc["kl"] < (1e-3 if exact else 5e-3) and fc["top1"] > .99 and chk["entries"] == 16 * len(TOLS) * len(run.moe) and chk["equal"] >= .98 * chk["entries"], (fc, chk)
    ref, rprof = reference(run, ids, ncal)
    K, nh = run.K, (len(ids) - ncal) * ids.shape[1]
    t_sum, t_hist = (1e-4, 2e-3) if exact else (3e-2, 2e-2)
    for j, b in enumerate(run.moe):
        g, r = got[j], ref[j]
        assert g["hz"].sum() == nh * K == g["hk"][0].sum() and (g["la_n"] == r["la_n"]).all(), (b, "pairs, tokens")
        assert (np.abs(g["ex"][EX.index("n")] - r["ex"][EX.index("n")]).sum() <= (0 if exact else .005 * nh * K)) and np.abs(g["cal"] - r["cal"]).sum() <= (0 if exact else .005 * ncal * ids.shape[1] * K), (b, "routing")
        assert exact <= (g["hz"] == r["hz"]).all() and same_hist(g["hz"], r["hz"], t_hist) and all(same_hist(g["hk"][t], r["hk"][t], t_hist) for t in range(len(TOLS))), (b, "zeros, columns kept")
        for name in EX:
            assert np.allclose(g["ex"][EX.index(name)], r["ex"][EX.index(name)], rtol=max(t_sum, 5e-3) if name[0] == "k" else t_sum, atol=2 if name[0] == "k" else 1e-3), (b, name)
        assert np.allclose(g["ssh"], r["ssh"], rtol=t_sum, atol=1e-3) and np.allclose(g["pr_s"], r["pr_s"], rtol=t_sum, atol=2) and same_hist(g["pr_h"].T, r["pr_h"].T, 2e-2), (b, "importance, predictor")
        assert all(same_hist(g["la_h"][d, i], r["la_h"][d, i], t_hist) for d in range(DMAX + 1) for i in range(g["la_h"].shape[1])) and exact <= (g["la_h"] == r["la_h"]).all(), (b, "look-ahead")
        assert np.allclose(g["la_w"], r["la_w"], rtol=1e-4, atol=1e-3) and same_hist(g["la_wh"].T, r["la_wh"].T, 2e-2), (b, "look-ahead weights")
        assert (np.asarray(prof[b]) == rprof[j]).mean() >= (1 if exact else .99), (b, "profile")
    pairs = nh * K
    assert got[0]["hz"][96] == pairs, "block 1 has 96 zeros of 192 in every pair"
    last = got[len(run.moe) - 1]
    assert all(last["la_h"][d, 0, K] == last["la_n"][d] and abs(last["la_w"][d, 0] - last["la_n"][d]) < 1e-3 * last["la_n"][d] for d in (0, 1, 2)) and last["la_h"][3, 0, K] < last["la_n"][3], "block 7's look-ahead"
    assert all(got[j]["la_h"][0, :, K].tolist() == [nh] * len(FETCH_X) for j in got), "d = 0 is the model's own choice"
    return got, ref


def check_decode(m, exact):
    """decode_streams: the inputs it keeps at each one-token step are the ones a forward over the whole sequence feeds the blocks at those positions (the same to float rounding: the embeddings exactly), and the
    routing recomputed from them is what the model chose while it decoded."""
    run, rng = nq.NemoRun(m), np.random.default_rng(2)
    prompts = [list(map(int, rng.integers(2, 256, 9))) for _ in range(3)]
    chose = {b: [] for b in run.moe}
    hooks = [run.layers[b].mixer.gate.register_forward_hook(lambda mod, a_, o, b=b: chose[b].append(o[2].cpu()) if o[2].shape[0] == 1 else None) for b in run.moe]  # (a hook of the test's own)
    streams, routes, gens = decode_streams(run, prompts, 7)
    for h in hooks:
        h.remove()
    assert all(torch.equal(routes[b], torch.cat(chose[b])) for b in run.moe), "the experts kept at each step are not the router's"
    steps = [len(g) - len(p) - 1 for g, p in zip(gens, prompts)]  # (the last token generated is never fed back)
    assert all(s >= 1 for s in steps) and all(len(v) == sum(steps) and v.shape[1] == run.width for v in streams.values()) and all(g[:len(p)] == p for g, p in zip(gens, prompts))
    off = 0
    for g, p, st in zip(gens, prompts, steps):
        tf, hooks = [None] * len(run.layers), []
        for b, l in enumerate(run.layers):
            hooks.append(l.register_forward_pre_hook(lambda mod, a_, b=b: tf.__setitem__(b, a_[0][0].float().cpu())))
        run.m(input_ids=torch.tensor(g[:-1])[None].to(run.dev0), use_cache=False)
        for h in hooks:
            h.remove()
        for b in range(len(run.layers)):
            want, have = tf[b][len(p):len(p) + st], streams[b][off:off + st].float()
            err = float(((have - want).norm(dim=1) / want.norm(dim=1)).max())
            assert err == 0 if b == 0 else err < (.03 if exact else .1), (b, err)
        off += st
    for b in run.moe:
        idx = run.block(b).gate(run.layers[b].norm(streams[b].to(run.dev(b))))[2].cpu()
        same = (idx.sort(1).values == torch.cat(chose[b]).sort(1).values).all(1).float().mean()
        assert same > (.99 if exact else .95), (b, same)
    return sum(steps)


def check_job(m, d, got):
    """run_probe on the model m with the files of test_data in d: what it writes, the numbers of probe.json against the arrays check_model compared with the reference (got), and the report of a stopped
    job's partial arrays."""
    out = tempfile.mkdtemp()
    run = nq.NemoRun(m)
    a = types.SimpleNamespace(data=d, out=out, cal_seqs=1, held_seqs=1, decode_new=6, chunk=5, deadline=0)
    prompts = [list(map(int, np.random.default_rng(4).integers(2, 256, 9))) for _ in range(4)]
    run_probe(m, a, prompts)
    assert sorted(os.listdir(out)) == ["decode.npz", "meta.json", "prefill.npz", "probe.json", "probe.md"], os.listdir(out)
    res, meta = json.load(open(f"{out}/probe.json")), json.load(open(f"{out}/meta.json"))
    assert meta["errors"] == [] and meta["decode"]["routing_agreement"]["min"] > .95 and meta["decode"]["prompts"] == 4 and meta["inter"] == 192 and meta["latent"] == 64 and meta["hidden"] == 128 and meta["expert_bytes_4bit"] == 2 * 64 * 192 // 2 and meta["moe"] == run.moe
    pre, dec = res["prefill"], res["decode"]
    assert pre["lookahead"]["d0_all_named"] == 1 and dec["lookahead"]["d0_all_named"] == 1, "d = 0 names the experts the model chose"
    hz = sum(got[j]["hz"] for j in range(len(run.moe)))
    assert pre["layers"] == 4 and dec["layers"] == 4 and pre["pairs"] == hz.sum() and abs(pre["sparsity"]["zero"]["mean"] - (hz * np.arange(193)).sum() / hz.sum() / 192) < 1e-12
    assert dec["pairs"] == meta["decode"]["steps"] * run.K * 4 and pre["sparsity"]["zero_by_layer"][0]["mean"] == .5 and pre["sparsity"]["zero_by_layer"][0]["p10"] == .5, "block 1: half the neurons are zero"
    assert pre["lookahead"]["by_layer"][3]["d_prev"] == 2 and pre["lookahead"]["by_layer"][3]["at_prev"]["recall"]["mean"] == 1 and pre["lookahead"]["by_layer"][3]["two_earlier"]["d"] == 4
    md = open(f"{out}/probe.md").read()
    assert "### Held-out text: relu2 sparsity" in md and "### Decode steps: relu2 sparsity" in md and "### Checks" in md and "**error:**" not in md and "NOT the model's forward" not in md and "while decoding for 100.0% of the steps of a layer on average" in md
    # a report that fails does not end the job: the decodes still run and their arrays are written, and the report step renders them after
    out2, real = tempfile.mkdtemp(), globals()["render"]
    globals()["render"] = lambda o: 1 / 0
    try:
        run_probe(m, types.SimpleNamespace(**{**vars(a), "out": out2}), prompts[:2])
    finally:
        globals()["render"] = real
    assert sorted(os.listdir(out2)) == ["decode.npz", "meta.json", "prefill.npz"], os.listdir(out2)
    render(out2)
    assert json.load(open(f"{out2}/probe.json"))["decode"]["layers"] == 4
    z = dict(np.load(f"{out}/prefill.npz"))
    assert int(z["done"]) == 4 and all(np.array_equal(z["hz"][j], got[j]["hz"]) for j in range(4))
    z["done"] = np.array(2)
    os.makedirs(f"{out}/partial")
    np.savez_compressed(f"{out}/partial/prefill.npz", **z)
    meta["errors"].append("prefill: deadline before block 5")
    json.dump(meta, open(f"{out}/partial/meta.json", "w"))
    render(f"{out}/partial")
    part = open(f"{out}/partial/probe.md").read()
    assert "in each of the 2 MoE layers measured (4 in the model)" in part and "**error:** prefill: deadline before block 5" in part and json.load(open(f"{out}/partial/probe.json"))["prefill"]["layers"] == 2
    return out


def selftest(_=None):
    import shutil
    for pkg in ("fla", "causal_conv1d", "mamba_ssm"):
        sys.modules[pkg] = None  # (the tiny model is on the CPU: with these installed, transformers binds their Triton / CUDA kernels when the model code is imported, and they take no CPU tensors)
    torch.manual_seed(0)
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() and os.environ.get("SELFTEST_GPU", "1") != "0" else 0
    t0 = time.time()
    check_hand("cpu")
    check_device("cpu")
    check_report()
    d = tempfile.mkdtemp()
    ids, ncal = test_data(d)
    m = probe_model().eval()
    m.set_experts_implementation("eager")
    with torch.inference_mode():
        got, ref = check_model(m, ids, ncal, True)
        steps = check_decode(m, True)
        out = check_job(m, d, got)
        # the cards: the hand-made cases and the random comparison on each, then the tiny model split over them as the real one is (accelerate hooks, layers on different cards)
        for i in range(n_cuda):
            check_hand(f"cuda:{i}")
            check_device(f"cuda:{i}")
        if n_cuda:
            from transformers import NemotronHForCausalLM
            ck = tempfile.mkdtemp()
            m.save_pretrained(ck)
            if n_cuda > 1:
                gq.log(f"direct GPU-to-GPU copies that came back wrong: {gq.host_hops() or 'none'} (the job's copies go through the host)")
            mg = NemotronHForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map=nq.device_map(n_cuda)).eval()
            mg.set_experts_implementation("eager")
            check_model(mg, ids, ncal, False)
            check_decode(mg, False)
            shutil.rmtree(ck)
    print(f"selftest ok ({time.time() - t0:.0f} s): zeros, columns kept, predictor, look-ahead hits, ranks, quantiles and correlations on hand-made cases (CPU{''.join(f', cuda:{i}' for i in range(n_cuda))}); "
          f"the report's arithmetic on hand-made arrays; a tiny random NemotronH against a naive reference in all {len(ids) * ids.shape[1]} tokens x {len(m.model.layers)} blocks (96 zeros of 192 in block 1, the look-ahead of block 7 exact "
          f"from the inputs of blocks 7, 6 and 5); {steps} decode steps against a forward over the whole sequences; the job end to end, and a stopped job's partial report"
          + (f"; the tiny model split over {n_cuda} cards" if n_cuda else ""))
    for p_ in (d, out):
        shutil.rmtree(p_, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--text", action="store_true")
    p.set_defaults(fn=plan)
    p = sub.add_parser("run")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--held-seqs", type=int, default=4, help="held-out sequences a domain")
    p.add_argument("--cal-seqs", type=int, default=12, help="calibration sequences a domain (the neuron profiles and the routing counts)")
    p.add_argument("--decode-prompts", type=int, default=20, help="decode prompts (glm_skew.py's, in order)")
    p.add_argument("--decode-new", type=int, default=64, help="new tokens a decode")
    p.add_argument("--chunk", type=int, default=0, help="experts read at a time (0: 16)")
    p.add_argument("--deadline", type=float, default=0, help="epoch seconds by which the job must be over")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
