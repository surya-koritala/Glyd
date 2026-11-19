"""Calibrated 4-bit variants for kld_study.py (GLY-37): GPTQ and AWQ rounding into the Glyd block-code format, runs of BLOCK weights along a row,
one bf16 scale a run, symmetric codes -q..q; nothing else is stored (no zero points, no groups beyond the run).

    python calib.py build MODEL_DIR OUT_DIR [--seqs 512] [--length 2048]     (writes OUT_DIR/calib.pt and calib.json; needs OUT_DIR/ref.pt from kld_study.py ref)
    python calib.py quantize MODEL_DIR OUT_DIR --bits 5|6 [--seqs 512]        (GPTQ with act-order, as scored below; writes OUT_DIR/codes: the codes and scales of every matrix the packer takes,
                                                                               for glyd-gpu pack --plan PLAN --codes OUT_DIR/codes; needs OUT_DIR/calib.pt and calib.json from build)
    python calib.py check PACKED_DIR PLAN.json CODES_DIR                      (a glyd-gpu pack --codes save decodes to its codes times its scales, bit for bit)
    python calib.py selftest                                                  (CPU: a tiny random qwen3_5 model, calibration and both variants end to end)

Variants of kld_study.py score (all matrices of the text decoder not in a decoder layer, the embedding and lm_head, are rounded as int:B:BLOCK:mse:bf16):

    gptq:B:BLOCK:bf16[:act]      per decoder layer, the Hessian X^T X of each group of linears sharing an input, from the calibration sequences run through the
                                 layers already rounded; columns rounded in order with the error fed to the columns after (GPTQ), the scale of each run
                                 the best of 16 clips (as int:..:mse) of the run's weights when its first column is reached. With :act the runs (not the
                                 columns: a run stays 32 neighbours) go in order of falling Hessian diagonal, so the format is the same
    hgptq:B:BLOCK:bf16[:act]     incoherence processing (GLY-127): each linear's input side is first multiplied by a randomized Hadamard transform T = H_R diag(s)
                                 (s: random signs, one vector a group of linears sharing an input, seed = layer * 8 + group; H_R: orthonormal Walsh-Hadamard over
                                 blocks of R = gcd(input dim, 128) values), W' = W T^T and H' = T H T^T, GPTQ as above in that basis, then Q = Q' T back
                                 (y = (Q' T)(x) = Q'(T x): the engine would rotate each activation instead). Format in the rotated basis: runs of BLOCK, one bf16 scale
    lmgptq:B:BLOCK:bf16[:act]    non-uniform codebook (GLY-127): a run keeps its bf16 scale (its absmax), and each output row a 2^B-entry codebook (bf16, in units of
                                 the scale) fitted by Lloyd-Max (1-D k-means of the row's w / scale, weighted by the Hessian diagonal of the column) replaces the
                                 uniform grid in the error-feedback loop; the codes are indices into the row's codebook
    hlmgptq:B:BLOCK:bf16[:act]   both: the codebook is fitted and used in the rotated basis
    awq:B:BLOCK:bf16             per input channel a scale s = a^alpha (a: the channel's rms activation, alpha on a grid of 21 in 0..1, the least output
                                 error X(Q(W s)/s - W)^T over the group), folded into the norm that feeds the group; the scaled weights then rounded as
                                 int:..:mse. Folds: input_layernorm into q/k/v (full attention) or in_proj_qkv/z/b/a (linear attention), and
                                 post_attention_layernorm into gate_proj/up_proj. o_proj, out_proj and down_proj have no norm before them (their input
                                 is the attention or gated-MLP output), so they are rounded as int:..:mse; scaling their columns and dividing back would
                                 leave the run format, so there is no such "kept" scale.

The layer reconstruction error of a matrix is tr(dW H dW^T), H = X^T X / tokens (the output error on the calibration inputs); it is logged for the
method and for plain mse rounding of the same matrix.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kld_study import fit_scale, group_of, int_tensor, windows  # noqa: E402

# the calibration mix: public, ungated, pinned by revision and by the file's hash. WikiText-2 train is a different split from the test windows
# kld_study.py scores; the code and chat text are checked against those windows for shared 32-token runs.
SOURCES = [
    {'name': 'wikitext-2-train', 'dataset': 'Salesforce/wikitext', 'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
     'file': 'wikitext-2-raw-v1/train-00000-of-00001.parquet', 'sha256': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7',
     'license': 'CC-BY-SA-3.0 and GFDL', 'rows': None},
    {'name': 'python-code', 'dataset': 'iamtarun/python_code_instructions_18k_alpaca', 'revision': '7cae181e29701a8663a07a3ea43c8e105b663ba1',
     'file': 'data/train-00000-of-00001-8b6e212f3e1ece96.parquet', 'sha256': 'bbc6fa528fcc17a845e863da8652bc02c2523373cd8cc0b9af1ffb374e9ae378',
     'license': 'CC-BY-NC-4.0 (research use)', 'rows': 4000},
    {'name': 'ultrachat-chat', 'dataset': 'HuggingFaceH4/ultrachat_200k', 'revision': '8049631c405ae6576f93f445c6b8166f76f5505a',
     'file': 'data/train_sft-00000-of-00003-a3ecf92756993583.parquet', 'sha256': 'afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727',
     'license': 'MIT', 'rows': 1000},
]
NGRAM = 32
BLOCK = 32  # (the engine's run: one scale for 32 values of a row)


def source_text(src):
    import pyarrow as pa
    import pyarrow.parquet as pqt
    data = urllib.request.urlopen(f'https://huggingface.co/datasets/{src["dataset"]}/resolve/{src["revision"]}/{src["file"]}', timeout=600).read()
    if hashlib.sha256(data).hexdigest() != src['sha256']:
        raise RuntimeError(f'{src["name"]}: dataset hash mismatch')
    t = pqt.read_table(pa.BufferReader(data))
    n = src['rows'] or t.num_rows
    if 'text' in t.column_names:
        rows = t.column('text').to_pylist()[:n]
    elif 'prompt' in t.column_names and 'output' in t.column_names:
        rows = t.column('prompt').to_pylist()[:n]
    else:
        rows = ['\n'.join(f'{m["role"]}: {m["content"]}' for m in c) for c in t.column('messages').to_pylist()[:n]]
    return '\n\n'.join(rows)


def grams(ids):
    return {hash(tuple(ids[i:i + NGRAM])) for i in range(len(ids) - NGRAM + 1)}


def pick(streams, count, length, banned):
    """`count` sequences, source by source in turn (so the first n are a mix too): disjoint `length`-token windows evenly spaced through each
    source's tokens, skipping any that share a 32-token run with `banned`. Returns the windows and how many were skipped."""
    out, skipped = [], 0
    need = [len(range(i, count, len(streams))) for i in range(len(streams))]
    taken = []
    for i, ids in enumerate(streams):
        c = [ids[j * length:(j + 1) * length] for j in range(len(ids) // length)]
        if len(c) < need[i] * 1.1:
            raise ValueError(f'source {i}: {len(c)} windows of {length} tokens for {need[i]}')
        spread = [int(j * len(c) / need[i]) for j in range(need[i])]
        clean = []
        for j in spread + [j for j in range(len(c)) if j not in spread]:  # (the spread first, then any window to replace one that is skipped)
            if len(clean) == need[i]:
                break
            if grams(c[j]) & banned:
                skipped += 1
            else:
                clean.append(c[j])
        if len(clean) < need[i]:
            raise ValueError(f'source {i}: only {len(clean)} windows free of the evaluation text')
        taken.append(clean)
    for j in range(max(need)):
        for i in range(len(streams)):
            if j < len(taken[i]):
                out.append(taken[i][j])
    return out, skipped


def cmd_build(a):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True)
    ref = torch.load(a.out / 'ref.pt', weights_only=True)
    banned = set()
    for w in ref['windows']:
        banned |= grams(w)
    streams = [tok.encode(source_text(s), add_special_tokens=False) for s in SOURCES]
    seqs, skipped = pick(streams, a.seqs, a.length, banned)
    shared = sum(1 for s in seqs if grams(s) & banned)
    assert shared == 0
    ids = torch.tensor(seqs, dtype=torch.int32)
    torch.save(ids, a.out / 'calib.pt')
    meta = {'sequences': len(seqs), 'length': a.length, 'tokens': int(ids.numel()), 'sources': SOURCES, 'order': 'round robin over sources',
            'source_tokens': [len(s) for s in streams], 'eval_windows': len(ref['windows']), 'skipped_for_shared_32gram': skipped,
            'sequences_sharing_a_32gram_with_eval': shared, 'ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
    (a.out / 'calib.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({k: v for k, v in meta.items() if k != 'sources'}), flush=True)


def quantize(model, seqs, bits, codes, sources):
    """GPTQ with act-order of `model`'s text decoder at `bits` (5 or 6) on `seqs`, as kld_study.py scores gptq:B:32:bf16:act, in place (ponytail: the embedding and lm_head are not calibrated, as scored: the packer rounds
    them from their weights; GPTQ of the head needs the final hidden states' Hessian and its own measurement); the codes and scales of every matrix the packer takes are written to the directory
    `codes` as `glyd-gpu pack --plan PLAN --codes` reads them: for each matrix NAME (its path in the model, the checkpoint's), NAME.codes (int8, rows x cols) and NAME.scales (bf16, rows x cols / 32), in shards of a layer
    named by model.safetensors.index.json, and calibration.json: what glyd.json records (the method, the datasets and their revisions, the sequences and their length in tokens)."""
    import torch
    from safetensors.torch import save_file
    assert bits in (5, 6), 'the packer has the six- and five-bit layouts'
    codes.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, [0]

    def sink(li, layer):
        flat = {f'{n}.{kind}': t for n, pair in layer.items() for kind, t in zip(('codes', 'scales'), pair)}
        shard = f'codes-{li:05d}.safetensors'
        save_file(flat, codes / shard, metadata={'format': 'pt'})
        weight_map.update({n: shard for n in flat})
        total[0] += sum(t.numel() * t.element_size() for t in flat.values())

    run(model, {'kind': 'gptq', 'bits': bits, 'block': BLOCK, 'act': True, 'sink': sink}, seqs, next(model.parameters()).device)
    (codes / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': total[0]}, 'weight_map': weight_map}, indent=1) + '\n')
    cal = {'method': 'gptq-act', 'data': [{'dataset': s['dataset'], 'revision': s['revision']} for s in sources], 'seqs': len(seqs), 'tokens': int(seqs.shape[1])}
    (codes / 'calibration.json').write_text(json.dumps(cal, indent=1) + '\n')
    return cal


def cmd_quantize(a):
    import torch
    from kld_study import load
    meta = json.loads((a.out / 'calib.json').read_text())
    seqs = torch.load(a.out / 'calib.pt', weights_only=True)[:a.seqs]
    cal = quantize(load(a.model_dir), seqs, a.bits, a.out / 'codes', meta['sources'])
    print(json.dumps(cal), flush=True)


def values(codes, name):
    """The bf16 matrix NAME.codes times NAME.scales make in the checkpoint-form directory `codes`: each run's int8 codes times its bf16 scale (exact in float32), rounded to bf16 once, as the packer's decode makes it."""
    import torch
    from safetensors import safe_open
    index = json.loads((codes / 'model.safetensors.index.json').read_text())['weight_map']
    got = {}
    for kind in ('codes', 'scales'):
        with safe_open(str(codes / index[f'{name}.{kind}']), framework='pt') as f:
            got[kind] = f.get_tensor(f'{name}.{kind}')
    c, s = got['codes'], got['scales']
    assert c.dtype == torch.int8 and s.dtype == torch.bfloat16 and c.shape == (s.shape[0], s.shape[1] * BLOCK), (name, c.shape, s.shape)
    return (c.float().view(s.shape[0], s.shape[1], BLOCK) * s.float()[:, :, None]).view(c.shape).bfloat16()


def check_pack(packed, plan, codes):
    """A save of `glyd-gpu pack --plan PLAN --codes CODES` (`packed`, verified apart: glyd-gpu verify) decodes to its codes times its scales, bit for bit: glyd.json's sha256 of each matrix the codes cover is the sha256 of the bf16
    bytes of values(); returns how many, and the names of the packs that have no codes (packed from their weights)."""
    import torch
    index = json.loads((codes / 'model.safetensors.index.json').read_text())['weight_map']
    held = {t['name']: t['sha256'] for e in json.loads((packed / 'glyd.json').read_text())['packs'].values() for t in e['tensors']}
    n, plain = 0, []
    for put in json.loads(plan.read_text())['puts']:
        if 'linear' not in put:
            continue
        sources = put.get('from', put['linear'])
        if not any(f'{f}.codes' in index for f in sources):
            plain.append(put['linear'][0])
            continue
        for member, f in zip(put['linear'], sources):
            want = hashlib.sha256(values(codes, f).contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            assert held.get(f'{member}.weight') == want, f'{member} does not decode to its codes times its scales'
            n += 1
    return n, plain


def cmd_check(a):
    n, plain = check_pack(a.packed, a.plan, a.codes)
    print(f'{n} matrices decode to their codes times their scales, bit for bit; packed from their weights: {", ".join(plain) or "none"}')


def h_err(dw, H):
    """tr(dW H dW^T): the output error summed over the calibration tokens (H = X^T X / tokens)."""
    dw = dw.float()
    return float(((dw @ H) * dw).sum())


def entropy(codes, q, n=None):
    import torch
    hist = torch.bincount((codes.to(torch.int64) + q).view(-1), minlength=n or 2 * q + 1).double()
    p = hist[hist > 0] / hist.sum()
    return float(-(p * torch.log2(p)).sum())


def fwht(x):
    """Orthonormal Walsh-Hadamard transform over blocks of R = gcd(n, 128) values of the last dim (n = x.shape[-1]); symmetric, so its own inverse."""
    import math
    import torch
    n = x.shape[-1]
    R = math.gcd(n, 128)
    y, h = x.reshape(-1, n // R, R), 1
    while h < R:
        y = y.reshape(-1, n // R, R // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
        h *= 2
    return y.reshape(x.shape) / R ** 0.5


def signs(n, seed, device):
    import torch
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)


def lloyd_max(x, wt, k, iters=25, chunk=2048):
    """A k-entry codebook for each row of x (rows, n): weighted 1-D k-means (weights wt (n,), Lloyd-Max), started on a uniform grid over -1..1; sorted, bf16."""
    import torch
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


def gptq_matrix(w, H, bits, block, act_order=False, damp=0.01, step=128, lm=False, rot_seed=None):
    """GPTQ of w (rows, cols) with H (cols, cols): returns the dequantized matrix (bf16), the codes (int8, rows x cols), the scales (rows, cols // block, bf16) and
    the codebooks (rows, 2^bits, bf16; None unless lm). With lm the grid is each row's Lloyd-Max codebook (units of the run's absmax scale) and codes are indices;
    with rot_seed the matrix is rotated (randomized Hadamard on the input side) first and the result rotated back: codes and scales are then of the rotated basis."""
    import torch
    rows, cols = w.shape
    q = (1 << (bits - 1)) - 1
    runs = cols // block
    W, H = w.float().clone(), H.float().clone()
    dead = H.diagonal() == 0
    W[:, dead] = 0
    wt = H.diagonal().clone()
    if rot_seed is not None:
        s = signs(cols, rot_seed, w.device)
        W = fwht(W * s)
        H = fwht(fwht(H * s).T * s).T  # (T H T^T: the rows, then the columns)
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
        cb = lloyd_max(norm.reshape(rows, cols), (wt / wt.mean().clamp_min(1e-30) + 1e-6)[None, :], 1 << bits)
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q, C = torch.zeros_like(W), torch.zeros(rows, cols, dtype=torch.int8, device=w.device)
    S = torch.zeros(rows, runs, device=w.device)
    for i1 in range(0, cols, step):
        i2 = min(i1 + step, cols)
        W1, E1, U1 = W[:, i1:i2].clone(), torch.zeros(rows, i2 - i1, device=w.device), U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % block == 0:
                if lm:
                    sc = W1[:, i:i + block].abs().amax(-1).clamp_min(1e-30).bfloat16().float()
                else:
                    sc = fit_scale(W1[:, i:i + block].reshape(rows, 1, block), q, 'mse', True).view(rows)
                S[:, (i1 + i) // block] = sc
            if lm:
                c = (W1[:, i, None] / sc[:, None] - cb).abs().argmin(-1)
                v = cb.gather(1, c[:, None])[:, 0] * sc
            else:
                c = torch.clamp(torch.round(W1[:, i] / sc), -q, q)
                v = c * sc
            Q[:, i1 + i], C[:, i1 + i] = v, c.to(torch.int8)
            e = (W1[:, i] - v) / U1[i, i]
            W1[:, i:] -= e[:, None] * U1[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    inv = torch.argsort(perm)
    scales = torch.empty_like(S)
    if act_order:
        scales[:, order] = S
    else:
        scales = S
    Q = Q[:, inv]
    if rot_seed is not None:
        Q = fwht(Q) * s
    return Q.bfloat16(), C[:, inv], scales.bfloat16(), (cb.bfloat16() if lm else None)


def awq_group(ws, H, norm, bits, block, grid=21):
    """AWQ for the linears `ws` (weights, all with input H's channels) fed by `norm` (zero-centred RMSNorm: x * (1 + weight)): returns the new norm weight
    (bf16) and the dequantized weights (bf16, to put in place), each the rounding of W times the scale the bf16 norm weight really folds in."""
    import torch
    W = torch.cat([w.float() for w in ws])
    a = (H.diagonal().clamp_min(0) ** 0.5).clamp_min(1e-8)
    base = 1.0 + norm.float()
    best = None
    for alpha in torch.linspace(0, 1, grid):
        s = a ** float(alpha)
        s = s / (s.max() * s.min()).sqrt()
        nw = (base / s - 1).bfloat16()
        se = base / (1.0 + nw.float())
        deq = int_tensor(W * se, bits, block, 'mse', scale_bf16=True)[0].bfloat16()
        err = h_err(deq.float() / se - W, H)
        if best is None or err < best[0]:
            best = (err, nw, deq, se)
    _, nw, deq, se = best
    return nw, list(deq.split([w.shape[0] for w in ws])), (deq.float() / se - W).split([w.shape[0] for w in ws])


def layer_groups(layer):
    """(norm or None, [linears]) for each set of linears sharing an input, in the order the layer runs them."""
    if hasattr(layer, 'linear_attn'):
        m = layer.linear_attn
        mix, out = [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a], m.out_proj
    else:
        m = layer.self_attn
        mix, out = [m.q_proj, m.k_proj, m.v_proj], m.o_proj
    return [(layer.input_layernorm, mix), (None, [out]), (layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj]), (None, [layer.mlp.down_proj])]


def run(model, spec, seqs, device):
    """(TF32 for the Hessian products only: restored after.)"""
    import torch
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return _run(model, spec, seqs, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _run(model, spec, seqs, device):
    """Rounds the text decoder in place by spec ({'kind': 'gptq'|'awq', 'bits', 'block', 'act'}), layer by layer on `seqs` (int tensor, N x length): each
    layer sees the outputs of the layers before it, already rounded. Returns the bits by group as apply_variant does and the reconstruction errors."""
    import torch
    bits, block = spec['bits'], spec['block']
    q = (1 << (bits - 1)) - 1
    cbk, rot = spec.get('lm', False), spec.get('rot', False)
    lm = model.model.language_model
    names = {id(m): n for n, m in model.named_modules()}
    by_group, mats, done, nominal = {}, [], set(), [0, 0.0]

    def account(name, p, ent, extra=None):
        """`extra`: the bits a weight beyond the codes (default the run's scale; the codebook and sign vector add theirs); the nominal total counts B-bit codes."""
        extra = 16.0 / block if extra is None else extra
        g = group_of(name)
        n, b = by_group.get(g, (0, 0.0))
        by_group[g] = (n + p.numel(), b + (ent + extra) * p.numel())
        nominal[0], nominal[1] = nominal[0] + p.numel(), nominal[1] + (bits + extra) * p.numel()
        done.add(name)

    kw = {}
    hooks = [l.register_forward_pre_hook(lambda m, args, k, i=i: kw.__setitem__(i, k), with_kwargs=True) for i, l in enumerate(lm.layers)]
    with torch.inference_mode():
        lm(input_ids=seqs[:1].to(device).long(), use_cache=False)
        acts = [lm.embed_tokens(s[None].to(device).long()).cpu() for s in seqs]
    for h in hooks:
        h.remove()
    t0 = time.time()
    for li, layer in enumerate(lm.layers):
        groups = layer_groups(layer)
        Hs, count, kept = [None] * len(groups), [0], {}

        def grab(m, inp, out, k):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            Hs[k] = x.T @ x if Hs[k] is None else Hs[k].addmm_(x.T, x)
            if k == 0:
                count[0] += x.shape[0]
        hooks = [lins[0].register_forward_hook(lambda m, i, o, k=k: grab(m, i, o, k)) for k, (_, lins) in enumerate(groups)]
        with torch.inference_mode():
            for x in acts:
                layer(x.to(device), **kw[li])
        for h in hooks:
            h.remove()
        with torch.no_grad():
            for gi, ((norm, lins), H) in enumerate(zip(groups, Hs)):
                H = H / count[0]
                ws = [m.weight.data for m in lins]
                plain = [int_tensor(w, bits, block, 'mse', scale_bf16=True) for w in ws]
                if spec['kind'] == 'gptq':
                    cat, sizes = torch.cat(ws), [w.shape[0] for w in ws]
                    deq, C, S, _ = gptq_matrix(cat, H, bits, block, spec['act'], lm=cbk, rot_seed=li * 8 + gi if rot else None)
                    news = list(deq.split(sizes))
                    ents = [entropy(c, 0, 1 << bits) if cbk else entropy(c, q) for c in C.split(sizes)]
                    extra = 16.0 / block + (16.0 * (1 << bits) / cat.shape[1] if cbk else 0) + (1.0 / cat.shape[0] if rot else 0)  # (scale, codebook, signs)
                    diffs = [n.float() - w.float() for n, w in zip(news, ws)]
                    if spec.get('sink'):  # (the matrices the packer takes: rows of a multiple of 64, so not in_proj_a, in_proj_b, which stay bf16)
                        assert not (cbk or rot), 'the packer takes plain GPTQ codes, not codebook indices or rotated ones'
                        kept.update({names[id(m)]: (c.cpu().clone(), sc.cpu().clone()) for m, c, sc in zip(lins, C.split(sizes), S.split(sizes)) if c.shape[0] % 64 == 0 and c.shape[1] % 32 == 0})
                elif norm is not None:
                    nw, news, diffs = awq_group(ws, H, norm.weight.data, bits, block)
                    norm.weight.data.copy_(nw)
                    extra, ents = None, [e[1] - 16.0 / block for e in plain]  # (the unscaled codes' entropy: the scaled codes' differ little)
                else:
                    extra, news, ents = None, [p[0] for p in plain], [p[1] - 16.0 / block for p in plain]
                    diffs = [n.float() - w.float() for n, w in zip(news, ws)]
                for m, w, n, e, d, p in zip(lins, ws, news, ents, diffs, plain):
                    mats.append({'name': names[id(m)], 'err_plain': h_err(p[0].float() - w.float(), H), 'err': h_err(d, H), 'ref': h_err(w, H)})
                    m.weight.data.copy_(n)
                    account(names[id(m)] + '.weight', m.weight, e, extra)
        with torch.inference_mode():
            for i, x in enumerate(acts):
                acts[i] = layer(x.to(device), **kw[li]).cpu()
        if kept:
            spec['sink'](li, kept)
        print(json.dumps({'layer': li, 'seconds': round(time.time() - t0, 1)}), flush=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if group_of(name) and name not in done:
                new, ent = int_tensor(p.data, bits, block, 'mse', scale_bf16=True)
                p.data.copy_(new)
                account(name, p, ent - 16.0 / block)
    spec['nominal'] = nominal[1] / nominal[0]
    return by_group, mats


def summary(mats):
    return {'matrices': len(mats), 'recon_err_method': sum(m['err'] for m in mats), 'recon_err_plain': sum(m['err_plain'] for m in mats),
            'recon_err_ratio_mean': sum(m['err'] / max(m['err_plain'], 1e-30) for m in mats) / max(len(mats), 1)}


def s_err_ok(mats):
    return sum(m['err'] for m in mats) <= sum(m['err_plain'] for m in mats) * 1.001


def selftest():
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from kld_study import apply_variant, parse_variant
    torch.manual_seed(0)
    assert parse_variant('gptq:4:32:bf16:act') == {'kind': 'gptq', 'bits': 4, 'block': 32, 'act': True, 'bf16': True}
    assert parse_variant('awq:3:32:bf16')['kind'] == 'awq' and parse_variant('gptq:5:32:bf16')['act'] is False
    import random
    rnd = random.Random(0)
    streams = [[rnd.randrange(1000) for _ in range(64 * 40)] for _ in range(3)]
    bad = streams[1][64 * 13:64 * 14]  # (a window of the evaluation text inside source 1)
    seqs3, skipped = pick(streams, 10, 64, grams(bad))
    assert len(seqs3) == 10 and skipped >= 1 and not any(grams(w) & grams(bad) for w in seqs3) and len({tuple(w) for w in seqs3}) == 10
    assert [len(range(i, 10, 3)) for i in range(3)] == [4, 3, 3]
    # one matrix: GPTQ in the block-code format, and no worse than plain rounding under H
    X = torch.randn(4096, 96) @ torch.randn(96, 96) + 0.3 * torch.randn(4096, 96)
    Wm = torch.randn(48, 96) * 0.05
    H = X.T @ X / 4096
    plain = int_tensor(Wm.bfloat16(), 4, 32, 'mse', scale_bf16=True)[0]
    for act in (False, True):
        deq, codes, scales, _ = gptq_matrix(Wm.bfloat16(), H, 4, 32, act)
        assert int(codes.abs().max()) <= 7 and codes.dtype == torch.int8, 'codes do not fit 4 bits'
        assert scales.shape == (48, 3) and scales.dtype == torch.bfloat16, 'not one bf16 scale a run of 32'
        assert torch.equal(deq, (codes.float().view(48, 3, 32) * scales.float().view(48, 3, 1)).view(48, 96).bfloat16()), 'values are not code * scale'
        assert h_err(deq.float() - Wm.bfloat16().float(), H) <= h_err(plain.float() - Wm.bfloat16().float(), H), act
    # GLY-127: the transform is orthonormal and an involution; on a matrix with outlier input channels (X with a few large, correlated channels) at 2 bits the
    # rotation beats plain GPTQ under H, and the codebook beats the uniform grid (which at 2 bits is ternary: codes -1..1) with and without it
    xt = torch.randn(64, 3, 128)
    assert torch.allclose(fwht(fwht(xt)), xt, atol=1e-5) and torch.allclose(fwht(xt).norm(dim=-1), xt.norm(dim=-1), atol=1e-4)
    Hd = torch.stack([fwht(torch.eye(128)[i]) for i in range(128)])  # (rows of the transform: +-1/sqrt(128), orthogonal)
    assert torch.allclose(Hd @ Hd.T, torch.eye(128), atol=1e-5) and torch.allclose(Hd.abs(), torch.full((128, 128), 128 ** -0.5), atol=1e-6)
    assert fwht(torch.randn(5, 96)).shape == (5, 96)  # (96 = 3 * 32: blocks of gcd(96, 128) = 32)
    Xo = torch.randn(4096, 96) * torch.where(torch.arange(96) % 17 == 0, 12.0, 1.0)
    Xo = Xo + 0.5 * Xo.roll(1, 1)
    Wo = torch.randn(48, 96) * 0.05 * torch.where(torch.arange(96) % 17 == 0, 4.0, 1.0)
    Ho = Xo.T @ Xo / 4096
    e = {}
    for lab, kw in (('u', {}), ('h', {'rot_seed': 7}), ('lm', {'lm': True}), ('hlm', {'lm': True, 'rot_seed': 7})):
        dq, cd, sc, cb = gptq_matrix(Wo.bfloat16(), Ho, 2, 32, True, **kw)
        e[lab] = h_err(dq.float() - Wo.bfloat16().float(), Ho)
        if kw.get('lm'):
            assert cb.shape == (48, 4) and int(cd.min()) >= 0 and int(cd.max()) <= 3 and (cb.float().diff(dim=-1) >= 0).all(), 'codebook not 4 sorted entries a row'
        if kw.get('lm') and 'rot_seed' not in kw:
            assert torch.equal(dq, (cb.float().gather(1, cd.long()).view(48, 3, 32) * sc.float().view(48, 3, 1)).view(48, 96).bfloat16()), 'values are not cb[code] * scale'
    print('2 bits, outlier matrix, error under H:', {k: round(v, 4) for k, v in e.items()}, flush=True)
    assert e['h'] < e['u'] and e['lm'] < e['u'] and e['hlm'] < e['h'], e
    # the model: tiny qwen3_5, calibration and both variants end to end through kld_study's apply_variant
    cfg = Qwen3_5Config(text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                                         linear_conv_kernel_dim=4, max_position_embeddings=256,
                                         layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']),
                        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))
    model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():  # (a trained model's norms and weights are not at their initial values)
            if 'layernorm' in n:
                p.copy_(torch.randn_like(p) * 0.3)
            elif p.dim() == 2 and '.visual.' not in n:
                p.copy_(torch.randn_like(p) * 0.05)
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    seqs = torch.randint(0, 256, (16, 64), dtype=torch.int32)
    ids = seqs[:2].long()
    with torch.inference_mode():
        before = model(input_ids=ids, use_cache=False).logits.float()
    res = {}
    for label in ('gptq:4:32:bf16', 'gptq:4:32:bf16:act', 'awq:4:32:bf16', 'gptq:2:32:bf16:act', 'hgptq:2:32:bf16:act', 'lmgptq:2:32:bf16:act', 'hlmgptq:2:32:bf16:act',
                  'hgptq:3:32:bf16:act', 'hlmgptq:3:32:bf16:act'):
        spec = parse_variant(label)
        spec['seqs'] = seqs
        changed, by_group = apply_variant(model, originals, spec)
        mats = spec['stats']['mats']
        B = spec['bits']
        assert changed > 0 and by_group and all(0 < v[1] / v[0] <= B + 3.0 for v in by_group.values()), (label, by_group)  # (tiny rows: the codebook costs up to 2 bits a weight)
        if not spec.get('rot'):
            for n, p in model.named_parameters():  # every rounded matrix: at most 2q+1 (uniform) or 2^B (codebook) distinct values in a run of 32 (codes * one scale)
                if group_of(n):
                    runs = p.data.float().view(-1, 32).sort(-1).values
                    assert int((runs.diff(dim=-1) != 0).sum(-1).max()) + 1 <= (1 << B if spec.get('lm') else 2 * ((1 << (B - 1)) - 1) + 1), (label, n)
        res[label] = {m['name']: m['err'] for m in mats}
        if label.startswith(('gptq', 'lmgptq')) and B > 2:  # (at 2 bits GPTQ's three-value grid can lose a tiny matrix to plain's mse-fit scale: there, in all no worse)
            for m in mats:
                assert m['err'] <= m['err_plain'] * 1.0001, (label, m)
        else:  # (awq's grid holds no scaling, but the bf16 norm weight moves a tiny scale: per matrix within noise, in all no worse than plain)
            assert s_err_ok(mats), label
        with torch.inference_mode():
            after = model(input_ids=ids, use_cache=False).logits.float()
        kl = float(torch.nn.functional.kl_div(after.log_softmax(-1), before.log_softmax(-1), log_target=True, reduction='batchmean'))
        s = summary(mats)
        print(f'{label}: layers 4, matrices {s["matrices"]}, recon err {s["recon_err_method"]:.4g} vs plain {s["recon_err_plain"]:.4g} '
              f'(mean ratio {s["recon_err_ratio_mean"]:.3f}), bits/weight {sum(v[1] for v in by_group.values()) / sum(v[0] for v in by_group.values()):.3f}, '
              f'KL vs bf16 {kl:.4f}', flush=True)
        if label == 'hlmgptq:3:32:bf16:act':
            assert spec['nominal'] > 3.5, spec['nominal']
        print(f'    nominal bits/weight (fixed-width codes + scale + codebook + signs) {spec["nominal"]:.3f}', flush=True)
        if label == 'awq:4:32:bf16':
            assert any(m['err'] < m['err_plain'] for m in mats), 'awq never beat plain'
            g = [m for m in mats if 'o_proj' in m['name'] or 'out_proj' in m['name'] or 'down_proj' in m['name']]
            assert all(abs(m['err'] - m['err_plain']) <= 1e-9 * max(m['ref'], 1e-30) for m in g), 'unfolded layers should be plain'
    # GLY-127 ordering at 2 bits on calibration data: codebook <= uniform (in total and on most matrices), rotation <= plain on most matrices (the tiny model's
    # weights are random, so already incoherent: rotation has little to win here; the outlier matrix above is its test)
    names = list(res['gptq:2:32:bf16:act'])
    for a, b, need in (('lmgptq:2:32:bf16:act', 'gptq:2:32:bf16:act', 0.9), ('hlmgptq:2:32:bf16:act', 'hgptq:2:32:bf16:act', 0.9), ('hgptq:2:32:bf16:act', 'gptq:2:32:bf16:act', 0.4)):
        wins = sum(res[a][n] <= res[b][n] * 1.0001 for n in names)
        print(f'{a} <= {b}: {wins}/{len(names)} matrices, total {sum(res[a].values()):.4g} vs {sum(res[b].values()):.4g}', flush=True)
        assert wins >= need * len(names), (a, b)
    assert sum(res['lmgptq:2:32:bf16:act'].values()) < sum(res['gptq:2:32:bf16:act'].values())
    print('calib selftest ok')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('model_dir', type=Path); b.add_argument('out', type=Path)
    b.add_argument('--seqs', type=int, default=512); b.add_argument('--length', type=int, default=2048)
    z = sub.add_parser('quantize'); z.add_argument('model_dir', type=Path); z.add_argument('out', type=Path)
    z.add_argument('--bits', type=int, choices=(5, 6), required=True); z.add_argument('--seqs', type=int, default=512)
    c = sub.add_parser('check'); c.add_argument('packed', type=Path); c.add_argument('plan', type=Path); c.add_argument('codes', type=Path)
    sub.add_parser('selftest')
    a = ap.parse_args()
    {'build': cmd_build, 'quantize': cmd_quantize, 'check': cmd_check, 'selftest': lambda _: selftest()}[a.cmd](a)


if __name__ == '__main__':
    main()
