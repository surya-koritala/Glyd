"""Weight representations of Qwen3.8-27B against its BF16 outputs, on one 80 GB GPU: the KL divergence of each next-token distribution
from BF16's, the top token's agreement and the perplexity, over pinned WikiText-2 test windows, with each representation's bits a weight.

    python kld_study.py ref    MODEL_DIR OUT_DIR [--windows 256] [--length 1024] [--skip 32]
    python kld_study.py score  MODEL_DIR OUT_DIR VARIANT [VARIANT ...]          (MODEL_DIR: the BF16 snapshot; fp8 takes --fp8-dir)
    python kld_study.py selftest

The reference pass keeps, for every scored position, BF16's 64 likeliest tokens and their log probabilities, the target and BF16's top
token. A variant's KL divergence is taken over those 64 and the rest of the vocabulary as one tail bucket (a lower bound on the full
divergence, by the log-sum inequality; with 64 tokens the tail holds little mass). Variants (each restored from BF16 first):

    bf16                         the BF16 weights (a check: divergence 0)
    frac:F                       every BF16 parameter rounded to F fraction bits (nearest, ties to even; F=3 is Candidate A)
    frac:F@GROUP=G               as frac:F, with GROUP's matrices at G fraction bits (GROUPS below)
    gptq:B:BLOCK:bf16[:act]      GPTQ and AWQ rounding into the same block codes (calib.py, GLY-37): calibration sequences from --calib (default OUT_DIR/calib.pt, built
    awq:B:BLOCK:bf16             by calib.py build), the first --calib-seqs of them (512)
    hgptq / lmgptq / hlmgptq:B:BLOCK:bf16[:act]   GPTQ with a randomized Hadamard rotation of the inputs (h), a per-row Lloyd-Max codebook (lm), or both (calib.py, GLY-127)
    int:B:BLOCK:FIT[:bf16]       every matrix (GROUPS' tensors) as B-bit symmetric integer codes times an fp16 scale (bf16 with :bf16, as
                                 glyd-gpu's pack_int6 holds it) for each run of BLOCK values of a row; FIT is absmax (the scale covers the
                                 block's largest value) or mse (the clip of 16 that gives the least squared error); the other parameters
                                 stay BF16
    fp8                          the official FP8 checkpoint (--fp8-dir), dequantized as transformers does

Bits a weight are over the matrices: for frac and int, the empirical entropy of each tensor's codes (a lower bound for a lossless coder of
codes taken one at a time; Glyd's compact layout holds frac:3 at 6.774) plus the scales' 16 bits a block.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TOPK = 64
# the text decoder's matrices by kind, for frac:F@GROUP=G and the int variants (the conv1d, the norms, A_log and dt_bias are not matrices)
GROUPS = {
    'embed': r'language_model\.embed_tokens\.weight$',
    'head': r'(^|\.)lm_head\.weight$',
    'attn_qkv': r'self_attn\.(q_proj|k_proj|v_proj)\.weight$',
    'attn_o': r'self_attn\.o_proj\.weight$',
    'lin_in': r'linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)\.weight$',
    'lin_out': r'linear_attn\.out_proj\.weight$',
    'mlp_gate_up': r'mlp\.(gate_proj|up_proj)\.weight$',
    'mlp_down': r'mlp\.down_proj\.weight$',
}


def group_of(name):
    for g, pat in GROUPS.items():
        if re.search(pat, name) and '.visual.' not in name and 'mtp' not in name:
            return g
    return None


def round_frac(words, f):
    """BF16 words (int32, 0..65535) rounded to f of their 7 fraction bits, nearest and ties to even; a finite value that would round past the
    largest finite keeps the sign and saturates to the largest finite value with f fraction bits; infinities and NaNs unchanged."""
    import torch
    d = 7 - f
    if d == 0:
        return words
    mask = (1 << d) - 1
    rounded = (words + ((1 << (d - 1)) - 1) + ((words >> d) & 1)) & (0xFFFF ^ mask)
    finite = (words & 0x7F80) != 0x7F80
    overflow = (rounded & 0x7FFF) >= 0x7F80
    rounded = torch.where(overflow, (words & 0x8000) | (0x7F7F & (0xFFFF ^ mask)), rounded)
    return torch.where(finite, rounded, words)


def frac_tensor(t, f):
    """A BF16 (or BF16-valued fp32) tensor rounded to f fraction bits, in its dtype, and the entropy in bits of its rounded words."""
    import torch
    b = t.to(torch.bfloat16)
    words = b.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    r = round_frac(words, f)
    hist = torch.bincount(r.view(-1), minlength=65536).double()
    p = hist[hist > 0] / r.numel()
    ent = float(-(p * torch.log2(p)).sum())
    out = r.to(torch.int16).view(torch.bfloat16).to(t.dtype)
    return out, ent


def fit_scale(w, q, fit, scale_bf16=False):
    """The scale of each run of w (..., BLOCK): the largest value over q (absmax), or of 16 clips of 0.70..1.0 that of least squared error (mse);
    held in fp16, or bf16 as the engine does."""
    import torch
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    ratios = torch.linspace(0.70, 1.0, 16, device=w.device) if fit == 'mse' else torch.ones(1, device=w.device)
    best_err, best_scale = None, None
    for ratio in ratios:
        scale = (amax * ratio / q).bfloat16().float() if scale_bf16 else (amax * ratio / q).half().float().clamp_min(6e-8)
        err = ((torch.clamp(torch.round(w / scale), -q, q) * scale - w) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, scale
        else:
            better = err < best_err
            best_err, best_scale = torch.where(better, err, best_err), torch.where(better, scale, best_scale)
    return best_scale


def int_tensor(t, bits, block, fit, rows_at_once=4096, scale_bf16=False):
    """A matrix as `bits`-bit symmetric codes (-q..q, q = 2^(bits-1)-1) times an fp16 scale for each run of `block` values of a row; the
    dequantized matrix in t's dtype, and the entropy in bits of its codes."""
    import torch
    out_rows, cols = t.shape
    if cols % block:
        raise ValueError(f'{cols} columns are not a multiple of {block}')
    q = (1 << (bits - 1)) - 1
    out = torch.empty_like(t)
    hist = torch.zeros(2 * q + 1, dtype=torch.float64, device=t.device)
    for r0 in range(0, out_rows, rows_at_once):
        w = t[r0:r0 + rows_at_once].float().view(-1, cols // block, block)
        best_scale = fit_scale(w, q, fit, scale_bf16)
        codes = torch.clamp(torch.round(w / best_scale), -q, q)
        out[r0:r0 + rows_at_once] = (codes * best_scale).view(-1, cols).to(t.dtype)
        hist += torch.bincount((codes + q).to(torch.int64).view(-1), minlength=2 * q + 1).double()
    p = hist[hist > 0] / hist.sum()
    return out, float(-(p * torch.log2(p)).sum()) + 16.0 / block


def parse_variant(v):
    if v in ('bf16', 'fp8'):
        return {'kind': v}
    m = re.fullmatch(r'frac:(\d)((?:@\w+=\d)*)', v)
    if m:
        over = dict((g, int(x)) for g, x in re.findall(r'@(\w+)=(\d)', m.group(2)))
        if any(g not in GROUPS for g in over) or not 0 <= int(m.group(1)) <= 7:
            raise ValueError(f'variant {v}: groups are {", ".join(GROUPS)}')
        return {'kind': 'frac', 'f': int(m.group(1)), 'groups': over}
    m = re.fullmatch(r'int:(\d):(\d+):(absmax|mse)(:bf16)?', v)
    if m:
        return {'kind': 'int', 'bits': int(m.group(1)), 'block': int(m.group(2)), 'fit': m.group(3), 'bf16': bool(m.group(4))}
    m = re.fullmatch(r'(awq|(h)?(lm)?gptq):(\d):(\d+):bf16(:act)?', v)
    if m and (m.group(1) != 'awq' or not m.group(6)):  # (calib.py; h: rotation, lm: Lloyd-Max codebook, GLY-127)
        kind = 'awq' if m.group(1) == 'awq' else 'gptq'
        return {'kind': kind, 'bits': int(m.group(4)), 'block': int(m.group(5)), 'bf16': True, **({'act': bool(m.group(6))} if kind == 'gptq' else {}),
                **({'rot': True} if m.group(2) else {}), **({'lm': True} if m.group(3) else {})}
    raise ValueError(f'variant {v}: bf16, fp8, frac:F, frac:F@GROUP=G, int:B:BLOCK:absmax|mse, [h][lm]gptq:B:BLOCK:bf16[:act], awq:B:BLOCK:bf16')


def windows(tokenizer, count, length):
    """Disjoint `length`-token windows evenly spaced through the pinned WikiText-2 test text (prepare_quality.py's source, checked by hash)."""
    import urllib.request
    import pyarrow as pa
    import pyarrow.parquet as pqt
    from prepare_quality import DATASET, DATA_REV, DATA_FILE, DATA_SHA
    data = urllib.request.urlopen(f'https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REV}/{DATA_FILE}', timeout=90).read(2 << 20)
    if hashlib.sha256(data).hexdigest() != DATA_SHA:
        raise RuntimeError('held-out dataset hash mismatch')
    text = '\n\n'.join(pqt.read_table(pa.BufferReader(data)).column('text').to_pylist())
    ids = tokenizer.encode(text, add_special_tokens=False)
    count = min(count, len(ids) // length)
    step = (len(ids) - length) // max(count - 1, 1)
    return [ids[i * step:i * step + length] for i in range(count)], len(ids)


def load(model_dir, fp8=False):
    import torch
    from transformers import AutoModelForImageTextToText, FineGrainedFP8Config
    extra = {'quantization_config': FineGrainedFP8Config(dequantize=True)} if fp8 else {}
    return AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
                                                       trust_remote_code=False, local_files_only=True, **extra).eval()


def forward_logp(model, ids):
    import torch
    x = torch.tensor([ids], device='cuda')
    with torch.inference_mode():
        logits = model(input_ids=x, use_cache=False).logits[0].float()
    return torch.log_softmax(logits, dim=-1)


def cmd_ref(a):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True)
    wins, total = windows(tok, a.windows, a.length)
    model = load(a.model_dir)
    tops, logps, targets, t0 = [], [], [], time.time()
    for w in wins:
        lp = forward_logp(model, w)[a.skip - 1:-1]
        v, i = lp.topk(TOPK, dim=-1)
        tops.append(i.int().cpu()); logps.append(v.cpu()); targets.append(torch.tensor(w[a.skip:]).int())
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({'windows': wins, 'skip': a.skip, 'top_ids': torch.cat(tops), 'top_logp': torch.cat(logps), 'targets': torch.cat(targets)}, a.out / 'ref.pt')
    meta = {'windows': len(wins), 'length': a.length, 'skip': a.skip, 'positions': int(sum(len(t) for t in targets)), 'dataset_tokens': total,
            'seconds': round(time.time() - t0, 1)}
    (a.out / 'ref.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta), flush=True)


def apply_variant(model, originals, spec):
    """Restores every transformed parameter from `originals`, then applies `spec`: (values changed, the matrices' bits a weight, by group)."""
    import torch
    by_group, changed = {}, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in originals:
                p.copy_(originals[name].to(p.device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if spec['kind'] == 'bf16':
            return 0, {}
        if spec['kind'] in ('gptq', 'awq'):  # rounded layer by layer on the calibration sequences spec['seqs'] (calib.py)
            from calib import run, summary
            by_group, mats = run(model, spec, spec['seqs'], next(model.parameters()).device)
            spec['stats'] = {'mats': mats, **summary(mats), 'nominal_bits_a_weight': spec.get('nominal')}
            return sum(int((p.data != originals[n].to(p.device)).sum()) for n, p in model.named_parameters() if n in originals and group_of(n)), by_group
        for name, p in model.named_parameters():
            g = group_of(name)
            if spec['kind'] == 'frac':
                if name not in originals:
                    continue
                f = spec['groups'].get(g, spec['f']) if g else spec['f']
                new, ent = frac_tensor(p.data, f)
            elif g is None:
                continue
            else:
                new, ent = int_tensor(p.data, spec['bits'], spec['block'], spec['fit'], scale_bf16=spec.get('bf16', False))
            changed += int((new != p.data).sum())
            p.data.copy_(new)
            if g:
                n, bits = by_group.get(g, (0, 0.0))
                by_group[g] = (n + p.numel(), bits + ent * p.numel())
    return changed, by_group


def score_one(model, ref, label, seconds_transform, changed, by_group, out, extra=None):
    import torch
    kl_sum, agree, nll, n, kls = 0.0, 0, 0.0, 0, []
    at, t0 = 0, time.time()
    for w in ref['windows']:
        lp = forward_logp(model, w)[ref['skip'] - 1:-1]
        m = lp.shape[0]
        ids, rp = ref['top_ids'][at:at + m].to('cuda').long(), ref['top_logp'][at:at + m].to('cuda')
        tg = ref['targets'][at:at + m].to('cuda').long()
        qk = lp.gather(1, ids)
        p = rp.exp()
        p_tail = (1 - p.sum(-1)).clamp_min(1e-12)
        q_tail = (1 - qk.exp().sum(-1)).clamp_min(1e-12)
        kl = (p * (rp - qk)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())
        kls.append(kl.cpu())
        agree += int((lp.argmax(-1) == ids[:, 0]).sum())
        nll -= float(lp.gather(1, tg[:, None]).sum())
        n += m
        at += m
    k = torch.cat(kls).double()
    mats = sum(v[0] for v in by_group.values())
    row = {'variant': label, 'positions': n, 'kl_mean': float(k.mean()), 'kl_median': float(k.median()), 'kl_p99': float(k.quantile(0.99)),
           'kl_p999': float(k.quantile(0.999)), 'kl_max': float(k.max()), 'top1_agreement': agree / n, 'perplexity': math.exp(nll / n),
           'bits_a_weight': (sum(v[1] for v in by_group.values()) / mats) if mats else None,
           'by_group_bits': {g: round(v[1] / v[0], 4) for g, v in sorted(by_group.items())}, 'values_changed': changed,
           'seconds': {'transform': round(seconds_transform, 1), 'score': round(time.time() - t0, 1)}, **(extra or {})}
    with open(out / 'results.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: row[k] for k in ('variant', 'kl_mean', 'kl_p99', 'top1_agreement', 'perplexity', 'bits_a_weight')}), flush=True)


def cmd_score(a):
    import torch
    ref = torch.load(a.out / 'ref.pt', weights_only=True)
    specs = [(v, parse_variant(v)) for v in a.variants]
    if any(s['kind'] == 'fp8' for _, s in specs) and len(specs) > 1:
        raise ValueError('fp8 is scored alone (its own model)')
    if specs[0][1]['kind'] == 'fp8':
        model = load(a.fp8_dir, fp8=True)
        score_one(model, ref, 'fp8', 0.0, 0, {}, a.out)
        return
    model = load(a.model_dir)
    # every BF16-held parameter of the text model, kept on the host to restore before each variant
    originals = {n: p.detach().to('cpu', copy=True) for n, p in model.named_parameters()
                 if p.dtype in (torch.bfloat16, torch.float32) and '.visual.' not in n and 'mtp' not in n}
    for label, spec in specs:
        t0 = time.time()
        if spec['kind'] in ('gptq', 'awq'):
            spec['seqs'] = torch.load(a.calib or a.out / 'calib.pt', weights_only=True)[:a.calib_seqs]
        changed, by_group = apply_variant(model, originals, spec)
        extra = {'calibration': {'sequences': len(spec['seqs']), **{k: v for k, v in spec['stats'].items() if k != 'mats'}}} if 'seqs' in spec else None
        score_one(model, ref, label, time.time() - t0, changed, by_group, a.out, extra)


def selftest():
    import torch
    from matched_quality import round_candidate_a
    words = torch.arange(65536, dtype=torch.int32)
    bf = words.to(torch.int16).view(torch.bfloat16)
    a3 = (round_candidate_a(bf).view(torch.int16).to(torch.int32) & 0xFFFF)
    assert torch.equal(round_frac(words, 3), a3), 'frac:3 is not Candidate A'
    assert torch.equal(round_frac(words, 7), words)
    for f in range(7):
        r = round_frac(words, f)
        finite = (words & 0x7F80) != 0x7F80
        assert int((r[finite] & ((1 << (7 - f)) - 1)).abs().sum()) == 0, f  # (no fraction bit below f left)
        normal = finite & ((words & 0x7F80) != 0)  # (a subnormal's step is not relative to its value)
        v, rv = bf[normal].float(), r[normal].to(torch.int16).view(torch.bfloat16).float()
        assert float(((rv - v).abs() / v.abs()).max()) <= 2.0 ** -(f + 1) * 1.0001, f  # (half a step at most, saturation included)
    w = torch.randn(256, 512, dtype=torch.bfloat16)
    for bits in (4, 5, 6):
        for fit in ('absmax', 'mse'):
            d, ent = int_tensor(w, bits, 32, fit)
            q = (1 << (bits - 1)) - 1
            amax = w.float().view(256, 16, 32).abs().amax(-1, keepdim=True)
            err = (d.float().view(256, 16, 32) - w.float().view(256, 16, 32)).abs()
            if fit == 'absmax':
                assert float((err / amax).max()) <= 0.5 / q * 1.001 + 2.0 ** -8, (bits, fit)  # (half a step, and the BF16 rounding of the dequantized value)
            assert 0 < ent <= bits + 0.5 + 1e-9, (bits, fit, ent)
        assert float(((int_tensor(w, bits, 32, 'mse')[0].float() - w.float()) ** 2).sum()) <= float(((int_tensor(w, bits, 32, 'absmax')[0].float() - w.float()) ** 2).sum()) * 1.0001
    # :bf16 holds the scale in bf16 as pack_int6 does; with absmax: s = round_bf16(amax / 31), codes clamp(round(w / s), -31, 31), value round_bf16(code * s)
    # (pack_int6's rule is int:6:32:mse:bf16: the same codes and values, s the one of 16 ratios of amax / 31 with the least squared error)
    d6, _ = int_tensor(w, 6, 32, 'absmax', scale_bf16=True)
    wf = w.float().view(256, 16, 32)
    s6 = (wf.abs().amax(-1, keepdim=True) / 31).bfloat16().float()
    ref = (torch.clamp(torch.round(wf / s6), -31, 31) * s6).view(256, 512).bfloat16()
    assert torch.equal(d6, ref), 'int:6:32:absmax:bf16 is not s = round_bf16(amax / 31)'
    assert parse_variant('int:6:32:absmax:bf16') == {'kind': 'int', 'bits': 6, 'block': 32, 'fit': 'absmax', 'bf16': True}
    assert parse_variant('hlmgptq:2:32:bf16:act') == {'kind': 'gptq', 'bits': 2, 'block': 32, 'act': True, 'bf16': True, 'rot': True, 'lm': True}
    assert parse_variant('hgptq:3:32:bf16') == {'kind': 'gptq', 'bits': 3, 'block': 32, 'act': False, 'bf16': True, 'rot': True}
    assert parse_variant('lmgptq:2:32:bf16:act')['lm'] and 'rot' not in parse_variant('lmgptq:2:32:bf16:act')
    try:
        parse_variant('hawq:3:32:bf16')
        raise AssertionError('hawq parsed')
    except ValueError:
        pass
    assert parse_variant('frac:2@attn_o=3@head=3') == {'kind': 'frac', 'f': 2, 'groups': {'attn_o': 3, 'head': 3}}
    assert group_of('model.language_model.layers.3.mlp.down_proj.weight') == 'mlp_down' and group_of('lm_head.weight') == 'head'
    assert group_of('model.language_model.layers.3.linear_attn.in_proj_qkv.weight') == 'lin_in' and group_of('model.visual.blocks.0.mlp.down_proj.weight') is None
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('ref'); r.add_argument('model_dir', type=Path); r.add_argument('out', type=Path)
    r.add_argument('--windows', type=int, default=256); r.add_argument('--length', type=int, default=1024); r.add_argument('--skip', type=int, default=32)
    s = sub.add_parser('score'); s.add_argument('model_dir', type=Path); s.add_argument('out', type=Path); s.add_argument('variants', nargs='+')
    s.add_argument('--fp8-dir', type=Path); s.add_argument('--calib', type=Path); s.add_argument('--calib-seqs', type=int, default=512)
    sub.add_parser('selftest')
    a = ap.parse_args()
    {'ref': cmd_ref, 'score': cmd_score, 'selftest': lambda _: selftest()}[a.cmd](a)


if __name__ == '__main__':
    main()
