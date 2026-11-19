"""PyTorch reference of Nemotron 3 Super's Mamba-2, attention and latent mixture-of-experts layers for glyd-engine's NemotronH path (Linear GLY-132).

    nemotron_ref.py selftest                         the reference against transformers' own functions and modules on random small tensors (CPU; needs transformers 5.19.0)
    nemotron_ref.py fetch OUT_DIR [--layers 0,2,7] [--tokens 333] [--seed 1]
                                                     a mini checkpoint of those layers' tensors (HTTP range reads of the pinned revision: no token, no whole shards) with the real
                                                     config.json, x.bin (the embeddings of random tokens, bf16 [tokens, 4096]: the layers' residual input) and OUT_DIR/toy, a model of
                                                     those layers (renumbered) with part of the real embedding and head
    nemotron_ref.py kernels gen DIR [--nh 16 --ng 2 --n 300 --split 150 --na 40 --seed 1] [--from MINI]
                                                     random inputs in DIR/in for `glyd-engine nhkernels DIR/in DIR/out` (a GPU) or tools/nh_emu.cpp's program (any computer); with --from, the
                                                     real activations of MINI's first Mamba-2 and attention layers instead (the model's heads and groups)
    nemotron_ref.py kernels check DIR                DIR/out against the reference, stage by stage
    nemotron_ref.py layers gen MINI [IO] [--level 6|5|4] [--kinds M*E]
                                                     IO/spec.json (default MINI/io): the cases `glyd-engine nhlayers MINI IO` runs on the mini checkpoint's layers (the kinds named)
    nemotron_ref.py layers check MINI [IO] [--dt-min 0.001]
                                                     the engine's dumps in IO against the reference, case by case and stage by stage
    nemotron_ref.py toy MINI                         `glyd-engine nhtoy MINI/toy MINI/toy`'s logits against the reference (top-1 token at least 98% the same; where the toy has mixture-of-experts layers, dropping them must move the logits)
    nemotron_ref.py moetoy MINI [--pattern ME*EME] [--level 6|5|4] [--seed 1] [--tokens 300] [--prefill 150] [--total 210]
                                                     a random tiny NemotronH of that pattern (M Mamba-2, * attention, E latent mixture of experts) with no Hub read: MINI as `fetch` makes it (the layers' tensors, x.bin) and MINI/toy;
                                                     the experts, the latent projections and the shared expert are on the grid of the six-, five- or four-bit block layout (--level; at 4 the experts alone are four-bit codes and the
                                                     latent projections and the shared expert six-bit ones, as the engine holds them at wren), so that the engine's packs hold them exactly
    nemotron_ref.py real MINI                        the reference against transformers' own Mamba-2 and attention modules on the mini checkpoint's real weights (CPU; needs transformers 5.19.0)
    nemotron_ref.py inventory OUT.json               the checkpoint's tensors from the shards' headers, as tests/fixtures/nemotron_h_tensors.json holds them
    nemotron_ref.py ids MODEL OUT [--prefill 64] [--total 160] [--level 5|4|6]
                                                     OUT/ids.txt and OUT/toy.txt for `glyd-engine nhtoy PACKED OUT` on the whole model: the first TOTAL tokens of a text (MODEL's tokenizer), PREFILL of them prefilled
    nemotron_ref.py full MODEL OUT                   OUT/toy_*_logits.bin (`glyd-engine nhtoy PACKED OUT` on the whole model packed at a level) against the bf16 checkpoint MODEL run layer by layer on the GPU (the
                                                     layers' weights read when asked: the model does not fit one): top-1 tokens and KL divergence, which are the level's cost, not a layer's error
    A layer of a glyd-v7 directory is checked against the weights its packs hold: `glyd-engine nhdecode PACKED MINI --layers 0,1,7` writes them as a mini checkpoint (and x.bin), then `layers gen`, `glyd-engine
    nhlayers PACKED IO` (with MINI/x.bin beside PACKED's files), `layers check MINI IO`.

The reference is vLLM's (the engine's correctness reference) written as plain torch, rounding to bf16 where vLLM stores a bf16 tensor: the projections' outputs, the convolution's output, the
scan's output, the gated norm's normed value and its result. The state is fp32. dt = softplus(dt + dt_bias), no lower limit (vLLM passes dt_limit=(0, inf); transformers 5.19.0's chunk path clamps dt to
time_step_min = 0.001, its recurrent step does not: `--dt-min 0.001` shows what that does). Each stage of a case is checked from the engine's own input to that stage, so a difference
is the stage's own; the layer's output is also checked from the layer's input through the reference alone (a looser bound: the engine's and torch's products differ in the last place).

A mixture-of-experts layer is transformers' NemotronHMoE with the roundings of src/moe.rs: the router in fp32 over the model's width (sigmoid scores; the k largest of scores + e_score_correction_bias, equal ones the smaller
expert; their scores, not the biased ones, divided by their sum + 1e-20 and times routed_scaling_factor; n_group = topk_group = 1, which the engine's config refuses to differ from), the latent projection down in bf16, each
chosen expert's up product and relu squared stored in bf16, its down product in fp32 times the pair's weight, summed in fp32 and rounded once, the projection up, and the shared expert (bf16, on the model's
width) added in fp32 and rounded once. A case's choices are checked against the engine's (a token whose experts differ is fine where the two sets' scores are equal to 1e-6), the stages after them on the engine's own.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import urllib.request

import torch
import torch.nn.functional as F

REPO = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
REV = "2dc98e2afe4face0e4ce40972a915c45368bd34a"
BF = torch.bfloat16
P, N = 64, 128  # a head's channels, the state's size: what the kernels are written for
CHUNK = 128


def bf(x):
    return x.to(BF)


def f32(x):
    return x.to(torch.float32)


# ---- the layer's pieces -------------------------------------------------------------------------------------------------------------------------------------------------------------

def rmsnorm(x, w, eps):
    """NemotronHRMSNorm / vLLM's RMSNorm: fp32 inside, the weight multiplied in fp32, one rounding."""
    xf = f32(x)
    return bf(f32(w) * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)))


def softplus(x):
    """torch's softplus (threshold 20)."""
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def conv_silu(x, w, b, state):
    """The causal depthwise convolution of kernel K and SiLU, rounded once. x [T, C] bf16, w [C, 1, K], b [C], state [K-1, C] (the last K-1 inputs, oldest first) -> (y [T, C], new state)."""
    T, C = x.shape
    K = w.shape[-1]
    xin = torch.cat([state, x], 0)
    acc = f32(b).expand(T, C).clone()
    for k in range(K):
        acc = acc + f32(w[:, 0, k]) * f32(xin[k:k + T])
    return bf(F.silu(acc)), xin[T:].clone()


def dt_of(dtr, dt_bias, dt_min):
    """dt [T, H] fp32 from the in_proj's dt columns (bf16) and dt_bias."""
    return torch.clamp(softplus(f32(dtr) + f32(dt_bias)), min=dt_min)


def ssm_recurrent(x, B, C, dtr, dt_bias, A_log, D, S, dt_min=0.0):
    """The selective state update token by token (vLLM's selective_state_update). x [T, H, P], B and C [T, G, N] (bf16), dtr [T, H]; S [H, P, N] fp32, updated in place -> y [T, H, P] bf16."""
    T, H, _ = x.shape
    G = B.shape[1]
    A = -torch.exp(f32(A_log))
    dt = dt_of(dtr, dt_bias, dt_min)
    rep = H // G
    out = torch.empty(T, H, P, dtype=torch.float32, device=x.device)
    for t in range(T):
        Bh = f32(B[t]).repeat_interleave(rep, 0)  # [H, N]
        Ch = f32(C[t]).repeat_interleave(rep, 0)
        xt = f32(x[t])
        S.mul_(torch.exp(dt[t] * A)[:, None, None]).add_((dt[t][:, None] * xt)[:, :, None] * Bh[:, None, :])
        out[t] = (S * Ch[:, None, :]).sum(-1) + f32(D)[:, None] * xt
    return bf(out)


def segsum(a):
    """[..., L] -> [..., L, L]: out[l, s] = sum of a[s+1..l] for s <= l, else -inf."""
    L = a.shape[-1]
    c = torch.cumsum(a, -1)
    d = c[..., :, None] - c[..., None, :]
    return d.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool, device=a.device)), -math.inf)


def ssm_chunked(x, B, C, dtr, dt_bias, A_log, D, S, dt_min=0.0, chunk=CHUNK):
    """The state-space-duality chunked scan (transformers' mamba2_chunk_scan; the engine's nh_ssd_* kernels): same arguments as ssm_recurrent; S [H, P, N] is the state at the start and is
    left holding the state at the end. Everything fp32."""
    T, H, _ = x.shape
    G = B.shape[1]
    rep = H // G
    pad = (-T) % chunk
    A = -torch.exp(f32(A_log))
    dt = dt_of(dtr, dt_bias, dt_min)  # [T, H]
    xf, Bf, Cf = f32(x), f32(B).repeat_interleave(rep, 1), f32(C).repeat_interleave(rep, 1)
    pd = lambda t: F.pad(t, (0, 0) * (t.dim() - 1) + (0, pad))  # pad the first dim
    xd = pd(xf * dt[:, :, None])  # x * dt
    a = pd(dt * A)  # [Tp, H]
    Bf, Cf, xp = pd(Bf), pd(Cf), pd(xf)
    nc = (T + pad) // chunk
    ch = lambda t: t.reshape(nc, chunk, *t.shape[1:])
    xd, a, Bf, Cf = ch(xd), ch(a), ch(Bf), ch(Cf)  # [c, l, H, ...]
    a = a.permute(0, 2, 1)  # [c, H, l]
    acum = torch.cumsum(a, -1)
    Lm = torch.exp(segsum(a))  # [c, H, l, s]
    Gm = torch.einsum("clhn,cshn->clsh", Cf, Bf)
    Mm = Gm * Lm.permute(0, 2, 3, 1)
    y_diag = torch.einsum("clsh,cshp->clhp", Mm, xd)
    decay = torch.exp(acum[:, :, -1:] - acum)  # [c, H, l]
    states = torch.einsum("clhn,clhp->chpn", Bf * decay.permute(0, 2, 1)[..., None], xd)
    start = [S.clone()]
    for c in range(nc):
        start.append(torch.exp(acum[c, :, -1])[:, None, None] * start[-1] + states[c])
    S.copy_(start[-1])
    start = torch.stack(start[:-1])  # [c, H, P, N]: the state each chunk starts from
    y_off = torch.einsum("clhn,chpn->clhp", Cf, start) * torch.exp(acum).permute(0, 2, 1)[..., None]
    y = (y_diag + y_off).reshape(nc * chunk, H, P) + f32(D)[None, :, None] * xp
    return bf(y[:T])


def gated_norm(y, z, w, eps, groups):
    """vLLM's Mixer2RMSNormGated.forward_native: x = y * silu(z) in fp32, normed over each group, rounded to bf16, times the weight, rounded again. y, z [T, D] bf16."""
    T, Dm = y.shape
    g = f32(y) * F.silu(f32(z))
    gv = g.reshape(T, groups, Dm // groups)
    n = bf((gv * torch.rsqrt(gv.pow(2).mean(-1, keepdim=True) + eps)).reshape(T, Dm))
    return bf(f32(w) * f32(n))


def rmsnorm_add(x, y, w, eps):
    """x + y in fp32 rounded to bf16 (the residual stream), and the norm of the fp32 sum (vLLM's fused add and norm, the engine's rmsnorm_add) -> (x', norm)."""
    z = f32(x) + f32(y)
    return bf(z), bf(f32(w) * (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)))


def mamba_mixer(h, W, cfg, st, chunked=True, dt_min=0.0, stages=None):
    """One layer's Mamba-2 mixer on h [T, hidden] bf16 (the normed input), continuing from st = {conv [3, C] bf16, ssm [H, P, N] fp32} (updated in place) -> [T, hidden] bf16.
    stages (a dict) takes each stage's output."""
    H, G, di = cfg["heads"], cfg["groups"], cfg["heads"] * P
    mix = F.linear(h, W["in_proj"])
    z, xbc, dtr = mix[:, :di], mix[:, di:di + di + 2 * G * N], mix[:, di + di + 2 * G * N:]
    cv, st["conv"] = conv_silu(xbc, W["conv_w"], W["conv_b"], st["conv"])
    xs, B, C = cv[:, :di].reshape(-1, H, P), cv[:, di:di + G * N].reshape(-1, G, N), cv[:, di + G * N:].reshape(-1, G, N)
    scan = ssm_chunked if chunked else ssm_recurrent
    y = scan(xs, B, C, dtr, W["dt_bias"], W["A_log"], W["D"], st["ssm"], dt_min).reshape(-1, di)
    gn = gated_norm(y, z, W["gnorm"], cfg["eps"], G)
    out = F.linear(gn, W["out_proj"])
    if stages is not None:
        stages.update(mix=mix, conv=cv, y=y, gn=gn, out=out)
    return out


def mamba_layer(x, W, cfg, st, chunked=True, dt_min=0.0, stages=None):
    """The layer on its residual input x: the norm, then the mixer."""
    return mamba_mixer(rmsnorm(x, W["norm"], cfg["eps"]), W, cfg, st, chunked, dt_min, stages)


def attn_core(q, k, v):
    """Causal attention of q [T, nq, d] over k, v [S, nk, d] (the T rows are the last T positions): the scores in fp32, the probabilities rounded to bf16 for the values, one rounding of the
    output [T, nq * d]. No positional encoding, no window, scale d^-0.5."""
    T, nq, d = q.shape
    S, nk, _ = k.shape
    kr, vr = f32(k).repeat_interleave(nq // nk, 1), f32(v).repeat_interleave(nq // nk, 1)  # [S, nq, d]
    sc = torch.einsum("tqd,sqd->qts", f32(q), kr) / math.sqrt(d)
    mask = torch.arange(S, device=q.device)[None, :] > (torch.arange(T, device=q.device) + (S - T))[:, None]
    sc = sc.masked_fill(mask[None], -math.inf)
    p = torch.exp(sc - sc.amax(-1, keepdim=True))  # (as FlashAttention: the unnormalized probabilities rounded for the values, the sum in fp32, one division)
    return bf(torch.einsum("qts,sqd->tqd", bf(p).float(), vr) / p.sum(-1).transpose(0, 1)[..., None]).reshape(T, nq * d)


def attention_mixer(h, W, cfg, kv=None, stages=None):
    """One layer's attention mixer (no positional encoding, no sliding window) on h [T, hidden] (the normed input), after kv = (k, v) of the earlier positions [S, nk, d] (or None) -> out, new kv."""
    nq, nk, d = cfg["att_heads"], cfg["kv_heads"], cfg["att_dim"]
    q = F.linear(h, W["q_proj"]).reshape(-1, nq, d)
    k = F.linear(h, W["k_proj"]).reshape(-1, nk, d)
    v = F.linear(h, W["v_proj"]).reshape(-1, nk, d)
    T = q.shape[0]
    if kv is not None:
        k, v = torch.cat([kv[0], k]), torch.cat([kv[1], v])
    o = attn_core(q, k, v)
    out = F.linear(o, W["o_proj"])
    if stages is not None:
        stages.update(qkv=torch.cat([q.reshape(T, -1), k[-T:].reshape(T, -1), v[-T:].reshape(T, -1)], 1), attn=o, out=out)
    return out, (k, v)


def attention_layer(x, W, cfg, kv=None, stages=None):
    return attention_mixer(rmsnorm(x, W["norm"], cfg["eps"]), W, cfg, kv, stages)


def relu2(x):
    """relu(x) squared (ACT2FN["relu2"]), rounded once to bf16."""
    return bf(F.relu(f32(x)).pow(2))


def moe_weights(s, idx, scale, norm_topk=True):
    """The weights of the chosen experts idx [T, k] from the scores s [T, E]: the chosen scores (not the biased ones) divided by their sum + 1e-20 where norm_topk, times scale."""
    w = s.gather(1, idx)
    if norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * scale


def moe_route(h, gate, bias, k, scale, norm_topk=True):
    """NemotronHTopkRouter on h [T, hidden] (n_group = topk_group = 1): fp32 logits over the model's width, sigmoid scores, the k largest of scores + bias (largest first, equal ones the smaller expert),
    their weights (`moe_weights`) -> (experts [T, k] int64, weights [T, k] fp32)."""
    s = torch.sigmoid(F.linear(f32(h), f32(gate)))
    idx = torch.sort(s + f32(bias), dim=-1, descending=True, stable=True).indices[:, :k]
    return idx, moe_weights(s, idx, scale, norm_topk)


def moe_experts(z, idx, w, up, down):
    """The chosen experts' down(relu2(up(z))) on z [T, latent], weighted and summed (NemotronHExperts, with src/moe.rs's roundings): each expert on the tokens that chose it, the up product and the activation
    stored in bf16, the down product in fp32 times the pair's weight w [T, k], the pairs summed in fp32, one rounding. up [E, inter, latent], down [E, latent, inter] -> [T, latent] bf16."""
    y = torch.zeros(z.shape[0], down.shape[1], dtype=torch.float32, device=z.device)
    for e in idx.unique().tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        y.index_add_(0, tok, F.linear(f32(relu2(F.linear(z[tok], up[e]))), f32(down[e])) * w[tok, slot, None])
    return bf(y)


def moe_shared(h, W):
    """The shared expert (NemotronHMLP on the model's width): down(relu2(up(h))), bf16 throughout."""
    return F.linear(relu2(F.linear(h, W["s_up"])), W["s_down"])


def moe_mixer(h, W, cfg, stages=None):
    """One layer's latent mixture-of-experts mixer on h [T, hidden] bf16 (the normed input) -> [T, hidden] bf16: z = fc1_latent_proj(h); y = the experts' weighted sum of z; fc2_latent_proj(y) + the shared
    expert's output (a sum in fp32, rounded once). stages (a dict) takes each stage's output (`ids` and `wts`: the choice)."""
    idx, w = moe_route(h, W["gate"], W["bias"], cfg["top_k"], cfg["scale"], cfg["norm_topk"])
    z = F.linear(h, W["fc1"])
    y = moe_experts(z, idx, w, W["up"], W["down"])
    out = bf(f32(F.linear(y, W["fc2"])) + f32(moe_shared(h, W)))
    if stages is not None:
        stages.update(z=z, y=y, out=out, ids=idx, wts=w)
    return out


def moe_layer(x, W, cfg, stages=None):
    return moe_mixer(rmsnorm(x, W["norm"], cfg["eps"]), W, cfg, stages)


def moe_choice(h, W, cfg, ids, wts, name="  router"):
    """The engine's experts ids [T, k] and weights wts against the reference's on h: a token whose experts differ is fine where the two sets' scores (+ bias) are equal to 1e-6 (a tie that falls the other way,
    as src/kcheck_moe.rs allows); the weights the reference makes of the engine's experts against the engine's (bound 1e-5). Returns whether both hold."""
    s = torch.sigmoid(F.linear(f32(h), f32(W["gate"])))
    choose = s + f32(W["bias"])
    mine, _ = moe_route(h, W["gate"], W["bias"], cfg["top_k"], cfg["scale"], cfg["norm_topk"])
    bad = ties = 0
    for t in range(h.shape[0]):
        if torch.equal(mine[t], ids[t]):
            continue
        if bool(((ids[t] < 0) | (ids[t] >= choose.shape[1])).any()):
            bad += 1
            continue
        a, b = choose[t, ids[t]].sort(descending=True).values, choose[t, mine[t]].sort(descending=True).values
        ties, bad = (ties + 1, bad) if torch.allclose(a, b, rtol=1e-6, atol=1e-12) else (ties, bad + 1)
    good = bad == 0
    print(f"{'ok  ' if good else 'FAIL'} {name}: the reference's experts equal the engine's for {h.shape[0] - bad - ties}/{h.shape[0]} tokens" + (f" (+{ties} by a tie the other way)" if ties else ""))
    return stats(f"{name}: weights of the engine's experts", wts, moe_weights(s, ids.clamp(min=0), cfg["scale"], cfg["norm_topk"]), 1e-5) and good


# ---- comparing -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def ulps(a, b):
    """bf16 places between two bf16 tensors: (largest, share equal)."""
    key = lambda t: torch.where((t.view(torch.int16) < 0), -(t.view(torch.int16).int() & 0x7FFF), t.view(torch.int16).int())
    d = (key(a.contiguous()) - key(b.contiguous())).abs()
    return int(d.max()) if d.numel() else 0, float((d == 0).float().mean()) if d.numel() else 1.0


def stats(name, got, want, bound, ulp_bound=None, quiet=False):
    """One line: the largest difference relative to the largest value of `want`, and (bf16 tensors) the places apart. Returns whether it is within `bound` (and `ulp_bound` places, if given)."""
    g, w = f32(got).detach(), f32(want).detach()
    scale = max(float(w.abs().max()), 1e-30)
    rel = float((g - w).abs().max()) / scale
    mean = float((g - w).abs().mean()) / max(float(w.abs().mean()), 1e-30)
    ok = bool(torch.isfinite(g).all()) and rel <= bound
    extra = ""
    if got.dtype == BF and want.dtype == BF:
        worst, same = ulps(got, want)
        extra = f", {100 * same:.2f}% equal, at most {worst} places apart"
        ok = ok and (ulp_bound is None or worst <= ulp_bound)
    if not quiet:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {w.numel()} values, largest difference {rel:.3e} of the largest value (bound {bound:g}), mean {mean:.3e} of the mean{extra}")
    return ok


def put(path, t):
    t = t.contiguous().cpu()
    (t.view(torch.int16) if t.dtype == BF else t).numpy().tofile(path)


def get(path, shape, dtype=BF):
    import numpy as np
    a = np.fromfile(path, dtype={BF: np.int16, torch.float32: np.float32, torch.int64: np.int64}[dtype])
    t = torch.from_numpy(a)
    return (t.view(BF) if dtype == BF else t).reshape(*shape)


# ---- the checkpoint, by range ---------------------------------------------------------------------------------------------------------------------------------------------------------

def hf_get(repo, rev, path, rng=None):
    """Bytes of a file of the (public) repo at a revision, the whole file or a byte range (start, end inclusive). No token: the repo is ungated; none is read or sent."""
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{path}", headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
    return b""


def shard_header(repo, rev, shard):
    """(offset of a safetensors shard's data, {tensor: {dtype, shape, data_offsets}}) from two range reads."""
    n = struct.unpack("<Q", hf_get(repo, rev, shard, (0, 7)))[0]
    h = json.loads(hf_get(repo, rev, shard, (8, 7 + n)))
    h.pop("__metadata__", None)
    return 8 + n, h


def weight_map(repo, rev):
    return json.loads(hf_get(repo, rev, "model.safetensors.index.json"))["weight_map"]


def write_safetensors(path, tensors):
    """tensors: [(name, dtype, shape, bytes)] -> one safetensors file."""
    header, at = {}, 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        at += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for _, _, _, data in tensors:
            f.write(data)


def fetch_mini(out, layers, tokens, seed, repo=REPO, rev=REV, toy=True):
    """OUT/config.json (the real one), OUT/model.safetensors (every tensor of these layers' mixers and norms but a mixture of experts' experts, with their real names, read by range from the shards that
    hold them), OUT/x.bin (bf16 [tokens, 4096]: the embeddings of random tokens, by range from the embedding's shard: the layers' residual input) and OUT/tokens.json."""
    cfg = hf_get(repo, rev, "config.json")
    hidden, vocab = json.loads(cfg)["hidden_size"], json.loads(cfg)["vocab_size"]
    pattern = json.loads(cfg)["hybrid_override_pattern"]
    if any(l >= len(pattern) or pattern[l] == "E" for l in layers):
        raise SystemExit(f"layers {layers}: a mixture-of-experts layer's experts are 5.6 GB, not read here (`moetoy` makes a random model that has such layers); pick Mamba-2 (M) and attention (*) layers of {pattern}")
    os.makedirs(out, exist_ok=True)
    open(f"{out}/config.json", "wb").write(cfg)
    wm = weight_map(repo, rev)
    names = sorted(n for n in wm if any(n.startswith(f"backbone.layers.{l}.") for l in layers) and ".experts." not in n)
    if not names:
        raise SystemExit(f"no tensors for layers {layers}")
    headers, tensors = {}, []
    for n in names:
        sh = wm[n]
        if sh not in headers:
            headers[sh] = shard_header(repo, rev, sh)
        base, h = headers[sh]
        t = h[n]
        a, b = t["data_offsets"]
        tensors.append((n, t["dtype"], t["shape"], hf_get(repo, rev, sh, (base + a, base + b - 1))))
        print(f"{n} {t['dtype']} {t['shape']} ({sh})")
    # the embeddings of random tokens: rows by range
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, vocab, (tokens,), generator=g).tolist()
    sh = wm["backbone.embeddings.weight"]
    base, h = shard_header(repo, rev, sh)
    t = h["backbone.embeddings.weight"]
    row = hidden * 2
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        rows = b"".join(ex.map(lambda i: hf_get(repo, rev, sh, (base + t["data_offsets"][0] + i * row, base + t["data_offsets"][0] + (i + 1) * row - 1)), ids))
    open(f"{out}/x.bin", "wb").write(rows)
    json.dump({"ids": ids}, open(f"{out}/tokens.json", "w"))
    write_safetensors(f"{out}/model.safetensors", tensors)
    print(f"{out}: {len(tensors)} tensors of layers {layers}, {sum(len(t[3]) for t in tensors) / 1e6:.0f} MB; {tokens} embeddings")
    if toy:
        from safetensors.torch import load_file
        toy_build(out, load_file(f"{out}/model.safetensors"), layers, repo, rev, seed)


def inventory(out, repo=REPO, rev=REV):
    """The checkpoint's tensors (name, dtype, shape) from its shards' headers, as tests/fixtures/nemotron_h_tensors.json holds them: every tensor but a mixture of experts' experts by name,
    those as one run for each layer (their prefix, how many, and the shape and dtype of each of their two tensors: the shape of every expert of a layer is checked to be the same)."""
    from concurrent.futures import ThreadPoolExecutor
    wm = weight_map(repo, rev)
    shards = sorted(set(wm.values()))
    with ThreadPoolExecutor(8) as ex:
        heads = dict(zip(shards, ex.map(lambda s: shard_header(repo, rev, s)[1], shards)))
    have = {n: (t["dtype"], t["shape"]) for h in heads.values() for n, t in h.items()}
    assert set(have) == set(wm), "the index and the shards' headers name different tensors"
    plain, runs = {}, {}
    for n, (d, sh) in sorted(have.items()):
        m = re.fullmatch(r"(.*\.experts\.)(\d+)\.(up_proj\.weight|down_proj\.weight)", n)
        if m:
            run = runs.setdefault(m.group(1), {"count": set(), "tensors": {}})
            run["count"].add(int(m.group(2)))
            assert run["tensors"].setdefault(m.group(3), [d, sh]) == [d, sh], f"{n}: not the shape of the layer's other experts"
        else:
            plain[n] = [d, sh]
    out_runs = []
    for prefix, run in sorted(runs.items()):
        assert run["count"] == set(range(len(run["count"]))), f"{prefix}: experts are not 0..{len(run['count'])}"
        out_runs.append({"prefix": prefix, "count": len(run["count"]), "tensors": run["tensors"]})
    json.dump({"repo": repo, "revision": rev, "tensors": plain, "runs": out_runs}, open(out, "w"), indent=0, separators=(",", ":"))
    print(f"{out}: {len(plain)} tensors by name and {len(out_runs)} runs of experts ({sum(r['count'] * 2 for r in out_runs)} tensors); {len(have)} in all")


# ---- the kernels on random inputs: `glyd-engine nhkernels DIR/in DIR/out`, or tools/nh_emu.cpp's program, against the reference -------------------------------------------------------------

CASES = ("chunked", "recurrent", "steps", "mixed")  # the scan in chunks (two calls), the recurrence (two calls), one token a call, chunks for the first call then single tokens


def kernels_gen(d, seed, nh, ng, n, split, nq, nk, na, dt_min, eps=1e-5, real=None):
    """Inputs of the kernels in DIR/in: random (statistics like the model's), or with `real` = (MINI, Mamba-2 layer, attention layer) the real layers' in_proj output, small tensors and qkv product of
    the mini checkpoint's embeddings (the model's heads and groups, its extremes of dt and A)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape, mean=0.0, std=1.0: torch.randn(*shape, generator=g) * std + mean
    u = lambda *shape, lo=0.0, hi=1.0: torch.rand(*shape, generator=g) * (hi - lo) + lo
    os.makedirs(f"{d}/in", exist_ok=True)
    if real:
        mini, lm, la = real
        cfg = model_cfg(mini)
        nh, ng, eps = cfg["heads"], cfg["groups"], cfg["eps"]
        nq, nk = cfg["att_heads"], cfg["kv_heads"]
        t = read_tensors(f"{mini}/model.safetensors")
        x = get(f"{mini}/x.bin", (-1, cfg["hidden"]))
        n, na = min(n, x.shape[0]), min(na, x.shape[0])
        W = load_layer(t, lm, "cpu")
        mix = F.linear(rmsnorm(x[:n], W["norm"], eps), W["in_proj"])
        di, cd = nh * P, nh * P + 2 * ng * N
        tens = dict(mix=mix, conv_w=W["conv_w"], conv_b=W["conv_b"], a_log=W["A_log"], d=W["D"], dt_bias=W["dt_bias"], norm_w=W["gnorm"], cstate0=torch.zeros(3, cd, dtype=BF))
        A = load_layer(t, la, "cpu")
        h = rmsnorm(x[:na], A["norm"], eps)
        tens["qkv"] = torch.cat([F.linear(h, A["q_proj"]), F.linear(h, A["k_proj"]), F.linear(h, A["v_proj"])], 1)
        state0 = torch.zeros(nh, P, N)
    else:
        di, cd = nh * P, nh * P + 2 * ng * N
        tens = dict(mix=torch.cat([r(n, di, std=2.0), r(n, cd, std=1.5), r(n, nh, std=1.2)], 1), conv_w=r(cd, 1, 4, std=0.4), conv_b=r(cd, std=0.1), a_log=u(nh, lo=0.0, hi=4.8), d=r(nh, mean=1.0, std=0.3),
                    dt_bias=u(nh, lo=-5.0, hi=-2.0), norm_w=u(di, lo=0.5, hi=1.5), cstate0=r(3, cd), qkv=r(na, (nq + 2 * nk) * 128, std=1.5))
        state0 = r(nh, P, N, std=0.3)
    for name, tt in tens.items():
        put(f"{d}/in/{name}.bin", bf(tt))
    put(f"{d}/in/state0.bin", state0)
    open(f"{d}/in/mamba.txt", "w").write(f"{nh} {ng} {n} {eps} {dt_min} {split}\n")
    open(f"{d}/in/attn.txt", "w").write(f"{nq} {nk} {na}\n")
    print(f"kernels: {nh} heads, {ng} groups, {n} tokens (a first call of {split}), attention {nq}/{nk} heads, {na} rows{' (real layers)' if real else ''}: inputs in {d}/in")


def kernels_check(d, dt_min_arg=None):
    I, O = f"{d}/in", f"{d}/out"
    nh, ng, n, eps, dt_min, split = open(f"{I}/mamba.txt").read().split()
    nh, ng, n, split, eps, dt_min = int(nh), int(ng), int(n), int(split), float(eps), float(dt_min)
    if dt_min_arg is not None:
        dt_min = dt_min_arg
    di, cd, W = nh * P, nh * P + 2 * ng * N, nh * P + nh * P + 2 * ng * N + nh
    rd = lambda name, *shape, dtype=BF: get(f"{I}/{name}.bin", shape, dtype)
    mix, cw, cb, al, dd, dtb, nw = rd("mix", n, W), rd("conv_w", cd, 1, 4), rd("conv_b", cd), rd("a_log", nh), rd("d", nh), rd("dt_bias", nh), rd("norm_w", di)
    state0, cstate0 = rd("state0", nh, P, N, dtype=torch.float32), rd("cstate0", 3, cd)
    ok = True
    ref_conv, ref_cst = conv_silu(mix[:, di:di + cd], cw, cb, cstate0)
    cases = open(f"{O}/cases.txt").read().split() if os.path.exists(f"{O}/cases.txt") else list(CASES)  # (only tools/nh_emu.cpp runs fewer cases: NH_EMU_CASES)
    for c in cases:
        if not os.path.exists(f"{O}/{c}_conv.bin"):
            print(f"no {O}/{c}_conv.bin")
            return False
        eng = lambda name, *shape, dtype=BF: get(f"{O}/{c}_{name}.bin", shape, dtype)
        conv, y, gn, cst, state = eng("conv", n, cd), eng("y", n, di), eng("gn", n, di), eng("cstate", 3, cd), eng("state", nh, P, N, dtype=torch.float32)
        ok &= stats(f"{c}: convolution", conv, ref_conv, 2e-2, 2)
        same = bool((cst.view(torch.int16) == ref_cst.view(torch.int16)).all())
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} {c}: the convolution's state is the last three inputs")
        # the scan from the engine's own convolution output
        xs, B, C = conv[:, :di].reshape(n, nh, P), conv[:, di:di + ng * N].reshape(n, ng, N), conv[:, di + ng * N:].reshape(n, ng, N)
        S = state0.clone()
        ry = ssm_recurrent(xs, B, C, mix[:, di + cd:], dtb, al, dd, S, dt_min).reshape(n, di)
        ok &= stats(f"{c}: scan outputs", y, ry, 8e-3)
        ok &= stats(f"{c}: scan final state", state, S, 1e-4)
        ok &= stats(f"{c}: gated norm", gn, gated_norm(y, mix[:, :di], nw, eps, ng), 1e-2, 2)
    # the cases against each other: the same sequence however it is cut
    base = {c: get(f"{O}/{c}_y.bin", (n, di)) for c in cases}
    for c in cases[1:]:
        ok &= stats(f"{c} against {cases[0]} (whole scan outputs)", base[c], base[cases[0]], 1e-2)
    # attention
    nq, nk, na = (int(x) for x in open(f"{I}/attn.txt").read().split())
    qkv = rd("qkv", na, (nq + 2 * nk) * 128)
    q, k, v = qkv[:, :nq * 128].reshape(na, nq, 128), qkv[:, nq * 128:(nq + nk) * 128].reshape(na, nk, 128), qkv[:, (nq + nk) * 128:].reshape(na, nk, 128)
    if os.path.exists(f"{O}/kv.bin"):
        kv = get(f"{O}/kv.bin", (na, nk, 256))
        ok &= stats("attention: keys and values in the cache", kv, torch.cat([k, v], -1), 0.0)
        want = attn_core(q, k, v)
        for name in ("dec", "pre"):
            if os.path.exists(f"{O}/{name}.bin"):
                ok &= stats(f"attention: {'decode steps' if name == 'dec' else 'the prompt kernel'}", get(f"{O}/{name}.bin", (na, nq * 128)), want, 2e-2)
    print("kernels:", "ok" if ok else "FAILED")
    return ok


# ---- layers of the real model through the engine: `glyd-engine nhlayers MINI IO` against the reference ---------------------------------------------------------------------------------

def load_layer(tensors, l, dev):
    """The reference's weights of layer l from a dict of the checkpoint's tensors (names as the checkpoint has them)."""
    g = lambda n: tensors[f"backbone.layers.{l}.{n}"].to(dev)
    W = dict(norm=g("norm.weight"))
    if f"backbone.layers.{l}.mixer.in_proj.weight" in tensors:
        W.update(in_proj=g("mixer.in_proj.weight"), out_proj=g("mixer.out_proj.weight"), conv_w=g("mixer.conv1d.weight"), conv_b=g("mixer.conv1d.bias"), A_log=g("mixer.A_log"), D=g("mixer.D"),
                 dt_bias=g("mixer.dt_bias"), gnorm=g("mixer.norm.weight"))
    elif f"backbone.layers.{l}.mixer.gate.weight" in tensors:
        n = 0
        while f"backbone.layers.{l}.mixer.experts.{n}.up_proj.weight" in tensors:
            n += 1
        W.update(gate=g("mixer.gate.weight"), bias=g("mixer.gate.e_score_correction_bias"), fc1=g("mixer.fc1_latent_proj.weight"), fc2=g("mixer.fc2_latent_proj.weight"),
                 s_up=g("mixer.shared_experts.up_proj.weight"), s_down=g("mixer.shared_experts.down_proj.weight"),
                 up=torch.stack([g(f"mixer.experts.{e}.up_proj.weight") for e in range(n)]), down=torch.stack([g(f"mixer.experts.{e}.down_proj.weight") for e in range(n)]))
    else:
        W.update({k: g(f"mixer.{k}.weight") for k in ("q_proj", "k_proj", "v_proj", "o_proj")})
    return W


def model_cfg(d):
    c = json.load(open(f"{d}/config.json"))
    return dict(heads=c["mamba_num_heads"], groups=c["n_groups"], eps=c.get("layer_norm_epsilon", 1e-5), att_heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                att_dim=c["head_dim"], hidden=c["hidden_size"], kinds=c["hybrid_override_pattern"], top_k=c["num_experts_per_tok"], scale=c["routed_scaling_factor"],
                norm_topk=c.get("norm_topk_prob", True), latent=c.get("moe_latent_size"))


def read_tensors(path):
    from safetensors.torch import load_file
    return load_file(path)


def layers_gen(mini, io_dir, level=None, kinds="M*E"):
    """IO/spec.json: for each Mamba-2 (M), attention (*) and mixture-of-experts (E) layer of the mini checkpoint that `kinds` names, the cases to run: the prompt in one call and in odd pieces, the scan token by
    token (Mamba-2), single steps (replayed from a CUDA graph, and not), a prompt then steps. level (4, 5 or 6): the level the engine holds a mixture-of-experts layer's matrices at (the one `moetoy` made the mini for)."""
    cfg = model_cfg(mini)
    rows = os.path.getsize(f"{mini}/x.bin") // 2 // cfg["hidden"]
    tensors = read_tensors(f"{mini}/model.safetensors")
    layers = sorted({int(m.group(1)) for n in tensors for m in [re.match(r"backbone\.layers\.(\d+)\.", n)] if m})
    odd = [100, 128, 5, rows - 233] if rows > 240 else [rows]
    steps = min(rows, 64)
    cases = []
    layers = [l for l in layers if cfg["kinds"][l] in kinds]
    for l in layers:
        mamba = cfg["kinds"][l] == "M"
        tag = f"{ {'M': 'm', '*': 'a', 'E': 'e'}[cfg['kinds'][l]] }{l}"
        cases += [dict(name=f"{tag}_prefill", layer=l, mode="prefill", calls=[rows]), dict(name=f"{tag}_chunks", layer=l, mode="prefill", calls=odd)]
        if mamba:
            cases.append(dict(name=f"{tag}_recurrent", layer=l, mode="prefill", calls=[rows], recurrent=True))
        cases += [dict(name=f"{tag}_steps", layer=l, mode="decode", tokens=steps), dict(name=f"{tag}_graph", layer=l, mode="decode", tokens=steps, graph=True),
                  dict(name=f"{tag}_mixed", layer=l, mode="mixed", prefill=min(200, rows - 1), tokens=rows, graph=True)]
    os.makedirs(io_dir, exist_ok=True)
    json.dump(dict(chunk=max(rows, 128), **({"level": level} if level else {}), cases=cases), open(f"{io_dir}/spec.json", "w"), indent=1)
    print(f"{io_dir}/spec.json: {len(cases)} cases on layers {layers}, {rows} rows")


def layers_check(mini, io_dir, dt_min=0.0, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = model_cfg(mini)
    hid = cfg["hidden"]
    tensors = read_tensors(f"{mini}/model.safetensors")
    spec = json.load(open(f"{io_dir}/spec.json"))
    x_all = get(f"{mini}/x.bin", (-1, hid)).to(dev)
    di, cd, nq, nk, d = cfg["heads"] * P, cfg["heads"] * P + 2 * cfg["groups"] * N, cfg["att_heads"], cfg["kv_heads"], cfg["att_dim"]
    ok = True
    Ws = {}
    for case in spec["cases"]:
        name, l = case["name"], case["layer"]
        mamba = cfg["kinds"][l] == "M"
        if l not in Ws:
            Ws[l] = load_layer(tensors, l, dev)
        W = Ws[l]
        if not os.path.exists(f"{io_dir}/{name}.out.bin"):
            print(f"FAIL {name}: the engine wrote nothing")
            ok = False
            continue
        calls = case.get("calls") or ([1] * case["tokens"] if case["mode"] == "decode" else [case["prefill"]] + [1] * (case["tokens"] - case["prefill"]))
        T = sum(calls)
        x = x_all[:T]
        eng = lambda stage, width, dtype=BF: get(f"{io_dir}/{name}.{stage}.bin", (T, width), dtype).to(dev)
        print(f"-- {name}: layer {l} ({ {'M': 'Mamba-2', '*': 'attention', 'E': 'mixture of experts'}[cfg['kinds'][l]] }), {len(calls)} calls, {T} rows")
        if mamba:
            mix, conv, gn, out = eng("mix", cfg["heads"] * P + cd + cfg["heads"]), eng("conv", cd), eng("gn", di), eng("out", hid)
            h = rmsnorm(x, W["norm"], cfg["eps"])
            ok &= stats("  in_proj output", mix, F.linear(h, W["in_proj"]), 2e-2)
            rconv, rcst = conv_silu(mix[:, di:di + cd], W["conv_w"], W["conv_b"], torch.zeros(3, cd, dtype=BF, device=dev))
            ok &= stats("  convolution", conv, rconv, 2e-2, 2)
            S = torch.zeros(cfg["heads"], P, N, device=dev)
            xs, B, C = conv[:, :di].reshape(T, -1, P), conv[:, di:di + cfg["groups"] * N].reshape(T, cfg["groups"], N), conv[:, di + cfg["groups"] * N:].reshape(T, cfg["groups"], N)
            y = ssm_recurrent(xs, B, C, mix[:, di + cd:], W["dt_bias"], W["A_log"], W["D"], S, dt_min).reshape(T, di)
            ok &= stats("  scan and gated norm", gn, gated_norm(y, mix[:, :di], W["gnorm"], cfg["eps"], cfg["groups"]), 2e-2)
            ok &= stats("  out_proj output", out, F.linear(gn, W["out_proj"]), 2e-2)
            ok &= stats("  final SSM state", get(f"{io_dir}/{name}.ssm.bin", (cfg["heads"], P, N), torch.float32).to(dev), S, 2e-3)
            same = bool((get(f"{io_dir}/{name}.cstate.bin", (3, cd)).to(dev).view(torch.int16) == rcst.view(torch.int16)).all())
            ok &= same
            print(f"{'ok  ' if same else 'FAIL'}   the convolution's state is the last three inputs")
            st = dict(conv=torch.zeros(3, cd, dtype=BF, device=dev), ssm=torch.zeros(cfg["heads"], P, N, device=dev))
            whole = mamba_layer(x, W, cfg, st, True, dt_min)
            ok &= stats("  the layer from its input, reference alone", out, whole, 1e-1)
        elif cfg["kinds"][l] == "E":
            h, z, y, out = eng("h", hid), eng("z", cfg["latent"]), eng("y", cfg["latent"]), eng("out", hid)
            ids, wts = (get(f"{io_dir}/{name}.{stage}.bin", (T, cfg["top_k"]), dtype).to(dev) for stage, dtype in (("ids", torch.int64), ("wts", torch.float32)))
            # (the router's and the stages' inputs are the engine's own normed input: a different rounding of it moves a score by more than a tie's tolerance)
            ok &= stats("  the layer's norm", h, rmsnorm(x, W["norm"], cfg["eps"]), 2e-2, 2)
            ok &= stats("  fc1_latent_proj output", z, F.linear(h, W["fc1"]), 2e-2)
            ok &= moe_choice(h, W, cfg, ids, wts)
            ok &= stats("  experts' sum (on the engine's z, experts and weights)", y, moe_experts(z, ids, wts, W["up"], W["down"]), 2e-2)
            ok &= stats("  fc2_latent_proj and the shared expert (on the engine's y)", out, bf(f32(F.linear(y, W["fc2"])) + f32(moe_shared(h, W))), 2e-2)
            # (no "reference alone" line as the other layers have: a router that picks the other of two experts at a near tie would fail it, and every hop is checked from the engine's own data)
        else:
            qkv, attn, out = eng("qkv", (nq + 2 * nk) * d), eng("attn", nq * d), eng("out", hid)
            h = rmsnorm(x, W["norm"], cfg["eps"])
            ref_qkv = torch.cat([F.linear(h, W["q_proj"]), F.linear(h, W["k_proj"]), F.linear(h, W["v_proj"])], 1)
            ok &= stats("  qkv products", qkv, ref_qkv, 2e-2)
            q, k, v = qkv[:, :nq * d].reshape(T, nq, d), qkv[:, nq * d:(nq + nk) * d].reshape(T, nk, d), qkv[:, (nq + nk) * d:].reshape(T, nk, d)
            ok &= stats("  attention", attn, attn_core(q, k, v), 2e-2)
            ok &= stats("  o product", out, F.linear(attn, W["o_proj"]), 2e-2)
            kvc = get(f"{io_dir}/{name}.kv.bin", (T, nk, 2 * d)).to(dev)
            ok &= stats("  keys and values in the cache", kvc, torch.cat([k, v], -1), 0.0)
            whole, _ = attention_layer(x, W, cfg)
            ok &= stats("  the layer from its input, reference alone", out, whole, 1e-1)
    # the graph replays and the eager steps are the same work
    by = {c["name"]: c for c in spec["cases"]}
    for n, c in by.items():
        if n.endswith("_graph") and n[:-6] + "_steps" in by and os.path.exists(f"{io_dir}/{n}.out.bin"):
            a, b = get(f"{io_dir}/{n}.out.bin", (c["tokens"], hid)), get(f"{io_dir}/{n[:-6]}_steps.out.bin", (c["tokens"], hid))
            ok &= stats(f"{n} against {n[:-6]}_steps (a CUDA graph's replays against the same steps launched one by one: the same bits where the products are deterministic)", a, b, 2e-3)
    print("layers:", "ok" if ok else "FAILED")
    return ok


def real_vs_transformers(mini, tokens=64):
    """The reference against transformers' own NemotronHMamba2Mixer and NemotronHAttention (fp32, their torch paths) on the real weights of the mini checkpoint's layers and its embeddings: what is left
    is bf16 rounding. Their chunk scan clamps dt to time_step_min = 0.001, so the Mamba-2 comparison is made with that limit (and, to show what it does, without)."""
    from transformers.models.nemotron_h import modeling_nemotron_h as M
    d = json.load(open(f"{mini}/config.json"))
    for k in ("architectures", "auto_map", "transformers_version"):
        d.pop(k, None)
    cfg = M.NemotronHConfig(**d)
    cfg._attn_implementation = "sdpa"
    cfgd = model_cfg(mini)
    t = read_tensors(f"{mini}/model.safetensors")
    layers = sorted({int(m.group(1)) for n in t for m in [re.match(r"backbone\.layers\.(\d+)\.", n)] if m})
    x = get(f"{mini}/x.bin", (-1, cfgd["hidden"]))[:tokens]
    ok = True
    for l in layers:
        W = load_layer(t, l, "cpu")
        h = rmsnorm(x, W["norm"], cfgd["eps"])
        if cfgd["kinds"][l] == "M":
            mixer = M.NemotronHMamba2Mixer(cfg, layer_idx=l).float().eval()
            with torch.no_grad():
                for n in ("in_proj.weight", "out_proj.weight", "conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight"):
                    dict(mixer.named_parameters())[n].copy_(t[f"backbone.layers.{l}.mixer.{n}"].float())
                want = mixer(f32(h)[None])[0]
            for dm, bound in ((0.001, 5e-2), (0.0, None)):
                st = dict(conv=torch.zeros(3, cfgd["heads"] * P + 2 * cfgd["groups"] * N, dtype=BF), ssm=torch.zeros(cfgd["heads"], P, N))
                got = mamba_mixer(h, W, cfgd, st, True, dm)
                good = stats(f"layer {l} (Mamba-2) against transformers' NemotronHMamba2Mixer, dt limit {dm}", got, want, bound or 1.0)
                ok &= good if bound else True
        else:
            att = M.NemotronHAttention(cfg, layer_idx=l).float().eval()
            with torch.no_grad():
                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    getattr(att, n).weight.copy_(t[f"backbone.layers.{l}.mixer.{n}.weight"].float())
                want = att(f32(h)[None])[0][0]
            ok &= stats(f"layer {l} (attention) against transformers' NemotronHAttention", attention_mixer(h, W, cfgd)[0], want, 5e-2)
    print("real:", "ok" if ok else "FAILED")
    return ok


# ---- a model of a few real layers: `glyd-engine nhtoy DIR IO` -------------------------------------------------------------------------------------------------------------------------------------

TOY_VOCAB = 2048


def toy_build(mini, tensors, layers, repo=REPO, rev=REV, seed=1, total=96, prefill=40):
    """MINI/toy: the layers (renumbered 0, 1, ...) with the first TOY_VOCAB rows of the embedding and of the LM head and the final norm: a model the engine loads as it is. The config's pattern is the layers' own."""
    cfg = json.load(open(f"{mini}/config.json"))
    kinds = "".join("M" if f"backbone.layers.{l}.mixer.in_proj.weight" in tensors else "*" for l in layers)
    cfg.update(hybrid_override_pattern=kinds, num_hidden_layers=len(layers), vocab_size=TOY_VOCAB)
    out = f"{mini}/toy"
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    wm = weight_map(repo, rev)
    hidden = cfg["hidden_size"]
    ts = []
    for i, l in enumerate(layers):
        for n, t in tensors.items():
            if n.startswith(f"backbone.layers.{l}."):
                ts.append((f"backbone.layers.{i}." + n[len(f"backbone.layers.{l}."):], "BF16", list(t.shape), put_bytes(t)))
    for name, rows in (("backbone.embeddings.weight", TOY_VOCAB), ("lm_head.weight", TOY_VOCAB), ("backbone.norm_f.weight", None)):
        sh = wm[name]
        base, h = shard_header(repo, rev, sh)
        t = h[name]
        a = base + t["data_offsets"][0]
        n_bytes = (rows * hidden * 2) if rows else (t["data_offsets"][1] - t["data_offsets"][0])
        ts.append((name, t["dtype"], [rows, hidden] if rows else t["shape"], hf_get(repo, rev, sh, (a, a + n_bytes - 1))))
    write_safetensors(f"{out}/model.safetensors", ts)
    g = torch.Generator().manual_seed(seed)
    open(f"{out}/ids.txt", "w").write(" ".join(str(i) for i in torch.randint(0, TOY_VOCAB, (total,), generator=g).tolist()))
    open(f"{out}/toy.txt", "w").write(f"{prefill} {total}\n")
    print(f"{out}: a model of pattern {kinds}, {TOY_VOCAB} tokens of vocabulary; {prefill} tokens of prefill, {total} in all")


def put_bytes(t):
    t = t.contiguous().cpu()
    return (t.view(torch.int16) if t.dtype == BF else t.view(torch.int32)).numpy().tobytes()


# ---- a random tiny model with mixture-of-experts layers: `nemotron_ref.py moetoy MINI`, then `glyd-engine nhlayers` and `nhtoy` -----------------------------------------------------------------------

# What `moetoy` changes in the real model's config (tests/fixtures/nemotron_h_config.json) to cut it down to a tenth of the width: the engine's kernels want heads of 64 channels, a state of 128, a convolution
# of 4 and attention heads of 128, so those stay; 64 Mamba-2 heads in 4 groups (a multiple of 64 heads makes the in_proj's rows whole blocks of 64, as the real model's 128 do), 4 attention heads of 2 key-value
# heads, 16 experts of which 4 a token (a prompt of 300 rows gives each 75 pairs: the library's prompt kernel, from 48 to 64 pairs an expert), a latent of 128, packs of whole blocks (rows a multiple of 64, columns
# of 64).
MOETOY = dict(hidden_size=256, mamba_num_heads=64, n_groups=4, num_attention_heads=4, num_key_value_heads=2, n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=128,
              moe_shared_expert_intermediate_size=256, moe_latent_size=128, vocab_size=TOY_VOCAB, max_position_embeddings=4096, num_nextn_predict_layers=0)


def moetoy_config(pattern):
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "nemotron_h_config.json")))
    cfg.update(MOETOY, hybrid_override_pattern=pattern, num_hidden_layers=len(pattern))
    cfg.pop("mtp_hybrid_override_pattern", None)
    return cfg


def moetoy_tensors(cfg, level, seed):
    """The tiny model's tensors, {name: tensor} in the checkpoint's names and dtypes (bf16 but the routers' score-correction biases, fp32): every layer's, the embedding, the LM head and the final norm. The experts,
    the latent projections and the shared expert are on the grid of the block layout the engine holds them in (level 6: six-bit codes, 5: five-bit; 4: the experts four-bit and the latent projections and the shared expert
    six-bit, as the engine holds them at wren), as `grid` makes them: its packs are exact (an expert's down matrix is on the grid of both its block layout and its row layout)."""
    g = torch.Generator().manual_seed(seed)
    h, nh, ng = cfg["hidden_size"], cfg["mamba_num_heads"], cfg["n_groups"]
    di = nh * P
    cd = di + 2 * ng * N
    nq, nk, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    E, lat, inter, sh = cfg["n_routed_experts"], cfg["moe_latent_size"], cfg["moe_intermediate_size"], cfg["moe_shared_expert_intermediate_size"]
    kmax = {6: 31, 5: 15, 4: 7}[level]
    kdense = 31 if level == 4 else kmax  # (the engine's pack_layout: at wren only the experts are four-bit codes)
    randn = lambda *shape, std=1.0: bf(torch.randn(*shape, generator=g) * std)
    unif = lambda *shape, lo, hi: bf(torch.rand(*shape, generator=g) * (hi - lo) + lo)

    def grid(rows, cols, std, kmax=kmax, both=False):
        """rows x cols weights of about that spread that a block layout holds exactly: integers of at most kmax times a power of two (the codes spread over a third of the range or less), with a code of kmax in
        every run of 32 along a row (so that the packer's fit of the scale lands on the power of two, and a code times it is the weight). `both`: with a code of kmax in every run of 32 down a column too
        (the codes of kmax on the diagonals where row + column is a multiple of 32), so that the matrix's transpose is held exactly as well: an experts' down matrix is held in the block layout of a directory of
        before, and in the row layout, which groups the runs of 32 along the other axis."""
        scale = 2.0 ** math.ceil(math.log2(3 * std / kmax))
        k = torch.round(torch.randn(rows, cols, generator=g) * std / scale).clamp(-kmax, kmax)
        top = kmax * (torch.randint(0, 2, (rows, (cols + 31) // 32), generator=g) * 2 - 1)
        if both:
            at = (32 * torch.arange((cols + 31) // 32)[None, :] + (-torch.arange(rows)[:, None]) % 32).clamp(max=cols - 1)
            k[torch.arange(rows)[:, None], at] = top.float()
        else:
            k[:, ::32] = top
        return bf(k * scale)

    T = {}
    for l, kind in enumerate(cfg["hybrid_override_pattern"]):
        n = f"backbone.layers.{l}"
        T[f"{n}.norm.weight"] = unif(h, lo=0.5, hi=1.5)
        if kind == "M":
            # statistics like the model's (kernels_gen's): the small tensors; the projections of a spread that keeps the activations about 1
            T.update({f"{n}.mixer.in_proj.weight": randn(di + cd + nh, h, std=MOETOY_STD["in_proj"]), f"{n}.mixer.out_proj.weight": randn(h, di, std=MOETOY_STD["out_proj"]),
                      f"{n}.mixer.conv1d.weight": randn(cd, 1, 4, std=0.4), f"{n}.mixer.conv1d.bias": randn(cd, std=0.1), f"{n}.mixer.A_log": unif(nh, lo=0.0, hi=4.8),
                      f"{n}.mixer.D": randn(nh, std=0.3) + 1.0, f"{n}.mixer.dt_bias": unif(nh, lo=-5.0, hi=-2.0), f"{n}.mixer.norm.weight": unif(di, lo=0.5, hi=1.5)})
        elif kind == "*":
            T.update({f"{n}.mixer.q_proj.weight": randn(nq * d, h, std=MOETOY_STD["qkv"]), f"{n}.mixer.k_proj.weight": randn(nk * d, h, std=MOETOY_STD["qkv"]),
                      f"{n}.mixer.v_proj.weight": randn(nk * d, h, std=MOETOY_STD["qkv"]), f"{n}.mixer.o_proj.weight": randn(h, nq * d, std=MOETOY_STD["o_proj"])})
        else:
            T.update({f"{n}.mixer.gate.weight": randn(E, h, std=MOETOY_STD["gate"]), f"{n}.mixer.gate.e_score_correction_bias": randn(E, std=0.1).float(),
                      f"{n}.mixer.fc1_latent_proj.weight": grid(lat, h, MOETOY_STD["fc1"], kdense), f"{n}.mixer.fc2_latent_proj.weight": grid(h, lat, MOETOY_STD["fc2"], kdense),
                      f"{n}.mixer.shared_experts.up_proj.weight": grid(sh, h, MOETOY_STD["s_up"], kdense), f"{n}.mixer.shared_experts.down_proj.weight": grid(h, sh, MOETOY_STD["s_down"], kdense)})
            for e in range(E):
                T[f"{n}.mixer.experts.{e}.up_proj.weight"] = grid(inter, lat, MOETOY_STD["up"])
                T[f"{n}.mixer.experts.{e}.down_proj.weight"] = grid(lat, inter, MOETOY_STD["down"], both=True)
    T["backbone.embeddings.weight"] = randn(cfg["vocab_size"], h)
    # (the head is the embedding scaled: a token's own embedding is then the largest logit by a margin the engine's rounding does not close, as a trained model's logits are peaked; random rows would leave a fifth of
    # the positions within 2% of a tie)
    T["lm_head.weight"] = bf(T["backbone.embeddings.weight"].float() * MOETOY_STD["head"])
    T["backbone.norm_f.weight"] = unif(h, lo=0.5, hi=1.5)
    return T


# The spread of each matrix of the tiny model (the standard deviation of its weights; `head`: the scale of the embedding that is the LM head), chosen so that every layer's mixer has about the size of the
# residual stream's embedding (a rms of 0.5 to 0.9 where the embedding's is 1): a layer that is wrong then moves the logits by tenths, and a token's own embedding is the largest logit by a margin.
MOETOY_STD = dict(in_proj=0.0625, out_proj=0.011, qkv=0.0625, o_proj=0.08, gate=0.1, fc1=0.0625, fc2=0.025, up=0.09, down=0.015, s_up=0.0625, s_down=0.012, head=0.0625)


def moetoy_build(out, pattern="ME*EME", seed=1, level=6, tokens=300, prefill=150, total=210):
    """OUT: as `fetch_mini` makes a mini checkpoint (config.json, model.safetensors of the layers' tensors, x.bin: the embeddings of `tokens` random tokens, tokens.json), and OUT/toy: the whole model
    (the layers, the embedding, the head, the final norm), ids.txt (`total` random tokens) and toy.txt (`prefill` `total` `level`)."""
    cfg = moetoy_config(pattern)
    T = moetoy_tensors(cfg, level, seed)
    dtype = lambda t: "BF16" if t.dtype == BF else "F32"
    os.makedirs(f"{out}/toy", exist_ok=True)
    layers = [(n, dtype(t), list(t.shape), put_bytes(t)) for n, t in T.items() if n.startswith("backbone.layers.")]
    g = torch.Generator().manual_seed(seed + 1)
    x_ids = torch.randint(0, cfg["vocab_size"], (tokens,), generator=g)
    for d in (out, f"{out}/toy"):
        json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    write_safetensors(f"{out}/model.safetensors", layers)
    put(f"{out}/x.bin", T["backbone.embeddings.weight"][x_ids])
    json.dump({"ids": x_ids.tolist()}, open(f"{out}/tokens.json", "w"))
    write_safetensors(f"{out}/toy/model.safetensors", layers + [(n, dtype(t), list(t.shape), put_bytes(t)) for n, t in T.items() if not n.startswith("backbone.layers.")])
    open(f"{out}/toy/ids.txt", "w").write(" ".join(str(i) for i in torch.randint(0, cfg["vocab_size"], (total,), generator=g).tolist()))
    open(f"{out}/toy/toy.txt", "w").write(f"{prefill} {total} {level}\n")
    experts = f", {cfg['n_routed_experts']} experts ({cfg['num_experts_per_tok']} a token) on the grid of {level}-bit codes" if "E" in pattern else ""
    print(f"{out}: a random model of pattern {pattern}, {sum(t.numel() for t in T.values()) / 1e6:.1f} M parameters{experts}; {tokens} embeddings in x.bin; {out}/toy: {prefill} tokens of prefill, {total} in all")


TOY_BOUND = 5e-2  # the toy's logits against the reference's, of the largest logit
TOY_TOP1 = 0.98  # the share of positions at which the engine's top-1 token is the reference's


def deviation(got, want):
    """Each position's largest difference of logits [positions, vocab] from the reference's, relative to the reference's largest at that position."""
    return (f32(got) - f32(want)).abs().max(-1).values / f32(want).abs().max(-1).values


def toy_check(mini, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    toy = f"{mini}/toy"
    cfg = model_cfg(toy)
    cfg["hidden"] = json.load(open(f"{toy}/config.json"))["hidden_size"]
    from safetensors.torch import load_file
    t = load_file(f"{toy}/model.safetensors")
    kinds = cfg["kinds"]
    Ws = [load_layer(t, i, dev) for i in range(len(kinds))]
    p, total = (int(v) for v in open(f"{toy}/toy.txt").read().split()[:2])
    ids = torch.tensor([int(v) for v in open(f"{toy}/ids.txt").read().split()][:total])
    emb, head, nf = t["backbone.embeddings.weight"].to(dev), t["lm_head.weight"].to(dev), t["backbone.norm_f.weight"].to(dev)

    def reference(drop_experts=False):
        """The logits from the position p - 1 on: the whole sequence in one pass, the residual stream as the engine keeps it (bf16 sums, the norm of the fp32 sum); drop_experts: a mixture-of-experts layer adds nothing."""
        x, o = emb[ids.to(dev)], None
        for i, W in enumerate(Ws):
            if o is None:
                h = rmsnorm(x, W["norm"], cfg["eps"])
            else:
                x, h = rmsnorm_add(x, o, W["norm"], cfg["eps"])
            if kinds[i] == "M":
                st = dict(conv=torch.zeros(3, cfg["heads"] * P + 2 * cfg["groups"] * N, dtype=BF, device=dev), ssm=torch.zeros(cfg["heads"], P, N, device=dev))
                o = mamba_mixer(h, W, cfg, st, True, 0.0)
            elif kinds[i] == "E":
                o = torch.zeros_like(x) if drop_experts else moe_mixer(h, W, cfg)
            else:
                o, _ = attention_mixer(h, W, cfg)
        _, hl = rmsnorm_add(x, o, nf, cfg["eps"])
        return F.linear(hl, head)[p - 1:total]

    ref = reference()
    ok = True
    # A router that scores two experts within the engine's rounding of each other picks the other of them in the engine (about 1 token-layer in a hundred at the roundings of bf16 in the toy), and that token's
    # layer then differs from the reference's by an expert's share of its output; the state a token leaves behind carries it on. With mixture-of-experts layers the bound is therefore the median position's,
    # not the worst's: a few positions are moved by a flip, by up to a fifth of the largest logit, and the median by about a hundredth, where a layer wrong in its structure (the scale, the choice, the weights,
    # the shared expert, the activation) moves the median by a tenth or more. The layers' own stages, with the engine's choices, are `layers check`'s.
    moe = "E" in kinds
    if moe:
        moved = float(deviation(reference(True), ref).median())
        sees = moved > 3 * TOY_BOUND
        ok &= sees
        print(f"{'ok  ' if sees else 'FAIL'} toy model: without its mixture-of-experts layers the median position's logits move by {moved:.3f} of the largest (the bound is {TOY_BOUND:g}: the check sees the layers)")
    runs = {}
    for tag in ("graph", "eager"):
        f = f"{toy}/toy_{tag}_logits.bin"
        if not os.path.exists(f):
            print(f"FAIL toy {tag}: the engine wrote nothing")
            return False
        runs[tag] = get(f, (total - p + 1, TOY_VOCAB)).to(dev)
        what = f"toy model, {tag} decode: logits of the prefill's last token and of {total - p} steps"
        if moe:
            dv = deviation(runs[tag], ref)
            good = bool(torch.isfinite(f32(runs[tag])).all()) and float(dv.median()) <= TOY_BOUND
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {what}: the median position differs by {float(dv.median()):.3e} of its largest logit (bound {TOY_BOUND:g}), the 90th by {float(dv.quantile(0.9)):.3e}, the worst by {float(dv.max()):.3e}")
        else:
            ok &= stats(what, runs[tag], ref, TOY_BOUND)
        top = (runs[tag].float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
        ok &= top >= TOY_TOP1
        print(f"{'    ' if top >= TOY_TOP1 else 'FAIL'} top-1 token the same as the reference's at {100 * top:.1f}% of the {total - p + 1} positions (at least {100 * TOY_TOP1:.0f}%)")
    ok &= stats("toy model: the decode graph's logits against the eager steps'", runs["graph"], runs["eager"], 1e-3)
    print("toy:", "ok" if ok else "FAILED")
    return ok


# ---- the whole model packed at a level, against the bf16 checkpoint: `glyd-engine nhtoy PACKED OUT`, then `full` -------------------------------------------------------------------------------

# A public-domain text (Dickens, A Tale of Two Cities, 1859) for the tokens: natural, and long enough for 160 tokens.
TEXT = ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
        "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
        "direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of "
        "comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. "
        "In both countries it was clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were settled for ever.")
FULL_TOP1 = 0.85  # the share of positions at which the packed model's top-1 token is the bf16 model's (a level costs a few per cent; a layer that is wrong costs nearly all)
FULL_KL = 0.1  # the mean KL divergence of the packed model's next-token distribution from the bf16 model's, in nats (a level's is about 0.01; a wrong layer's is several)


def make_ids(model, out, prefill=64, total=160, level=5):
    """OUT/ids.txt (the first `total` tokens of TEXT in MODEL's tokenizer) and OUT/toy.txt (`prefill` `total` `level`), the files `glyd-engine nhtoy` reads."""
    from tokenizers import Tokenizer
    ids = Tokenizer.from_file(f"{model}/tokenizer.json").encode(TEXT, add_special_tokens=False).ids
    if len(ids) < total or not 0 < prefill <= total:
        raise SystemExit(f"{len(ids)} tokens in the text, {prefill} of {total} wanted")
    os.makedirs(out, exist_ok=True)
    open(f"{out}/ids.txt", "w").write(" ".join(str(i) for i in ids[:total]))
    open(f"{out}/toy.txt", "w").write(f"{prefill} {total} {level}\n")
    print(f"{out}: {total} tokens, {prefill} of them prefilled, level {level}")


class Shards:
    """A checkpoint's tensors by name, each read when asked (the model's 247 GB do not fit a GPU): the sharded checkpoint's index says where."""

    def __init__(self, d):
        self.d, self.open = d, {}
        self.wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]

    def __contains__(self, name):
        return name in self.wm

    def __getitem__(self, name):
        from safetensors import safe_open
        sh = self.wm[name]
        if sh not in self.open:
            self.open[sh] = safe_open(f"{self.d}/{sh}", "pt")
        return self.open[sh].get_tensor(name)


def full_reference(model, ids, p, dev):
    """The logits of the bf16 checkpoint for the position p - 1 on, [len(ids) - p + 1, vocab] bf16: the whole sequence in one pass, one layer's weights on the GPU at a time (as `toy_check`'s reference, the
    residual stream as the engine keeps it)."""
    cfg = model_cfg(model)
    t = Shards(model)
    x, o = t["backbone.embeddings.weight"][ids].to(dev), None
    for i, kind in enumerate(cfg["kinds"]):
        W = load_layer(t, i, dev)
        if o is None:
            h = rmsnorm(x, W["norm"], cfg["eps"])
        else:
            x, h = rmsnorm_add(x, o, W["norm"], cfg["eps"])
        if kind == "M":
            st = dict(conv=torch.zeros(3, cfg["heads"] * P + 2 * cfg["groups"] * N, dtype=BF, device=dev), ssm=torch.zeros(cfg["heads"], P, N, device=dev))
            o = mamba_mixer(h, W, cfg, st, True, 0.0)
        elif kind == "E":
            o = moe_mixer(h, W, cfg)
        else:
            o, _ = attention_mixer(h, W, cfg)
        del W
        if i % 8 == 7:
            print(f"reference: {i + 1} of {len(cfg['kinds'])} layers", flush=True)
    _, hl = rmsnorm_add(x, o, t["backbone.norm_f.weight"].to(dev), cfg["eps"])
    return F.linear(hl[p - 1:], t["lm_head.weight"].to(dev))


def full_check(model, toy, device=None):
    """OUT/toy_graph_logits.bin and toy_eager_logits.bin (bf16 [total - p + 1, vocab], the logits of the prefill's last token and of each teacher-forced step) against `full_reference`: per position the largest
    logit's token, and the KL divergence of the softmax from the reference's. The first position is the first token the model would say. A model run through the expert cache (`glyd-engine nhtoy PACKED OUT --expert-cache
    auto`) has no decode graph, its steps reading the router's choices on the host: it writes the eager file alone, which is then the one checked."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p, total = (int(v) for v in open(f"{toy}/toy.txt").read().split()[:2])
    ids = torch.tensor([int(v) for v in open(f"{toy}/ids.txt").read().split()][:total])
    ref = full_reference(model, ids, p, dev)
    ok, runs = True, {}
    for tag in ("graph", "eager"):
        f = f"{toy}/toy_{tag}_logits.bin"
        if not os.path.exists(f):
            if tag == "graph" and os.path.exists(f"{toy}/toy_eager_logits.bin"):
                print("full: no decode graph (the engine ran through the expert cache): the eager steps are checked alone")
                continue
            print(f"FAIL full {tag}: the engine wrote nothing")
            return False
        runs[tag] = get(f, (total - p + 1, ref.shape[-1])).to(dev)
    lp = F.log_softmax(f32(ref), -1)
    for tag, eng in runs.items():
        kl = (lp.exp() * (lp - F.log_softmax(f32(eng), -1))).sum(-1)
        top = (eng.float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
        dv = deviation(eng, ref)
        good = bool(torch.isfinite(f32(eng)).all()) and top >= FULL_TOP1 and float(kl.mean()) <= FULL_KL
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} the whole model, {tag} decode: top-1 token the bf16 model's at {100 * top:.1f}% of {total - p + 1} positions (at least {100 * FULL_TOP1:.0f}%), "
              f"KL {float(kl.mean()):.4f} nats on average (at most {FULL_KL:g}; the 90th position {float(kl.quantile(0.9)):.4f}, the worst {float(kl.max()):.4f}), the logits differ by {float(dv.median()):.3e} "
              f"of the largest at the median position, {float(dv.max()):.3e} at the worst")
    shown = runs.get("graph", runs["eager"])
    for k in range(min(3, total - p + 1)):
        e, r = shown[k].float(), ref[k].float()
        print(f"  position {p - 1 + k}: the engine's five largest logits' tokens {e.topk(5).indices.tolist()} ({e.max().item():.2f}), the reference's {r.topk(5).indices.tolist()} ({r.max().item():.2f})")
    if "graph" in runs:
        ok &= stats("the whole model: the decode graph's logits against the eager steps'", runs["graph"], runs["eager"], 1e-3)
    print("full:", "ok" if ok else "FAILED")
    return ok


def compare_logits(got, want):
    """The logits of two runs of `glyd-engine nhtoy` over the same tokens (OUT/toy_eager_logits.bin of each, bf16 [total - p + 1, vocab]; `got` the run with the host computing experts, `want` the one that copies them): per position
    the largest difference of the logits from `want`'s relative to `want`'s largest, whether the largest logit's token is the same, and the KL divergence of `want`'s softmax from `got`'s. Reported, no bound set (the tiny model's
    bound, in `logits_within`, is 1/64 of the largest); false only if a logit is not finite."""
    p, total = (int(v) for v in open(f"{want}/toy.txt").read().split()[:2])
    n = total - p + 1
    runs = []
    for d in (got, want):
        f = f"{d}/toy_eager_logits.bin"
        if not os.path.exists(f):
            print(f"FAIL compare: {f} is not there")
            return False
        runs.append(get(f, (n, os.path.getsize(f) // 2 // n)))
    g, w = runs
    if g.shape != w.shape:
        print(f"FAIL compare: logits {tuple(g.shape)} against {tuple(w.shape)}")
        return False
    dv = deviation(g, w)
    same = (g.float().argmax(-1) == w.float().argmax(-1)).float().mean().item()
    lw = F.log_softmax(f32(w), -1)
    kl = (lw.exp() * (lw - F.log_softmax(f32(g), -1))).sum(-1)
    ok = bool(torch.isfinite(f32(g)).all())
    print(f"{'ok  ' if ok else 'FAIL'} compare: {n} positions: the largest logit's token the same at {100 * same:.1f}%, the logits differ by {float(dv.median()):.3e} of the largest at the median position and {float(dv.max()):.3e} at the worst, "
          f"KL {float(kl.mean()):.5f} nats on average (the worst position {float(kl.max()):.5f})")
    return ok


# ---- the checks that need no GPU and no weights: the reference against transformers ----------------------------------------------------------------------------------------------------

def selftest():
    from transformers.models.nemotron_h import modeling_nemotron_h as M
    torch.manual_seed(0)
    ok = True
    T, H, G = 77, 8, 2
    x = torch.randn(T, H, P) * 0.7
    B, C = torch.randn(T, G, N) * 0.5, torch.randn(T, G, N) * 0.5
    dtr = torch.randn(T, H) * 1.5 - 2.0
    dt_bias, A_log, D = torch.randn(H) * 0.5, torch.rand(H) * 3.0, torch.randn(H)
    xb, Bb, Cb, dtb = bf(x), bf(B), bf(C), bf(dtr)
    S0 = torch.randn(H, P, N) * 0.1
    for dt_min in (0.0, 0.001):
        Sa, Sb = S0.clone(), S0.clone()
        ya = ssm_recurrent(xb, Bb, Cb, dtb, bf(dt_bias), bf(A_log), bf(D), Sa, dt_min)
        yb = ssm_chunked(xb, Bb, Cb, dtb, bf(dt_bias), bf(A_log), bf(D), Sb, dt_min)
        ok &= stats(f"chunked scan against the recurrence (dt_min {dt_min}), outputs", yb, ya, 8e-3)
        ok &= stats(f"chunked scan against the recurrence (dt_min {dt_min}), final state", Sb, Sa, 1e-5)
        # transformers' chunk scan on the same bf16 inputs (it works on the fp32 values, dt as bf16 would make it; ours keeps dt fp32: tolerances for the bf16 dt)
        Af = -torch.exp(f32(bf(A_log)))
        ref, final = M.mamba2_chunk_scan(f32(xb)[None], f32(dtb)[None], Af, f32(Bb)[None], f32(Cb)[None], CHUNK, D=f32(bf(D)), dt_bias=f32(bf(dt_bias)), dt_softplus=True,
                                         dt_limit=(dt_min, float("inf")), return_final_states=True, initial_states=S0[None].clone())
        ok &= stats(f"chunked scan against transformers' mamba2_chunk_scan (dt_min {dt_min}), outputs", ssm_chunked(xb, Bb, Cb, dtb, bf(dt_bias), bf(A_log), bf(D), S0.clone(), dt_min),
                    bf(ref[0]), 8e-3, 2)
        ok &= stats(f"chunked scan against transformers' mamba2_chunk_scan (dt_min {dt_min}), final state", final[0], Sb, 1e-5)
    # transformers' recurrent step (it has no limit)
    S1, S2 = S0.clone(), S0.clone()
    A = -torch.exp(f32(bf(A_log)))
    for t in range(5):
        o = M.mamba2_selective_state_update(S1[None], f32(xb[t])[None], f32(dtb[t])[None, :, None].expand(1, H, P), A[:, None, None].expand(H, P, N), f32(Bb[t])[None], f32(Cb[t])[None],
                                           f32(bf(D))[:, None].expand(H, P), dt_bias=f32(bf(dt_bias))[:, None].expand(H, P), dt_softplus=True)
        ok &= stats(f"recurrent step {t} against transformers' mamba2_selective_state_update", ssm_recurrent(xb[t:t + 1], Bb[t:t + 1], Cb[t:t + 1], dtb[t:t + 1], bf(dt_bias), bf(A_log), bf(D), S2)[0],
                    bf(o[0]), 8e-3, 2, quiet=t > 0)
    ok &= stats("recurrent steps' state against transformers'", S2, S1, 1e-5)
    # the gated norm against transformers' Zamba2RMSNormGated (the prefill path keeps fp32 up to the weight; vLLM's rounds the normed value first: one place apart at most)
    from transformers.models.zamba2.modeling_zamba2 import Zamba2RMSNormGated
    y, z, w = bf(torch.randn(5, 512)), bf(torch.randn(5, 512) * 2), bf(torch.rand(512) + 0.5)
    nrm = Zamba2RMSNormGated(512, group_size=128, eps=1e-5)
    nrm.weight.data = w.clone()
    ok &= stats("gated norm against transformers' Zamba2RMSNormGated", gated_norm(y, z, w, 1e-5, 4), nrm(y, z), 1e-2, 2)
    # a whole layer against transformers' NemotronHMamba2Mixer, tiny (random weights, fp32 end to end)
    cfgd = dict(hidden_size=96, mamba_num_heads=6, mamba_head_dim=P, ssm_state_size=N, n_groups=2, chunk_size=16, conv_kernel=4, hybrid_override_pattern="M", num_hidden_layers=1, vocab_size=64,
                layer_norm_epsilon=1e-5, time_step_min=0.001)
    cfg = M.NemotronHConfig(**cfgd)
    mixer = M.NemotronHMamba2Mixer(cfg, layer_idx=0).float().eval()
    with torch.no_grad():
        for p in mixer.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        mixer.A_log.copy_(torch.rand(6) * 2)
        mixer.dt_bias.copy_(torch.randn(6) - 1)
        mixer.norm.weight.copy_(torch.rand_like(mixer.norm.weight) + 0.5)
    xin = torch.randn(1, 40, 96)
    with torch.no_grad():
        want = mixer(xin)[0]
    Hh = 6
    Wd = dict(in_proj=mixer.in_proj.weight.data, out_proj=mixer.out_proj.weight.data, conv_w=mixer.conv1d.weight.data, conv_b=mixer.conv1d.bias.data, A_log=mixer.A_log.data, D=mixer.D.data,
              dt_bias=mixer.dt_bias.data, gnorm=mixer.norm.weight.data, norm=torch.ones(96))
    # (the reference rounds to bf16 where vLLM does; the layer test runs it on values that are bf16 already and compares at bf16 level)
    cfg2 = dict(heads=Hh, groups=2, eps=1e-5)
    Wb = {k: bf(v) for k, v in Wd.items()}
    st = dict(conv=torch.zeros(3, Hh * P + 2 * 2 * N, dtype=BF), ssm=torch.zeros(Hh, P, N))
    ours = mamba_layer(bf(xin[0]), {**Wb, "norm": bf(torch.ones(96))}, cfg2, st, True, dt_min=0.001)
    # transformers applies its mixer to the unnormalised input directly (the block does the norm): feed it the same (norm weight 1 -> the rms-normed input)
    with torch.no_grad():
        xn = f32(rmsnorm(bf(xin[0]), bf(torch.ones(96)), 1e-5))
        want = mixer(xn[None])[0]
    ok &= stats("a Mamba-2 layer against transformers' NemotronHMamba2Mixer (tiny, bf16 weights in ours, fp32 in its)", ours, want, 4e-2)
    ok &= moe_vs_transformers(M)
    # the tiny model `moetoy` builds is the config the engine's tests hold (tests/fixtures/nemotron_h_moetoy_config.json), and its weights are on the block layouts' grids
    fixture = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "nemotron_h_moetoy_config.json")))
    same = moetoy_config("ME*EME") == fixture
    ok &= same
    print(f"{'ok  ' if same else 'FAIL'} the tiny model's config is tests/fixtures/nemotron_h_moetoy_config.json")
    print("selftest:", "ok" if ok else "FAILED")
    return ok


def moe_vs_transformers(M):
    """The mixture-of-experts reference against transformers' NemotronHMoE (tiny, random weights, bf16-valued in ours and fp32 in its): the experts it chooses and their weights (compared as dense [T, E]
    matrices), and the layer's output (bf16 rounding is what is left)."""
    torch.manual_seed(1)
    E, k, hid, lat, inter, sh = 16, 4, 96, 32, 48, 80
    cfg = M.NemotronHConfig(hidden_size=hid, hybrid_override_pattern="E", num_hidden_layers=1, vocab_size=64, n_routed_experts=E, num_experts_per_tok=k, moe_intermediate_size=inter,
                            moe_shared_expert_intermediate_size=sh, moe_latent_size=lat, routed_scaling_factor=5.0, norm_topk_prob=True, n_group=1, topk_group=1, mlp_hidden_act="relu2", mlp_bias=False)
    moe = M.NemotronHMoE(cfg, layer_idx=0).float().eval()
    with torch.no_grad():
        for prm in moe.parameters():
            prm.copy_(f32(bf(torch.randn_like(prm) * 0.3)))
        moe.gate.e_score_correction_bias.copy_(torch.randn(E) * 0.1)
        x = bf(torch.randn(40, hid))
        want = moe(f32(x)[None])[0]
        _, tw, ti = moe.gate(f32(x))
    W = dict(gate=bf(moe.gate.weight.data), bias=moe.gate.e_score_correction_bias.data.clone(), fc1=bf(moe.fc1_latent_proj.weight.data), fc2=bf(moe.fc2_latent_proj.weight.data),
             s_up=bf(moe.shared_experts.up_proj.weight.data), s_down=bf(moe.shared_experts.down_proj.weight.data), up=bf(moe.experts.up_proj.data), down=bf(moe.experts.down_proj.data))
    mc = dict(top_k=k, scale=5.0, norm_topk=True)
    idx, w = moe_route(x, W["gate"], W["bias"], k, 5.0)
    dense = lambda i, v: torch.zeros(x.shape[0], E).scatter_(1, i, v)
    ok = bool((dense(idx, w) - dense(ti, tw)).abs().max() <= 1e-5 * float(tw.abs().max())) and bool(torch.equal(idx.sort(-1).values, ti.sort(-1).values))
    print(f"{'ok  ' if ok else 'FAIL'} the mixture-of-experts reference's router against transformers' NemotronHTopkRouter: the same {k} of {E} experts and weights for {x.shape[0]} tokens")
    return stats("a mixture-of-experts layer against transformers' NemotronHMoE (tiny, bf16 weights in ours, fp32 in its)", moe_mixer(x, W, mc), want[0] if want.dim() == 3 else want, 4e-2) and ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    k = sub.add_parser("kernels")
    k.add_argument("what", choices=["gen", "check"])
    k.add_argument("dir")
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--nh", type=int, default=16)
    k.add_argument("--ng", type=int, default=2)
    k.add_argument("--n", type=int, default=300)
    k.add_argument("--split", type=int, default=150)
    k.add_argument("--nq", type=int, default=32)
    k.add_argument("--nk", type=int, default=2)
    k.add_argument("--na", type=int, default=40)
    k.add_argument("--dt-min", type=float, default=0.0)
    k.add_argument("--from", dest="mini", help="gen: the mini checkpoint whose layers (the first Mamba-2 and the first attention layer in it) give the inputs: real activations, the model's heads and groups")
    f = sub.add_parser("fetch")
    f.add_argument("dir")
    f.add_argument("--layers", default="0,2,7")
    f.add_argument("--tokens", type=int, default=333)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--repo", default=REPO)
    f.add_argument("--rev", default=REV)
    ly = sub.add_parser("layers")
    ly.add_argument("what", choices=["gen", "check"])
    ly.add_argument("mini")
    ly.add_argument("io", nargs="?")
    ly.add_argument("--dt-min", type=float, default=0.0)
    ly.add_argument("--level", type=int, choices=[4, 5, 6], help="gen: the level the engine holds a mixture-of-experts layer's matrices at (the mini checkpoint is the grid of one)")
    ly.add_argument("--kinds", default="M*E", help="gen: the kinds of layer to make cases for (M Mamba-2, * attention, E mixture of experts)")
    ty = sub.add_parser("toy")
    ty.add_argument("mini")
    mt = sub.add_parser("moetoy")
    mt.add_argument("mini")
    mt.add_argument("--pattern", default="ME*EME")
    mt.add_argument("--level", type=int, choices=[4, 5, 6], default=6)
    mt.add_argument("--seed", type=int, default=1)
    mt.add_argument("--tokens", type=int, default=300)
    mt.add_argument("--prefill", type=int, default=150)
    mt.add_argument("--total", type=int, default=210)
    rl = sub.add_parser("real")
    rl.add_argument("mini")
    i = sub.add_parser("inventory")
    i.add_argument("out")
    idp = sub.add_parser("ids")
    idp.add_argument("model")
    idp.add_argument("out")
    idp.add_argument("--prefill", type=int, default=64)
    idp.add_argument("--total", type=int, default=160)
    idp.add_argument("--level", type=int, choices=[4, 5, 6], default=5)
    fl = sub.add_parser("full")
    fl.add_argument("model")
    fl.add_argument("out")
    cm = sub.add_parser("compare")
    cm.add_argument("got")
    cm.add_argument("want")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "kernels":
        if a.what == "gen":
            real = None
            if a.mini:
                ls = sorted({int(m.group(1)) for n in read_tensors(f"{a.mini}/model.safetensors") for m in [re.match(r"backbone\.layers\.(\d+)\.", n)] if m})
                kinds = model_cfg(a.mini)["kinds"]
                real = (a.mini, next(l for l in ls if kinds[l] == "M"), next(l for l in ls if kinds[l] == "*"))
            kernels_gen(a.dir, a.seed, a.nh, a.ng, a.n, a.split, a.nq, a.nk, a.na, a.dt_min, real=real)
            return True
        return kernels_check(a.dir)
    if a.cmd == "fetch":
        fetch_mini(a.dir, [int(x) for x in a.layers.split(",")], a.tokens, a.seed, a.repo, a.rev)
        return True
    if a.cmd == "inventory":
        inventory(a.out)
        return True
    if a.cmd == "layers":
        io_dir = a.io or f"{a.mini}/io"
        if a.what == "gen":
            layers_gen(a.mini, io_dir, a.level, a.kinds)
            return True
        return layers_check(a.mini, io_dir, a.dt_min)
    if a.cmd == "toy":
        return toy_check(a.mini)
    if a.cmd == "moetoy":
        moetoy_build(a.mini, a.pattern, a.seed, a.level, a.tokens, a.prefill, a.total)
        return True
    if a.cmd == "real":
        return real_vs_transformers(a.mini)
    if a.cmd == "ids":
        make_ids(a.model, a.out, a.prefill, a.total, a.level)
        return True
    if a.cmd == "full":
        return full_check(a.model, a.out)
    if a.cmd == "compare":
        return compare_logits(a.got, a.want)
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
