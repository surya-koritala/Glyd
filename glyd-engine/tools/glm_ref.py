"""PyTorch reference of GLM-5.3-Flash's Kimi delta attention, hyper-connection (mHC), dense MLP and mixture-of-experts layers for glyd-engine's GLM path (Linear GLY-133).

    glm_ref.py selftest                              the reference against transformers 5.19.0's own functions, modules and model: its delta-rule functions, the hyper-connection, the Kimi delta mixer, the MLP, the router and
                                                     the mixture of experts, the decoder layer and a random tiny model whole (CPU; needs transformers 5.19.0)
    glm_ref.py kernels gen DIR [--heads 4 --n 150 --split 70 --d 512 --seed 1]
                                                     random inputs in DIR/in for `glyd-engine glmkernels DIR/in DIR/out` (a GPU) or tools/glm_emu.cpp's program (any computer)
    glm_ref.py kernels check DIR                     DIR/out against the reference, stage by stage
    glm_ref.py tiny MINI [--level 6|5|4] [--seed 1] [--tokens 300] [--prefill 150] [--total 210]
                                                     a random tiny GLM-5.3-Flash (four layers of Kimi delta attention, a dense MLP in the first and mixture-of-experts layers after it; no Hub read) laid out as the real
                                                     checkpoint is (the first layer's MLP as E4M3 bytes with block scales, the rest bf16, the fp32 parameters fp32): MINI (config.json, model.safetensors of the layers'
                                                     tensors, x.bin: the layers' input, bf16 [tokens, 4, 256]) and MINI/toy (the whole model, ids.txt, toy.txt); the experts and the shared experts are on the grid of the block
                                                     layout of --level, so that the engine's packs hold them exactly
    glm_ref.py fetch OUT_DIR [--layers 0,1,44] [--experts N] [--tokens 333] [--seed 1]
                                                     a mini checkpoint of those layers' tensors (HTTP range reads of the pinned revision: no token, no whole shards) with the real config.json and x.bin (bf16
                                                     [tokens, 4, 4096]: the layers' input); --experts N keeps a mixture-of-experts layer's first N experts only
    glm_ref.py layers gen MINI [IO] [--level 6|5|4] [--kinds DE]
                                                     IO/spec.json (default MINI/io): the cases `glyd-engine glmlayers MINI IO` runs on the mini checkpoint's layers
    glm_ref.py layers check MINI [IO] [--decoded DIR]
                                                     the engine's dumps in IO against the reference, case by case and stage by stage; --decoded: the weights `glyd-engine glmdecode` made of a mixture-of-experts layer's packs
                                                     (what the engine multiplies by at a level), where the mini checkpoint's own FP8 experts are not
    glm_ref.py toy MINI                              `glyd-engine glmtoy MINI/toy MINI/toy`'s logits against the reference
    glm_ref.py fp8 check MINI OUT                    `glyd-engine glmdequant MINI OUT`: the FP8 matrices of the mini checkpoint as the engine reads them (bf16), bit for bit against transformers' Fp8Dequantize
    glm_ref.py mtp MINI                              `glyd-engine glmmtp MINI/toy MINI/toy`'s MTP head against the reference
    glm_ref.py chain OUT REF [--layers A:B] [--teacher]   `glyd-engine glmreal SNAPSHOT REF OUT`'s real layers against the layer-by-layer reference REF (research/glm-data's ref128); the logits of a full chain
    glm_ref.py real MINI                             the reference against transformers' own modules on the mini checkpoint's real weights (CPU; needs transformers 5.19.0)

The reference is transformers 5.19.0's (modeling_glm5_next.py) written as plain torch, rounding to bf16 where transformers does: the projections' outputs, beta (the sigmoid of the input gate's product), the delta rule's
outputs, the gated norm's result, the layer norms' two roundings, the streams' update; and, as the engine's `conv_silu` (qwen35.cu) does, one rounding of the convolution's output after the SiLU (transformers' torch path rounds
after the convolution and again after the SiLU: selftest shows the two a place apart at most). The delta rule's state, the forget gate (-5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias)), fp32) and the mixing weights are fp32.
Each stage of a case is checked from the engine's own input to that stage, so a difference is the stage's own; the layer's output is also checked from the layer's input through the reference alone (a looser bound: the
engine's and torch's products differ in the last place).

The mixture-of-experts layer is transformers' Glm5NextTextMoE with the roundings of src/moe.rs: the router in fp32 (sigmoid scores; the k largest of scores + e_score_correction_bias, equal ones the smaller expert; their
scores, not the biased ones, divided by their sum + 1e-20 and times routed_scaling_factor), each chosen expert's gate and up products stored in bf16 and the clamped SwiGLU rounded once, its down product in fp32 times the pair's
weight, summed in fp32 and rounded once, and the shared expert (bf16) added in fp32 and rounded once.
"""
import argparse
import importlib.util
import json
import math
import os
import re
import struct
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _tool(name):
    """A sibling tool's module, loaded by its path (python -I does not look in the script's directory): the comparisons, the file formats and the by-range reads that tools/nemotron_ref.py has."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nr = _tool("nemotron_ref")
BF = torch.bfloat16
bf, f32, stats, put, get = nr.bf, nr.f32, nr.stats, nr.put, nr.get
hf_get, shard_header, write_safetensors, put_bytes = nr.hf_get, nr.shard_header, nr.write_safetensors, nr.put_bytes
moe_route, moe_choice, moe_weights = nr.moe_route, nr.moe_choice, nr.moe_weights

REPO = "zai-org/GLM-5.3-Flash"
REV = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
P = 128  # a Kimi delta head's channels, the state's side: what the kernels are written for
HC = 4  # residual streams
CHUNK = 32  # tokens a chunk of the chunked delta rule (the engine's GM_L)
PREFIX = "model.language_model."


# ---- the layer's pieces -------------------------------------------------------------------------------------------------------------------------------------------------------------

def rmsnorm(x, w, eps):
    """Glm5NextTextRMSNorm: fp32 inside, the normed value rounded to bf16, then times the weight (bf16 by bf16, rounded again)."""
    xf = f32(x)
    return bf(f32(w) * f32(bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))))


def sigmoid(x):
    return torch.sigmoid(x)


def tokens_within(name, got, want, bound, share):
    """Each row (token) of got against want: its largest difference relative to the largest value of want's rows (all of them); returns whether at least `share` of the rows are within `bound`. For a layer of experts, where a
    router that scores two experts within the rounding of each other picks the other of them for a token and that token's output moves by an expert's share."""
    g, w = f32(got).detach().reshape(got.shape[0], -1), f32(want).detach().reshape(got.shape[0], -1)
    dv = (g - w).abs().max(-1).values / max(float(w.abs().max()), 1e-30)
    good = float((dv <= bound).float().mean())
    ok = bool(torch.isfinite(g).all()) and good >= share
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {g.shape[0]} rows, {100 * good:.1f}% within {bound:g} of the largest value (at least {100 * share:.0f}%), the median row differs by {float(dv.median()):.3e}, the worst by {float(dv.max()):.3e}")
    return ok


def conv_silu(x, w, state):
    """The causal depthwise convolution of kernel K and SiLU, rounded once (causal-conv1d's kernel; the engine's conv_silu): the taps oldest first, no bias. x [T, C] bf16, w [C, 1, K], state [K-1, C] (the last K-1 inputs,
    oldest first) -> (y [T, C], new state)."""
    T, C = x.shape
    K = w.shape[-1]
    xin = torch.cat([state, x], 0)
    acc = f32(w[:, 0, 0]) * f32(xin[0:T])
    for k in range(1, K):
        acc = acc + f32(w[:, 0, k]) * f32(xin[k:k + T])
    return bf(F.silu(acc)), xin[T:].clone()


def l2norm(x, eps=1e-6):
    """transformers' (flash-linear-attention's) l2norm: x / sqrt(sum x^2 + eps), over the last dimension, fp32."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate(fg, dt_bias, A_log, bound, heads):
    """The forget gate g [T, H, P] fp32: bound * sigmoid(exp(A_log) * (fg + dt_bias)), fg [T, H P] bf16 (f_b(f_a(x))), dt_bias [H P] and A_log [H] fp32 (Glm5NextTextForgetGate)."""
    g = (f32(fg) + f32(dt_bias)).reshape(-1, heads, P)
    return bound * sigmoid(torch.exp(f32(A_log)).reshape(1, heads, 1) * g)


def kda_beta(bb):
    """The input gate beta [T, H]: sigmoid of b_proj's output, rounded to bf16 (torch.sigmoid of a bf16 tensor)."""
    return f32(bf(sigmoid(f32(bb))))


def kda_recurrent(q, k, v, g, beta, S):
    """The delta rule with a decay for each key channel, token by token (transformers' recurrent_kimi_delta_attention, flash-linear-attention's fused_recurrent_kda): q, k, v [T, H, P] bf16 (the convolution's output), g [T, H, P] fp32,
    beta [T, H] fp32; S [H, P, P] fp32 (S[h][i][j]: i over the keys' channels) is updated in place -> core [T, H, P] bf16."""
    T, H, _ = q.shape
    qn, kn = l2norm(f32(q)) * (P ** -0.5), l2norm(f32(k))
    vf = f32(v)
    out = torch.empty(T, H, P, dtype=torch.float32, device=q.device)
    for t in range(T):
        S.mul_(torch.exp(g[t])[:, :, None])
        d = (vf[t] - (S * kn[t][:, :, None]).sum(1)) * beta[t][:, None]
        S.add_(kn[t][:, :, None] * d[:, None, :])
        out[t] = (S * qn[t][:, :, None]).sum(1)
    return bf(out)


def kda_chunked(q, k, v, g, beta, S, chunk=CHUNK):
    """The same rule in chunks (the WY form: transformers' chunk_kimi_delta_attention, flash-linear-attention's chunk_kda; the engine's glm_kda_chunk), fp32: arguments as `kda_recurrent`; S is the state at the start and is
    left holding the state at the end. Inside a chunk, with G the cumulative sum of g: A[t][s] = beta_t sum_i k_t k_s exp(G_t - G_s) (s < t), Vn = (I + A)^-1 beta (V - (k exp(G)) S0), the outputs (q exp(G)) S0 + P Vn with
    P[t][s] = sum_i q_t k_s exp(G_t - G_s) (s <= t), the state exp(G_last) S0 + sum_s (k_s exp(G_last - G_s)) Vn_s^T."""
    T, H, _ = q.shape
    qn, kn = l2norm(f32(q)) * (P ** -0.5), l2norm(f32(k))
    vf = f32(v)
    out = torch.empty(T, H, P, dtype=torch.float32, device=q.device)
    for t0 in range(0, T, chunk):
        tc = min(chunk, T - t0)
        sl = slice(t0, t0 + tc)
        Qc, Kc, Vc, Gc, Bc = qn[sl].transpose(0, 1), kn[sl].transpose(0, 1), vf[sl].transpose(0, 1), g[sl].transpose(0, 1).cumsum(1), beta[sl].transpose(0, 1)  # [H, tc, ...]
        causal = torch.tril(torch.ones(tc, tc, dtype=torch.bool, device=q.device))
        D = torch.exp((Gc[:, :, None, :] - Gc[:, None, :, :]).masked_fill(~causal[None, :, :, None], -math.inf))  # [H, t, s, P]: exp(G_t - G_s) for s <= t (a pair the other way round is not a decay)
        A = (Kc[:, :, None, :] * Kc[:, None, :, :] * D).sum(-1) * Bc[:, :, None] * torch.tril(torch.ones(tc, tc, device=q.device), -1)
        Y = Bc[:, :, None] * (Vc - torch.einsum("htp,hpj->htj", Kc * torch.exp(Gc), S))
        Vn = torch.linalg.solve_triangular(torch.eye(tc, device=q.device) + A, Y, upper=False, unitriangular=True)
        Pm = (Qc[:, :, None, :] * Kc[:, None, :, :] * D).sum(-1)
        out[sl] = (torch.einsum("htp,hpj->htj", Qc * torch.exp(Gc), S) + Pm @ Vn).transpose(0, 1)
        S.mul_(torch.exp(Gc[:, -1])[:, :, None]).add_(torch.einsum("htp,htj->hpj", Kc * torch.exp(Gc[:, -1:, :] - Gc), Vn))
    return bf(out)


def kda_norm(core, og, w, eps):
    """The gated norm of the delta rule's outputs (Glm5NextTextRMSNormGated with the sigmoid activation): per head the rms norm of the 128 channels of core [T, H P] (bf16), fp32, times the weight w [P], times sigmoid(og)
    (og [T, H P] bf16), one rounding."""
    T = core.shape[0]
    c = f32(core).reshape(T, -1, P)
    n = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps)
    return bf((f32(w) * n * sigmoid(f32(og).reshape(T, -1, P))).reshape(T, -1))


def kda_mixer(h, W, cfg, st, chunked=True, stages=None):
    """One Kimi delta attention layer's mixer on h [T, hidden] bf16 (the normed input), continuing from st = {conv [3, 3 H P] bf16, ssm [H, P, P] fp32} (updated in place) -> [T, hidden] bf16. stages (a dict) takes each
    stage's output."""
    H = cfg["heads"]
    mix = F.linear(h, W["qkv"])
    cv, st["conv"] = conv_silu(mix, W["conv"], st["conv"])
    qd = H * P
    q, k, v = (cv[:, i * qd:(i + 1) * qd].reshape(-1, H, P) for i in range(3))
    fg = F.linear(F.linear(h, W["f_a"]), W["f_b"])
    bb = F.linear(h, W["b_proj"])
    og = F.linear(F.linear(h, W["g_a"]), W["g_b"])
    core = (kda_chunked if chunked else kda_recurrent)(q, k, v, kda_gate(fg, W["dt_bias"], W["A_log"], cfg["bound"], H), kda_beta(bb), st["ssm"]).reshape(-1, qd)
    att = kda_norm(core, og, W["o_norm"], cfg["eps"])
    out = F.linear(att, W["o_proj"])
    if stages is not None:
        stages.update(mix=mix, conv=cv, fg=fg, bb=bb, og=og, core=core, att=att, out=out)
    return out


def layernorm(x, w, b, eps):
    """nn.LayerNorm (the indexer's k_norm) on a bf16 tensor: fp32 inside (mean, biased variance), weight and bias, one rounding."""
    xf = f32(x)
    mu = xf.mean(-1, keepdim=True)
    var = (xf - mu).pow(2).mean(-1, keepdim=True)
    return bf((xf - mu) * torch.rsqrt(var + eps) * f32(w) + f32(b))


# ---- latent attention over the keys an indexer picks (Glm5NextTextAttention and Glm5NextTextIndexer) ------------------------------------------------------------------------------------------
# No rotary anywhere. A token's cache is its latent c (kv_a_proj_with_mqa's product normed, kv_lora_rank wide), which is both its key and its value through kv_b_proj: the head h's key is Wk_h c and its value Wv_h c with
# Wk_h = kv_b_proj's rows h (nope + v) .. + nope and Wv_h the v rows after them. Attention is run in that latent space (the absorbed form): the query q_h (nope wide) becomes q_abs = Wk_h^T q_h (kv_lora_rank wide), the scores
# q_abs . c_t, and the output sum_t p_t c_t is multiplied by Wv_h at the end: the same numbers as transformers' expanded keys and values, the cache 16 times smaller.

def mla_project(h, W, cfg):
    """The projections of a latent attention layer's mixer on h [T, hidden] bf16 (the normed input): (q_resid, q, c, iq, ik, gs, wts), all bf16: the query's latent normed, the query [T, heads nope], the token's latent normed, the
    indexer's queries [T, ih idim], its keys normed by a LayerNorm, its pooling gate scores [T, idim] and its head weights [T, ih]."""
    eps = cfg["eps"]
    qr = rmsnorm(F.linear(h, W["q_a"]), W["q_a_ln"], eps)
    q = F.linear(qr, W["q_b"])
    c = rmsnorm(F.linear(h, W["kv_a"]), W["kv_a_ln"], eps)
    iq = F.linear(qr, W["wq_b"])
    ik = layernorm(F.linear(h, W["wk"]), W["k_norm_w"], W["k_norm_b"], 1e-6)
    gs = F.linear(h, W["ix_gate"])
    wts = F.linear(h, W["weights_proj"])
    return qr, q, c, iq, ik, gs, wts


def mla_pool_keys(ik, gs, ape, kpool):
    """The indexer's keys of the complete pools of a sequence (ik, gs [T, idim] bf16: the normed keys, the gate scores): a pool of `kpool` consecutive tokens (from position 0) has per channel the softmax over its tokens of
    gs + ape (ape [kpool, idim]) as weights of the keys, summed -> [T // kpool, idim] bf16."""
    T, D = ik.shape
    n = T // kpool
    if n == 0:
        return ik[:0]
    g = f32(gs[:n * kpool]).reshape(n, kpool, D) + f32(ape)[None]
    return bf((torch.softmax(g, dim=1) * f32(ik[:n * kpool]).reshape(n, kpool, D)).sum(1))


def mla_index_scores(iq, wts, pk, ih, idim):
    """The score of each pool for each query row: sum over the indexer's heads of w_h relu(iq_h . pk / sqrt(idim)), w = wts / sqrt(ih); iq [Tq, ih idim], wts [Tq, ih], pk [P, idim] -> [Tq, P] fp32."""
    iq, wts, pkf = f32(iq).reshape(-1, ih, idim), f32(wts) * ih ** -0.5, f32(pk)
    out = []
    for r in range(0, iq.shape[0], 256):  # (a block of rows at a time: the scores of a long sequence are rows x heads x pools)
        s = torch.relu(torch.einsum("thd,pd->thp", iq[r:r + 256], pkf) * idim ** -0.5)
        out.append(torch.einsum("th,thp->tp", wts[r:r + 256], s))
    return torch.cat(out) if out else torch.zeros(0, pkf.shape[0])


def mla_select(scores, q, topk, kpool, tail):
    """The tokens a query at position q attends to, ascending: the pools whose last token is at most q (q + 1 // kpool of them) are scored (scores [P] fp32, the first of them used); the best topk // kpool (equal ones the smaller
    pool) with all their tokens, and the incomplete last pool's tokens (q + 1 % kpool) where `tail`. Where the pools are no more than topk // kpool, all of them: every token up to q."""
    P = (q + 1) // kpool
    k = min(topk // kpool, P)
    pools = torch.arange(P) if k == P else torch.sort(torch.sort(-scores[:P], stable=True).indices[:k]).values
    toks = (pools[:, None] * kpool + torch.arange(kpool)[None]).reshape(-1)
    return torch.cat([toks, torch.arange(P * kpool, q + 1)]) if tail else toks


def mla_attend(q, c, sel, W, cfg):
    """The attention of each row (a query at position pos = its index) over the tokens sel[i] of the latents c [T, kv_rank] (bf16), in the latent space: q_abs = Wk_h^T q_h (fp32), softmax of q_abs . c_t / sqrt(nope) in fp32,
    olat = the weighted sum of the latents (fp32 [T, heads, kv_rank]), ov = olat times Wv_h rounded to bf16 [T, heads v]."""
    NH, nope, vd, KR = cfg["nh"], cfg["nope"], cfg["vd"], cfg["kv_rank"]
    Wkv = f32(W["kv_b"]).reshape(NH, nope + vd, KR)
    q_abs = torch.einsum("thd,hdj->thj", f32(q).reshape(-1, NH, nope), Wkv[:, :nope])
    cf = f32(c)
    olat = torch.empty(q.shape[0], NH, KR)
    for i in range(q.shape[0]):
        ci = cf[sel[i]]
        olat[i] = torch.softmax(torch.einsum("hj,tj->ht", q_abs[i], ci) * nope ** -0.5, -1) @ ci
    return olat, bf(torch.einsum("thj,hej->the", olat, Wkv[:, nope:]).reshape(q.shape[0], NH * vd))


def mla_select_all(iq, wts, ik, gs, W, cfg):
    """Every row's selected tokens, for a whole sequence from position 0: the pool keys, the scores, the choice (`mla_select`) -> (pk [T // kpool, idim] bf16, [tokens of row i])."""
    T, kp = iq.shape[0], cfg["kpool"]
    pk = mla_pool_keys(ik, gs, W["ape"], kp)
    sc = mla_index_scores(iq, wts, pk, cfg["ih"], cfg["idim"]) if pk.shape[0] else None
    return pk, [mla_select(sc[i] if sc is not None else None, i, cfg["topk"], kp, cfg["tail"]) for i in range(T)]


def mla_mixer(h, W, cfg, stages=None, sel=None):
    """One latent attention layer's mixer on h [T, hidden] bf16 (the normed input; a whole sequence from position 0) -> [T, hidden] bf16. stages (a dict) takes each stage's output. sel: the tokens of each row to
    attend to (a list of index tensors), the engine's, instead of the reference's own choice."""
    qr, q, c, iq, ik, gs, wts = mla_project(h, W, cfg)
    if sel is None:
        pk, sel = mla_select_all(iq, wts, ik, gs, W, cfg)
    else:
        pk = None
    olat, ov = mla_attend(q, c, sel, W, cfg)
    out = F.linear(ov, W["o_proj"])
    if stages is not None:
        stages.update(qr=qr, q=q, c=c, iq=iq, ik=ik, gs=gs, wts=wts, pk=pk, sel=sel, olat=olat, ov=ov, out=out)
    return out


def hc_mix(x, fn, base, scale, hc_eps, iters, eps):
    """Glm5NextTextHyperConnection on the streams x [T, 4, D] bf16: fn [24, 4D] bf16, base [24] and scale [3] fp32 -> (pre [T, 4], post [T, 4], comb [T, 4, 4]) fp32 and the collapse sum of pre * x over the streams, bf16
    [T, D]. The 4D values are normed without a weight (fp32), projected, the three groups of the 24 turned into weights (sigmoid + hc_eps; 2 sigmoid; softmax over each row of the matrix + hc_eps), the matrix made doubly
    stochastic by `iters` rounds of Sinkhorn-Knopp (the columns first, then each round the rows and the columns, every division by sum + hc_eps)."""
    T = x.shape[0]
    flat = f32(x).reshape(T, -1)
    proj = (flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)) @ f32(fn).T
    pw, qw, cw = proj[:, :HC], proj[:, HC:2 * HC], proj[:, 2 * HC:].reshape(T, HC, HC)
    base, scale = f32(base), f32(scale)
    pre = sigmoid(pw * scale[0] + base[:HC]) + hc_eps
    post = 2 * sigmoid(qw * scale[1] + base[HC:2 * HC])
    comb = torch.softmax(cw * scale[2] + base[2 * HC:].reshape(HC, HC), -1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb, bf((pre[:, :, None] * f32(x)).sum(1))


def hc_post(x, o, post, comb):
    """The streams after a layer's block: bf16(bf16(post_i * o) + bf16(sum over j of comb[j][i] * x_j)) with post and comb rounded to bf16 first (post.to(dtype) * out + comb.to(dtype)^T @ residual), x [T, 4, D], o [T, D]."""
    a = bf(f32(bf(post))[:, :, None] * f32(o)[:, None, :])
    m = bf(torch.einsum("tji,tjd->tid", f32(bf(comb)), f32(x)))
    return bf(f32(a) + f32(m))


def hc_head(x, w, eps):
    """The model's last step: the mean of the four streams (fp32 sums, rounded to bf16) and the final norm."""
    return rmsnorm(bf(f32(x).sum(1) / HC), w, eps)


def swiglu(gu, limit):
    """The clamped SwiGLU of a product of gate and up rows gu [T, 2 I] (bf16): silu(min(gate, limit)) * clamp(up, -limit, limit), fp32, one rounding (src/moe.rs, moe_act)."""
    I = gu.shape[-1] // 2
    g, u = f32(gu[..., :I]).clamp(max=limit), f32(gu[..., I:]).clamp(-limit, limit)
    return bf(F.silu(g) * u)


def mlp(h, gate_up, down, limit):
    """A dense MLP (or the shared expert): down(swiglu(gate_up(h))), bf16 products."""
    return F.linear(swiglu(F.linear(h, gate_up), limit), down)


def moe_experts(h, idx, w, gate_up, down, limit):
    """The chosen experts' down(swiglu(gate_up(h))) on h [T, hidden], weighted and summed (Glm5NextTextExperts with src/moe.rs's roundings): each expert on the tokens that chose it, the gate and up products and the activation
    stored in bf16, the down product in fp32 times the pair's weight w [T, k], the pairs summed in fp32, one rounding. gate_up(e) -> [2 I, hidden], down(e) -> [hidden, I] -> [T, hidden] bf16."""
    y = torch.zeros(h.shape[0], h.shape[1], dtype=torch.float32, device=h.device)
    for e in idx.unique().tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        y.index_add_(0, tok, F.linear(f32(swiglu(F.linear(h[tok], gate_up(e)), limit)), f32(down(e))) * w[tok, slot, None])
    return bf(y)


def moe_mixer(h, W, cfg, idx=None, w=None, stages=None):
    """One layer's mixture-of-experts mixer on h [T, hidden] bf16 (the normed input) -> [T, hidden] bf16: the routed experts' weighted sum + the shared expert's output (a sum in fp32, rounded once). idx, w: the
    experts and weights to use (the engine's, to check the rest from its own choice), else the reference's router's."""
    if idx is None:
        idx, w = moe_route(h, W["gate"], W["bias"], cfg["top_k"], cfg["scale"], cfg["norm_topk"])
    elif w is None:  # (the engine's experts: their weights from this reference's own scores)
        w = moe_weights(torch.sigmoid(F.linear(f32(h), f32(W["gate"]))), idx, cfg["scale"], cfg["norm_topk"])
    y = moe_experts(h, idx, w, W["expert_gu"], W["expert_down"], cfg["limit"])
    out = bf(f32(y) + f32(mlp(h, W["s_gate_up"], W["s_down"], cfg["limit"])))
    if stages is not None:
        stages.update(y=y, out=out, ids=idx, wts=w)
    return out


def layer(x, W, cfg, st, chunked=True, stages=None, drop_ffn=False, force=None):
    """One decoder layer on its streams x [T, 4, D] bf16: the attention half (mHC mix, the input norm, the Kimi delta or the latent attention mixer, the streams' update) and the feed-forward half (the same around the MLP or the
    experts) -> the streams. stages (a dict) takes the intermediate results. drop_ffn: the feed-forward adds nothing (to show that a check sees it). force: {"ids": the experts of each row, "sel": the tokens of each
    row} of the engine's, to run the discrete choices of a router or an indexer as it made them (they flip on near ties: the rest is then rounding alone)."""
    out = {}
    pre, post, comb, col = hc_mix(x, W["hc_a_fn"], W["hc_a_base"], W["hc_a_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
    h = rmsnorm(col, W["norm_a"], cfg["eps"])
    force = force or {}
    o = mla_mixer(h, W, cfg, out, force.get("sel")) if "q_a" in W else kda_mixer(h, W, cfg, st, chunked, out)
    x1 = hc_post(x, o, post, comb)
    pre2, post2, comb2, col2 = hc_mix(x1, W["hc_f_fn"], W["hc_f_base"], W["hc_f_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
    h2 = rmsnorm(col2, W["norm_f"], cfg["eps"])
    o2 = moe_mixer(h2, W, cfg, force.get("ids"), stages=out) if "gate" in W else mlp(h2, W["gate_up"], W["down"], cfg["limit"])
    x2 = hc_post(x1, torch.zeros_like(o2) if drop_ffn else o2, post2, comb2)
    if stages is not None:
        stages.update(out, h1=h, o1=o, x1=x1, h2=h2, o2=o2, x2=x2)
    return x2


# ---- the MTP head (checkpoint layer `layers`; transformers does not implement it, vLLM's mtp.py and Glm5NextDecoderLayer(is_mtp_layer) are the description) --------------------------------------------------------
# For a row j: the previous position's final hidden state h_j (normed, bf16) and the token t_{j+1} that follows it; enorm(embedding(t_{j+1})) and hnorm(h_j) side by side are projected to the width by eh_proj; a latent
# attention layer with experts on a plain residual stream (input_layernorm, the mixer, post_attention_layernorm, the experts and the shared expert, no hyper-connections) and shared_head.norm give the head's normed
# hidden state, whose LM head is the model's; that hidden state and the token drafted from it are the next draft step's inputs. The norms that fuse an addition (vLLM's fused add and RMSNorm: the sum rounded to bf16
# as the residual, the norm taken from the sum before its rounding, one rounding) are the engine's `rmsnorm_add`, the layer's input norm its `rmsnorm` (one rounding), the two norms of the inputs the layers' own (two roundings).

def rmsnorm1(x, w, eps):
    """One rounding (vLLM's RMSNorm, the engine's `rmsnorm`): x * rsqrt(mean(x^2) + eps) * w in fp32, rounded once."""
    xf = f32(x)
    return bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * f32(w))


def add_rmsnorm1(res, y, w, eps):
    """The residual plus y rounded to bf16, and the norm of the sum before its rounding (one rounding) -> (residual, normed)."""
    sm = f32(res) + f32(y)
    return bf(sm), bf(sm * torch.rsqrt(sm.pow(2).mean(-1, keepdim=True) + eps) * f32(w))


def mtp_forward(prev, toks, embed, W, cfg, stages=None):
    """The head on rows (prev [T, hidden] bf16: the final normed hidden states of positions 0..T-1, toks [T]: the tokens after them; the rows are a sequence from position 0) -> the head's normed hidden states [T, hidden]
    bf16. W: `load_layer` of the MTP layer plus enorm, hnorm, eh_proj, head_norm."""
    eps = cfg["eps"]
    x = F.linear(torch.cat([rmsnorm(embed[toks], W["enorm"], eps), rmsnorm(prev, W["hnorm"], eps)], -1), W["eh_proj"])
    a = mla_mixer(rmsnorm1(x, W["norm_a"], eps), W, cfg)
    res, m_in = add_rmsnorm1(x, a, W["norm_f"], eps)
    mo = moe_mixer(m_in, W, cfg, stages=stages)
    _, out = add_rmsnorm1(res, mo, W["head_norm"], eps)
    if stages is not None:
        stages.update(x=x, a=a, m_in=m_in, mo=mo)
    return out


def mtp_weights(tensors, cfg, dev):
    """The MTP head's weights: its layer's (`load_layer`) and its own tensors."""
    l = cfg["layers"]
    W = load_layer(tensors, l, dev)
    p = f"{PREFIX}layers.{l}"
    W.update(enorm=tensors[f"{p}.enorm.weight"].to(dev), hnorm=tensors[f"{p}.hnorm.weight"].to(dev), eh_proj=tensors[f"{p}.eh_proj.weight"].to(dev), head_norm=tensors[f"{p}.shared_head.norm.weight"].to(dev))
    return W


# ---- the kernels on random inputs: `glyd-engine glmkernels DIR/in DIR/out`, or tools/glm_emu.cpp's program, against the reference ----------------------------------------------------------

CASES = ("chunked", "recurrent", "steps", "mixed")  # the delta rule in chunks (two calls), token by token (two calls), one token a call, chunks for the first call then single tokens


def kernels_gen(d, seed, heads, n, split, dim, mla="small"):
    """Inputs of the kernels in DIR/in: random, with spreads like the model's (gates from fast decay, -5 a token, to none; q, k and v of about unit size; streams and mixing weights of a size that makes the weights vary)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape, mean=0.0, std=1.0: torch.randn(*shape, generator=g) * std + mean
    u = lambda *shape, lo=0.0, hi=1.0: torch.rand(*shape, generator=g) * (hi - lo) + lo
    os.makedirs(f"{d}/in", exist_ok=True)
    qd = heads * P
    tens = dict(xc=r(n, 3 * qd, std=1.0), fg=r(n, qd, std=1.5), bb=r(n, heads, std=1.5), og=r(n, qd, std=1.5), ow=u(P, lo=0.5, hi=1.5), hx=r(n, HC, dim) * u(1, HC, 1, lo=0.5, hi=2.0)[0], hfn=r(24, HC * dim, std=0.03),
                hnw=u(dim, lo=0.5, hi=1.5), ho=r(n, dim), cx=r(n, 3 * qd, std=1.5), cw=r(3 * qd, 1, 4, std=0.4), cst0=r(3, 3 * qd))
    for name, t in tens.items():
        put(f"{d}/in/{name}.bin", bf(t))
    for name, t in dict(dtb=r(qd, std=1.0), alog=r(heads, std=0.4), state0=r(heads, P, P, std=0.3), hbase=r(24, std=0.5), hscale=torch.tensor([0.9, 1.1, 0.8])).items():
        t.contiguous().numpy().tofile(f"{d}/in/{name}.bin")
    open(f"{d}/in/kda.txt", "w").write(f"{heads} {n} {split} -5.0 1e-5\n")
    open(f"{d}/in/hc.txt", "w").write(f"{dim} {n} 1e-5 1e-6 20\n")
    print(f"kernels: {heads} heads, {n} tokens (a first call of {split}), streams of {dim} columns: inputs in {d}/in")
    mla_kernels_gen(d, seed, mla, n=2600 if mla == "model" else 60, first=700 if mla == "model" else 23)


def kernels_check(d):
    I, O = f"{d}/in", f"{d}/out"
    heads, n, split, bound, eps = open(f"{I}/kda.txt").read().split()
    heads, n, split, bound, eps = int(heads), int(n), int(split), float(bound), float(eps)
    qd = heads * P
    rd = lambda name, *shape, dtype=BF: get(f"{I}/{name}.bin", shape, dtype)
    xc, fg, bb, og, ow = rd("xc", n, 3 * qd), rd("fg", n, qd), rd("bb", n, heads), rd("og", n, qd), rd("ow", P)
    dtb, alog, state0 = rd("dtb", qd, dtype=torch.float32), rd("alog", heads, dtype=torch.float32), rd("state0", heads, P, P, dtype=torch.float32)
    ok = True
    mla_ok = mla_kernels_check(d) if os.path.exists(f"{O}/mla_ov.bin") else True
    if not os.path.exists(f"{O}/chunked_core.bin") and os.path.exists(f"{O}/mla_ov.bin") and not os.path.exists(f"{O}/cases.txt"):
        return mla_ok  # (GLM_EMU_NO_KDA: the latent attention's kernels alone)
    ok = bool(mla_ok)
    cases = open(f"{O}/cases.txt").read().split() if os.path.exists(f"{O}/cases.txt") else list(CASES)  # (only tools/glm_emu.cpp runs fewer cases: GLM_EMU_CASES)
    q, k, v = (xc[:, i * qd:(i + 1) * qd].reshape(n, heads, P) for i in range(3))
    g, beta = kda_gate(fg, dtb, alog, bound, heads), kda_beta(bb)
    S = state0.clone()
    ref_core = kda_recurrent(q, k, v, g, beta, S).reshape(n, qd)
    S2 = state0.clone()
    chunk_core = kda_chunked(q, k, v, g, beta, S2).reshape(n, qd)
    ok &= stats("reference: the chunked rule against the recurrence, outputs", chunk_core, ref_core, 8e-3)
    ok &= stats("reference: the chunked rule against the recurrence, final state", S2, S, 1e-4)
    for c in cases:
        if not os.path.exists(f"{O}/{c}_core.bin"):
            print(f"no {O}/{c}_core.bin")
            return False
        eng = lambda name, *shape, dtype=BF: get(f"{O}/{c}_{name}.bin", shape, dtype)
        core, att, state = eng("core", n, qd), eng("att", n, qd), eng("state", heads, P, P, dtype=torch.float32)
        ok &= stats(f"{c}: delta rule outputs", core, ref_core, 8e-3)
        ok &= stats(f"{c}: delta rule final state", state, S, 1e-4)
        ok &= stats(f"{c}: gated norm (on the engine's outputs)", att, kda_norm(core, og, ow, eps), 1e-2, 2)
    base = {c: get(f"{O}/{c}_core.bin", (n, qd)) for c in cases}
    for c in cases[1:]:
        ok &= stats(f"{c} against {cases[0]} (whole delta rule outputs)", base[c], base[cases[0]], 1e-2)
    # the convolution (qwen35.cu's conv_silu: only the GPU runs it; three calls, the state passed on)
    if os.path.exists(f"{O}/conv.bin"):
        rconv, rcst = conv_silu(rd("cx", n, 3 * qd), rd("cw", 3 * qd, 1, 4), rd("cst0", 3, 3 * qd))
        ok &= stats("convolution", get(f"{O}/conv.bin", (n, 3 * qd)), rconv, 2e-2, 2)
        same = bool((get(f"{O}/cstate.bin", (3, 3 * qd)).view(torch.int16) == rcst.view(torch.int16)).all())
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} the convolution's state is the last three inputs")
    # the streams
    dim, n2, heps, hc_eps, iters = open(f"{I}/hc.txt").read().split()
    dim, n2, heps, hc_eps, iters = int(dim), int(n2), float(heps), float(hc_eps), int(iters)
    hx, hfn, hnw, ho = rd("hx", n2, HC, dim), rd("hfn", 24, HC * dim), rd("hnw", dim), rd("ho", n2, dim)
    hbase, hscale = rd("hbase", 24, dtype=torch.float32), rd("hscale", 3, dtype=torch.float32)
    if os.path.exists(f"{O}/hc_params.bin"):
        params = get(f"{O}/hc_params.bin", (n2, 24), torch.float32)
        pre, post, comb, col = hc_mix(hx, hfn, hbase, hscale, hc_eps, iters, heps)
        ok &= stats("hyper-connection: pre, post and comb", params, torch.cat([pre, post, comb.reshape(n2, 16)], 1), 1e-4)
        epre = params[:, :4]
        ecol = bf((epre[:, :, None] * f32(hx)).sum(1))
        if os.path.exists(f"{O}/hc_params_s.bin"):  # (a decode step's: the split projection of the first rows)
            m8 = min(n2, 8)
            ps = get(f"{O}/hc_params_s.bin", (m8, 24), torch.float32)
            pre_s, post_s, comb_s, _ = hc_mix(hx[:m8], hfn, hbase, hscale, hc_eps, iters, heps)
            ok &= stats("hyper-connection, few rows (split projection): pre, post and comb", ps, torch.cat([pre_s, post_s, comb_s.reshape(m8, 16)], 1), 1e-4)
            ok &= stats("hyper-connection, few rows: the normed collapse (on the engine's pre)", get(f"{O}/hc_h_s.bin", (m8, dim)), rmsnorm(bf((ps[:, :4, None] * f32(hx[:m8])).sum(1)), hnw, heps), 1e-2, 2 if dim <= 1024 else None)
        ok &= stats("hyper-connection: the normed collapse (on the engine's pre)", get(f"{O}/hc_h.bin", (n2, dim)), rmsnorm(ecol, hnw, heps), 1e-2, 2 if dim <= 1024 else None)  # (at the model's width the sums of squares add up in another order: a place or two more on the small values)
        ok &= stats("hyper-connection: the streams' update (on the engine's post and comb)", get(f"{O}/hc_post.bin", (n2, HC, dim)), hc_post(hx, ho, params[:, 4:8], params[:, 8:].reshape(n2, 4, 4)), 1e-2, 1)
        ok &= stats("hyper-connection: the mean of the streams and the final norm", get(f"{O}/hc_head.bin", (n2, dim)), hc_head(hx, hnw, heps), 1e-2, 1)
        ok &= stats("hyper-connection: the embedding spread over the streams", get(f"{O}/hc_streams.bin", (n2, HC, dim)), ho[:, None, :].expand(n2, HC, dim), 0.0)
    print("kernels:", "ok" if ok else "FAILED")
    return ok


# ---- the latent attention's kernels on random inputs (`kernels gen` also makes them; `glyd-engine glmkernels` and tools/glm_emu.cpp run them) ----------------------------------------------------------------------

MLA_DIMS = dict(small=dict(NH=2, nope=64, vd=64, KR=64, IH=4, ID=32, kp=4, topk=8), model=dict(NH=64, nope=256, vd=256, KR=512, IH=32, ID=128, kp=4, topk=2048),
                # (the edges: a latent width that does not divide 2,048, and no tail, so that a query before the first whole pool attends to nothing)
                edge=dict(NH=3, nope=96, vd=64, KR=192, IH=4, ID=32, kp=4, topk=8, tail=False))


def mla_calls(n, first):
    """The rows of a sequence of n in calls: a first of `first`, then odd sizes (single steps among them) that put the calls' ends in the middle of pools."""
    sizes, rest, k = [first], n - first, 0
    cyc = [1, 3, 1, 5, 2, 1, 4, 7, 1, 1]
    while rest > 0:
        sizes.append(min(cyc[k % len(cyc)], rest))
        rest -= sizes[-1]
        k += 1
    return sizes


def mla_kernels_gen(d, seed, which="small", n=60, first=23):
    """Inputs of the latent attention's kernels in DIR/in: the projections' raw outputs of n rows (the token latents before their norm, the indexer's keys before theirs, its gate scores, queries and head weights, the
    attention's queries) and the matrix kv_b_proj, random with spreads like the model's, and mla.txt (the dimensions, the calls: the rows of a sequence in calls of those sizes)."""
    g = torch.Generator().manual_seed(seed + 100)
    r = lambda *shape, std=1.0: torch.randn(*shape, generator=g) * std
    u = lambda *shape, lo=0.0, hi=1.0: torch.rand(*shape, generator=g) * (hi - lo) + lo
    D = MLA_DIMS[which]
    NH, nope, vd, KR, IH, ID, kp = (D[k] for k in ("NH", "nope", "vd", "KR", "IH", "ID", "kp"))
    os.makedirs(f"{d}/in", exist_ok=True)
    tens = dict(kvraw=r(n, KR, std=1.5), kvw=u(KR, lo=0.5, hi=1.5), ikraw=r(n, ID, std=1.2), knw=u(ID, lo=0.5, hi=1.5), knb=r(ID, std=0.2), gsraw=r(n, ID), ape=r(kp, ID, std=0.5), iq=r(n, IH * ID),
                wts=r(n, IH), q=r(n, NH * nope), wkv=torch.cat([r(NH, nope, KR, std=0.25), r(NH, vd, KR, std=0.1)], 1).reshape(NH * (nope + vd), KR))
    for name, t in tens.items():
        put(f"{d}/in/mla_{name}.bin", bf(t))
    calls = mla_calls(n, min(first, n))
    open(f"{d}/in/mla.txt", "w").write(" ".join(str(x) for x in [NH, nope, vd, KR, IH, ID, kp, D["topk"], n, len(calls)] + calls) + f" 1e-5 {int(D.get('tail', True))}\n")
    print(f"mla kernels ({which}): {NH} heads of {nope}/{vd}, latents of {KR}, an indexer of {IH} heads of {ID} picking {D['topk']} tokens' worth of pools of {kp}; {n} rows in {len(calls)} calls: inputs in {d}/in")


def mla_kernels_check(d):
    """DIR/out's latent attention kernels' results against the reference, stage by stage (each from the engine's own input to it)."""
    I, O = f"{d}/in", f"{d}/out"
    m = open(f"{I}/mla.txt").read().split()
    NH, nope, vd, KR, IH, ID, kp, topk, n, nc = (int(x) for x in m[:10])
    eps = float(m[10 + nc])
    tail = bool(int(m[11 + nc])) if len(m) > 11 + nc else True
    rd = lambda name, *shape, dtype=BF: get(f"{I}/mla_{name}.bin", shape, dtype)
    kvraw, kvw, ikraw, knw, knb, gsraw, ape, iq, wts, q, wkv = (rd("kvraw", n, KR), rd("kvw", KR), rd("ikraw", n, ID), rd("knw", ID), rd("knb", ID), rd("gsraw", n, ID), rd("ape", kp, ID), rd("iq", n, IH * ID), rd("wts", n, IH),
                                                                  rd("q", n, NH * nope), rd("wkv", NH * (nope + vd), KR))
    cfg = dict(nh=NH, nope=nope, vd=vd, kv_rank=KR, ih=IH, idim=ID, kpool=kp, topk=topk, tail=tail, eps=eps)
    W = dict(kv_b=wkv, ape=ape)
    Pmax = n // kp + 1
    ok = True
    lat, ikc, gsc = get(f"{O}/mla_lat.bin", (n, KR)), get(f"{O}/mla_ikc.bin", (n, ID)), get(f"{O}/mla_gsc.bin", (n, ID))
    ok &= stats("mla: token latents (norm, written to the cache at the rows' positions)", lat, rmsnorm(kvraw, kvw, eps), 1e-2, 3)
    ok &= stats("mla: indexer keys (LayerNorm)", ikc, layernorm(ikraw, knw, knb, 1e-6), 1e-2, 2)
    ok &= stats("mla: indexer gate scores (copied)", gsc, gsraw, 0.0, 0)
    pk = get(f"{O}/mla_pk.bin", (Pmax, ID))[:n // kp]
    ok &= stats("mla: pool keys (on the engine's keys and scores)", pk, mla_pool_keys(ikc, gsc, ape, kp), 1e-2, 2) if n >= kp else True
    qa = get(f"{O}/mla_qa.bin", (n, NH, KR), torch.float32)
    qa_ref = torch.einsum("thd,hdj->thj", f32(q).reshape(n, NH, nope), f32(wkv).reshape(NH, nope + vd, KR)[:, :nope])
    ok &= stats("mla: absorbed queries (Wk_h^T q_h)", qa, qa_ref, 1e-5)
    cnt = get_i32(f"{O}/mla_cnt.bin", (n,))
    sel = get_i32(f"{O}/mla_sel.bin", (n, topk + kp - 1))
    ok &= mla_choice(iq, wts, pk, sel, cnt, cfg, "mla: indexer")
    sel_rows = [sel[i, :int(cnt[i])].long() for i in range(n)]
    olat_ref, ov_ref = mla_attend(q, lat, sel_rows, W, cfg)
    olat = get(f"{O}/mla_olat.bin", (n, NH, KR), torch.float32)
    ok &= stats("mla: attention over the latents (on the engine's tokens and latents)", olat, olat_ref, 1e-4)
    ov = get(f"{O}/mla_ov.bin", (n, NH * vd))
    ok &= stats("mla: values (olat times kv_b_proj's v rows)", ov, bf(torch.einsum("thj,hej->the", olat, f32(wkv).reshape(NH, nope + vd, KR)[:, nope:]).reshape(n, NH * vd)), 1e-2, None if n > 100 or KR > 64 else 2)
    print("mla kernels:", "ok" if ok else "FAILED")
    return ok


# ---- the weights ------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Tensors:
    """A safetensors file's tensors by name, each read when asked (a real mixture-of-experts layer's are 8 GB)."""

    def __init__(self, path):
        from safetensors import safe_open
        self.f = safe_open(path, "pt")
        self.names = set(self.f.keys())

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name):
        return self.f.get_tensor(name)


def fp8_dequantize(q, scales):
    """transformers 5.19.0's Fp8Dequantize on one matrix: the E4M3 bytes q [rows, cols] times the block scales (F32, [ceil(rows / bm), ceil(cols / bn)], the blocks derived from the grid), the f32 products rounded to bf16."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    return Fp8Dequantize(None)._dequantize_one(q, scales, torch.bfloat16)


class Weights:
    """One layer's matrices as the engine multiplies by them: bf16 as the checkpoint has them, FP8 ones as the bf16 they dequantise to, and where `decoded` (a mini checkpoint `glyd-engine glmdecode` made) holds a
    tensor, that one (the engine's packs hold it at a level, not the checkpoint's own)."""

    def __init__(self, tensors, dev, decoded=None):
        self.t, self.dev, self.decoded, self.cache = tensors, dev, decoded, {}

    def matrix(self, name):
        if self.decoded is not None and name in self.decoded:
            return self.decoded[name].to(self.dev)
        w = self.t[name]
        if w.dtype == torch.float8_e4m3fn:
            return fp8_dequantize(w.to(self.dev), self.t[name + "_scale_inv"].to(self.dev))
        return w.to(self.dev)

    def cached(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]


def load_layer(tensors, l, dev, decoded=None):
    """The reference's weights of layer l from the checkpoint's tensors (names as the checkpoint has them): the norms, the mixing weights, the Kimi delta attention's matrices (q, k, v as one, the three convolutions as one),
    and the dense MLP's (gate and up as one, down) or the mixture of experts' (the router and its bias, the shared expert's, the experts' as functions of the expert)."""
    p = f"{PREFIX}layers.{l}"
    W = Weights(tensors, dev, decoded)
    g = lambda n: tensors[f"{p}.{n}"].to(dev)
    m = lambda n: W.matrix(f"{p}.{n}")
    out = dict(norm_a=g("input_layernorm.weight"), norm_f=g("post_attention_layernorm.weight"))
    for k, nm in (("a", "hc_attn"), ("f", "hc_ffn")):
        if f"{p}.{nm}_fn" in tensors:  # (the MTP head's layer has none)
            out.update({f"hc_{k}_fn": g(f"{nm}_fn"), f"hc_{k}_base": f32(g(f"{nm}_base")), f"hc_{k}_scale": f32(g(f"{nm}_scale"))})
    if f"{p}.self_attn.q_a_proj.weight" in tensors:
        # latent attention: the indexer's tensors are the checkpoint's bf16 as they are, kv_b_proj too (the absorbed form multiplies by its rows)
        out.update(q_a=m("self_attn.q_a_proj.weight"), q_a_ln=g("self_attn.q_a_layernorm.weight"), q_b=m("self_attn.q_b_proj.weight"), kv_a=m("self_attn.kv_a_proj_with_mqa.weight"),
                   kv_a_ln=g("self_attn.kv_a_layernorm.weight"), kv_b=m("self_attn.kv_b_proj.weight"), o_proj=m("self_attn.o_proj.weight"), wq_b=g("self_attn.indexer.wq_b.weight"),
                   wk=g("self_attn.indexer.wk.weight"), k_norm_w=g("self_attn.indexer.k_norm.weight"), k_norm_b=g("self_attn.indexer.k_norm.bias"), weights_proj=g("self_attn.indexer.weights_proj.weight"),
                   ape=g("self_attn.indexer.index_kpool_compress_ape"), ix_gate=g("self_attn.indexer.index_kpool_compress_gate"))
    else:
        out.update(qkv=torch.cat([m(f"self_attn.{n}_proj.weight") for n in "qkv"], 0), o_proj=m("self_attn.o_proj.weight"), f_a=m("self_attn.f_a_proj.weight"), f_b=m("self_attn.f_b_proj.weight"),
                   g_a=m("self_attn.g_a_proj.weight"), g_b=m("self_attn.g_b_proj.weight"), b_proj=m("self_attn.b_proj.weight"), conv=torch.cat([g(f"self_attn.{n}_conv1d.weight") for n in "qkv"], 0),
                   A_log=f32(g("self_attn.A_log")), dt_bias=f32(g("self_attn.dt_bias")), o_norm=g("self_attn.o_norm.weight"))
    if f"{p}.mlp.gate_proj.weight" in tensors:
        out.update(gate_up=torch.cat([m("mlp.gate_proj.weight"), m("mlp.up_proj.weight")], 0), down=m("mlp.down_proj.weight"))
    else:
        out.update(gate=g("mlp.gate.weight"), bias=f32(g("mlp.gate.e_score_correction_bias")), s_gate_up=torch.cat([m("mlp.shared_experts.gate_proj.weight"), m("mlp.shared_experts.up_proj.weight")], 0),
                   s_down=m("mlp.shared_experts.down_proj.weight"),
                   expert_gu=lambda e: W.cached(("gu", e), lambda: torch.cat([m(f"mlp.experts.{e}.gate_proj.weight"), m(f"mlp.experts.{e}.up_proj.weight")], 0)),
                   expert_down=lambda e: W.cached(("down", e), lambda: m(f"mlp.experts.{e}.down_proj.weight")))
    return out


def model_cfg(d):
    """The numbers of the reference from a checkpoint directory's config.json."""
    c = json.load(open(f"{d}/config.json"))
    t = c["text_config"]
    la = t.get("linear_attn_config", {})
    kinds = t.get("layer_types") or ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(t["num_hidden_layers"])]
    moe = [x == "sparse" for x in t["mlp_layer_types"]] if t.get("mlp_layer_types") else [i >= t.get("first_k_dense_replace", 3) for i in range(t["num_hidden_layers"])]
    return dict(hidden=t["hidden_size"], eps=t["rms_norm_eps"], heads=la.get("num_heads", 64), bound=la.get("gate_lower_bound") or -5.0, hc_eps=t.get("hc_eps", 1e-6), hc_iters=t.get("hc_sinkhorn_iters", 20),
                top_k=t["num_experts_per_tok"], scale=t["routed_scaling_factor"], norm_topk=t.get("norm_topk_prob", True), limit=t.get("swiglu_limit", 10.0), inter=t["intermediate_size"],
                experts=t["n_routed_experts"], vocab=t["vocab_size"], kinds=kinds, moe=moe, layers=t["num_hidden_layers"], nh=t["num_attention_heads"], q_rank=t["q_lora_rank"], kv_rank=t["kv_lora_rank"],
                nope=t["qk_nope_head_dim"], vd=t["v_head_dim"], ih=t["index_n_heads"], idim=t["index_head_dim"], topk=t["index_topk"], kpool=t["index_kpool"], tail=t.get("index_kpool_always_select_tail", True))


def read_tensors(path):
    return Tensors(path)


# ---- a random tiny model: `glm_ref.py tiny MINI`, then `glyd-engine glmlayers` and `glmtoy` ---------------------------------------------------------------------------------------------------------

TOY_VOCAB = 2048
TOY_ID = {"pad": 0, "eos": [2, 3]}

# What `tiny` changes in the real model's config (tests/fixtures/glm5_next_config.json) to cut it down: eight layers in the real model's pattern (three of Kimi delta attention with a dense MLP, then a latent attention layer and
# three of Kimi delta attention, a latent attention layer again; mixture-of-experts layers from the fourth), a width of 256 (Kimi delta heads of 128 channels, 2 of them), latent attention of 2 heads (q rank 64, kv rank 64,
# heads of 64) with an indexer of 4 heads of 32 channels that picks 8 tokens' worth of pools of 4 (so that a prompt of a few tokens attends to all of its past and a longer one to the pools it scores best, and the tail),
# 16 experts of which 4 a token (a prompt of 150 rows gives each 37 pairs), an MLP of 384 (blocks of 128 for its FP8 scales), a vocabulary of 2,048.
TINY = dict(hidden_size=256, intermediate_size=384, moe_intermediate_size=192, num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2, n_routed_experts=16, num_experts_per_tok=4, n_shared_experts=1,
            vocab_size=TOY_VOCAB, max_position_embeddings=4096, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=64, qk_head_dim=64, v_head_dim=64, index_n_heads=4, index_head_dim=32, index_topk=8,
            num_nextn_predict_layers=1, first_k_dense_replace=3, eos_token_id=TOY_ID["eos"], pad_token_id=TOY_ID["pad"])
TINY_KINDS = ["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"] * 3 + ["deepseek_sparse_attention"]


def tiny_config():
    cfg = json.load(open(os.path.join(HERE, "..", "tests", "fixtures", "glm5_next_config.json")))
    t = cfg["text_config"]
    cfg["quantization_config"] = dict(activation_scheme="dynamic", fmt="e4m3", quant_method="fp8", weight_block_size=[128, 128])
    t.update(TINY)
    n = TINY["num_hidden_layers"]
    t.update(layer_types=TINY_KINDS, mlp_layer_types=["dense"] * TINY["first_k_dense_replace"] + ["sparse"] * (n - TINY["first_k_dense_replace"]), indexer_types=["full"] * n,
             linear_attn_config=dict(num_heads=2, gate_lower_bound=-5.0, head_dim=P, short_conv_kernel_size=4, kda_layers=[i for i, k in enumerate(TINY_KINDS) if k == "linear_attention"],
                                     full_attn_layers=[i for i, k in enumerate(TINY_KINDS) if k != "linear_attention"]))
    cfg["vision_config"].update(depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=TINY["hidden_size"], projection_intermediate_size=64)
    return cfg


# The spread of each matrix of the tiny model (the standard deviation of its weights), chosen so that every layer's block has about the size of the streams' embedding (a rms of 0.5 to 0.9 where the embedding's is 1):
# a layer that is wrong then moves the logits by tenths. `head`: the scale of the embedding that is the LM head, so that a token's own embedding is the largest logit by a margin.
TINY_STD = dict(qkv=0.0625, fa=0.0625, fb=0.09, ga=0.0625, gb=0.09, b=0.09, o=0.07, hc=0.03, gate=0.1, up=0.0625, down=0.03, s_up=0.0625, s_down=0.03, dense_up=0.0625, dense_down=0.04, head=0.0625,
                qa=0.0625, qb=0.125, kva=0.0625, kvb_k=0.25, kvb_v=0.1, mo=0.08, iwq=0.125, iwk=0.0625, igate=0.0625, iw=0.0625)


def tiny_tensors(cfg, level, seed):
    """The tiny model's tensors, {name: (dtype, shape, bytes)} in the checkpoint's names and dtypes (bf16; the router's bias, A_log, dt_bias and the mixing weights' base and scales fp32; the first layer's MLP as E4M3 bytes
    with F32 block scales): every layer's, the embedding, the head and the final norm. The experts and the shared experts are on the grid of the block layout of `level` (6: six-bit codes, 5: five-bit, 4: four-bit), as
    `grid` makes them: the engine's packs hold them exactly."""
    t = cfg["text_config"]
    g = torch.Generator().manual_seed(seed)
    h, heads, E, inter, dense = t["hidden_size"], t["linear_attn_config"]["num_heads"], t["n_routed_experts"], t["moe_intermediate_size"], t["intermediate_size"]
    qd = heads * P
    kmax = {6: 31, 5: 15, 4: 7}[level]
    randn = lambda *shape, std=1.0: bf(torch.randn(*shape, generator=g) * std)
    unif = lambda *shape, lo, hi: bf(torch.rand(*shape, generator=g) * (hi - lo) + lo)

    def grid(rows, cols, std):
        """rows x cols weights of about that spread that a block layout holds exactly: integers of at most kmax times a power of two (the codes spread over a third of the range or less), with a code of kmax in every run of 32
        along a row (so that the packer's fit of the scale lands on the power of two, and a code times it is the weight)."""
        scale = 2.0 ** math.ceil(math.log2(3 * std / kmax))
        k = torch.round(torch.randn(rows, cols, generator=g) * std / scale).clamp(-kmax, kmax)
        k[:, ::32] = kmax * (torch.randint(0, 2, (rows, (cols + 31) // 32), generator=g) * 2 - 1)
        return bf(k * scale)

    def fp8(w):
        """w [rows, cols] as E4M3 bytes and the F32 scales of its 128 x 128 blocks (a power of two covering the block's largest value in 448)."""
        rows, cols = w.shape
        gr, gc = rows // 128, cols // 128
        blocks = f32(w).reshape(gr, 128, gc, 128)
        amax = blocks.abs().amax((1, 3)).clamp(min=1e-6)
        scale = 2.0 ** torch.ceil(torch.log2(amax / 448.0))
        q = (blocks / scale[:, None, :, None]).reshape(rows, cols).to(torch.float8_e4m3fn)
        return q, scale

    T = {}
    nl = t["num_hidden_layers"]
    for l in range(nl + t["num_nextn_predict_layers"]):
        p = f"{PREFIX}layers.{l}"
        mtp = l >= nl  # (the MTP head's layer: latent attention and experts on a plain residual stream, with its own norms and projection)
        T[f"{p}.input_layernorm.weight"] = unif(h, lo=0.5, hi=1.5)
        T[f"{p}.post_attention_layernorm.weight"] = unif(h, lo=0.5, hi=1.5)
        if mtp:
            T[f"{p}.enorm.weight"] = unif(h, lo=0.5, hi=1.5)
            T[f"{p}.hnorm.weight"] = unif(h, lo=0.5, hi=1.5)
            T[f"{p}.eh_proj.weight"] = randn(h, 2 * h, std=0.04)
            T[f"{p}.shared_head.norm.weight"] = unif(h, lo=0.5, hi=1.5)
        for nm in ("hc_attn", "hc_ffn"):
            if mtp:
                break
            T[f"{p}.{nm}_fn"] = randn(24, HC * h, std=TINY_STD["hc"])
            T[f"{p}.{nm}_base"] = f32(randn(24, std=0.5))
            T[f"{p}.{nm}_scale"] = f32(unif(3, lo=0.7, hi=1.3))
        if not mtp and t["layer_types"][l] == "linear_attention":
            for n in "qkv":
                T[f"{p}.self_attn.{n}_proj.weight"] = randn(qd, h, std=TINY_STD["qkv"])
                T[f"{p}.self_attn.{n}_conv1d.weight"] = randn(qd, 1, 4, std=0.4)
            T[f"{p}.self_attn.o_proj.weight"] = randn(h, qd, std=TINY_STD["o"])
            T[f"{p}.self_attn.f_a_proj.weight"] = randn(P, h, std=TINY_STD["fa"])
            T[f"{p}.self_attn.f_b_proj.weight"] = randn(qd, P, std=TINY_STD["fb"])
            T[f"{p}.self_attn.g_a_proj.weight"] = randn(P, h, std=TINY_STD["ga"])
            T[f"{p}.self_attn.g_b_proj.weight"] = randn(qd, P, std=TINY_STD["gb"])
            T[f"{p}.self_attn.b_proj.weight"] = randn(heads, h, std=TINY_STD["b"])
            T[f"{p}.self_attn.A_log"] = f32(randn(heads, std=0.4))
            T[f"{p}.self_attn.dt_bias"] = f32(randn(qd, std=1.0))
            T[f"{p}.self_attn.o_norm.weight"] = unif(P, lo=0.5, hi=1.5)
        else:
            NH, QR, KR, nope, vd, IH, ID, kp = t["num_attention_heads"], t["q_lora_rank"], t["kv_lora_rank"], t["qk_nope_head_dim"], t["v_head_dim"], t["index_n_heads"], t["index_head_dim"], t["index_kpool"]
            a = f"{p}.self_attn"
            T[f"{a}.q_a_proj.weight"] = randn(QR, h, std=TINY_STD["qa"])
            T[f"{a}.q_a_layernorm.weight"] = unif(QR, lo=0.5, hi=1.5)
            T[f"{a}.q_b_proj.weight"] = randn(NH * nope, QR, std=TINY_STD["qb"])
            T[f"{a}.kv_a_proj_with_mqa.weight"] = randn(KR, h, std=TINY_STD["kva"])
            T[f"{a}.kv_a_layernorm.weight"] = unif(KR, lo=0.5, hi=1.5)
            kv_b = torch.cat([randn(NH, nope, KR, std=TINY_STD["kvb_k"]), randn(NH, vd, KR, std=TINY_STD["kvb_v"])], 1)
            T[f"{a}.kv_b_proj.weight"] = kv_b.reshape(NH * (nope + vd), KR)
            T[f"{a}.o_proj.weight"] = randn(h, NH * vd, std=TINY_STD["mo"])
            T[f"{a}.indexer.wq_b.weight"] = randn(IH * ID, QR, std=TINY_STD["iwq"])
            T[f"{a}.indexer.wk.weight"] = randn(ID, h, std=TINY_STD["iwk"])
            T[f"{a}.indexer.k_norm.weight"] = unif(ID, lo=0.5, hi=1.5)
            T[f"{a}.indexer.k_norm.bias"] = randn(ID, std=0.1)
            T[f"{a}.indexer.weights_proj.weight"] = randn(IH, h, std=TINY_STD["iw"])
            T[f"{a}.indexer.index_kpool_compress_ape"] = randn(kp, ID, std=0.5)
            T[f"{a}.indexer.index_kpool_compress_gate"] = randn(ID, h, std=TINY_STD["igate"])
        if not mtp and t["mlp_layer_types"][l] == "dense":
            for n, shape, std in (("gate_proj", (dense, h), TINY_STD["dense_up"]), ("up_proj", (dense, h), TINY_STD["dense_up"]), ("down_proj", (h, dense), TINY_STD["dense_down"])):
                T[f"{p}.mlp.{n}.weight"], T[f"{p}.mlp.{n}.weight_scale_inv"] = fp8(randn(*shape, std=std))
        else:
            T[f"{p}.mlp.gate.weight"] = randn(E, h, std=TINY_STD["gate"])
            T[f"{p}.mlp.gate.e_score_correction_bias"] = f32(randn(E, std=0.1))
            for n, shape, std in (("gate_proj", (inter, h), TINY_STD["s_up"]), ("up_proj", (inter, h), TINY_STD["s_up"]), ("down_proj", (h, inter), TINY_STD["s_down"])):
                T[f"{p}.mlp.shared_experts.{n}.weight"] = grid(*shape, std)
            for e in range(E):
                for n, shape, std in (("gate_proj", (inter, h), TINY_STD["up"]), ("up_proj", (inter, h), TINY_STD["up"]), ("down_proj", (h, inter), TINY_STD["down"])):
                    T[f"{p}.mlp.experts.{e}.{n}.weight"] = grid(*shape, std)
    T[f"{PREFIX}embed_tokens.weight"] = randn(t["vocab_size"], h)
    # (the head is the embedding scaled: a token's own embedding is then the largest logit by a margin the engine's rounding does not close, as a trained model's logits are peaked)
    T["lm_head.weight"] = bf(f32(T[f"{PREFIX}embed_tokens.weight"]) * TINY_STD["head"])
    T[f"{PREFIX}norm.weight"] = unif(h, lo=0.5, hi=1.5)
    return T


def entry(name, t):
    """A tensor as (name, safetensors dtype, shape, bytes)."""
    if t.dtype == torch.float8_e4m3fn:
        return name, "F8_E4M3", list(t.shape), t.contiguous().view(torch.uint8).numpy().tobytes()
    return name, "BF16" if t.dtype == BF else "F32", list(t.shape), put_bytes(t)


def tiny_build(out, level=6, seed=1, tokens=300, prefill=150, total=210):
    """OUT: the layers' tensors as a mini checkpoint (config.json, model.safetensors, x.bin: the streams of `tokens` random tokens, the embedding plus a noise of 0.5 on each stream, bf16 [tokens, 4, hidden]; tokens.json) and
    OUT/toy: the whole model (the layers, the embedding, the head, the final norm), ids.txt (`total` random tokens) and toy.txt (`prefill` `total` `level`)."""
    cfg = tiny_config()
    T = tiny_tensors(cfg, level, seed)
    os.makedirs(f"{out}/toy", exist_ok=True)
    layers = [entry(n, t) for n, t in T.items() if n.startswith(f"{PREFIX}layers.")]
    g = torch.Generator().manual_seed(seed + 1)
    h = cfg["text_config"]["hidden_size"]
    x_ids = torch.randint(0, TOY_VOCAB, (tokens,), generator=g)
    for d in (out, f"{out}/toy"):
        json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    write_safetensors(f"{out}/model.safetensors", layers)
    put(f"{out}/x.bin", bf(f32(T[f"{PREFIX}embed_tokens.weight"][x_ids])[:, None, :] + 0.5 * torch.randn(tokens, HC, h, generator=g)))
    json.dump({"ids": x_ids.tolist()}, open(f"{out}/tokens.json", "w"))
    write_safetensors(f"{out}/toy/model.safetensors", layers + [entry(n, t) for n, t in T.items() if not n.startswith(f"{PREFIX}layers.")])
    open(f"{out}/toy/ids.txt", "w").write(" ".join(str(i) for i in torch.randint(0, TOY_VOCAB, (total,), generator=g).tolist()))
    open(f"{out}/toy/toy.txt", "w").write(f"{prefill} {total} {level}\n")
    print(f"{out}: a random model of {cfg['text_config']['num_hidden_layers']} layers (Kimi delta attention and latent attention), {sum(t.numel() for t in T.values()) / 1e6:.1f} M parameters, {cfg['text_config']['n_routed_experts']} experts "
          f"({cfg['text_config']['num_experts_per_tok']} a token) on the grid of {level}-bit codes; {tokens} rows of streams in x.bin; {out}/toy: {prefill} tokens of prefill, {total} in all")


# ---- layers through the engine: `glyd-engine glmlayers MINI IO` against the reference ----------------------------------------------------------------------------------------------------------

def layers_gen(mini, io_dir, level=None, kinds="DE"):
    """IO/spec.json: for each layer of the mini checkpoint (Kimi delta or latent attention) with a dense MLP (D) or experts (E) that `kinds` names, the cases to run: the prompt in one call and in odd pieces, the delta rule
    token by token (Kimi delta layers), single steps (replayed from a CUDA graph, and not), a prompt then steps. level (4, 5 or 6): the level the engine holds a mixture-of-experts layer's matrices at (the one `tiny` made the mini for)."""
    cfg = model_cfg(mini)
    hid = cfg["hidden"]
    rows = os.path.getsize(f"{mini}/x.bin") // 2 // (HC * hid)
    t = read_tensors(f"{mini}/model.safetensors")
    layers = [l for l in range(cfg["layers"]) if (f"{PREFIX}layers.{l}.self_attn.q_proj.weight" in t or f"{PREFIX}layers.{l}.self_attn.q_a_proj.weight" in t) and ("E" if cfg["moe"][l] else "D") in kinds]
    odd = [100, 128, 5, rows - 233] if rows > 240 else [rows]
    steps = min(rows, 64)
    cases = []
    for l in layers:
        tag = f"{'e' if cfg['moe'][l] else 'd'}{l}"
        cases += [dict(name=f"{tag}_prefill", layer=l, mode="prefill", calls=[rows]), dict(name=f"{tag}_chunks", layer=l, mode="prefill", calls=odd)]
        if cfg["kinds"][l] == "linear_attention":  # (the delta rule token by token: the latent attention has no such mode)
            cases += [dict(name=f"{tag}_recurrent", layer=l, mode="prefill", calls=[rows], chunked=False)]
        cases += [dict(name=f"{tag}_steps", layer=l, mode="decode", tokens=steps), dict(name=f"{tag}_graph", layer=l, mode="decode", tokens=steps, graph=True),
                  dict(name=f"{tag}_mixed", layer=l, mode="mixed", prefill=min(200, rows - 1), tokens=rows, graph=True)]
    os.makedirs(io_dir, exist_ok=True)
    json.dump(dict(chunk=max(rows, 128), **({"level": level} if level else {}), cases=cases), open(f"{io_dir}/spec.json", "w"), indent=1)
    print(f"{io_dir}/spec.json: {len(cases)} cases on layers {layers}, {rows} rows")


def get_i32(path, shape):
    import numpy as np
    return torch.from_numpy(np.fromfile(path, dtype=np.int32)).reshape(*shape)


def sample_rows(case, T):
    """The rows of a case's sequence of T rows that the engine dumped: None for all of them, else the positions in the windows of the case's "sample" ([[first row, rows], ...]), ascending."""
    w = case.get("sample")
    if not w:
        return None
    return torch.tensor([r for r in range(T) if any(a <= r < a + k for a, k in w)], dtype=torch.long)


def mla_choice(iq, wts, pk, sel, cnt, cfg, name="  indexer", pos=None):
    """The engine's selected tokens (sel [T, Tmax] int32, the first cnt [T] of a row) against the reference's from the engine's own scores' inputs (iq, wts, the pool keys pk): the same set of tokens for every row, but where the engine's
    pools and the reference's differ by pools whose scores are within 1e-3 of the cut (a tie that falls the other way, as the router's check allows). Returns whether every row is so."""
    T, kp, topk = iq.shape[0], cfg["kpool"], cfg["topk"]
    sc = mla_index_scores(iq, wts, pk, cfg["ih"], cfg["idim"]) if pk.shape[0] else None
    exact = ties = bad = 0
    for i in range(T):
        qpos = i if pos is None else int(pos[i])  # (the row's position in the sequence: rows of a sample are not the first ones)
        want = mla_select(sc[i] if sc is not None else None, qpos, topk, kp, cfg["tail"])
        got = sel[i, :int(cnt[i])].long().sort().values
        if torch.equal(got, want):
            exact += 1
            continue
        P = (qpos + 1) // kp
        # (the differing pools: tokens of the one set and not the other, as pools)
        a, b = set(got.tolist()), set(want.tolist())
        diff = sorted({t // kp for t in (a ^ b) if t < P * kp})
        cut = float(torch.sort(sc[i, :P], descending=True).values[min(topk // kp, P) - 1]) if sc is not None and P > 0 else 0.0
        if all(abs(float(sc[i, d]) - cut) <= 1e-3 * max(abs(cut), 1e-6) for d in diff) and len(a) == len(b):
            ties += 1
        else:
            bad += 1
    ok = bad == 0
    print(f"{'ok  ' if ok else 'FAIL'} {name}: the reference's tokens equal the engine's for {exact}/{T} rows" + (f" (+{ties} by a tie the other way)" if ties else "") + (f", {bad} differ" if bad else ""))
    return ok


def mla_half_check(io_dir, name, T, x, W, cfg, dev, rows=None):
    """The latent attention layer's mixer half of the engine's case `name`, stage by stage from the engine's own input to each: the pool keys, the choice of tokens, the attention, the product by o_proj, the streams after it.
    `rows` (a case with a sample): the positions the engine dumped the per-row stages of; the caches of the latents, the indexer's keys and gate scores, and the pool keys are of all T rows.
    Returns (ok, mixing weights, h1, o1, x1), the per-row ones of the dumped rows."""
    hid, nh, kr, vd, tmax = cfg["hidden"], cfg["nh"], cfg["kv_rank"], cfg["vd"], cfg["topk"] + cfg["kpool"] - 1
    Ts = T if rows is None else len(rows)
    pick = (lambda t: t) if rows is None else (lambda t: t[rows.to(t.device)])
    eng = lambda stage, *width, dtype=BF: get(f"{io_dir}/{name}.{stage}.bin", (Ts, *width), dtype).to(dev)
    whole = lambda stage, *width, dtype=BF: get(f"{io_dir}/{name}.{stage}.bin", (T, *width), dtype).to(dev)
    ok = True
    xs = pick(x)
    params = eng("hcA", 24, dtype=torch.float32)
    pre, post, comb, col = hc_mix(xs, W["hc_a_fn"], W["hc_a_base"], W["hc_a_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
    ok &= stats("  mixing weights (mHC) of the mixer's half", params, torch.cat([pre, post, comb.reshape(Ts, 16)], 1), 1e-4)
    h1, qr, q, iq, wts = eng("h1", hid), eng("qr", cfg["q_rank"]), eng("q", nh * cfg["nope"]), eng("iq", cfg["ih"] * cfg["idim"]), eng("iw", cfg["ih"])
    c, ik, gs = whole("c", kr), whole("ik", cfg["idim"]), whole("gs", cfg["idim"])
    ecol = bf((params[:, :HC, None] * f32(xs)).sum(1))
    ok &= stats("  collapse and the input norm", h1, rmsnorm(ecol, W["norm_a"], cfg["eps"]), 1e-2, 4)
    ok &= stats("  query latent (q_a_proj, norm)", qr, rmsnorm(F.linear(h1, W["q_a"]), W["q_a_ln"], cfg["eps"]), 2e-2)
    ok &= stats("  query (q_b_proj)", q, F.linear(qr, W["q_b"]), 2e-2)
    ok &= stats("  token latent (kv_a_proj_with_mqa, norm)", pick(c), rmsnorm(F.linear(h1, W["kv_a"]), W["kv_a_ln"], cfg["eps"]), 2e-2)
    ok &= stats("  indexer queries", iq, F.linear(qr, W["wq_b"]), 1e-2)
    ok &= stats("  indexer keys (wk, LayerNorm)", pick(ik), layernorm(F.linear(h1, W["wk"]), W["k_norm_w"], W["k_norm_b"], 1e-6), 2e-2)
    ok &= stats("  indexer pooling gate", pick(gs), F.linear(h1, W["ix_gate"]), 1e-2)
    ok &= stats("  indexer head weights", wts, F.linear(h1, W["weights_proj"]), 1e-2)
    pk_ref = mla_pool_keys(ik, gs, W["ape"], cfg["kpool"])
    ok &= stats("  indexer pool keys (on the engine's keys and gates)", get(f"{io_dir}/{name}.pk.bin", (T // cfg["kpool"], cfg["idim"])).to(dev), pk_ref, 1e-2) if pk_ref.shape[0] else True
    pk = get(f"{io_dir}/{name}.pk.bin", (T // cfg["kpool"], cfg["idim"])).to(dev) if pk_ref.shape[0] else pk_ref
    cnt = get_i32(f"{io_dir}/{name}.cnt.bin", (Ts,)).to(dev)
    sel = get_i32(f"{io_dir}/{name}.sel.bin", (Ts, tmax)).to(dev)
    ok &= mla_choice(iq, wts, pk, sel, cnt, cfg, pos=rows)
    sel_rows = [sel[i, :int(cnt[i])].long() for i in range(Ts)]
    olat_ref, ov_ref = mla_attend(q, c, sel_rows, W, cfg)
    olat = eng("olat", nh, kr, dtype=torch.float32)
    ok &= stats("  attention over the latents (on the engine's query, latents and tokens)", olat, olat_ref, 5e-3)
    ov = eng("ov", nh * vd)
    ok &= stats("  values (olat times kv_b_proj's v rows)", ov, bf(torch.einsum("thj,hej->the", olat, f32(W["kv_b"]).reshape(nh, cfg["nope"] + vd, kr)[:, cfg["nope"]:]).reshape(Ts, nh * vd)), 1e-2)  # (no bound in places: a sum of 512 products of both signs has values near zero, where the accumulation order moves many places)
    o1 = eng("o1", hid)
    ok &= stats("  o product", o1, F.linear(ov, W["o_proj"]), 2e-2)
    x1 = eng("x1", HC, hid)
    ok &= stats("  streams after the mixer", x1, hc_post(xs, o1, params[:, HC:2 * HC], params[:, 2 * HC:].reshape(Ts, HC, HC)), 1e-2, 1)
    return ok, params, h1, o1, x1


# The bound on a stage of a layer whose experts are packs: the engine multiplies by the weights its packs decode to, which `--decoded` gives the reference; the stages are then checked as the others.
def layers_check(mini, io_dir, device=None, decoded_dir=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = model_cfg(mini)
    hid, heads = cfg["hidden"], cfg["heads"]
    qd = heads * P
    tensors = read_tensors(f"{mini}/model.safetensors")
    decoded = Tensors(f"{decoded_dir}/model.safetensors") if decoded_dir else None
    spec = json.load(open(f"{io_dir}/spec.json"))
    x_all = get(f"{mini}/x.bin", (-1, HC, hid)).to(dev)
    ok = True
    Ws = {}
    for case in spec["cases"]:
        name, l = case["name"], case["layer"]
        moe = cfg["moe"][l]
        if l not in Ws:
            Ws[l] = load_layer(tensors, l, dev, decoded)
        W = Ws[l]
        if not os.path.exists(f"{io_dir}/{name}.x2.bin"):
            print(f"FAIL {name}: the engine wrote nothing")
            ok = False
            continue
        calls = case.get("calls") or ([1] * case["tokens"] if case["mode"] == "decode" else [case["prefill"]] + [1] * (case["tokens"] - case["prefill"]))
        T = sum(calls)
        x = x_all[:T]
        # (a case with a "sample" has the per-row stages of those windows of the sequence only, `Ts` rows; the stages of Kimi delta attention that the recurrences run over, and the latent attention's caches, are of all T rows)
        rows = sample_rows(case, T)
        Ts = T if rows is None else len(rows)
        pick = (lambda t: t) if rows is None else (lambda t: t[rows.to(t.device)])
        xs = pick(x)
        eng = lambda stage, *width, dtype=BF: get(f"{io_dir}/{name}.{stage}.bin", (Ts, *width), dtype).to(dev)
        whole = lambda stage, *width, dtype=BF: get(f"{io_dir}/{name}.{stage}.bin", (T, *width), dtype).to(dev)
        print(f"-- {name}: layer {l} ({'latent attention' if cfg['kinds'][l] != 'linear_attention' else 'Kimi delta attention'} and {'experts' if moe else 'a dense MLP'}), {len(calls)} calls, {T} rows" + ("" if rows is None else f", {Ts} of them dumped"))
        # the token mixer's half, each stage from the engine's own input to it
        mla = cfg["kinds"][l] != "linear_attention"
        if mla:
            ok_m, params, h1, o1, x1 = mla_half_check(io_dir, name, T, x, W, cfg, dev, rows)
            ok &= ok_m
        else:
            params = eng("hcA", 24, dtype=torch.float32)
            pre, post, comb, col = hc_mix(xs, W["hc_a_fn"], W["hc_a_base"], W["hc_a_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
            ok &= stats("  mixing weights (mHC) of the mixer's half", params, torch.cat([pre, post, comb.reshape(Ts, 16)], 1), 1e-4)
            ecol = bf((params[:, :HC, None] * f32(xs)).sum(1))
            h1, og, core, att, o1, x1 = (eng("h1", hid), eng("og", qd), eng("core", qd), eng("att", qd), eng("o1", hid), eng("x1", HC, hid))
            mix, conv, fg, bb = whole("mix", 3 * qd), whole("conv", 3 * qd), whole("fg", qd), whole("bb", heads)
            ok &= stats("  collapse and the input norm", h1, rmsnorm(ecol, W["norm_a"], cfg["eps"]), 1e-2, 4)
            ok &= stats("  q, k, v product", pick(mix), F.linear(h1, W["qkv"]), 2e-2)
            rconv, rcst = conv_silu(mix, W["conv"], torch.zeros(3, 3 * qd, dtype=BF, device=dev))
            ok &= stats("  convolution", conv, rconv, 2e-2, 2)
            ok &= stats("  forget gate's products", pick(fg), F.linear(F.linear(h1, W["f_a"]), W["f_b"]), 2e-2)
            ok &= stats("  input gate's product", pick(bb), F.linear(h1, W["b_proj"]), 2e-2)
            ok &= stats("  output gate's products", og, F.linear(F.linear(h1, W["g_a"]), W["g_b"]), 2e-2)
            S = torch.zeros(heads, P, P, device=dev)
            q, k, v = (conv[:, i * qd:(i + 1) * qd].reshape(T, heads, P) for i in range(3))
            rcore = kda_recurrent(q, k, v, kda_gate(fg, W["dt_bias"], W["A_log"], cfg["bound"], heads), kda_beta(bb), S).reshape(T, qd)
            ok &= stats("  delta rule", core, pick(rcore), 2e-2)
            ok &= stats("  final delta rule state", get(f"{io_dir}/{name}.ssm.bin", (heads, P, P), torch.float32).to(dev), S, 2e-3)
            same = bool((get(f"{io_dir}/{name}.cstate.bin", (3, 3 * qd)).to(dev).view(torch.int16) == rcst.view(torch.int16)).all())
            ok &= same
            print(f"{'ok  ' if same else 'FAIL'}   the convolution's state is the last three inputs")
            ok &= stats("  gated norm", att, kda_norm(core, og, W["o_norm"], cfg["eps"]), 1e-2, 2)
            ok &= stats("  o product", o1, F.linear(att, W["o_proj"]), 2e-2)
            ok &= stats("  streams after the mixer", x1, hc_post(xs, o1, params[:, HC:2 * HC], params[:, 2 * HC:].reshape(Ts, HC, HC)), 1e-2, 1)
        # the feed-forward's half
        params2 = eng("hcF", 24, dtype=torch.float32)
        pre2, post2, comb2, _ = hc_mix(x1, W["hc_f_fn"], W["hc_f_base"], W["hc_f_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
        ok &= stats("  mixing weights (mHC) of the feed-forward's half", params2, torch.cat([pre2, post2, comb2.reshape(Ts, 16)], 1), 1e-4)
        ecol2 = bf((params2[:, :HC, None] * f32(x1)).sum(1))
        h2, o2, x2 = eng("h2", hid), eng("o2", hid), eng("x2", HC, hid)
        ok &= stats("  collapse and the post-attention norm", h2, rmsnorm(ecol2, W["norm_f"], cfg["eps"]), 1e-2, 4)
        if moe:
            ids, wts = (get(f"{io_dir}/{name}.{stage}.bin", (Ts, cfg["top_k"]), dtype).to(dev) for stage, dtype in (("ids", torch.int64), ("wts", torch.float32)))
            # (the router's and the stages' inputs are the engine's own normed input: a different rounding of it moves a score by more than a tie's tolerance)
            ok &= moe_choice(h2, W, cfg, ids, wts)
            ok &= stats("  routed experts and the shared expert (on the engine's input, experts and weights)", o2, moe_mixer(h2, W, cfg, ids, wts), 2e-2)
        else:
            gu, act = eng("gu", 2 * cfg["inter"]), eng("act", cfg["inter"])
            ok &= stats("  gate and up product", gu, F.linear(h2, W["gate_up"]), 2e-2)
            ok &= stats("  clamped SwiGLU", act, swiglu(gu, cfg["limit"]), 1e-2, 1)
            ok &= stats("  down product", o2, F.linear(act, W["down"]), 2e-2)
        ok &= stats("  streams after the layer", x2, hc_post(x1, o2, params2[:, HC:2 * HC], params2[:, 2 * HC:].reshape(Ts, HC, HC)), 1e-2, 1)
        if not moe and rows is None:
            # (no "reference alone" line for a layer of experts: a router that picks the other of two experts at a near tie would fail it, and every hop is checked from the engine's own data)
            st = dict(conv=torch.zeros(3, 3 * qd, dtype=BF, device=dev), ssm=torch.zeros(heads, P, P, device=dev))
            ok &= stats("  the layer from its input, reference alone", x2, layer(x, W, cfg, st, True), 1e-1 if not mla else 2e-1)
    # the graph replays and the eager steps are the same work
    by = {c["name"]: c for c in spec["cases"]}
    for n, c in by.items():
        if n.endswith("_graph") and n[:-6] + "_steps" in by and os.path.exists(f"{io_dir}/{n}.x2.bin"):
            a, b = get(f"{io_dir}/{n}.x2.bin", (c["tokens"], HC, hid)), get(f"{io_dir}/{n[:-6]}_steps.x2.bin", (c["tokens"], HC, hid))
            ok &= stats(f"{n} against {n[:-6]}_steps (a CUDA graph's replays against the same steps launched one by one: the same bits where the products are deterministic)", a, b, 2e-3)
    print("layers:", "ok" if ok else "FAILED")
    return ok


# ---- the tiny model whole: `glyd-engine glmtoy MINI/toy MINI/toy` ------------------------------------------------------------------------------------------------------------------------------

TOY_BOUND = 1.5e-1  # the toy's logits against the reference's own choices, of the largest logit (the median position: the routers and the indexers flip at near ties in about half of the positions; the rest sit at 1e-2)
TOY_FORCED_BOUND = 3e-2  # and against the reference run with the engine's choices
TOY_TOP1 = 0.98  # the share of positions at which the engine's top-1 token is the reference's


def deviation(got, want):
    """Each position's largest difference of logits [positions, vocab] from the reference's, relative to the reference's largest at that position."""
    return (f32(got) - f32(want)).abs().max(-1).values / f32(want).abs().max(-1).values


def forward(tensors, cfg, ids, dev, drop=None, decoded=None, chunked=True, force=None):
    """The reference's logits [len(ids), vocab] bf16: the whole sequence in one pass through the layers, the embedding spread over the four streams, the mean and the final norm, the head. drop: a layer whose feed-forward
    adds nothing (to show that a check sees it)."""
    x = tensors[f"{PREFIX}embed_tokens.weight"].to(dev)[ids.to(dev)]
    x = x[:, None, :].expand(-1, HC, -1).contiguous()
    qd = cfg["heads"] * P
    for l in range(cfg["layers"]):
        W = load_layer(tensors, l, dev, decoded)
        st = dict(conv=torch.zeros(3, 3 * qd, dtype=BF, device=dev), ssm=torch.zeros(cfg["heads"], P, P, device=dev))
        x = layer(x, W, cfg, st, chunked, drop_ffn=l == drop, force=(force or {}).get(l))
    return F.linear(hc_head(x, tensors[f"{PREFIX}norm.weight"].to(dev), cfg["eps"]), tensors["lm_head.weight"].to(dev))


def toy_check(mini, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    toy = f"{mini}/toy"
    cfg = model_cfg(toy)
    t = read_tensors(f"{toy}/model.safetensors")
    p, total = (int(v) for v in open(f"{toy}/toy.txt").read().split()[:2])
    ids = torch.tensor([int(v) for v in open(f"{toy}/ids.txt").read().split()][:total])
    ref = forward(t, cfg, ids, dev)[p - 1:total]
    ok = True
    # A router that scores two experts within the engine's rounding of each other picks the other of them in the engine, and that token's layer then differs from the reference's by an expert's share of its output; the
    # state a token leaves behind carries it on. With experts the bound is therefore the median position's, not the worst's: a few positions are moved by a flip, and the median by about a hundredth, where a layer wrong in
    # its structure (the scale, the choice, the weights, the shared expert, the activation) moves the median by a tenth or more. The layers' own stages, with the engine's choices, are `layers check`'s.
    moved = float(deviation(forward(t, cfg, ids, dev, drop=3)[p - 1:total], ref).median())
    sees = moved > 10 * TOY_FORCED_BOUND
    ok &= sees
    print(f"{'ok  ' if sees else 'FAIL'} toy model: without the fourth layer's feed-forward (the first with experts, after the first latent attention) the median position's logits move by {moved:.3f} of the largest (the bound is {TOY_FORCED_BOUND:g}: the check sees the layers)")
    runs = {}
    for tag in ("graph", "eager"):
        f = f"{toy}/toy_{tag}_logits.bin"
        if not os.path.exists(f):
            print(f"FAIL toy {tag}: the engine wrote nothing")
            return False
        runs[tag] = get(f, (total - p + 1, cfg["vocab"])).to(dev)
        dv = deviation(runs[tag], ref)
        good = bool(torch.isfinite(f32(runs[tag])).all()) and float(dv.median()) <= TOY_BOUND
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} toy model, {tag} decode: logits of the prefill's last token and of {total - p} steps: the median position differs by {float(dv.median()):.3e} of its largest logit (bound {TOY_BOUND:g}), "
              f"the 90th by {float(dv.quantile(0.9)):.3e}, the worst by {float(dv.max()):.3e}")
        top = (runs[tag].float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
        ok &= top >= TOY_TOP1
        print(f"{'    ' if top >= TOY_TOP1 else 'FAIL'} top-1 token the same as the reference's at {100 * top:.1f}% of the {total - p + 1} positions (at least {100 * TOY_TOP1:.0f}%)")
    ok &= stats("toy model: the decode graph's logits against the eager steps'", runs["graph"], runs["eager"], 1e-3)
    # The same with the engine's discrete choices run in the reference (the experts every router chose, the tokens every indexer picked, read from the eager pass's trace): a router or an indexer that
    # scores two candidates within the roundings of each other picks the other in one of the two, and that position's layer then differs by a candidate's share: what is left is rounding alone.
    force = {}
    tmax = cfg["topk"] + cfg["kpool"] - 1
    for l in range(cfg["layers"]):
        f = {}
        if os.path.exists(f"{toy}/toy_ids_L{l}.bin"):
            f["ids"] = get(f"{toy}/toy_ids_L{l}.bin", (total, cfg["top_k"]), torch.int64).to(dev)
        if os.path.exists(f"{toy}/toy_sel_L{l}.bin"):
            sel, cnt = get_i32(f"{toy}/toy_sel_L{l}.bin", (total, tmax)), get_i32(f"{toy}/toy_cnt_L{l}.bin", (total,))
            f["sel"] = [sel[i, :int(cnt[i])].long().sort().values.to(dev) for i in range(total)]
        if f:
            force[l] = f
    if force:
        reff = forward(t, cfg, ids, dev, force=force)[p - 1:total]
        for tag in ("graph", "eager"):
            dv = deviation(runs[tag], reff)
            good = bool(torch.isfinite(f32(runs[tag])).all()) and float(dv.median()) <= TOY_FORCED_BOUND and float(dv.quantile(0.9)) <= 3 * TOY_FORCED_BOUND
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} toy model, {tag} decode, with the engine's choices of experts ({sum('ids' in f for f in force.values())} routers) and of tokens ({sum('sel' in f for f in force.values())} indexers) run in the reference: "
                  f"the median position differs by {float(dv.median()):.3e} of its largest logit (bound {TOY_FORCED_BOUND:g}), the 90th by {float(dv.quantile(0.9)):.3e} (bound {3 * TOY_FORCED_BOUND:g}), the worst by {float(dv.max()):.3e}")
            lp, lq = torch.log_softmax(f32(reff), -1), torch.log_softmax(f32(runs[tag]), -1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            print(f"     KL of the engine's next-token distribution from the reference's (the engine's choices run in it): mean {float(kl.mean()):.2e} nats a position, median {float(kl.median()):.2e}, worst {float(kl.max()):.2e}")
            top = (runs[tag].float().argmax(-1) == reff.float().argmax(-1)).float().mean().item()
            ok &= top >= 0.99
            print(f"{'    ' if top >= 0.99 else 'FAIL'} top-1 token the same as the reference's at {100 * top:.1f}% of the {total - p + 1} positions (at least 99%)")
    print("toy:", "ok" if ok else "FAILED")
    return ok


# ---- the checkpoint, by range ------------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_mini(out, layers, tokens, seed, experts=None, repo=REPO, rev=REV):
    """OUT/config.json (the real one; with `experts` N, n_routed_experts N), OUT/model.safetensors (every tensor of these layers, with their real names, dtypes (FP8 with its block scales) and values, read by range from the shards
    that hold them; with `experts` N the routers' first N rows and biases' first N entries and the first N experts only: a mixture-of-experts layer of N experts, 25 MB each, not 288), OUT/x.bin (bf16 [tokens, 4,
    4096]: the streams of random tokens, the embedding plus a noise of 0.5 on each stream, the embeddings by range from the embedding's shard) and OUT/tokens.json."""
    from concurrent.futures import ThreadPoolExecutor
    cfg = json.loads(hf_get(repo, rev, "config.json"))
    t = cfg["text_config"]
    hidden, vocab = t["hidden_size"], t["vocab_size"]
    if experts is not None:
        t["n_routed_experts"] = experts
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    wm = json.loads(hf_get(repo, rev, "model.safetensors.index.json"))["weight_map"]
    keep = lambda n: experts is None or not (m := re.search(r"\.mlp\.experts\.(\d+)\.", n)) or int(m.group(1)) < experts
    names = sorted(n for n in wm if any(n.startswith(f"{PREFIX}layers.{l}.") for l in layers) and keep(n))
    if not names:
        raise SystemExit(f"no tensors for layers {layers}")
    headers = {}
    for sh in sorted({wm[n] for n in names} | {wm[f"{PREFIX}embed_tokens.weight"]}):
        headers[sh] = shard_header(repo, rev, sh)
    entries = []
    for n in names:
        base, h = headers[wm[n]]
        info = h[n]
        shape = info["shape"]
        if experts is not None and n.endswith((".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")):
            shape = [experts] + shape[1:]
        a, b = info["data_offsets"]
        entries.append((n, info["dtype"], shape, wm[n], base + a, (b - a) * shape[0] // info["shape"][0]))
    # the tensors are read ahead by 16 threads and written in order, a window at a time (the experts of one layer are 7 GB)
    header, at = {}, 0
    for n, dtype, shape, _, _, size in entries:
        header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + size]}
        at += size
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(f"{out}/model.safetensors", "wb") as f, ThreadPoolExecutor(16) as ex:
        f.write(struct.pack("<Q", len(hb)) + hb)
        done = 0
        for i in range(0, len(entries), 64):
            window = entries[i:i + 64]
            for data in ex.map(lambda e: hf_get(repo, rev, e[3], (e[4], e[4] + e[5] - 1)), window):
                f.write(data)
            done += sum(e[5] for e in window)
            print(f"{min(i + 64, len(entries))} of {len(entries)} tensors, {done / 1e6:.0f} MB", flush=True)
    # the streams: the embeddings of random tokens by range, plus a noise on each stream
    g = torch.Generator().manual_seed(seed)
    # (at most 3,000 rows are asked of the Hub, one request each; more rows draw from those with the noise of each stream, which makes them differ)
    pool = min(tokens, 3000)
    ids = torch.randint(1000, vocab, (pool,), generator=g).tolist()
    sh = wm[f"{PREFIX}embed_tokens.weight"]
    base, h = headers[sh]
    info = h[f"{PREFIX}embed_tokens.weight"]
    row = hidden * 2
    with ThreadPoolExecutor(16) as ex:
        rows = b"".join(ex.map(lambda i: hf_get(repo, rev, sh, (base + info["data_offsets"][0] + i * row, base + info["data_offsets"][0] + (i + 1) * row - 1)), ids))
    emb = torch.frombuffer(bytearray(rows), dtype=torch.int16).view(BF).reshape(pool, hidden)
    if tokens > pool:
        pick = torch.randint(0, pool, (tokens,), generator=g)
        emb, ids = emb[pick], [ids[i] for i in pick.tolist()]
    put(f"{out}/x.bin", bf(f32(emb)[:, None, :] + 0.5 * torch.randn(tokens, HC, hidden, generator=g)))
    json.dump({"ids": ids}, open(f"{out}/tokens.json", "w"))
    print(f"{out}: {len(entries)} tensors of layers {layers}, {at / 1e6:.0f} MB; {tokens} rows of streams")


# ---- the FP8 matrices: `glyd-engine glmdequant MINI OUT`, then `fp8 check MINI OUT` ----------------------------------------------------------------------------------------------------------

def fp8_check(mini, out):
    """The FP8 matrices the engine wrote (`glmdequant`: the bf16 `Checkpoint::read_bf16` makes of each) against transformers' Fp8Dequantize on the same bytes and block scales, bit for bit."""
    t = read_tensors(f"{mini}/model.safetensors")
    names = open(f"{out}/names.txt").read().split()
    if not names:
        print("FAIL fp8: the engine wrote no matrix")
        return False
    bad = 0
    for n in names:
        q = t[n]
        want = fp8_dequantize(q, t[n + "_scale_inv"]).contiguous()
        got = torch.from_numpy(__import__("numpy").fromfile(f"{out}/{n}.bf16", dtype="int16")).view(BF).reshape(want.shape)
        same = bool(torch.equal(got.view(torch.int16), want.view(torch.int16)))
        bad += not same
        if not same or n == names[0]:
            print(f"{'ok  ' if same else 'FAIL'} {n}: {tuple(q.shape)} E4M3 with scales {tuple(t[n + '_scale_inv'].shape)}: the engine's bf16 {'is' if same else 'is not'} transformers' bit for bit")
    print(f"{'ok  ' if not bad else 'FAIL'} fp8: {len(names) - bad} of {len(names)} matrices equal transformers' dequantisation bit for bit")
    return bad == 0


# ---- the checks that need no GPU: the reference against transformers ---------------------------------------------------------------------------------------------------------------------

def text_config(mini):
    """transformers' Glm5NextTextConfig of a checkpoint directory's config.json."""
    from transformers.models.glm5_next import configuration_glm5_next as C
    d = {k: v for k, v in json.load(open(f"{mini}/config.json")).items() if k not in ("architectures", "transformers_version", "model_type")}
    return C.Glm5NextConfig(**d).text_config


def hf_layer_state(W, l, prefix="", experts=True):
    """Layer l's weights (the reference's, from `load_layer`) as the names and shapes of transformers' Glm5NextTextDecoderLayer, fp32: transformers' conversion of the checkpoint (hc_attn_fn to attn_hc.fn, the forget gate's
    tensors into forget_gate, the three convolutions into one, the experts stacked)."""
    sd = {"input_layernorm.weight": W["norm_a"], "post_attention_layernorm.weight": W["norm_f"], "self_attn.o_proj.weight": W["o_proj"]}
    if "q_a" in W:
        sd.update({"self_attn.q_a_proj.weight": W["q_a"], "self_attn.q_a_layernorm.weight": W["q_a_ln"], "self_attn.q_b_proj.weight": W["q_b"], "self_attn.kv_a_proj_with_mqa.weight": W["kv_a"],
                   "self_attn.kv_a_layernorm.weight": W["kv_a_ln"], "self_attn.kv_b_proj.weight": W["kv_b"], "self_attn.indexer.wq_b.weight": W["wq_b"], "self_attn.indexer.wk.weight": W["wk"],
                   "self_attn.indexer.k_norm.weight": W["k_norm_w"], "self_attn.indexer.k_norm.bias": W["k_norm_b"], "self_attn.indexer.weights_proj.weight": W["weights_proj"],
                   "self_attn.indexer.index_kpool_compress_ape": W["ape"], "self_attn.indexer.index_kpool_compress_gate": W["ix_gate"]})
    else:
        heads = W["conv"].shape[0] // (3 * P)
        sd.update({"self_attn.b_proj.weight": W["b_proj"], "self_attn.g_a_proj.weight": W["g_a"], "self_attn.g_b_proj.weight": W["g_b"], "self_attn.o_norm.weight": W["o_norm"], "self_attn.conv1d.weight": W["conv"],
                   "self_attn.forget_gate.f_a_proj.weight": W["f_a"], "self_attn.forget_gate.f_b_proj.weight": W["f_b"], "self_attn.forget_gate.dt_bias": W["dt_bias"], "self_attn.forget_gate.A_log": W["A_log"]})
        qd = heads * P
        for i, n in enumerate("qkv"):
            sd[f"self_attn.{n}_proj.weight"] = W["qkv"][i * qd:(i + 1) * qd]
    for k, nm in (("a", "attn_hc"), ("f", "ffn_hc")):
        sd.update({f"{nm}.fn": W[f"hc_{k}_fn"], f"{nm}.base": W[f"hc_{k}_base"], f"{nm}.scale": W[f"hc_{k}_scale"]})
    if "gate_up" in W:
        inter = W["gate_up"].shape[0] // 2
        sd.update({"mlp.gate_proj.weight": W["gate_up"][:inter], "mlp.up_proj.weight": W["gate_up"][inter:], "mlp.down_proj.weight": W["down"]})
    else:
        E = W["gate"].shape[0]
        sd.update({"mlp.gate.weight": W["gate"], "mlp.gate.e_score_correction_bias": W["bias"], "mlp.shared_experts.gate_proj.weight": W["s_gate_up"][:W["s_gate_up"].shape[0] // 2],
                   "mlp.shared_experts.up_proj.weight": W["s_gate_up"][W["s_gate_up"].shape[0] // 2:], "mlp.shared_experts.down_proj.weight": W["s_down"],
                   })
        if experts:
            sd.update({"mlp.experts.gate_up_proj": torch.stack([W["expert_gu"](e) for e in range(E)]), "mlp.experts.down_proj": torch.stack([W["expert_down"](e) for e in range(E)])})
    return {prefix + k: f32(v) for k, v in sd.items()}


def vs_transformers(mini, layers=None, tokens=48, whole=False, experts_module=True):
    """The reference against transformers' own modules (their torch paths, fp32) on the mini checkpoint's weights and the first `tokens` rows of its streams: the hyper-connection of each half (the weights exactly, the collapse
    to a bf16 place), the Kimi delta mixer, the dense MLP and the mixture of experts (the router's choice and weights, the output), the whole decoder layer; where `whole`, the whole model (the layers are all of the
    checkpoint's) on its ids. What is left is bf16 rounding. `experts_module`: the module of experts too (its stacked experts are 3 H I floats each: not for the real model's)."""
    from transformers.models.glm5_next import modeling_glm5_next as M
    cfg, tc = model_cfg(mini), text_config(mini)
    hid, heads = cfg["hidden"], cfg["heads"]
    t = read_tensors(f"{mini}/model.safetensors")
    ok = True
    x_all = get(f"{mini}/x.bin", (-1, HC, hid))[:tokens]
    for l in layers if layers is not None else [i for i in range(cfg["layers"]) if f"{PREFIX}layers.{i}.self_attn.q_proj.weight" in t or f"{PREFIX}layers.{i}.self_attn.q_a_proj.weight" in t]:
        moe = cfg["moe"][l]
        W = load_layer(t, l, "cpu")
        small = moe and not experts_module
        if small:
            # (the experts' module of a real layer is 29 GB of fp32: the layer is made without it, and the router and the shared expert are compared)
            made = M.Glm5NextTextExperts.__init__
            M.Glm5NextTextExperts.__init__ = lambda self, config: torch.nn.Module.__init__(self)
        try:
            dec = M.Glm5NextTextDecoderLayer(tc, l).float().eval()
        finally:
            if small:
                M.Glm5NextTextExperts.__init__ = made
        sd = hf_layer_state(W, l, experts=not small)
        missing = dec.load_state_dict(sd, strict=False)
        ok &= not missing.unexpected_keys and not [k for k in missing.missing_keys if "experts" not in k]
        print(f"{'ok  ' if not missing.unexpected_keys and not [k for k in missing.missing_keys if 'experts' not in k] else 'FAIL'} layer {l}: the checkpoint's tensors load into transformers' decoder layer"
              f"{'' if not missing.missing_keys else ' (without ' + str(missing.missing_keys) + ')'}")
        x = x_all
        with torch.no_grad():
            # the hyper-connection of the mixer's half
            post_t, comb_t, col_t = dec.attn_hc(x[None])
            pre, post, comb, col = hc_mix(x, W["hc_a_fn"], W["hc_a_base"], W["hc_a_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
            ok &= stats(f"layer {l}: mHC weights against transformers' HyperConnection (post, comb)", torch.cat([post, comb.reshape(-1, 16)], 1), torch.cat([post_t[0], comb_t[0].reshape(-1, 16)], 1), 1e-6)
            ok &= stats(f"layer {l}: the collapse", col, col_t[0], 0.0)
            # the mixer
            h = rmsnorm(col, W["norm_a"], cfg["eps"])
            qd = heads * P
            mask = torch.ones(1, h.shape[0], dtype=torch.bool)
            if "q_a" in W:
                want = dec.self_attn(f32(h)[None], mask)[0][0]
                # the indexer's choice of tokens flips between two pools scoring within the roundings of each other: transformers' own choice is run in ours for the attention's numbers, and the share of rows where the
                # two choices are the same is its own line
                q_resid = dec.self_attn.q_a_layernorm(dec.self_attn.q_a_proj(f32(h)[None]))
                topk = dec.self_attn.indexer(f32(h)[None], q_resid, mask, None)[0]
                sel_hf = [topk[i][topk[i] >= 0].long().sort().values for i in range(h.shape[0])]
                ours = mla_mixer(h, W, cfg, sel=sel_hf)
                ok &= stats(f"layer {l}: latent attention mixer, with transformers' choice of tokens, against transformers' Glm5NextTextAttention (bf16 roundings in ours, fp32 in its)", ours, want, 5e-2)
                iq_, ik_, gs_, wts_ = mla_project(h, W, cfg)[3:]
                _, sel_ours = mla_select_all(iq_, wts_, ik_, gs_, W, cfg)
                same = sum(bool(torch.equal(a.sort().values, b)) for a, b in zip(sel_ours, sel_hf)) / h.shape[0]
                ok &= same >= 0.8
                print(f"{'ok  ' if same >= 0.8 else 'FAIL'} layer {l}: the indexer against transformers' Glm5NextTextIndexer: the same tokens for {100 * same:.1f}% of {h.shape[0]} rows (at least 80%; the others by a pool within the roundings of the cut)")
            else:
                want = dec.self_attn(f32(h)[None])[0]
                st = dict(conv=torch.zeros(3, 3 * qd, dtype=BF), ssm=torch.zeros(heads, P, P))
                ok &= stats(f"layer {l}: Kimi delta mixer against transformers' Glm5NextTextLinearAttention (bf16 roundings in ours, fp32 in its)", kda_mixer(h, W, cfg, st, True), want, 5e-2)
            # the feed-forward
            x1 = hc_post(x, bf(want), post, comb)
            _, post2, comb2, col2 = hc_mix(x1, W["hc_f_fn"], W["hc_f_base"], W["hc_f_scale"], cfg["hc_eps"], cfg["hc_iters"], cfg["eps"])
            h2 = rmsnorm(col2, W["norm_f"], cfg["eps"])
            if moe:
                _, tw, ti = dec.mlp.gate(f32(h2))
                idx, w = moe_route(h2, W["gate"], W["bias"], cfg["top_k"], cfg["scale"], cfg["norm_topk"])
                dense = lambda i, v: torch.zeros(h2.shape[0], cfg["experts"]).scatter_(1, i, v)
                same = bool((dense(idx, w) - dense(ti, tw)).abs().max() <= 1e-5 * float(tw.abs().max())) and bool(torch.equal(idx.sort(-1).values, ti.sort(-1).values))
                ok &= same
                print(f"{'ok  ' if same else 'FAIL'} layer {l}: the router against transformers' Glm5NextTextTopkRouter: the same {cfg['top_k']} of {cfg['experts']} experts and weights for {h2.shape[0]} tokens")
                shared = dec.mlp.shared_experts(f32(h2)[None])[0]
                ok &= stats(f"layer {l}: the shared expert against transformers' Glm5NextTextMLP", mlp(h2, W["s_gate_up"], W["s_down"], cfg["limit"]), shared, 5e-2)
                if experts_module:
                    want2 = dec.mlp(f32(h2)[None])[0]
                    ok &= stats(f"layer {l}: the mixture of experts against transformers' Glm5NextTextMoE (bf16 roundings in ours, fp32 in its)", moe_mixer(h2, W, cfg), want2, 5e-2)
            else:
                want2 = dec.mlp(f32(h2)[None])[0]
                ok &= stats(f"layer {l}: the dense MLP against transformers' Glm5NextTextMLP (bf16 roundings in ours, fp32 in its)", mlp(h2, W["gate_up"], W["down"], cfg["limit"]), want2, 5e-2)
            # the whole layer
            if not small:
                got = layer(x, W, cfg, dict(conv=torch.zeros(3, 3 * qd, dtype=BF), ssm=torch.zeros(heads, P, P)), True)
                want3, _ = dec(f32(x)[None], attention_mask=mask)
                if moe:
                    ok &= tokens_within(f"layer {l}: the whole layer against transformers' Glm5NextTextDecoderLayer (the streams; a token whose router is at a near tie may take the other expert)", got, want3[0], 1e-1, 0.9)
                else:
                    ok &= stats(f"layer {l}: the whole layer against transformers' Glm5NextTextDecoderLayer (the streams)", got, want3[0], 1e-1)
    if whole:
        t = read_tensors(f"{mini}/toy/model.safetensors")
        model = M.Glm5NextTextModel(tc).float().eval()
        sd = {k[len(PREFIX):]: f32(v) for k, v in {PREFIX + "embed_tokens.weight": t[PREFIX + "embed_tokens.weight"], PREFIX + "norm.weight": t[PREFIX + "norm.weight"]}.items()}
        for l in range(cfg["layers"]):
            sd.update(hf_layer_state(load_layer(t, l, "cpu"), l, f"layers.{l}."))
        missing = model.load_state_dict(sd, strict=False)
        ok &= not missing.unexpected_keys and not missing.missing_keys
        print(f"{'ok  ' if not missing.unexpected_keys and not missing.missing_keys else 'FAIL'} the checkpoint's tensors load into transformers' Glm5NextTextModel (missing {missing.missing_keys}, unexpected {missing.unexpected_keys})")
        ids = torch.tensor([int(v) for v in open(f"{mini}/toy/ids.txt").read().split()][:tokens]) if os.path.exists(f"{mini}/toy/ids.txt") else torch.randint(0, cfg["vocab"], (tokens,))
        with torch.no_grad():
            ref_h = model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
            ref = F.linear(ref_h, f32(t["lm_head.weight"]))
        mine = f32(forward(t, cfg, ids, "cpu"))
        dv = deviation(mine, ref)
        top = (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()
        good = float(dv.median()) <= TOY_BOUND and top >= 0.9
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} the whole model against transformers' Glm5NextTextModel (fp32): the median position's logits differ by {float(dv.median()):.3e} of the largest (bound {TOY_BOUND:g}), the worst by {float(dv.max()):.3e}; top-1 the same at {100 * top:.1f}% of {len(ids)} positions (at least 90%)")
    return ok


def loaded_as_transformers_does(toy, tokens):
    """The tiny checkpoint loaded by transformers itself (`Glm5NextForConditionalGeneration.from_pretrained`, its conversion of the names, the FP8 matrices dequantised by `FineGrainedFP8Config(dequantize=True)`, bf16):
    no tensor of the language model missing or unexpected, the dequantised matrices equal to `fp8_dequantize`'s bit for bit, and its logits against the reference's (both bf16)."""
    import transformers
    from transformers import FineGrainedFP8Config, Glm5NextForConditionalGeneration
    transformers.logging.set_verbosity_error()
    model, info = Glm5NextForConditionalGeneration.from_pretrained(toy, quantization_config=FineGrainedFP8Config(dequantize=True), dtype=BF, device_map="cpu", output_loading_info=True)
    nl = model_cfg(toy)["layers"]
    miss, extra = sorted(k for k in info["missing_keys"] if ".visual." not in k), sorted(k for k in info["unexpected_keys"] if f"layers.{nl}." not in k)  # (the MTP head's layer: transformers does not load it)
    ok = not miss and not extra
    print(f"{'ok  ' if ok else 'FAIL'} transformers loads the tiny checkpoint (from_pretrained, FP8 dequantised): missing {miss}, unexpected {extra} (but for its vision tower, which the checkpoint does not have)")
    t = read_tensors(f"{toy}/model.safetensors")
    cfg = model_cfg(toy)
    fp8 = sorted(n for n in t.names if t[n].dtype == torch.float8_e4m3fn)
    lm = model.model.language_model
    same = 0
    for n in fp8:
        l, rest = re.match(rf"{re.escape(PREFIX)}layers\.(\d+)\.(.*)", n).groups()
        w = lm.layers[int(l)]
        for part in rest.split("."):
            w = getattr(w, part)
        same += bool(torch.equal(w.data.view(torch.int16), fp8_dequantize(t[n], t[n + "_scale_inv"]).view(torch.int16)))
    ok &= same == len(fp8) > 0
    print(f"{'ok  ' if same == len(fp8) > 0 else 'FAIL'} the FP8 matrices as transformers loads them are fp8_dequantize's, bit for bit: {same} of {len(fp8)}")
    ids = torch.tensor([int(v) for v in open(f"{toy}/ids.txt").read().split()][:tokens])
    with torch.no_grad():
        theirs = model(input_ids=ids[None], use_cache=False).logits[0]
    mine = forward(t, cfg, ids, "cpu")
    dv = deviation(mine, theirs)
    top = (f32(mine).argmax(-1) == f32(theirs).argmax(-1)).float().mean().item()
    good = float(dv.median()) <= TOY_BOUND and top >= 0.9  # (transformers' bf16 path rounds the latent attention's scores to bf16, and a router or an indexer at a near tie picks the other in one of the two)
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} the whole model against transformers' Glm5NextForConditionalGeneration as loaded (bf16): the median position's logits differ by {float(dv.median()):.3e} of the largest (bound {TOY_BOUND:g}), the worst by {float(dv.max()):.3e}; "
          f"top-1 the same at {100 * top:.1f}% of {len(ids)} positions (at least 90%)")
    return ok


def selftest():
    import tempfile
    from transformers.models.glm5_next import modeling_glm5_next as M
    torch.manual_seed(0)
    ok = True
    T, H = 70, 2
    q, k, v = (bf(torch.randn(T, H, P)) for _ in range(3))
    g = -5.0 * torch.sigmoid(torch.randn(T, H, P) * 2.0)
    beta = f32(bf(torch.rand(T, H)))
    S0 = torch.randn(H, P, P) * 0.2
    Sa, Sb = S0.clone(), S0.clone()
    ya = kda_recurrent(q, k, v, g, beta, Sa)
    yb = kda_chunked(q, k, v, g, beta, Sb)
    ok &= stats("the chunked delta rule against the recurrence, outputs", yb, ya, 8e-3)
    ok &= stats("the chunked delta rule against the recurrence, final state", Sb, Sa, 1e-4)
    # transformers' own functions on the same tensors (their torch paths): (batch, tokens, heads, channels)
    ref, final = M.recurrent_kimi_delta_attention(q[None], k[None], v[None], g[None], beta[None], S0[None].clone(), True, use_qk_l2norm_in_kernel=True)
    ok &= stats("the recurrence against transformers' recurrent_kimi_delta_attention, outputs", ya, ref[0], 8e-3, 2)
    ok &= stats("the recurrence against transformers' recurrent_kimi_delta_attention, final state", Sa, final[0], 1e-5)
    ref, final = M.chunk_kimi_delta_attention(q[None], k[None], v[None], g[None], beta[None], initial_state=S0[None].clone(), output_final_state=True, use_qk_l2norm_in_kernel=True)
    ok &= stats("the chunked rule against transformers' chunk_kimi_delta_attention, outputs", yb, ref[0], 8e-3)
    ok &= stats("the chunked rule against transformers' chunk_kimi_delta_attention, final state", Sb, final[0], 1e-4)
    # the engine's convolution (one rounding after the SiLU) against transformers' two roundings (the convolution's output to bf16, then the SiLU): a place apart at most
    x = bf(torch.randn(23, 64) * 2)
    w = bf(torch.randn(64, 1, 4) * 0.4)
    y, _ = conv_silu(x, w, torch.zeros(3, 64, dtype=BF))
    ok &= stats("the convolution against transformers' causal_conv1d_fn", y, M.causal_conv1d_fn(x.T[None], w[:, 0, :], None, "silu")[0].T, 2e-2, 2)
    nrm = M.Glm5NextTextRMSNorm(64, 1e-5).to(BF)
    nrm.weight.data = bf(torch.rand(64) + 0.5)
    ok &= stats("the layer norm against transformers' Glm5NextTextRMSNorm", rmsnorm(x, nrm.weight.data, 1e-5), nrm(x), 0.0, 0)
    # the tiny model's config is the fixture the engine's tests hold, and a tiny random model checks out against transformers' modules and model
    fixture = json.load(open(os.path.join(HERE, "..", "tests", "fixtures", "glm5_next_toy_config.json")))
    same = tiny_config() == fixture
    ok &= same
    print(f"{'ok  ' if same else 'FAIL'} the tiny model's config is tests/fixtures/glm5_next_toy_config.json")
    with tempfile.TemporaryDirectory() as d:
        tiny_build(d, level=6, tokens=64, prefill=20, total=50)
        ok &= vs_transformers(d, tokens=48, whole=True)
        ok &= loaded_as_transformers_does(f"{d}/toy", tokens=48)
        # the FP8 matrices of the tiny model: transformers' Fp8Dequantize is the f32 product of the byte's value and its block's scale, rounded to bf16
        t = read_tensors(f"{d}/model.safetensors")
        n = f"{PREFIX}layers.0.mlp.gate_proj.weight"
        by_hand = bf(f32(t[n]) * f32(t[n + "_scale_inv"]).repeat_interleave(128, 0).repeat_interleave(128, 1))
        same = bool(torch.equal(by_hand.view(torch.int16), fp8_dequantize(t[n], t[n + "_scale_inv"]).view(torch.int16)))
        ok &= same
        print(f"{'ok  ' if same else 'FAIL'} transformers' Fp8Dequantize is the byte's value times its block's scale in f32, rounded to bf16 ({tuple(t[n].shape)} with scales {tuple(t[n + '_scale_inv'].shape)})")
    print("selftest:", "ok" if ok else "FAILED")
    return ok


def mtp_check(mini, device=None):
    """`glyd-engine glmmtp MINI/toy MINI/toy`'s results against `mtp_forward` on the engine's own inputs (the model's hidden states it kept, the tokens after them): the head's normed hidden states row by row (a router at a
    near tie takes the other expert in one of the two: most rows within the bound), and the token its LM head picks."""
    dev = device or "cpu"
    toy = f"{mini}/toy"
    cfg = model_cfg(toy)
    t = read_tensors(f"{toy}/model.safetensors")
    ids = torch.tensor([int(v) for v in open(f"{toy}/ids.txt").read().split()])
    hid = cfg["hidden"]
    rows = os.path.getsize(f"{toy}/mtp_out.bin") // 2 // hid
    prev, got = get(f"{toy}/mtp_prev.bin", (rows, hid)).to(dev), get(f"{toy}/mtp_out.bin", (rows, hid)).to(dev)
    W = mtp_weights(t, cfg, dev)
    embed = t[f"{PREFIX}embed_tokens.weight"].to(dev)
    stages = {}
    want = mtp_forward(prev, ids[1:rows + 1].to(dev), embed, W, cfg, stages)
    ok = tokens_within("mtp head: normed hidden states against the reference's (on the engine's hidden states and tokens)", got, want, 1e-1, 0.9)
    ok &= stats("mtp head: the median row", got, want, 0.5)
    lm = t["lm_head.weight"].to(dev)
    rec = [l.split() for l in open(f"{toy}/mtp_tok.txt").read().splitlines()]
    eng_tok = torch.tensor([int(a) for a, _ in rec])
    ref_tok = F.linear(got, lm).float().argmax(-1)
    same = float((eng_tok == ref_tok.cpu()).float().mean())
    ok &= same >= 0.99
    print(f"{'ok  ' if same >= 0.99 else 'FAIL'} mtp head: the token its LM head picks from the engine's own hidden states is the reference's for {100 * same:.1f}% of {rows} rows (at least 99%)")
    top = float((F.linear(want, lm).float().argmax(-1).cpu() == eng_tok).float().mean())
    print(f"     (and the token the reference's own hidden states pick is the engine's for {100 * top:.1f}% of the rows)")
    print("mtp:", "ok" if ok else "FAILED")
    return ok


# ---- the real model's layers through the engine, one at a time, against research/glm-data's ref128 (`glyd-engine glmreal`) -----------------------------------------------------------------------------------------

def fetch_head(out, repo=REPO, rev=REV):
    """OUT/head.safetensors: the LM head and the final norm of the pinned snapshot (by range from the shards that hold them), for the logits of a chain of layers."""
    from safetensors import safe_open
    if os.path.exists(f"{out}/head.safetensors"):
        return
    wm = json.loads(hf_get(repo, rev, "model.safetensors.index.json"))["weight_map"]
    entries = []
    for name in ("lm_head.weight", f"{PREFIX}norm.weight"):
        base, h = shard_header(repo, rev, wm[name])
        a, b = h[name]["data_offsets"]
        entries.append((name, h[name]["dtype"], h[name]["shape"], hf_get(repo, rev, wm[name], (base + a, base + b - 1))))
    os.makedirs(out, exist_ok=True)
    write_safetensors(f"{out}/head.safetensors", entries)


def real_chain_check(out, ref, first, end, teacher, logits=True):
    """The engine's per-layer dumps in OUT against the reference's layers in REF (ref128_layerNN.safetensors: `h_mid` after the mixer, `h_out` after the layer, `topk_idx` of a layer of experts): the difference of
    each layer's streams from the reference's, relative to the largest value of the reference's (the engine's layer on the reference's input where `teacher`, else on its own chain's), the share of positions whose
    experts are the reference's; and, for a chain that reaches the last layer, the logits of its final hidden state against the reference's (KL and top-1)."""
    from safetensors import safe_open
    ok = True
    print(f"{'layer':>5} {'streams after the mixer':>26} {'streams after the layer':>26} {'mean':>10} {'experts same':>13}")
    rows_worst = 0.0
    for l in range(first, end):
        r = safe_open(f"{ref}/layers/ref128_layer{l:02d}.safetensors", "pt")
        mid_r, out_r = r.get_tensor("h_mid")[0], r.get_tensor("h_out")[0]
        T = mid_r.shape[0]
        mid, got = get(f"{out}/real_L{l}.mid.bin", (T, HC, mid_r.shape[-1])), get(f"{out}/real_L{l}.out.bin", (T, HC, mid_r.shape[-1]))
        sc = lambda g, w: float((f32(g) - f32(w)).abs().max() / f32(w).abs().max())
        mean = float((f32(got) - f32(out_r)).abs().mean() / f32(out_r).abs().mean())
        same = ""
        if "topk_idx" in r.keys():
            ids = get(f"{out}/real_L{l}.ids.bin", (T, r.get_tensor("topk_idx").shape[1]), torch.int64)
            ri = r.get_tensor("topk_idx").long()
            share = float(torch.tensor([bool(torch.equal(ids[t].sort().values, ri[t].sort().values)) for t in range(T)]).float().mean())
            same = f"{100 * share:.1f}%"
        print(f"{l:>5} {sc(mid, mid_r):>26.3e} {sc(got, out_r):>26.3e} {mean:>10.3e} {same:>13}")
        rows_worst = max(rows_worst, sc(got, out_r))
    print(f"the worst layer's streams differ from the reference's by {rows_worst:.3e} of the largest value ({'each layer on the reference' if teacher else 'a chain on its own'} input)")
    final = safe_open(f"{ref}/final_ref128.safetensors", "pt")
    if logits and end == 45:
        fetch_head(out)
        h = safe_open(f"{out}/head.safetensors", "pt")
        lm, nw = h.get_tensor("lm_head.weight"), h.get_tensor(f"{PREFIX}norm.weight")
        last = get(f"{out}/real_L44.out.bin", (T, HC, nw.shape[0]))
        hid = hc_head(last, nw, 1e-5)
        lg = F.linear(f32(hid), f32(lm))
        lr = safe_open(f"{ref}/logits_ref128.safetensors", "pt").get_tensor("logits_f32")
        lp, lq = torch.log_softmax(lr, -1), torch.log_softmax(lg, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        top = float((lg.argmax(-1) == lr.argmax(-1)).float().mean())
        fh = final.get_tensor("final_hidden")
        print(f"final hidden state: differs from the reference's by {float((f32(hid) - f32(fh)).abs().max() / f32(fh).abs().max()):.3e} of the largest value")
        print(f"logits of {T} positions against the reference's: KL {float(kl.mean()):.4f} nats a position (median {float(kl.median()):.4f}, 90th {float(kl.quantile(0.9)):.4f}, worst {float(kl.max()):.4f}), top-1 token the same at {100 * top:.1f}%")
    return ok


def mtp_real_check(out, ref):
    """`glyd-engine glmmtpreal`'s hidden states of the real MTP head (rows: the model's final hidden state of position j, the token j + 1) turned into tokens by the LM head (j's draft is the token j + 2): how many are the
    model's own pick for position j + 2 (what a verified draft accepts: the model's logits of the reference at j + 1) and how many are the text's token."""
    from safetensors import safe_open
    fetch_head(out)
    h = safe_open(f"{out}/head.safetensors", "pt")
    lm = h.get_tensor("lm_head.weight")
    fin = safe_open(f"{ref}/final_ref128.safetensors", "pt")
    tokens = fin.get_tensor("tokens").long()
    lg = safe_open(f"{ref}/logits_ref128.safetensors", "pt")
    own = lg.get_tensor("argmax").long()  # the model's pick after each position
    rows = tokens.shape[0] - 1
    hid = get(f"{out}/mtp_real_out.bin", (rows, lm.shape[1]))
    pred = F.linear(f32(hid), f32(lm)).argmax(-1)  # pred[j]: the head's draft for position j + 2
    accept = float((pred[:rows - 1] == own[1:rows]).float().mean())  # the model's own pick after position j + 1
    text = float((pred[:rows - 1] == tokens[2:rows + 1]).float().mean())
    own_text = float((own[1:rows] == tokens[2:rows + 1]).float().mean())
    print(f"the real MTP head on {rows} positions of the reference text: its draft is the model's own next pick for {100 * accept:.1f}% of {rows - 1} positions (what a verified draft would accept), the text's token for {100 * text:.1f}% "
          f"(the model's own pick: {100 * own_text:.1f}%)")
    # the head's hidden states against the reference's own (REF/mtp_head_ref128.bin, the CPU reference's `mtp_forward` on the same rows; the engine holds the head's experts in a few bits)
    if os.path.exists(f"{ref}/mtp_head_ref128.bin"):
        want = get(f"{ref}/mtp_head_ref128.bin", (rows, lm.shape[1]))
        d = (f32(hid) - f32(want)).abs()
        pr = F.linear(f32(want), f32(lm)).argmax(-1)
        print(f"the head's normed hidden states differ from the reference's by {float(d.max() / f32(want).abs().max()):.3e} of the largest value (mean {float(d.mean() / f32(want).abs().mean()):.3e} of the mean); "
              f"its draft is the reference head's draft for {100 * float((pred == pr).float().mean()):.1f}% of the positions (the reference head's own: {100 * float((pr[:rows - 1] == own[1:rows]).float().mean()):.1f}% the model's pick)")
    return accept


def real(mini):
    """The reference against transformers' own modules on the real weights of the mini checkpoint's layers (the mixture of experts' router and shared expert, its experts too where they are few)."""
    cfg = model_cfg(mini)
    ok = vs_transformers(mini, tokens=48, experts_module=cfg["experts"] <= 64)
    print("real:", "ok" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    k = sub.add_parser("kernels")
    k.add_argument("what", choices=["gen", "check"])
    k.add_argument("dir")
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--heads", type=int, default=4)
    k.add_argument("--n", type=int, default=150)
    k.add_argument("--split", type=int, default=70)
    k.add_argument("--d", type=int, default=512)
    k.add_argument("--mla", choices=sorted(MLA_DIMS), default="small", help="gen: the shape of the latent attention's kernels (small: the tiny model's, model: the real one's, 2,600 rows)")
    f = sub.add_parser("fetch")
    f.add_argument("dir")
    f.add_argument("--layers", default="0,1,44")
    f.add_argument("--experts", type=int, help="the first N experts of a mixture-of-experts layer only (and the routers' first N rows): a layer of N experts, not 288")
    f.add_argument("--tokens", type=int, default=333)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--repo", default=REPO)
    f.add_argument("--rev", default=REV)
    ly = sub.add_parser("layers")
    ly.add_argument("what", choices=["gen", "check"])
    ly.add_argument("mini")
    ly.add_argument("io", nargs="?")
    ly.add_argument("--level", type=int, choices=[4, 5, 6], help="gen: the level the engine holds a mixture-of-experts layer's matrices at (the mini checkpoint is the grid of one)")
    ly.add_argument("--kinds", default="DE", help="gen: the layers to make cases for (D: a dense MLP, E: experts)")
    ly.add_argument("--decoded", help="check: the mini checkpoint `glyd-engine glmdecode` made (the weights the engine's packs hold)")
    ty = sub.add_parser("toy")
    ty.add_argument("mini")
    tn = sub.add_parser("tiny")
    tn.add_argument("mini")
    tn.add_argument("--level", type=int, choices=[4, 5, 6], default=6)
    tn.add_argument("--seed", type=int, default=1)
    tn.add_argument("--tokens", type=int, default=300)
    tn.add_argument("--prefill", type=int, default=150)
    tn.add_argument("--total", type=int, default=210)
    fp = sub.add_parser("fp8")
    fp.add_argument("what", choices=["check"])
    fp.add_argument("mini")
    fp.add_argument("out")
    mt = sub.add_parser("mtp")
    mt.add_argument("mini")
    mr = sub.add_parser("mtpreal")
    mr.add_argument("out")
    mr.add_argument("ref")
    rc = sub.add_parser("chain")
    rc.add_argument("out")
    rc.add_argument("ref")
    rc.add_argument("--layers", default="0:45")
    rc.add_argument("--teacher", action="store_true")
    rl = sub.add_parser("real")
    rl.add_argument("mini")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "kernels":
        if a.what == "gen":
            kernels_gen(a.dir, a.seed, a.heads, a.n, a.split, a.d, a.mla)
            return True
        return kernels_check(a.dir)
    if a.cmd == "fetch":
        fetch_mini(a.dir, [int(x) for x in a.layers.split(",")], a.tokens, a.seed, a.experts, a.repo, a.rev)
        return True
    if a.cmd == "layers":
        io_dir = a.io or f"{a.mini}/io"
        if a.what == "gen":
            layers_gen(a.mini, io_dir, a.level, a.kinds)
            return True
        return layers_check(a.mini, io_dir, decoded_dir=a.decoded)
    if a.cmd == "toy":
        return toy_check(a.mini)
    if a.cmd == "tiny":
        tiny_build(a.mini, a.level, a.seed, a.tokens, a.prefill, a.total)
        return True
    if a.cmd == "fp8":
        return fp8_check(a.mini, a.out)
    if a.cmd == "mtp":
        return mtp_check(a.mini)
    if a.cmd == "mtpreal":
        return mtp_real_check(a.out, a.ref) > 0.0
    if a.cmd == "chain":
        f, e = (int(x) for x in a.layers.split(":"))
        return real_chain_check(a.out, a.ref, f, e, a.teacher)
    if a.cmd == "real":
        return real(a.mini)
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
