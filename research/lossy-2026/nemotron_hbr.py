"""Hummingbird for Nemotron 3 Super with neuron rows ("hbr", Linear GLY-98): the recipe, the export of its codes, and the reader that checks the export.
  selftest              the recipe's pieces on the CPU (rotation, bit packing, nearest and GPTQ rounding against the existing code and a brute-force least squares, the export written, read and compared)
  check DIR [--quick] [--hash] [--layers 0,1,..]   read an export: its meta.json, every tensor's dtype and shape, the codes unpacked and the rows dequantised finite; --quick: names, dtypes and shapes only;
                        --hash: the files' sha256 against meta.json
nemotron_quality.py runs it (variants hbr40 and hbr40p; --export DIR writes hbr40's codes).

THE RECIPE. An expert of a latent-MoE layer is z (1,024, the latent) -> up [2,688 x 1,024] -> relu^2 -> down [1,024 x 2,688] -> times the router's weight, summed over the token's 22 experts. Both matrices
become 2,688 NEURON ROWS of 1,024 values: up's own rows (neuron n's input weights u_n) and down transposed (neuron n's output weights d_n = down[:, n]). A row is stored as codes (indices into the row's
codebook), a bf16 scale for each run of 32 of its values (the run's absmax, rounded to bf16), and a codebook of 2^bits bf16 entries in units of the scale (Lloyd-Max of the row's run-normalised values),
all in the basis of ONE rotation R_l = H diag(s_l) of the 1,024 latent axis for the whole layer l: H the orthonormal Walsh-Hadamard matrix of order 1,024 in natural (Sylvester) order, s_l a layer's
random signs. The rows stored are R_l u_n and R_l d_n. The engine runs an expert in that basis: z' = R_l z once for the token, a_n = relu(<R_l u_n, z'>)^2 (R_l is orthogonal: <R_l u, R_l z> = <u, z>),
y' = sum over the ACTIVE neurons n of a_n R_l d_n, and y = R_l^T y' = s_l * (H y') once after the weighted sum over the token's experts (R_l is shared and linear). No rotation touches the neuron axis, so the
92% of the 2,688 activations that are exactly 0 still need no row of down. hbp is the same with a rotation of its own for each expert (seed per layer and expert): what the sharing costs.
Bits a weight at 1,024 values a row: 3 + 16/32 + 8 * 16/1024 = 3.625 (3 bits), 2 + 16/32 + 4 * 16/1024 = 2.5625 (2 bits); hbr40 = 40% of a layer's experts (the most used: 205 of 512) at 3 bits, the rest at 2.

GPTQ ON NEURON ROWS (act-order, error feedback). For the calibration tokens t = 1..T the experts' down gets the activations a_t (2,688) and gives D a_t, D = [d_1 .. d_N] (1,024 x N). Replacing D by Q
changes the outputs by (D - Q) a_t, in total T tr((D - Q) H (D - Q)^T) with H = (1/T) sum_t a_t a_t^T (N x N). R_l is orthogonal, so that is also the error of the rotated rows d'_n = R_l d_n: H needs no rotation.
The neurons are the index of this quadratic form, so GPTQ's recursion runs over neurons and rounds one WHOLE row (1,024 values: its own run scales, its own codebook) a step. Take the neurons in the order of falling
H_nn (act-order) and write H for the damped H + 0.01 mean(diag H) I. After neuron i is rounded to q_i, with delta_i = d_i - q_i, the least-squares re-fit of the neurons r not yet rounded, minimising
|(Q - D) a|^2 over d_r with q_i fixed, solves the normal equations H_rr X = H_ri delta_i^T: d_r <- d_r + (H_rr^-1 H_ri) delta_i. With U the upper Cholesky factor of H^-1 (H^-1 = U^T U, taken in the
act-order) this is d_r <- d_r - (delta_i / U_ii) U[i, r]: the block-inverse identity (H^-1)_ri / (H^-1)_ii = -H_rr^-1 H_ri applied to the inverse of the Hessian of the neurons from i on, whose rows are U's.
That is the recursion of glm_quality.gptq_hlm, with a vector (the row) in place of a scalar per row of the matrix. It is done in blocks of 128 neurons: inside a block each step's update reaches the rest of the
block, after it one product updates all the neurons after. A row's run scales are taken from the row as it stands at its turn (error of the neurons before it already in); its codebook is fitted before the
loop on the rows as they were (a Lloyd-Max fit per row costs 25 passes: not once a step). A neuron no calibration token activates (H_nn = 0) is rounded like the rest: its damped H row has no cross terms, so it
feeds nothing back, and a token the calibration did not see may use it. Experts with too few calibration tokens, or a Hessian that is not positive definite even damped, are rounded to nearest (as hlm3g's).
Up is GPTQ on columns as gq.gptq_hlm has it (the 1,024 latent channels are its inputs, H = E[z' z'^T] of the rotated z), with the rotation given and the codes returned.

THE EXPORT (--export DIR, hbr40): DIR/meta.json and DIR/layer-{L:03d}.safetensors for the MoE layers L = 0..39 in order, each with, for every expert e (0..511, the router's index) and matrix m in {up, downT}:
  e{e:03d}.{m}.codes     uint8 [rows, row_length * bits / 8], row-major, bit-packed. Row n's codes are one bit stream: code j takes stream bits [bits*j, bits*j + bits), stream bit k being bit (k % 8) of
                         byte (k // 8) (little-endian within a byte). 2 bits: code j = (byte[j // 4] >> (2 * (j % 4))) & 3 (a row is 256 bytes). 3 bits: for a group of 8 codes (j = 8g .. 8g + 7),
                         v = byte[3g] | byte[3g + 1] << 8 | byte[3g + 2] << 16 and code 8g + i = (v >> (3 * i)) & 7 (a row is 384 bytes). A code is the index of the row's codebook entry.
  e{e:03d}.{m}.scales    bf16 [rows, row_length / 32]
  e{e:03d}.{m}.codebook  bf16 [rows, 2^bits], ascending
  The dequantisation law, in the rotated basis:  w'[n, j] = scales[n, j // 32] * codebook[n, code[n, j]]   (float32 product of two bf16 values: exact)
  up row n = R_l u_n (u_n = up_proj[e][n, :], neuron n's 1,024 input weights); downT row n = R_l d_n (d_n = down_proj[e][:, n], neuron n's 1,024 output weights); rows = 2,688, row_length = 1,024.
  meta.json: the format, the recipe, the model's repository and revision, the dequantisation law and packing in words, the rotation (the Hadamard's ordering and size, z' = H (s * z), y = s * (H y'), how the signs are
  drawn and the signs themselves as uint8 [1024], 1 -> +1 and 0 -> -1), and for each MoE layer: its index among the model's 88 blocks, the seed, the signs, the bits (2 or 3) of each expert, its file, the file's bytes and sha256.
The Exporter checks each expert as it takes it (the packed bytes unpack to the codes, and dequantise to the job's own float32 weights, bit for bit); read_layer and check_export are the reader the engine's packer can follow.
"""
import argparse, hashlib, json, math, os, sys, tempfile, types

import glm_quality as gq
import glm_skew as gs

np, torch, F = gq.np, gq.torch, gq.F
BLOCK = gq.BLOCK
FORMAT = "glyd-hbr-codes/1"
MATS = ("up", "downT")


# ------------------------------------------------------------------------------------------------------------------------------------------------- rotation
def had(x):
    """x (..., n) times the orthonormal Walsh-Hadamard matrix of order n (a power of 2) in natural (Sylvester) order, H[i, k] = (-1)^popcount(i & k) / sqrt(n): over the whole last dim (gq.fwht's blocks of 128 are
    not used here). Symmetric and orthonormal, so also its own inverse."""
    return x @ gq._hadamard(x.shape[-1], x.device)


def layer_seed(li):
    """The seed of the rotation shared by the experts of the MoE layer at block li (up and down use the same; hlm's seeds are li * 8 and li * 8 + 1)."""
    return li * 8 + 2


def expert_seed(li, e):
    """The seed of expert e's own rotation in hbp."""
    return (li * 8 + 3) * 1024 + e


def _s(s):
    return s if s.dim() == 1 else s[:, None, :]


def rot(x, s):
    """x (n, rows, L) in the rotated basis: each row w -> H (s * w); s (L,) shared or (n, L) one an expert."""
    return had(x * _s(s))


def unrot(x, s):
    """The inverse of rot: each row w' -> s * (H w')."""
    return had(x) * _s(s)


def sign_bits(seed, n, dev="cpu"):
    """uint8 [n] of the draw behind signs(n, seed): torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(seed)); 1 means +1, 0 means -1."""
    return ((gq.signs(n, seed, torch.device(dev)) + 1) / 2).to(torch.uint8)


# ------------------------------------------------------------------------------------------------------------------------------------------ rounding of rows
def code_rows(x, cb, sc):
    """Rows x (m, L) rounded to nearest in their codebook cb (m, k, ascending) in units of the scales sc (m, L / 32): the indices (m, L, int64) and the float32 weights cb[idx] * sc."""
    m, L = x.shape
    mids = ((cb[:, 1:] + cb[:, :-1]) / 2).contiguous()
    idx = torch.searchsorted(mids, (x.view(m, L // BLOCK, BLOCK) / sc[:, :, None]).view(m, L).contiguous())
    return idx, (cb.gather(1, idx).view(m, L // BLOCK, BLOCK) * sc[:, :, None]).view(m, L)


def rtn_rows(x, bits):
    """Rows x (m, L), already in the rotated basis, rounded to nearest (gq.quant_hlm's rule): a dict q (m, L, float32), codes (uint8), scales (m, L / 32, float32 holding bf16 values), cb (m, 2^bits)."""
    m, L = x.shape
    r = x.view(m, L // BLOCK, BLOCK)
    amax = r.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    cb = gq.lloyd_max((r / amax).view(m, L), None, 1 << bits)
    sc = amax.bfloat16().float().view(m, L // BLOCK)
    idx, q = code_rows(x, cb, sc)
    return dict(q=q, codes=idx.to(torch.uint8), scales=sc, cb=cb)


def gptq_up(W, Hs, bits, s, hd=had, damp=0.01, step=128):
    """gq.gptq_hlm for the first matrix (inputs are the columns), with the rotation given and the codes returned: W (n, rows, cols) float32 and Hs, n Hessians (cols, cols) (both consumed); s: the signs, (cols,) shared or
    (n, cols); hd: the Hadamard (had; gq.fwht makes it gq.gptq_hlm's). The rounding is gq.gptq_hlm's (checked bit for bit by the selftest). Returns a dict q (n, rows, cols, float32: the rotated basis, the columns in their
    own order), codes (uint8), scales (n, rows, cols / 32), cb (n, rows, 2^bits), and ok (n,): False where the damped Hessian was not positive definite (that matrix is for the caller to round to nearest)."""
    n, rows, cols = W.shape
    runs, dev = cols // BLOCK, W.device
    sj = lambda j: s if s.dim() == 1 else s[j]
    U, wt, perm, ok = torch.empty(n, cols, cols, device=dev), torch.empty(n, cols, device=dev), torch.empty(n, cols, dtype=torch.long, device=dev), [True] * n
    for j in range(n):
        H, Hs[j] = Hs[j], None
        dead = H.diagonal() == 0
        w = hd(W[j].masked_fill(dead[None, :], 0) * sj(j))
        H = hd(hd(H * sj(j)).T * sj(j)).T  # T H T^T
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
    cb = gq.lloyd_max((w4 / w4.abs().amax(-1, keepdim=True).clamp_min(1e-30)).view(n * rows, cols), wt / wt.mean(-1, keepdim=True).clamp_min(1e-30) + 1e-6, 1 << bits, per=rows)
    Q, mids = torch.empty_like(W), ((cb[:, 1:] + cb[:, :-1]) / 2).contiguous()
    C, S = torch.empty(n, rows, cols, dtype=torch.uint8, device=dev), torch.empty(n, rows, runs, device=dev)
    for i1 in range(0, cols, step):
        i2 = min(i1 + step, cols)
        W1, E1, U1 = W[:, :, i1:i2].clone(), torch.empty(n, rows, i2 - i1, device=dev), U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % BLOCK == 0:
                sc = W1[:, :, i:i + BLOCK].abs().amax(-1).clamp_min(1e-30).bfloat16().float()  # (n, rows)
                S[:, :, (i1 + i) // BLOCK] = sc
            col, v, e = W1[:, :, i], Q[:, :, i1 + i], E1[:, :, i]
            idx = torch.searchsorted(mids, (col / sc).reshape(-1, 1))
            torch.mul(cb.gather(1, idx).view(n, rows), sc, out=v)
            C[:, :, i1 + i] = idx.view(n, rows)
            torch.sub(col, v, out=e)
            e.div_(U1[:, i, i][:, None])
            W1[:, :, i:].addcmul_(e[:, :, None], U1[:, i, None, i:], value=-1)
        if i2 < cols:
            W[:, :, i2:] -= torch.bmm(E1, U[:, i1:i2, i2:])
    inv = perm.argsort(-1)
    run_of = perm[:, ::BLOCK] // BLOCK  # (n, runs): the original run at each place of the act-order
    sc_out = torch.empty_like(S).scatter_(2, run_of[:, None, :].expand(n, rows, runs), S)
    take = lambda t: t.gather(2, inv[:, None, :].expand(n, rows, cols))
    return dict(q=take(Q), codes=take(C), scales=sc_out, cb=cb.view(n, rows, -1)), torch.tensor(ok, device=dev)


def gptq_neurons(W, Hs, bits, damp=0.01, step=128):
    """GPTQ on neuron rows (the math is in the module's note): W (n, N, L) float32, the rows (already rotated) of n experts' N neurons, and Hs, n Hessians (N, N) of the neurons' activations (both consumed).
    Returns a dict q (n, N, L, float32: rotated basis, the neurons in their own order), codes (uint8), scales (n, N, L / 32), cb (n, N, 2^bits), and ok (n,) as gptq_up's."""
    n, N, L = W.shape
    runs, dev = L // BLOCK, W.device
    U, perm, ok = torch.empty(n, N, N, device=dev), torch.empty(n, N, dtype=torch.long, device=dev), [True] * n
    for j in range(n):
        H, Hs[j] = Hs[j], None
        perm[j] = H.diagonal().argsort(descending=True)  # act-order: the most active neurons first
        W[j] = W[j][perm[j]]
        H = H[perm[j]][:, perm[j]]
        H.diagonal().add_(damp * H.diagonal().mean())
        Lc, info = torch.linalg.cholesky_ex(H)
        del H
        if info.item():
            ok[j] = False
            U[j] = torch.eye(N, device=dev)
            continue
        Uj, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(Lc), upper=True)
        ok[j] = not info.item()
        U[j] = Uj if ok[j] else torch.eye(N, device=dev)
    r = W.view(n * N, runs, BLOCK)
    cb = gq.lloyd_max((r / r.abs().amax(-1, keepdim=True).clamp_min(1e-30)).view(n * N, L), None, 1 << bits).view(n, N, -1).permute(1, 0, 2).contiguous()  # (N, n, k): a neuron's codebooks together
    mids = ((cb[:, :, 1:] + cb[:, :, :-1]) / 2).contiguous()
    Q, C, S = torch.empty_like(W), torch.empty(n, N, L, dtype=torch.uint8, device=dev), torch.empty(n, N, runs, device=dev)
    for i1 in range(0, N, step):
        i2 = min(i1 + step, N)
        W1, E1, U1 = W[:, i1:i2].clone(), torch.empty(n, i2 - i1, L, device=dev), U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            row = W1[:, i]  # (n, L): the neuron's vector, the error of the neurons before it already in
            r3 = row.view(n, runs, BLOCK)
            sc = r3.abs().amax(-1).clamp_min(1e-30).bfloat16().float()  # (n, runs)
            idx = torch.searchsorted(mids[i1 + i], (r3 / sc[:, :, None]).view(n, L))
            v = (cb[i1 + i].gather(1, idx).view(n, runs, BLOCK) * sc[:, :, None]).view(n, L)
            Q[:, i1 + i], C[:, i1 + i], S[:, i1 + i] = v, idx, sc
            e = (row - v) / U1[:, i, i][:, None]
            W1[:, i:].addcmul_(U1[:, i, i:, None], e[:, None, :], value=-1)
            E1[:, i] = e
        if i2 < N:
            W[:, i2:] -= torch.bmm(U[:, i1:i2, i2:].transpose(1, 2), E1)
    inv = perm.argsort(-1)
    take = lambda t: t.gather(1, inv[:, :, None].expand(n, N, t.shape[2]))
    return dict(q=take(Q), codes=take(C), scales=take(S), cb=take(cb.permute(1, 0, 2))), torch.tensor(ok, device=dev)


def hbr_round(e, up, dn, ids, X, rt, ctx, li, bits, gptq, kind):
    """Experts `ids` of layer li (float32 up (n, N, L), dn (n, L, N)) as hbr<bits>[g] (kind hbr: the layer's rotation) or hbp (one an expert): the bf16 weights in the original basis, (qup, qdn). With GPTQ, experts with
    fewer than ctx.min_tok calibration tokens (or a Hessian not positive definite) are rounded to nearest. If ctx has an exporter for this variant it gets the codes of all the experts, before the rotation is undone."""
    n, N, L = up.shape
    dev = up.device
    s = gq.signs(L, layer_seed(li), dev) if kind == "hbr" else torch.stack([gq.signs(L, expert_seed(li, i), dev) for i in ids])
    toks = [rt.of(i)[0] for i in ids]
    toks = [t[t < ctx.ncal] for t in toks]
    use = [j for j in range(n) if gptq and len(toks[j]) >= ctx.min_tok]
    dnt = dn.transpose(1, 2)  # (n, N, L): neuron rows
    k = 1 << bits
    res = {m: dict(q=torch.empty(n, N, L, device=dev), codes=torch.empty(n, N, L, dtype=torch.uint8, device=dev), scales=torch.empty(n, N, L // BLOCK, device=dev), cb=torch.empty(n, N, k, device=dev)) for m in MATS}

    def put(m, sel, r):
        for key, v in r.items():
            res[m][key][sel] = v.reshape(len(sel), N, -1)
    fall = [j for j in range(n) if j not in use]
    if use:
        Hi, Hd = [], []
        with gq.tf32():
            for j in use:
                x = X[toks[j]]
                xf = x.float()
                Hi.append(xf.T @ xf / len(x))
                act = gq.activate(e, F.linear(x, up[j].bfloat16())).float()  # (down's input, from the released up as the layer's forward has it)
                Hd.append(act.T @ act / len(x))
        sel = torch.tensor(use, device=dev)
        s_sel = s if s.dim() == 1 else s[sel]
        r1, ok1 = gptq_up(up[sel], Hi, bits, s_sel)
        r2, ok2 = gptq_neurons(rot(dnt[sel], s_sel), Hd, bits)
        put("up", use, r1)
        put("downT", use, r2)
        bad = (~(ok1 & ok2)).tolist()
        fall += [j for j, b in zip(use, bad) if b]
        ctx.done += len(use) - sum(bad)
    if fall:
        sel = torch.tensor(fall, device=dev)
        s_sel = s if s.dim() == 1 else s[sel]
        put("up", fall, rtn_rows(rot(up[sel], s_sel).reshape(-1, L), bits))
        put("downT", fall, rtn_rows(rot(dnt[sel], s_sel).reshape(-1, L), bits))
        ctx.fell += len(fall)
    ex = getattr(ctx, "export", None)
    if ex is not None and ex.variant == getattr(ctx, "label", None) and kind == "hbr":
        ex.add(li, ids, bits, res["up"], res["downT"])
    return unrot(res["up"]["q"], s).bfloat16(), unrot(res["downT"]["q"], s).bfloat16().transpose(1, 2).contiguous()


gq.ROWS["hbr"] = gq.ROWS["hbp"] = hbr_round


# ------------------------------------------------------------------------------------------------------------------------------------------- the export
def pack_codes(c, bits):
    """c (..., L) uint8 codes (L a multiple of 8) -> (..., L * bits / 8) uint8, as the module's note says: code j at stream bits [bits * j, bits * j + bits), little-endian within a byte."""
    g = c.reshape(*c.shape[:-1], -1, 8).long()
    v = (g << (bits * torch.arange(8, device=c.device))).sum(-1)  # the 8 codes of a group as one number of 8 * bits bits
    return ((v[..., None] >> (8 * torch.arange(bits, device=c.device))) & 255).to(torch.uint8).reshape(*c.shape[:-1], -1)


def unpack_codes(p, bits):
    g = p.reshape(*p.shape[:-1], -1, bits).long()
    v = (g << (8 * torch.arange(bits, device=p.device))).sum(-1)
    return ((v[..., None] >> (bits * torch.arange(8, device=p.device))) & ((1 << bits) - 1)).to(torch.uint8).reshape(*p.shape[:-1], -1)


def dequant(codes, scales, cb):
    """The dequantisation law: w'[n, j] = scales[n, j // 32] * codebook[n, code[n, j]] in float32 (codes (rows, L) uint8, scales (rows, L / 32) and cb (rows, 2^bits) bf16 or float32)."""
    return scales.float().repeat_interleave(BLOCK, -1) * cb.float().gather(-1, codes.long())


LAW = "w'[n, j] = scales[n, j // 32] * codebook[n, code[n, j]], the float32 product of two bf16 values (exact), in the rotated basis; matrices: up row n = R_l u_n, downT row n = R_l d_n"
PACKING = ("codes: uint8 [rows, row_length * bits / 8], row-major; row n is one bit stream, code j at stream bits [bits*j, bits*j+bits), stream bit k = bit (k % 8) of byte (k // 8) (little-endian within a byte). "
           "2 bits: code j = (byte[j // 4] >> (2 * (j % 4))) & 3. 3 bits: for the group of 8 codes j = 8g..8g+7, v = byte[3g] | byte[3g+1] << 8 | byte[3g+2] << 16, code 8g+i = (v >> (3*i)) & 7. "
           "A code indexes the row's codebook (ascending).")
ROTATION = {"R_l": "H diag(s_l), one for all experts of MoE layer l and both matrices; z' = H (s * z); y = R_l^T y' = s * (H y')",
            "hadamard": "orthonormal, order 1024 (the whole axis), natural (Sylvester) order: H[i, k] = (-1)^popcount(i & k) / 32; H = [[H', H'], [H', -H']] / sqrt(2) recursively; the butterfly (h = 1, 2, 4 .. 512: a, b -> a + b, a - b) then / 32",
            "signs": "s_j = 2 * b_j - 1 with b = torch.randint(0, 2, (1024,), generator=torch.Generator().manual_seed(seed)) (CPU, int64; torch 2.11.0); signs_u8 is b: the saved signs are the authority, another torch may draw otherwise"}


class Exporter:
    """Takes the codes of one variant's experts as the pass rounds them, a MoE layer at a time (add), and writes the layer's file when the layer is whole (flush), with meta.json rewritten after each layer
    (complete: false until the last). Each expert is checked as it arrives: its packed bytes unpack to its codes and dequantise (the law) to the job's own float32 weights, bit for bit."""

    def __init__(self, out, variant="hbr40", repo=None, rev=None, keep=False):
        self.out, self.variant, self.repo, self.rev = out, variant, repo, rev
        self.buf, self.layers, self.moe, self.shape, self.keep = {}, [], None, None, ({} if keep else None)
        os.makedirs(out, exist_ok=True)

    def add(self, li, ids, bits, up, dnt):
        """The experts `ids` of block li at `bits`: up and dnt are the rounded rows of each, dicts as rtn_rows' with a leading dimension of experts."""
        d = self.buf.setdefault(li, {})
        self.shape = (up["q"].shape[1], up["q"].shape[2])
        for j, e in enumerate(ids):
            ent = {}
            for m, r in ((MATS[0], up), (MATS[1], dnt)):
                codes, q = r["codes"][j], r["q"][j]
                p = pack_codes(codes, bits)
                sc, cb = r["scales"][j].bfloat16(), r["cb"][j].bfloat16()
                assert torch.equal(sc.float(), r["scales"][j]) and torch.equal(cb.float(), r["cb"][j]), "scales and codebooks are not bf16 values"
                assert torch.equal(unpack_codes(p, bits), codes) and torch.equal(dequant(unpack_codes(p, bits), sc, cb), q), f"layer block {li} expert {e} {m}: the packed codes do not dequantise to the job's weights"
                ent[m] = dict(codes=p.cpu(), scales=sc.cpu(), codebook=cb.cpu())
                if self.keep is not None:
                    self.keep[(li, e, m)] = q.cpu().clone()
            ent["bits"] = bits
            d[e] = ent

    def flush(self, li, moe, E):
        """Layer li is whole: write its file (E experts) and meta.json. moe: the model's MoE blocks in order."""
        from safetensors.torch import save_file
        self.moe = moe
        L, got = moe.index(li), self.buf.pop(li)
        assert sorted(got) == list(range(E)), f"block {li}: {len(got)} of {E} experts"
        tensors = {f"e{e:03d}.{m}.{k}": got[e][m][k].contiguous() for e in range(E) for m in MATS for k in ("codes", "scales", "codebook")}
        name = f"layer-{L:03d}.safetensors"
        save_file(tensors, f"{self.out}/{name}.tmp", metadata={"format": "pt", "glyd": FORMAT, "moe_layer": str(L), "block": str(li)})
        os.replace(f"{self.out}/{name}.tmp", f"{self.out}/{name}")
        h = hashlib.sha256()
        with open(f"{self.out}/{name}", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        rows, length = self.shape
        self.layers.append(dict(moe_layer=L, block=li, seed=layer_seed(li), signs_u8=sign_bits(layer_seed(li), length).tolist(), expert_bits=[got[e]["bits"] for e in range(E)],
                                file=name, bytes=os.path.getsize(f"{self.out}/{name}"), sha256=h.hexdigest()))
        self.layers.sort(key=lambda r: r["moe_layer"])
        self.write_meta(len(self.layers) == len(moe))

    def write_meta(self, complete):
        rows, length = self.shape
        meta = dict(format=FORMAT, recipe=self.variant, complete=complete, model=dict(repo=self.repo, revision=self.rev), rows=rows, row_length=length, block=BLOCK, experts=len(self.layers[0]["expert_bits"]),
                    moe_layers=len(self.moe), dequantization=LAW, packing=PACKING, rotation=ROTATION, layers=self.layers)
        json.dump(meta, open(f"{self.out}/meta.json.tmp", "w"))
        os.replace(f"{self.out}/meta.json.tmp", f"{self.out}/meta.json")


def read_layer(d, rec, rows, length):
    """The layer's file as {(e, m): dict(bits, codes (rows, length) uint8, scales, codebook)} (codes unpacked)."""
    from safetensors import safe_open
    out = {}
    with safe_open(f"{d}/{rec['file']}", "pt") as f:
        for e, bits in enumerate(rec["expert_bits"]):
            for m in MATS:
                t = {k: f.get_tensor(f"e{e:03d}.{m}.{k}") for k in ("codes", "scales", "codebook")}
                assert t["codes"].dtype == torch.uint8 and t["codes"].shape == (rows, length * bits // 8), (e, m, t["codes"].shape)
                assert t["scales"].dtype == torch.bfloat16 and t["scales"].shape == (rows, length // BLOCK) and t["codebook"].dtype == torch.bfloat16 and t["codebook"].shape == (rows, 1 << bits), (e, m)
                out[(e, m)] = dict(bits=bits, codes=unpack_codes(t["codes"], bits), scales=t["scales"], codebook=t["codebook"])
    return out


def check_shapes(d, rec, rows, length):
    """A layer's file without reading its data: the tensors there are exactly the three of each expert and matrix, with the dtypes and shapes of the format."""
    from safetensors import safe_open
    want = {f"e{e:03d}.{m}.{k}": (dt, (rows, w)) for e, bits in enumerate(rec["expert_bits"]) for m in MATS
            for k, dt, w in (("codes", "U8", length * bits // 8), ("scales", "BF16", length // BLOCK), ("codebook", "BF16", 1 << bits))}
    with safe_open(f"{d}/{rec['file']}", "pt") as f:
        assert set(f.keys()) == set(want), f"{rec['file']}: tensors"
        for k, (dt, shape) in want.items():
            sl = f.get_slice(k)
            assert sl.get_dtype() == dt and tuple(sl.get_shape()) == shape, (rec["file"], k, sl.get_dtype(), sl.get_shape())


def check_export(d, expect=None, layers=None, hash_files=False, deep=True):
    """Read the export in d and check it: meta.json, every tensor's dtype and shape (deep=False: nothing more, the data unread), the dequantised rows finite and, with expect ({(block, expert, matrix): the job's
    float32 weights}), equal to them bit for bit; hash_files: the files' sha256 against meta.json. Returns a summary dict."""
    meta = json.load(open(f"{d}/meta.json"))
    assert meta["format"] == FORMAT, meta["format"]
    rows, length, n, regen = meta["rows"], meta["row_length"], 0, True
    for rec in meta["layers"]:
        if layers is not None and rec["moe_layer"] not in layers:
            continue
        assert os.path.getsize(f"{d}/{rec['file']}") == rec["bytes"], f"{rec['file']}: size"
        if hash_files:
            h = hashlib.sha256()
            with open(f"{d}/{rec['file']}", "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            assert h.hexdigest() == rec["sha256"], f"{rec['file']}: sha256"
        assert len(rec["signs_u8"]) == length and set(rec["signs_u8"]) <= {0, 1}, f"{rec['file']}: signs"
        regen = regen and rec["signs_u8"] == sign_bits(rec["seed"], length).tolist()
        check_shapes(d, rec, rows, length)
        for (e, m), t in (read_layer(d, rec, rows, length).items() if deep else ()):
            w = dequant(t["codes"], t["scales"], t["codebook"])
            assert torch.isfinite(w).all(), (rec["file"], e, m)
            if expect is not None:
                assert torch.equal(w, expect[(rec["block"], e, m)]), f"{rec['file']} expert {e} {m}: not the job's weights"
            n += 1
    return dict(layers=len(meta["layers"]), matrices_read=n, complete=meta["complete"], recipe=meta["recipe"], signs_regenerate_with_this_torch=regen)


# ----------------------------------------------------------------------------------------------------------------------------------------------- selftest
def brute_gptq(W, A, cb, damp=0.01):
    """The same rounding by least squares, float64, one neuron at a time: W (N, L) rows, A (T, N) the activations of T calibration tokens, cb (N, k) the rows' codebooks. In act-order the next
    neuron's current row is rounded (bf16 absmax scale per run of 32, nearest codebook entry), and the neurons after it are re-fitted by np.linalg.lstsq on the calibration activations, ridge damp * mean(diag H) as GPTQ's
    damping: min over d_r of |A_r (d_r - w_r) + A_i (q_i - w_i)|^2 + T * lam |d_r - w_r|^2, with q_i fixed."""
    T, N = A.shape
    H = A.T @ A / T
    lam = damp * np.diag(H).mean()
    order = np.argsort(-np.diag(H), kind="stable")
    W, Q = W.astype(np.float64).copy(), np.zeros(W.shape)
    for pos, i in enumerate(order):
        row = W[i]
        r = torch.from_numpy(row).view(-1, BLOCK)
        sc = r.abs().amax(-1, keepdim=True).clamp_min(1e-30).bfloat16().double()
        mids = torch.from_numpy((cb[i][1:] + cb[i][:-1]) / 2)
        v = (torch.from_numpy(cb[i])[torch.searchsorted(mids, (r / sc).contiguous())] * sc).view(-1).numpy()
        Q[i] = v
        rest = order[pos + 1:]
        if len(rest):
            Ar, Ai = A[:, rest], A[:, i:i + 1]
            lhs = np.vstack([Ar, np.sqrt(T * lam) * np.eye(len(rest))])
            rhs = np.vstack([-Ai @ (v - row)[None, :], np.zeros((len(rest), W.shape[1]))])
            W[rest] += np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return Q


def selftest(_=None):
    import shutil
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    # ---- the Hadamard: natural order, orthonormal, its own inverse, the butterfly
    for n in (64, 1024):
        Hm = had(torch.eye(n))
        pc = lambda x: bin(x).count("1")
        assert torch.allclose(Hm, torch.tensor([[(-1.0) ** pc(i & k) for k in range(n)] for i in range(n)]) / n ** 0.5, atol=1e-6) and torch.allclose(Hm @ Hm, torch.eye(n), atol=1e-5)
    x = torch.randn(3, 5, 64, generator=g)
    y, h = x.clone().reshape(-1, 64), 1
    while h < 64:  # (the butterfly the engine can run)
        y = y.reshape(-1, 64 // (2 * h), 2, h)
        y = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), 2)
        h *= 2
    assert torch.allclose(had(x), y.reshape(x.shape) / 8, atol=1e-5)
    s = gq.signs(64, 5, torch.device("cpu"))
    assert torch.allclose(unrot(rot(x, s), s), x, atol=1e-5) and torch.equal(sign_bits(5, 64).float() * 2 - 1, s) and not torch.equal(gq.signs(64, layer_seed(3), torch.device("cpu")), gq.signs(64, expert_seed(3, 0), torch.device("cpu")))
    # ---- the bit packing: against the bit stream written bit by bit
    for bits in (2, 3, 4):
        c = torch.randint(0, 1 << bits, (5, 64), dtype=torch.uint8, generator=g)
        p = pack_codes(c, bits)
        assert p.shape == (5, 64 * bits // 8) and torch.equal(unpack_codes(p, bits), c)
        for r in range(5):
            stream = [(int(c[r, j]) >> b) & 1 for j in range(64) for b in range(bits)]
            ref = [sum(stream[8 * t + b] << b for b in range(8)) for t in range(len(stream) // 8)]
            assert ref == p[r].tolist(), (bits, r)
    p3 = pack_codes(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.uint8), 3)  # (the note's 3-bit rule on one group)
    v = int(p3[0, 0]) | int(p3[0, 1]) << 8 | int(p3[0, 2]) << 16
    assert [(v >> (3 * i)) & 7 for i in range(8)] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert pack_codes(torch.tensor([[3, 0, 1, 2] * 2], dtype=torch.uint8), 2).tolist() == [[3 | 0 << 2 | 1 << 4 | 2 << 6] * 2]
    # ---- nearest rounding and GPTQ on columns are gq's, bit for bit, with gq's Hadamard
    seed, W = 9, (torch.randn(3, 40, 128, generator=g) * 0.05).bfloat16().float()
    sg = gq.signs(128, seed, torch.device("cpu"))
    for bits in (2, 3):
        r = rtn_rows(gq.fwht(W * sg).reshape(-1, 128), bits)
        assert torch.equal((gq.fwht(r["q"].view(3, 40, 128)) * sg).bfloat16(), gq.quant_hlm(W, bits, seed)), bits
    X = torch.randn(1500, 128, generator=g) * torch.where(torch.arange(128) % 17 == 0, 6.0, 1.0)
    Hs = [X.T @ X / 1500, X.roll(7, 0).T @ X.roll(7, 0) / 1500, (X * 0.5).T @ (X * 0.5) / 1500]
    for bits in (2, 3):
        r, ok = gptq_up(W.clone(), [h.clone() for h in Hs], bits, sg, gq.fwht)
        want, ok2 = gq.gptq_hlm(W.clone(), [h.clone() for h in Hs], bits, seed)
        assert ok.all() and ok2.all() and torch.equal((gq.fwht(r["q"]) * sg).bfloat16(), want), ("gptq_up is not gptq_hlm", bits)
        assert torch.equal(dequant(r["codes"][1], r["scales"][1], r["cb"][1]), r["q"][1]), "the codes, scales and codebooks are not the weights"
    sp = torch.stack([gq.signs(128, 20 + j, torch.device("cpu")) for j in range(3)])  # a rotation an expert: the same as each expert alone
    r3, _ = gptq_up(W.clone(), [h.clone() for h in Hs], 3, sp, gq.fwht)
    for j in range(3):
        r1, _ = gptq_up(W[j:j + 1].clone(), [Hs[j].clone()], 3, sp[j], gq.fwht)
        assert (r3["q"][j] - r1["q"][0]).abs().max() < 1e-6 and torch.equal(r3["codes"][j], r1["codes"][0])
    # ---- GPTQ on neuron rows against least squares: 12 neurons (one never active, so a dead row) of 64 values, 2 and 3 bits, blocks of 4 neurons (3 blocks) and of 128
    N, Lr, T = 12, 64, 300
    Wn = (torch.randn(2, N, Lr, generator=g) * 0.05).bfloat16().float()
    A = torch.relu(torch.randn(T, N, generator=g) + 0.7 * torch.randn(T, 1, generator=g)) ** 2 * (1 + 3.0 * (torch.arange(N) % 4 == 0))
    A[:, 5] = 0
    A2 = A.roll(11, 0) * 0.8
    for bits in (2, 3):
        for step in (4, 128):
            got, ok = gptq_neurons(Wn.clone(), [(A.T @ A / T), (A2.T @ A2 / T)], bits, step=step)
            assert ok.all()
            for j, Aj in enumerate((A, A2)):
                want = brute_gptq(Wn[j].double().numpy(), Aj.double().numpy(), got["cb"][j].double().numpy())
                assert np.abs(got["q"][j].double().numpy() - want).max() < 1e-4 * np.abs(want).max(), ("neuron GPTQ differs from least squares", bits, step, j, np.abs(got["q"][j].double().numpy() - want).max())
                assert torch.equal(dequant(got["codes"][j], got["scales"][j], got["cb"][j]), got["q"][j])
    # and what it is for: under H its error is below nearest rounding's, at 2 bits, for every neuron order the codebooks allow
    got, _ = gptq_neurons(Wn.clone(), [(A.T @ A / T), (A2.T @ A2 / T)], 2)
    near = rtn_rows(Wn.reshape(-1, Lr), 2)["q"].view(2, N, Lr)
    for j, Aj in enumerate((A, A2)):
        herr = lambda Q: float(((Aj @ (Q[j] - Wn[j])) ** 2).sum())
        assert herr(got["q"]) < herr(near), (herr(got["q"]), herr(near))
    one, _ = gptq_neurons(Wn[:1].clone(), [(A.T @ A / T)], 2)
    assert (one["q"][0] - got["q"][0]).abs().max() < 1e-6, "an expert alone differs from the same expert in a batch"
    bad, okb = gptq_neurons(Wn.clone(), [-torch.eye(N), (A2.T @ A2 / T)], 2)  # a Hessian that is no use: flagged, the other expert untouched
    assert okb.tolist() == [False, True] and (bad["q"][1] - got["q"][1]).abs().max() < 1e-6
    # ---- the export: a layer of 5 experts written, read and compared; the checks that refuse a wrong byte
    d = tempfile.mkdtemp()
    ex = Exporter(d, "hbr40", "repo", "rev", keep=True)
    rows_, len_ = 24, 64
    for bits, ids in ((3, [0, 1, 2]), (2, [3, 4])):
        mk = lambda: rtn_rows((torch.randn(len(ids) * rows_, len_, generator=g) * 0.05), bits)
        sh = lambda r: {k: v.view(len(ids), rows_, *v.shape[1:]) for k, v in r.items()}
        ex.add(7, ids, bits, sh(mk()), sh(mk()))
    ex.flush(7, [7, 9], 5)
    assert not json.load(open(f"{d}/meta.json"))["complete"]
    ex.add(9, [0, 1, 2, 3, 4], 2, *[{k: v.view(5, rows_, *v.shape[1:]) for k, v in rtn_rows(torch.randn(5 * rows_, len_, generator=g) * 0.05, 2).items()} for _ in range(2)])
    ex.flush(9, [7, 9], 5)
    meta = json.load(open(f"{d}/meta.json"))
    assert meta["complete"] and [r["expert_bits"] for r in meta["layers"]] == [[3, 3, 3, 2, 2], [2] * 5] and [r["block"] for r in meta["layers"]] == [7, 9] and meta["layers"][0]["seed"] == layer_seed(7)
    summ = check_export(d, expect=ex.keep, hash_files=True)
    assert summ["matrices_read"] == 20 and summ["complete"] and check_export(d, deep=False, hash_files=True)["matrices_read"] == 0, summ
    from safetensors import safe_open
    with safe_open(f"{d}/layer-000.safetensors", "pt") as f:
        assert f.get_tensor("e000.up.codes").shape == (rows_, 64 * 3 // 8) and f.get_tensor("e004.downT.codes").shape == (rows_, 64 * 2 // 8) and f.get_tensor("e004.up.scales").dtype == torch.bfloat16
    ex.keep[(7, 0, "up")][0, 0] += 1e-3
    try:
        check_export(d, expect=ex.keep)
        raise AssertionError("a wrong weight passed")
    except AssertionError as err:
        assert "not the job's weights" in str(err)
    with open(f"{d}/layer-001.safetensors", "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\xff\xff\xff\xff")
    try:
        check_export(d, hash_files=True)
        raise AssertionError("a wrong byte passed")
    except AssertionError as err:
        assert "sha256" in str(err)
    shutil.rmtree(d)
    # ---- bits a weight, from the job's accounting
    mats = [(2688, 1024), (1024, 2688)]
    b3, b2, hbp = gq.spec_bpw("hbr3g", mats), gq.spec_bpw("hbr2g", mats), gq.spec_bpw("hbp2g", mats)
    assert abs(b3 - 3.625) < 1e-12 and abs(b2 - 2.5625) < 1e-12 and abs(hbp - b2 - 1024 / 5505024) < 1e-12 and gq.spec_bpw("hbr2", mats) == b2
    v40 = dict(share=40, hot="hbr3g", rest="hbr2g")
    assert abs(gq.variant_bpw(v40, mats) - (0.4 * 3.625 + 0.6 * 2.5625)) < 1e-12 and gq.variant_bpw(v40, mats) <= 3.0
    print(f"nemotron_hbr selftest ok: Hadamard natural order and its butterfly; bit packing at 2, 3, 4 bits against a bit stream; nearest and column GPTQ equal gq's bit for bit, per-expert rotation equal to alone; "
          f"neuron GPTQ equal to a float64 least-squares fit at 2 and 3 bits (blocks of 4 and 128, a dead neuron), below nearest under H; export written, read, equal bit for bit, corruption refused; "
          f"bits/weight hbr3 {b3:.4f}, hbr2 {b2:.4f}, hbr40 {gq.variant_bpw(v40, mats):.4f}")


def cuda_check(devices):
    """The row rounding and the packing on each of the devices against the CPU's (a CUDA op that behaves differently would not raise, it would make the numbers wrong): nearest rounding to the same codes but for a few at a
    boundary, GPTQ's error under its Hessian within 10%, the codes, scales and codebooks dequantising to the weights bit for bit, the packed bytes the same."""
    g = torch.Generator().manual_seed(1)
    W = (torch.randn(2, 192, 64, generator=g) * 0.05).bfloat16().float()
    A = torch.relu(torch.randn(500, 192, generator=g)) ** 2 * (1 + 3.0 * (torch.arange(192) % 7 == 0))
    X = torch.randn(1500, 64, generator=g) * (1 + 4.0 * (torch.arange(64) % 9 == 0))
    H, Hz, s = A.T @ A / 500, X.T @ X / 1500, gq.signs(64, 3, torch.device("cpu"))

    def run(d):
        dev = torch.device(d)
        with gq.on(dev):
            w, sg = W.to(dev), s.to(dev)
            n = rtn_rows(rot(w, sg).reshape(-1, 64), 2)
            gn, ok1 = gptq_neurons(rot(w, sg), [H.to(dev).clone() for _ in range(2)], 2)
            gu, ok2 = gptq_up(w.clone(), [Hz.to(dev).clone() for _ in range(2)], 2, sg)
            assert ok1.all() and ok2.all(), d
            for r in (n, gn, gu):
                assert torch.equal(dequant(r["codes"].reshape(-1, 64), r["scales"].reshape(-1, 2), r["cb"].reshape(-1, 4)), r["q"].reshape(-1, 64)), (d, "codes and scales are not the weights")
                assert torch.equal(unpack_codes(pack_codes(r["codes"], 2), 2), r["codes"])
            return [{k: v.cpu() for k, v in r.items()} for r in (n, gn, gu)], pack_codes(n["codes"], 3).cpu()
    want, wpack = run("cpu")
    herr = lambda q, M, Wr: float((((q - Wr) @ M) * (q - Wr)).sum())
    for d in devices:
        got, gpack = run(d)
        assert float((got[0]["q"] != want[0]["q"]).float().mean()) < 0.02 and torch.equal(gpack, wpack), (d, "nearest rounding")
        wrot = rot(W, s)
        for j in range(2):
            assert abs(herr(got[1]["q"][j].T, H, wrot[j].T) / herr(want[1]["q"][j].T, H, wrot[j].T) - 1) < 0.1, (d, "neuron GPTQ error under H")
        Hr = rot(rot(Hz, s).T, s).T  # (T Hz T^T: the rotated basis's Hessian)
        assert abs(herr(got[2]["q"][0], Hr, wrot[0]) / herr(want[2]["q"][0], Hr, wrot[0]) - 1) < 0.1, (d, "column GPTQ error under H")
    return len(devices)


def cmd_check(a):
    r = check_export(a.dir, layers=None if a.layers is None else {int(x) for x in a.layers.split(",")}, hash_files=a.hash, deep=not a.quick)
    print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=selftest)
    p = sub.add_parser("check")
    p.add_argument("dir")
    p.add_argument("--hash", action="store_true")
    p.add_argument("--quick", action="store_true", help="tensors' names, dtypes and shapes only (the full read unpacks and dequantises every matrix: minutes a layer on a laptop)")
    p.add_argument("--layers")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
