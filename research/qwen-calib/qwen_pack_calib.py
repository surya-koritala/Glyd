#!/usr/bin/env python3
"""The routed experts packed as research/qwen-data/qwen_pack.py packs them (same files, formats and codebooks: the engine reads them as they are),
with each code chosen by GPTQ instead of rounding to the nearest codebook entry: the error of a column is fed to the columns after it, weighted by
the expert's own calibration Hessian (calib_capture.py's routed inputs, each token weighted by its routing weight squared).

- up (gate and up rows, 1280 x 2560, the input side rotated): H = X'^T X' of the rotated inputs routed to the expert; each row keeps its own
  codebook (fitted as the packer fits it), the columns (input features) quantized in order.
- down (down_proj, 2560 outputs x 640 inputs, the outputs rotated: downT's rows are its columns): H = A^T A of the intermediate activations
  silu(gate x) * up x of the bf16 weights; each column (downT row) keeps its own codebook.
An expert with no calibration token is packed as the packer packs it (nearest entry). The MTP layer (48) too: the reference does not run it.

  QWEN_SNAP=.../flash-next python qwen_pack_calib.py --plan plan.json --calib DIR --out PACK [--layers 0:49] [--damp 0.01] [--beta 1.0] [--batch 16]
  QWEN_SNAP=.../flash-next python qwen_pack_calib.py --side-of PACK --only cb4 --fmt cb3 --calib DIR --out PACK/hot    (the GPU copies, calibrated too)
  QWEN_SNAP=.../flash-next python qwen_pack_calib.py --damage DAMAGE.json --calib DIR --out DIR    (no pack: each expert's damage at cb2, cb3, cb4)
  python qwen_pack_calib.py --allocate DAMAGE.json --budget-gb G --out PLAN.json    (the formats that cost the least damage in G GB of experts)
An expert's damage at a format is its GPTQ output error tr(dW H dW^T) (up and down, each H per routed token) times its use (the sum of its
routing weights squared over the calibration tokens, per token): what the format costs the model's output on the calibration text.
Logs per layer the output error tr(dW H dW^T) summed over the experts, nearest-entry vs GPTQ."""
import argparse, json, os, sys, time
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen-data"))
import qwen_pack as P

DEV = P.DEV


def calib_of(d, l):
    """The layer's routed inputs: x (N, 2560) bf16, idx (N, 10), w (N, 10), all batches."""
    parts = [torch.load(f"{d}/{f}", weights_only=True) for f in sorted(os.listdir(d)) if f.startswith(f"L{l:02d}_b")]
    if not parts:
        return None
    return {k: torch.cat([p[k] for p in parts]) for k in ("x", "idx", "w")}


def hessians(x, xr, idx, wt, gue, e, beta, hu_all):
    """Expert e's Hessians (up: rotated inputs, 2560 x 2560; down: intermediate activations, 640 x 640): the routed tokens' (each weighted by
    its routing weight squared, normalized by the weights' sum) plus beta times the prior from every calibration token (up: the layer's inputs,
    hu_all; down: the expert's own activations on every token), each normalized to a mean over tokens. ~800 routed tokens in 2560 dimensions
    overfit alone (held out, GPTQ was worse than nearest-entry rounding); the prior keeps H full rank."""
    hit = idx == e
    rows = hit.any(-1).nonzero()[:, 0]
    Hu = beta * hu_all
    gate, up = (x @ gue.T).float().chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate) * up                                    # (N, 640) on every token
    Hd = beta * (act.T @ act) / len(x)
    if len(rows):
        g = (wt * hit).sum(-1)[rows]
        xs = xr[rows] * g[:, None]
        Hu = Hu + xs.T @ xs / (g * g).sum()
        ar = act[rows] * g[:, None]
        Hd = Hd + ar.T @ ar / (g * g).sum()
    return Hu, Hd, len(rows), (float((g * g).sum()) / len(x) if len(rows) else 0.0)


def gptq(W, H, cb, per_row, damp, step=128):
    """GPTQ over the columns of W (B, R, C) with H (B, C, C): per_row: cb (B, R, k), each row its codebook; else cb (B, C, k), each column its own.
    Returns the codes (B, R, C) int64."""
    Bn, R_, C = W.shape
    W = W.clone()
    H = H.clone()
    md = H.diagonal(dim1=1, dim2=2).mean(-1)
    eye = torch.eye(C, device=W.device)
    H = torch.where((md > 0)[:, None, None], H + (damp * md)[:, None, None] * eye, eye)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros(Bn, R_, C, dtype=torch.long, device=W.device)
    for i1 in range(0, C, step):
        i2 = min(i1 + step, C)
        W1, E1, U1 = W[:, :, i1:i2].clone(), torch.zeros(Bn, R_, i2 - i1, device=W.device), U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, :, i]
            if per_row:
                c = (w[..., None] > (cb[..., 1:] + cb[..., :-1]) * 0.5).sum(-1)
                q = cb.gather(2, c[..., None])[..., 0]
            else:
                ci = cb[:, i1 + i, :]
                c = (w[..., None] > ((ci[:, 1:] + ci[:, :-1]) * 0.5)[:, None, :]).sum(-1)
                q = ci.gather(1, c)
            codes[:, :, i1 + i] = c
            e = (w - q) / U1[:, i, i][:, None]
            W1[:, :, i:] -= e[:, :, None] * U1[:, i, i:][:, None, :]
            E1[:, :, i] = e
        W[:, :, i2:] -= E1 @ U[:, i1:i2, i2:]
    return codes


def deq(codes, cb, per_row):
    return cb.gather(2, codes) if per_row else cb[:, None, :, :].expand(-1, codes.shape[1], -1, -1).gather(3, codes[..., None])[..., 0]


def err(W, Q, H):
    d = W - Q
    return torch.einsum("brc,bcd,brd->", d, H, d).item()


def err_b(W, Q, H):
    """err for each of the batch's experts."""
    d = W - Q
    return torch.einsum("brc,bcd,brd->b", d, H, d)


def expert_bytes(fmt):
    """An expert's bytes at a format: up (1,280 rows) and downT (640) of 2,560 codes, each row its bf16 codebook."""
    return (1280 + 640) * (P.ROW_BYTES[fmt] + 2 * P.K_ENTRIES[fmt])


def damage(a):
    """--damage: every expert's damage at the formats of --fmts (see the docstring), layer by layer into a.damage (resumed where it stopped).
    An existing table gains the formats it lacks, layer by layer; its formats stay ordered by size (allocate's upgrades go up that list)."""
    want = a.fmts.split(",")
    out = json.load(open(a.damage)) if os.path.exists(a.damage) else {"fmts": [], "layers": {}, "calib": os.path.abspath(a.calib)}
    have = out["fmts"]
    fmts = sorted(set(have) | set(want), key=expert_bytes)
    ck = P.Ckpt()
    l0, l1 = (int(x) for x in a.layers.split(":"))
    for l in range(l0, min(l1, P.NL)):
        old = out["layers"].get(str(l))
        if old is not None and old.get("fmts", have) == fmts:
            continue
        todo = [f for f in fmts if old is None or f not in old.get("fmts", have)]
        t0 = time.time()
        gu = ck.get(P.prefix(l) + "mlp.experts.gate_up_proj")
        dn = ck.get(P.prefix(l) + "mlp.experts.down_proj")
        s = P.layer_signs(l)
        cal = calib_of(a.calib, l)
        got = {f: [0.0] * P.NE for f in todo}
        use = [0.0] * P.NE
        with P.Lock():
            gug, dng = gu.to(DEV), dn.to(DEV)
            x = cal["x"].to(DEV)
            xr = P.rotate(x, s)
            idx, wt = cal["idx"].to(DEV).long(), cal["w"].to(DEV)
            hu_all = xr.T @ xr / len(xr)
            for b0 in range(0, P.NE, a.batch):
                eb = list(range(b0, min(b0 + a.batch, P.NE)))
                Wu = torch.stack([P.rotate(gug[e], s) for e in eb])
                Wd = torch.stack([P.rotate(dng[e].T, s) for e in eb])
                Hu = torch.zeros(len(eb), P.HID, P.HID, device=DEV)
                Hd = torch.zeros(len(eb), P.INTER, P.INTER, device=DEV)
                for j, e in enumerate(eb):
                    Hu[j], Hd[j], _, use[e] = hessians(x, xr, idx, wt, gug[e], e, a.beta, hu_all)
                WdT = Wd.transpose(1, 2).contiguous()
                for fmt in todo:
                    k = P.K_ENTRIES[fmt]
                    cbu = torch.stack([P.fit_codebooks(w, k) for w in Wu])
                    cbd = torch.stack([P.fit_codebooks(w, k) for w in Wd])
                    cu, cd = gptq(Wu, Hu, cbu, True, a.damp), gptq(WdT, Hd, cbd, False, a.damp)
                    d = err_b(Wu, deq(cu, cbu, True), Hu) + err_b(WdT, deq(cd, cbd, False), Hd)
                    for j, e in enumerate(eb):
                        got[fmt][e] = float(d[j]) * use[e]
            del gug, dng, x, xr
            torch.cuda.empty_cache()
        prev = {} if old is None else {f: [d[i] for d in old["damage"]] for i, f in enumerate(old.get("fmts", have))}
        prev.update(got)
        dmg = [[prev[f][e] for f in fmts] for e in range(P.NE)]
        out["layers"][str(l)] = {"damage": dmg, "use": use, "fmts": fmts}
        out["fmts"] = fmts
        json.dump(out, open(a.damage + ".part", "w")); os.replace(a.damage + ".part", a.damage)
        tot = ", ".join(f"{f} {sum(d[i] for d in dmg):.3e}" for i, f in enumerate(fmts))
        print(f"layer {l}: {time.time() - t0:.0f} s, damage {tot}, unused {use.count(0.0)}", flush=True)


def allocate(dm, budget, per_layer=False):
    """The formats (per layer, per expert) that cost the least summed damage in `budget` bytes of experts: every expert from cb2, then the
    upgrades that save the most damage per byte (cb2 -> cb3 before cb3 -> cb4 of the same expert), while they fit. Returns (plan, bytes, damage).
    per_layer: each layer gets an equal share of the budget (damage chooses within a layer only; its scale grows with depth, KL's does not)."""
    if per_layer:
        plan, used, dmg = {}, 0, 0.0
        for l, v in dm["layers"].items():
            p, u, d = allocate({"fmts": dm["fmts"], "layers": {l: v}}, budget / len(dm["layers"]))
            plan.update(p); used += u; dmg += d
        return plan, used, dmg
    import heapq
    fmts = dm["fmts"]
    cost = [expert_bytes(f) for f in fmts]
    assert all(v.get("fmts", fmts) == fmts for v in dm["layers"].values()), "layers of the table with other formats: run --damage again"
    lay = {int(l): v["damage"] for l, v in dm["layers"].items()}
    at = {l: [0] * len(d) for l, d in lay.items()}
    used = sum(cost[0] * len(d) for d in lay.values())
    heap = []
    def push(l, e):
        f = at[l][e]
        if f + 1 < len(fmts):
            save = lay[l][e][f] - lay[l][e][f + 1]
            heapq.heappush(heap, (-save / (cost[f + 1] - cost[f]), l, e))
    for l, d in lay.items():
        for e in range(len(d)):
            push(l, e)
    while heap:
        _, l, e = heapq.heappop(heap)
        f = at[l][e]
        step = cost[f + 1] - cost[f]
        if used + step > budget:
            continue
        at[l][e], used = f + 1, used + step
        push(l, e)
    plan = {str(l): [fmts[f] for f in at[l]] for l in sorted(lay)}
    return plan, used, sum(lay[l][e][at[l][e]] for l in lay for e in range(len(lay[l])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan"); ap.add_argument("--calib"); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0:49"); ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0, help="the weight of the all-token prior in each expert's Hessians")
    ap.add_argument("--side-of", default=None, help="a sidecar of this (calibrated) pack: its --only experts again at --fmt (as qwen_pack.py sidecar)")
    ap.add_argument("--only", default="cb4"); ap.add_argument("--fmt", default="cb3")
    ap.add_argument("--damage", help="(no pack) each expert's damage at cb2, cb3, cb4 into this JSON"); ap.add_argument("--allocate", help="a --damage JSON: write the plan for --budget-gb to --out")
    ap.add_argument("--budget-gb", type=float); ap.add_argument("--fmts", default="cb2,cb3,cb4", help="--damage's formats (tern, cb2, cb3, cb4)")
    ap.add_argument("--per-layer", action="store_true", help="--allocate: an equal share of the budget per layer")
    a = ap.parse_args()
    if a.allocate:
        plan, used, dmg = allocate(json.load(open(a.allocate)), a.budget_gb * 1e9, a.per_layer)
        share = {f: sum(v.count(f) for v in plan.values()) for f in P.K_ENTRIES}
        json.dump({"layers": plan, "bytes": used, "damage": dmg, "from": os.path.abspath(a.allocate)}, open(a.out, "w"))
        print(f"plan {a.out}: {used / 1e9:.2f} GB of experts, damage {dmg:.4e}, {share}")
        return
    if a.damage:
        return damage(a)
    os.makedirs(a.out, exist_ok=True)
    mp = f"{a.out}/meta.json"
    if a.side_of:
        main_meta = json.load(open(f"{a.side_of}/meta.json"))
        plan = {L: [a.fmt if P.FMT_IN[f] in a.only.split(",") else None for f in m["f"]] for L, m in main_meta["layers"].items()}
        meta = json.load(open(mp)) if os.path.exists(mp) else {"format": P.FORMAT, "of": os.path.abspath(a.side_of), "fmt": P.FMT_OUT[a.fmt], "layers": {}, "calibrated": vars(a)}
    else:
        plan = json.load(open(a.plan))["layers"]
        meta = json.load(open(mp)) if os.path.exists(mp) else {"format": P.FORMAT, "hidden": P.HID, "intermediate": P.INTER, "layers": {}, "calibrated": vars(a)}
    ck = P.Ckpt()
    l0, l1 = (int(x) for x in a.layers.split(":"))
    for l in range(l0, l1):
        if str(l) in meta["layers"] or str(l) not in plan or not any(plan[str(l)]):
            continue
        t0 = time.time()
        gu = ck.get(P.prefix(l) + "mlp.experts.gate_up_proj")   # [512, 1280, 2560] bf16
        dn = ck.get(P.prefix(l) + "mlp.experts.down_proj")                  # [512, 2560, 640]
        s = P.layer_signs(l)
        cal = calib_of(a.calib, l) if a.calib and l < P.NL else None
        out, e_rtn, e_gptq, tokens = {}, [0.0, 0.0], [0.0, 0.0], []
        tm = {"wait": 0.0, "load": 0.0, "hess": 0.0, "gptq": 0.0, "check": 0.0}
        tick = lambda k, t: tm.__setitem__(k, tm[k] + time.time() - t) or time.time()
        t = time.time()
        with P.Lock():
            t = tick("wait", t)
            gug, dng = gu.to(DEV), dn.to(DEV)
            if cal is not None:
                x = cal["x"].to(DEV)
                xr = P.rotate(x, s)
                idx, wt = cal["idx"].to(DEV).long(), cal["w"].to(DEV)
                hu_all = xr.T @ xr / len(xr)
            t = tick("load", t)
            fmts = plan[str(l)]
            for fmt in sorted(set(f for f in fmts if f)):
                k = P.K_ENTRIES[fmt]
                es = [e for e in range(P.NE) if fmts[e] == fmt]
                for b0 in range(0, len(es), a.batch):
                    eb = es[b0:b0 + a.batch]
                    Wu = torch.stack([P.rotate(gug[e], s) for e in eb])                 # (B, 1280, 2560)
                    Wd = torch.stack([P.rotate(dng[e].T, s) for e in eb])               # (B, 640, 2560): downT
                    cbu = torch.stack([P.fit_codebooks(w, k) for w in Wu])              # (B, 1280, k)
                    cbd = torch.stack([P.fit_codebooks(w, k) for w in Wd])              # (B, 640, k)
                    Hu = torch.zeros(len(eb), P.HID, P.HID, device=DEV)
                    Hd = torch.zeros(len(eb), P.INTER, P.INTER, device=DEV)
                    if cal is not None:
                        for j, e in enumerate(eb):
                            Hu[j], Hd[j], n, _ = hessians(x, xr, idx, wt, gug[e], e, a.beta, hu_all)
                            tokens.append(n)
                    torch.cuda.synchronize(); t = tick("hess", t)
                    cu = gptq(Wu, Hu, cbu, True, a.damp)
                    cd = gptq(Wd.transpose(1, 2).contiguous(), Hd, cbd, False, a.damp)   # down as (2560 rotated outputs, 640 inputs)
                    torch.cuda.synchronize(); t = tick("gptq", t)
                    if cal is not None:
                        ru = torch.stack([P.codes_of(Wu[j], cbu[j]).long() for j in range(len(eb))])   # (one expert at a time: the comparison is (rows, cols, k) wide)
                        rd = torch.stack([P.codes_of(Wd[j], cbd[j]).long() for j in range(len(eb))])
                        e_rtn[0] += err(Wu, deq(ru, cbu, True), Hu); e_gptq[0] += err(Wu, deq(cu, cbu, True), Hu)
                        WdT = Wd.transpose(1, 2)
                        e_rtn[1] += err(WdT, deq(rd.transpose(1, 2), cbd, False), Hd); e_gptq[1] += err(WdT, deq(cd, cbd, False), Hd)
                    torch.cuda.synchronize(); t = tick("check", t)
                    for j, e in enumerate(eb):
                        out[P.tname(e, "up", "codes")] = P.pack_bits(cu[j].to(torch.uint8), fmt).cpu()
                        out[P.tname(e, "up", "codebook")] = cbu[j].bfloat16().cpu()
                        out[P.tname(e, "downT", "codes")] = P.pack_bits(cd[j].T.contiguous().to(torch.uint8), fmt).cpu()
                        out[P.tname(e, "downT", "codebook")] = cbd[j].bfloat16().cpu()
            del gug, dng
            torch.cuda.empty_cache()
        fn = f"layer-{l:03d}.safetensors"
        save_file(out, f"{a.out}/{fn}.part")
        os.replace(f"{a.out}/{fn}.part", f"{a.out}/{fn}")
        meta["layers"][str(l)] = {"file": fn, "bytes": os.path.getsize(f"{a.out}/{fn}"), "sha256": P.sha256(f"{a.out}/{fn}")}
        meta["layers"][str(l)].update({"experts": [e for e, f in enumerate(fmts) if f]} if a.side_of else {"s": s.tolist(), "f": [P.FMT_OUT[x] for x in fmts]})
        json.dump(meta, open(mp + ".part", "w")); os.replace(mp + ".part", mp)
        gain = "" if cal is None else (f", output error up {e_gptq[0] / max(e_rtn[0], 1e-30):.3f}x / down {e_gptq[1] / max(e_rtn[1], 1e-30):.3f}x of nearest-entry"
                                         f", tokens per expert median {sorted(tokens)[len(tokens) // 2]}, none {tokens.count(0)}")
        print(f"layer {l}: {time.time() - t0:.0f} s ({', '.join(f'{k} {v:.0f}' for k, v in tm.items())}){gain}", flush=True)


def selftest():
    """CPU: on correlated inputs GPTQ's output error is below nearest-entry rounding's, per-row and per-column codebooks, and the codes stay in range."""
    torch.manual_seed(0)
    B, R_, C, k = 2, 64, 128, 4
    X = torch.randn(B, 512, C) @ torch.randn(C, C) * 0.1 + torch.randn(B, 512, C)
    H = X.transpose(1, 2) @ X
    W = torch.randn(B, R_, C)
    for per_row in (True, False):
        src = W if per_row else W.transpose(1, 2)
        cb = torch.stack([torch.quantile(w, torch.linspace(0.1, 0.9, k), dim=-1).T.contiguous() for w in src])
        c = gptq(W, H, cb, per_row, 0.01)
        mids = (cb[..., 1:] + cb[..., :-1]) * 0.5
        r = (W[..., None] > mids[:, :, None, :]).sum(-1) if per_row else (W[..., None] > mids[:, None, :, :]).sum(-1)
        eg, er = err(W, deq(c, cb, per_row), H), err(W, deq(r, cb, per_row), H)
        assert c.min() >= 0 and c.max() < k and eg < 0.9 * er, (per_row, eg, er)
        print(f"selftest per_row={per_row}: GPTQ {eg / er:.3f}x of nearest-entry")
    # allocate: within the budget, the upgrade that saves the most damage per byte first, cb3 before cb4 of an expert
    c2, c3, c4 = (expert_bytes(f) for f in ("cb2", "cb3", "cb4"))
    dm = {"fmts": ["cb2", "cb3", "cb4"], "layers": {"0": {"damage": [[10.0, 2.0, 1.0], [1.0, 0.9, 0.85], [8.0, 7.9, 0.1]]}}}
    plan, used, dmg = allocate(dm, 3 * c2 + (c3 - c2))
    assert plan["0"] == ["cb3", "cb2", "cb2"] and used == 3 * c2 + (c3 - c2), (plan, used)
    plan, used, dmg = allocate(dm, 3 * c4)
    assert plan["0"] == ["cb4", "cb4", "cb4"] and abs(dmg - 1.95) < 1e-9, (plan, dmg)
    plan, _, _ = allocate(dm, 3 * c2 + (c3 - c2) + (c4 - c3) + 1)
    assert plan["0"] == ["cb4", "cb2", "cb2"], plan
    two = {"fmts": dm["fmts"], "layers": {"0": dm["layers"]["0"], "1": {"damage": [[100.0, 50.0, 0.0]] * 3}}}
    plan, used, _ = allocate(two, 2 * (3 * c2 + (c3 - c2)), per_layer=True)
    assert plan["0"] == ["cb3", "cb2", "cb2"] and plan["1"].count("cb3") == 1 and used == 2 * (3 * c2 + (c3 - c2)), (plan, used)
    print("selftest allocate: budget kept, most damage saved per byte first, per layer")
    # tern: five base-3 digits a byte, code 5 b + d at digit d of byte b (the engine's layout), round trip; its size; allocation from it
    codes = torch.randint(0, 3, (7, 2560), dtype=torch.uint8)
    b = P.pack_bits(codes, "tern")
    assert b.shape == (7, P.ROW_BYTES["tern"]) and int(b[0, 0]) == sum(int(codes[0, d]) * 3 ** d for d in range(5))
    assert torch.equal(P.unpack_bits(b, "tern").to(torch.uint8), codes)
    assert expert_bytes("tern") == (1280 + 640) * (512 + 6) and expert_bytes("cb2") == 1244160
    dm = {"fmts": ["tern", "cb2"], "layers": {"0": {"damage": [[5.0, 1.0], [0.2, 0.1]]}}}
    plan, used, _ = allocate(dm, 2 * expert_bytes("tern") + (expert_bytes("cb2") - expert_bytes("tern")))
    assert plan["0"] == ["cb2", "tern"], plan
    print("selftest tern: packing round trip, size, allocation from tern")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
