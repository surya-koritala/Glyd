# Held-out check of the calibrated packer: a random 75% of the calibration tokens make the Hessians, the output error is measured on the other 25%
# (routed tokens, routing weight squared); GPTQ vs nearest-entry rounding, per prior weight beta and damping. 16 experts a format.
import sys, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import qwen_pack_calib as C
P = C.P
l = int(sys.argv[1])   # holdout.py LAYER CALIB_DIR
cal = C.calib_of(sys.argv[2], l)
ck = P.Ckpt(); s = P.layer_signs(l)
gu, dn = ck.get(P.prefix(l) + "mlp.experts.gate_up_proj"), ck.get(P.prefix(l) + "mlp.experts.down_proj")
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(cal["x"]), generator=g)
cut = len(perm) * 3 // 4
with P.Lock():
    split = {}
    for name, ix in (("tr", perm[:cut]), ("te", perm[cut:])):
        x = cal["x"][ix].cuda(); xr = P.rotate(x, s)
        split[name] = (x, xr, cal["idx"][ix].cuda().long(), cal["w"][ix].cuda(), xr.T @ xr / len(xr))
    variants = [(0.0, 0.5), (0.25, 0.01), (1.0, 0.01), (4.0, 0.01)]
    for fmt, k in (("cb2", 4), ("cb3", 8)):
        res = {v: [0.0] * 4 for v in variants}
        for e in range(0, 512, 32):
            gue = gu[e].cuda()
            Wu = P.rotate(gue, s)[None]; WdT = P.rotate(dn[e].cuda().T, s)[None].transpose(1, 2).contiguous()
            cbu, cbd = P.fit_codebooks(Wu[0], k)[None], P.fit_codebooks(WdT[0].T.contiguous(), k)[None]
            Hte, Hdte, _ = C.hessians(*split["te"][:4], gue, e, 0.0, split["te"][4])
            ru = P.codes_of(Wu[0], cbu[0]).long()[None]; rd = P.codes_of(WdT[0].T.contiguous(), cbd[0]).long().T[None]
            bu, bd = C.err(Wu, C.deq(ru, cbu, True), Hte[None]), C.err(WdT, C.deq(rd, cbd, False), Hdte[None])
            for beta, damp in variants:
                Htr, Hdtr, _ = C.hessians(*split["tr"][:4], gue, e, beta, split["tr"][4])
                cu, cd = C.gptq(Wu, Htr[None], cbu, True, damp), C.gptq(WdT, Hdtr[None], cbd, False, damp)
                r = res[(beta, damp)]
                r[0] += C.err(Wu, C.deq(cu, cbu, True), Hte[None]); r[1] += bu; r[2] += C.err(WdT, C.deq(cd, cbd, False), Hdte[None]); r[3] += bd
        for (beta, damp), r in res.items():
            print(f"layer {l} {fmt} beta {beta} damp {damp}: held-out output error up {r[0] / r[1]:.3f}x, down {r[2] / r[3]:.3f}x of nearest-entry", flush=True)
