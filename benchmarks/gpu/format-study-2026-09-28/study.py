"""Packed formats for bf16 weights, measured on a model's Linear weights on
the CPU: bits a weight of each candidate and its exceptions (escapes), per
tensor and per model, and how the exceptions fall in the kernels' steps
(64 rows by 16 columns) and stages (4 steps).

    python study.py MODEL_DIR [MODEL_DIR ...] > study.txt

A weight is its sign s, exponent e (8 bits) and mantissa m (7 bits). The
candidates keep 8 bits a weight as they are and code the rest:

  mma12   today's 12-bit layout: the sign-and-mantissa byte, e a 4-bit code
          into the tensor's 15 commonest exponents, the rest in a step's list
          (32-bit entries)
  v12     the same by value: e = base + code over the best 15 contiguous
          exponents (no table), code 15 the list
  fast    the fast format: e a 3-bit code into the 7 commonest, code 7 an
          escape, its exponent byte in order (no position stored), a base a
          1024 weights of a row
  v11     the same by value: the best 7 contiguous exponents
  v11r    v11 with each row's own best 7
  v11L    8 contiguous exponents by value, all 8 codes used, the rest in a
          step's list (32-bit entries)
  sb12    split byte: the bf16's low byte (e's lowest bit and m) as it is;
          its high byte (s and e >> 1) as s and a 3-bit offset into the best
          8 values of e >> 1 (16 exponents), the rest in a step's list
  sb12a   sb12 with the 8 values aligned to 8 (e >> 1 = the base | the offset)
  sb11    split byte, s and a 2-bit offset into the best 4 values of e >> 1
          (8 exponents), the rest in a step's list (24-bit entries: the
          position in the step and the high byte's 7 bits)
  sb11r   sb11 with each row's own best 4
  sb11c   split byte, s and a 2-bit code: 3 values of e >> 1, code 3 an
          escape (its 7 bits in order, as fast's bytes)
  mma     the tiered layout (reference)
  floor   8 + the entropy of the tensor's exponents

The weights are read from safetensors on the CPU; nothing is written."""
import json, math, os, sys, time
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("THREADS", 8)))
CANDS = ("floor", "mma", "mma12", "v12", "fast", "v11", "v11r", "v11L", "sb12", "sb12a", "sb11", "sb11r", "sb11c")


def linears(d):
    """(name, tensor) of the Linear weights sizes.py packs, a 3-D experts' tensor as [E out, in]; lm_head apart."""
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else [f for f in os.listdir(d) if f.endswith(".safetensors")]
    for fn in files:
        with safe_open(os.path.join(d, fn), "pt") as f:
            for k in f.keys():
                parts = k.split(".")
                lin = (parts[-1] == "weight" and len(parts) > 1 and ("proj" in parts[-2] or parts[-2] in ("input_linear", "output_linear"))) or parts[-1].endswith("_proj") or ("experts" in parts[:-1] and not parts[-1].endswith("bias"))
                if not lin and k != "lm_head.weight":
                    continue
                t = f.get_tensor(k)
                if t.dtype != torch.bfloat16 or t.dim() not in (2, 3):
                    continue
                w = t.reshape(-1, t.shape[-1])
                if w.shape[0] % 64 or w.shape[1] % 64:
                    continue
                yield k, w


def best_window(h, width, align=1):
    """The window [b, b + width) of h's bins (the last axis) holding the most, b a multiple of align: (count, b)."""
    c = torch.nn.functional.pad(h.cumsum(-1), (1, 0))
    s = c[..., width:] - c[..., :-width]
    if align > 1:
        s = s[..., ::align]
    v, b = s.max(-1)
    return v, b * align


def per_step(mask, O, K):
    """A mask's count a step (64 rows by 16 columns) and a stage (4 steps along K)."""
    st = mask.view(O // 64, 64, K // 16, 16).sum((1, 3), dtype=torch.int32)
    sg = st.view(O // 64, K // 64, 4).sum(-1)
    return st.flatten(), sg.flatten()


def ent(h):
    p = h[h > 0].double() / h.sum()
    return float(-(p * p.log2()).sum())


def study(name, w):
    O, K = w.shape
    n, steps = O * K, O * K // 1024
    u = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    del u
    h = torch.bincount(e.flatten(), minlength=256)
    rows = torch.arange(O, dtype=torch.int32).unsqueeze(1) * 256
    hr = torch.bincount((e + rows).flatten(), minlength=256 * O).view(O, 256)
    hrow_ent = sum(ent(hr[r]) * int(hr[r].sum()) for r in range(0, O, max(1, O // 256))) / sum(int(hr[r].sum()) for r in range(0, O, max(1, O // 256)))
    h2, hr2 = h.view(128, 2).sum(1), hr.view(O, 128, 2).sum(2)
    r = {"floor": 8 + ent(h)}
    esc, stats = {}, {}

    def lut_mask(lut):  # lut: bool [256], True where e escapes
        return lut[e]

    # tiered: ranks by count; digits in tiers of 3
    order = torch.argsort(h, descending=True, stable=True)
    rank = torch.empty(256, dtype=torch.int64)
    rank[order] = torch.arange(256)
    rk = rank[e]
    t1, t2, t3 = per_step(rk >= 3, O, K)[0], per_step(rk >= 6, O, K)[0], per_step(rk >= 9, O, K)[0]
    del rk
    size = (2 * t2 + 7) // 8 + t3 + 4 * ((t1 + 15) // 16)
    r["mma"] = 8 * (steps * 1280 + 128 + int(size.sum()) + 256 + 4 * (steps + 1) + 12) / n

    def list_bits(width_bits, E, entry, pad4=True):
        pad = 4 - E % 4 if pad4 else 0
        return (width_bits * n / 8 + entry * (E + pad) + 4 * (steps + 1) + 16) * 8 / n

    # mma12: top 15 by count
    top15 = torch.zeros(256, dtype=torch.bool)
    top15[order[:15]] = True
    m = lut_mask(~top15)
    E = int(m.sum())
    esc["mma12"], r["mma12"] = E, list_bits(12, E, 4)
    stats["mma12"] = per_step(m, O, K)
    # v12: best 15 contiguous
    v, b = best_window(h, 15)
    esc["v12"], r["v12"] = n - int(v), list_bits(12, n - int(v), 4)
    # fast: top 7 by count, escape bytes, a base a 1024 of a row
    segs = (K + 1023) // 1024
    E = n - int(h[order[:7]].sum())
    esc["fast"], r["fast"] = E, (n + 3 * n / 8 + E + 4 * O * segs + 8) * 8 / n
    # v11: best 7 contiguous; per row
    v, b7 = best_window(h, 7)
    E = n - int(v)
    esc["v11"], r["v11"] = E, (n + 3 * n / 8 + E + 4 * O * segs + 8) * 8 / n
    lut = torch.ones(256, dtype=torch.bool)
    lut[int(b7): int(b7) + 7] = False
    stats["v11"] = per_step(lut_mask(lut), O, K)
    vr, br = best_window(hr, 7)
    E = n - int(vr.sum())
    esc["v11r"], r["v11r"] = E, (n + 3 * n / 8 + E + 4 * O * segs + O + 8) * 8 / n
    # v11L: best 8 contiguous, a step's list
    v, b8 = best_window(h, 8)
    esc["v11L"], r["v11L"] = n - int(v), list_bits(11, n - int(v), 4, False)
    # sb12: best 8 of e >> 1; aligned
    v, hb12 = best_window(h2, 8)
    E = n - int(v)
    esc["sb12"], r["sb12"] = E, list_bits(12, E, 4)
    lut = torch.ones(256, dtype=torch.bool)
    lut[2 * int(hb12): 2 * int(hb12) + 16] = False
    stats["sb12"] = per_step(lut_mask(lut), O, K)
    v, hb12a = best_window(h2, 8, 8)
    esc["sb12a"], r["sb12a"] = n - int(v), list_bits(12, n - int(v), 4)
    # sb11: best 4 of e >> 1, a step's list of 24-bit entries; per row
    v, hb11 = best_window(h2, 4)
    E = n - int(v)
    esc["sb11"], r["sb11"] = E, list_bits(11, E, 3, False)
    lut = torch.ones(256, dtype=torch.bool)
    lut[2 * int(hb11): 2 * int(hb11) + 8] = False
    stats["sb11"] = per_step(lut_mask(lut), O, K)
    vr, hbr = best_window(hr2, 4)
    E = n - int(vr.sum())
    esc["sb11r"], r["sb11r"] = E, list_bits(11, E, 3, False) + 8 * O / n
    lutr = torch.ones(O, 256, dtype=torch.bool)
    lo = 2 * hbr.unsqueeze(1)
    ar = torch.arange(256).unsqueeze(0)
    lutr &= ~((ar >= lo) & (ar < lo + 8))
    stats["sb11r"] = per_step(lutr.gather(1, e.to(torch.int64)), O, K)
    del lutr
    # sb11c: best 3 of e >> 1 and an escape code; the escapes' 7 bits in order (bytes), a base a 1024 of a row
    v, hb11c = best_window(h2, 3)
    E = n - int(v)
    esc["sb11c"], r["sb11c"] = E, (n + 3 * n / 8 + E + 4 * O * segs + 8) * 8 / n
    zero = int(h[0]) + int(h[255])
    return {"name": name, "O": O, "K": K, "n": n, "H": ent(h), "Hrow": hrow_ent, "bits": r, "esc": esc, "stats": stats,
            "base": {"v11": int(b7), "v11L": int(b8), "sb12": int(hb12), "sb12a": int(hb12a), "sb11": int(hb11), "sb11c": int(hb11c), "mode": int(order[0])}, "zero_or_special": zero}


def fmt_stats(st):
    s, g = st
    return f"{float(s.float().mean()):.2f}/{float((s == 0).float().mean()):.3f}/{int(s.max())}/{int(g.max())}"


def main():
    print("# per tensor: name OxK | H(e) H(e|row) | bits a weight (escapes a million weights) for each candidate | exceptions a step (mean/share of steps with none/most a step/most a stage of 4 steps) for mma12, v11, sb12, sb11, sb11r | window bases")
    for d in sys.argv[1:]:
        model = os.path.basename(os.path.normpath(d))
        if model.startswith("snapshots") or len(model) == 40:
            model = [p for p in d.split("/") if p.startswith("models--")][0][8:].replace("--", "/")
        t0 = time.time()
        tot = {c: 0.0 for c in CANDS}
        te = {c: 0 for c in CANDS}
        N = 0
        per = []
        head = None
        for name, w in linears(d):
            res = study(name, w)
            line = f"{model} {name} {res['O']}x{res['K']} | H {res['H']:.3f} Hrow {res['Hrow']:.3f} | " + " ".join(
                f"{c} {res['bits'][c]:.3f}" + (f" ({1e6 * res['esc'][c] / res['n']:.0f})" if c in res["esc"] else "") for c in CANDS
            ) + " | " + " ".join(f"{k} {fmt_stats(v)}" for k, v in res["stats"].items()) + " | " + " ".join(f"{k} {v}" for k, v in res["base"].items()) + f" special {res['zero_or_special']}"
            print(line, flush=True)
            if name == "lm_head.weight":
                head = res
                continue
            N += res["n"]
            for c in CANDS:
                tot[c] += res["bits"][c] * res["n"]
                te[c] += res["esc"].get(c, 0)
            per.append(res)
        print(f"== {model}: {len(per)} Linear tensors, {N / 1e9:.3f} B weights ({time.time() - t0:.0f} s)")
        for c in CANDS:
            rates = sorted(1e6 * p["esc"][c] / p["n"] for p in per) if c in per[0]["esc"] else []
            dist = f"; escapes {1e6 * te[c] / N:.0f} a million (tensors: min {rates[0]:.0f}, median {rates[len(rates) // 2]:.0f}, max {rates[-1]:.0f})" if rates else ""
            print(f"==   {c:6s} {tot[c] / N:.3f} bits a weight, {100 * (1 - tot[c] / N / 16):.2f}% smaller{dist}")
        if head:
            print(f"==   lm_head {head['O']}x{head['K']}: " + " ".join(f"{c} {head['bits'][c]:.3f}" for c in CANDS))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
