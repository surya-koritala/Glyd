#!/usr/bin/env python3
"""Qwen3.8-Flash-Next's routed experts as codebook rows (glyd-hbr-codes/2 at 2560 x 640) from the BF16 checkpoint, each expert's format by
how much the router uses it.

  python qwen_pack.py plan   --profile P.npz --hot 0.5 [--hot4 0.0] --out plan.json   per layer: the `hot` share of experts by routed weight mass
                                                                                      in cb3 (the top `hot4` share in cb4), the rest cb2; MTP cb3
                             [--layer-weights W,...]                                  the cb4 share over all layers by mass times the layer's weight
  (the trellis encoder, glyd-engine qxtq encode --plan, reads cb2, cb3, cb4 as format q at 2, 3, 4 bits)
  python qwen_pack.py pack   --plan plan.json --out DIR [--layers 0:49]               layer-LLL.safetensors + meta.json (GPU; the lock per layer)
  python qwen_pack.py verify --dir DIR [--layers 0,24,47]                             dequantised rows against the BF16 weights, per format

Layout (agreed with the engine): for expert e, `eEEE.up.codes [1280, rowbytes] U8` and `eEEE.up.codebook [1280, K] BF16` (gate_up_proj[e]:
640 gate rows, then 640 up rows), and `eEEE.downT.*` [640, ...] (down_proj[e] transposed: one row per intermediate neuron). Every row is
in the layer's rotated hidden basis, R x = 512^-0.5 (I_5 (x) H_512)(s * x), and has its own codebook (sorted, absolute units). Weight of row n,
column j = codebook[n][code[n][j]]. Code bitstreams: cb2 4 codes a byte (code 4b+m at bits 2m), cb3 8 codes in 3 bytes little-endian (code m
at bits 3m), cb4 2 a byte (low nibble first). meta.json: {"format", "layers": {"L": {"file", "bytes", "sha256", "signs_u8", "expert_fmt"}}};
L = 48 is the MTP layer.
"""
import argparse, fcntl, functools, hashlib, json, os, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = os.environ.get("QWEN_SNAP", "/home/surya-koritala/glyd-scratch/qwen/flash-next")
LOCK = os.environ.get("QWEN_GPU_LOCK", "/home/surya-koritala/glyd-scratch/gpu.lock")
HID, INTER, NE, NL = 2560, 640, 512, 48
K_ENTRIES = {"tern": 3, "cb2": 4, "cb3": 8, "cb4": 16}
# (a row's code bytes at each format, 2,560 codes: tern five base-3 digits a byte, code 5 b + d at digit d of byte b, as the engine's
# kernels and its host path read it; then the row's codebook, K_ENTRIES bf16 values)
ROW_BYTES = {"tern": 512, "cb2": 640, "cb3": 960, "cb4": 1280}
# The names the pack is written with (format glyd-x/1, the engine's): neutral, so files and messages say what, not how. The plan and this
# script keep their own names (cb2, cb3, cb4; downT, up; codes, codebook) and map them here.
FORMAT = "glyd-x/1"
FMT_OUT = {"cb2": "a", "cb3": "b", "cb4": "c", "tern": "t"}
FMT_IN = {v: k for k, v in FMT_OUT.items()}
PART = {"downT": "d", "up": "u"}


def tname(e, name, what):   # what: "codes" or "codebook"
    return f"e{e:04d}.{PART[name]}.{0 if what == 'codes' else 1}"
DEV = "cuda"


# ---------------------------------------------------------------------------------------------------------------------------- the rotation
@functools.lru_cache(maxsize=None)
def hadamard(n=512):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H.to(DEV)


def layer_signs(l):
    g = torch.Generator().manual_seed(7919 * (l + 1))
    return torch.randint(0, 2, (HID,), generator=g).to(torch.uint8)   # 1 = +1, 0 = -1


def rotate(x, s01):
    """Rows of x (..., 2560) to the rotated basis: five 512-point natural-order Walsh-Hadamard transforms of (s * x), times 512^-0.5."""
    s = s01.to(DEV).float() * 2 - 1
    y = (x.float() * s).reshape(*x.shape[:-1], 5, 512) @ hadamard()
    return (y * 512 ** -0.5).reshape(x.shape)


# --------------------------------------------------------------------------------------------------------------------------- the codebooks
def _gauss_levels(k, iters=400):
    x = torch.linspace(-8, 8, 40001, dtype=torch.float64)
    w = torch.exp(-x * x / 2)
    c = torch.linspace(-2, 2, k, dtype=torch.float64)
    for _ in range(iters):
        cell = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        c = torch.zeros(k, dtype=torch.float64).scatter_add_(0, cell, x * w) / torch.zeros(k, dtype=torch.float64).scatter_add_(0, cell, w)
    return c.float()


GAUSS = {k: _gauss_levels(k) for k in (3, 4, 8, 16)}


def fit_codebooks(x, k, iters=3):
    """Per-row Lloyd-Max codebooks (rows, k) of x (rows, 2560) in the rotated basis, started at the unit-Gaussian levels times the row's RMS,
    fitted on a strided subsample of the row; sorted bf16 values held in float32."""
    sub = x[:, :: max(1, HID // (2048 if k >= 8 else 1024))]
    sig = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
    xs = sub / sig
    c = GAUSS[k].to(x.device).expand(x.shape[0], k).clone()
    ones = torch.ones_like(xs)
    for _ in range(iters):
        mids = (c[:, 1:] + c[:, :-1]) * 0.5
        idx = (xs[:, :, None] > mids[:, None, :]).sum(-1)
        num = torch.zeros_like(c).scatter_add_(1, idx, xs)
        den = torch.zeros_like(c).scatter_add_(1, idx, ones)
        c = torch.where(den > 0, num / den.clamp_min(1e-12), c).sort(-1).values
    return (c * sig).bfloat16().float().contiguous()


def codes_of(x, cb):
    """The code of every value: how many of the row's codebook midpoints lie below it (ties go down)."""
    mids = (cb[:, 1:] + cb[:, :-1]) * 0.5
    return (x[:, :, None] > mids[:, None, :]).sum(-1).to(torch.uint8)


def pack_bits(codes, fmt):
    rows, K = codes.shape
    c = codes.to(torch.int32)
    if fmt == "cb2":
        c = c.view(rows, K // 4, 4)
        return (c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6).to(torch.uint8)
    if fmt == "cb4":
        c = c.view(rows, K // 2, 2)
        return (c[..., 0] | c[..., 1] << 4).to(torch.uint8)
    if fmt == "cb3":
        c = c.view(rows, K // 8, 8)
        w = sum(c[..., m] << (3 * m) for m in range(8))
        return torch.stack([w & 255, (w >> 8) & 255, (w >> 16) & 255], -1).reshape(rows, K * 3 // 8).to(torch.uint8)
    if fmt == "tern":
        c = c.view(rows, K // 5, 5)
        return sum(c[..., d] * 3 ** d for d in range(5)).to(torch.uint8)
    raise ValueError(fmt)


def unpack_bits(b, fmt, K=HID):
    rows = b.shape[0]
    x = b.to(torch.int32)
    if fmt == "cb2":
        return torch.stack([x & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3], -1).reshape(rows, K)
    if fmt == "cb4":
        return torch.stack([x & 15, (x >> 4) & 15], -1).reshape(rows, K)
    if fmt == "cb3":
        x = x.view(rows, K // 8, 3)
        w = x[..., 0] | x[..., 1] << 8 | x[..., 2] << 16
        return torch.stack([(w >> (3 * m)) & 7 for m in range(8)], -1).reshape(rows, K)
    if fmt == "tern":
        return torch.stack([(x // 3 ** d) % 3 for d in range(5)], -1).reshape(rows, K)
    raise ValueError(fmt)


def quantize(W, fmt):
    cb = fit_codebooks(W, K_ENTRIES[fmt])
    return pack_bits(codes_of(W, cb), fmt), cb.to(torch.bfloat16)


# -------------------------------------------------------------------------------------------------------------------------- checkpoint
class Ckpt:
    def __init__(self):
        self.map = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
        self.files = {}

    def get(self, name):
        fn = self.map[name]
        if fn not in self.files:
            self.files[fn] = safe_open(f"{SNAP}/{fn}", "pt", device="cpu")
        return self.files[fn].get_tensor(name)


def prefix(l):
    return "mtp.layers.0." if l == NL else f"model.language_model.layers.{l}."


class Lock:
    def __enter__(self):
        self.fh = open(LOCK, "a")
        fcntl.flock(self.fh, fcntl.LOCK_EX)

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ commands
def cmd_plan(a):
    pr = np.load(a.profile)
    w = pr["weights"]                       # (48, 512) routed weight mass
    plan = {"hot": a.hot, "hot4": a.hot4, "profile_tokens": int(pr["tokens"]), "layers": {}}
    for l in range(NL):
        order = np.argsort(-w[l])
        fmt = ["cb2"] * NE
        for rank, e in enumerate(order):
            if rank < int(round(a.hot4 * NE)):
                fmt[e] = "cb4"
            elif rank < int(round(a.hot * NE)):
                fmt[e] = "cb3"
        plan["layers"][str(l)] = fmt
    if a.layer_weights:
        # (cb4 by mass times the layer's weight over all the layers' cb3 and cb4 experts, the same count: more where the layers are more sensitive)
        lw = [float(x) for x in a.layer_weights.split(",")]
        assert NL % len(lw) == 0, f"{len(lw)} weights do not cut {NL} layers evenly"
        score = w * np.repeat(lw, NL // len(lw))[:, None]
        pool = [(score[l, e], l, e) for l in range(NL) for e in range(NE) if plan["layers"][str(l)][e] != "cb2"]
        n4 = sum(f == "cb4" for l in range(NL) for f in plan["layers"][str(l)])
        for rank, (_, l, e) in enumerate(sorted(pool, reverse=True)):
            plan["layers"][str(l)][e] = "cb4" if rank < n4 else "cb3"
        plan["layer_weights"] = lw
    plan["layers"][str(NL)] = ["cb3"] * NE   # (the MTP layer: transformers does not run it, so it has no profile)
    total = sum((1280 + 640) * (ROW_BYTES[f] + 2 * K_ENTRIES[f]) for fm in plan["layers"].values() for f in fm)
    plan["bytes"] = total
    json.dump(plan, open(a.out, "w"))
    share = {f: sum(fm.count(f) for fm in plan["layers"].values()) for f in K_ENTRIES}
    print(f"plan: {share}, {total / 2**30:.1f} GiB of experts (MTP layer included)")


def cmd_pack(a):
    plan = json.load(open(a.plan))["layers"]
    os.makedirs(a.out, exist_ok=True)
    mp = f"{a.out}/meta.json"
    meta = json.load(open(mp)) if os.path.exists(mp) else {"format": FORMAT, "hidden": HID, "intermediate": INTER, "layers": {}}
    ck = Ckpt()
    l0, l1 = (int(x) for x in a.layers.split(":"))
    for l in range(l0, l1):
        if str(l) in meta["layers"]:
            print(f"layer {l}: done before"); continue
        t0 = time.time()
        gu = ck.get(prefix(l) + "mlp.experts.gate_up_proj")   # [512, 1280, 2560] bf16
        dn = ck.get(prefix(l) + "mlp.experts.down_proj")      # [512, 2560, 640]
        s = layer_signs(l)
        out = {}
        with Lock():
            gug, dng = gu.to(DEV), dn.to(DEV)
            for e, fmt in enumerate(plan[str(l)]):
                for name, W in (("downT", dng[e].T), ("up", gug[e])):
                    codes, cb = quantize(rotate(W, s), fmt)
                    out[tname(e, name, "codes")] = codes.cpu()
                    out[tname(e, name, "codebook")] = cb.cpu()
            del gug, dng
            torch.cuda.empty_cache()
        fn = f"layer-{l:03d}.safetensors"
        save_file(out, f"{a.out}/{fn}.part")
        os.replace(f"{a.out}/{fn}.part", f"{a.out}/{fn}")
        meta["layers"][str(l)] = {"file": fn, "bytes": os.path.getsize(f"{a.out}/{fn}"), "sha256": sha256(f"{a.out}/{fn}"),
                                  "s": s.tolist(), "f": [FMT_OUT[x] for x in plan[str(l)]]}
        json.dump(meta, open(mp + ".part", "w")); os.replace(mp + ".part", mp)
        print(f"layer {l}: {time.time() - t0:.0f} s, {meta['layers'][str(l)]['bytes'] / 2**20:.0f} MiB", flush=True)


def cmd_sidecar(a):
    """Lower-bit copies of the pack's experts of the --only formats (comma-separated; e.g. the cb4 hot set as cb3, so that more of them fit in the GPU's places),
    in the same names and the same rotation (the main pack's signs), one file a layer: DIR/layer-LLL.safetensors + DIR/meta.json."""
    main = json.load(open(f"{a.pack}/meta.json"))
    os.makedirs(a.out, exist_ok=True)
    mp = f"{a.out}/meta.json"
    meta = json.load(open(mp)) if os.path.exists(mp) else {"format": FORMAT, "of": os.path.abspath(a.pack), "fmt": FMT_OUT[a.fmt], "layers": {}}
    ck = Ckpt()
    for L, m in sorted(main["layers"].items(), key=lambda kv: int(kv[0])):
        l = int(L)
        experts = [e for e, f in enumerate(m["f"]) if FMT_IN[f] in a.only.split(",")]
        if not experts or L in meta["layers"]:
            continue
        s = torch.tensor(m["s"], dtype=torch.uint8)
        assert torch.equal(s, layer_signs(l)), f"layer {l}: signs differ from the main pack's"
        gu, dn = ck.get(prefix(l) + "mlp.experts.gate_up_proj"), ck.get(prefix(l) + "mlp.experts.down_proj")
        out = {}
        with Lock():
            for e in experts:
                for name, W in (("downT", dn[e].T), ("up", gu[e])):
                    codes, cb = quantize(rotate(W.to(DEV), s), a.fmt)
                    out[tname(e, name, "codes")] = codes.cpu()
                    out[tname(e, name, "codebook")] = cb.cpu()
        fn = f"layer-{l:03d}.safetensors"
        save_file(out, f"{a.out}/{fn}.part"); os.replace(f"{a.out}/{fn}.part", f"{a.out}/{fn}")
        meta["layers"][L] = {"file": fn, "bytes": os.path.getsize(f"{a.out}/{fn}"), "sha256": sha256(f"{a.out}/{fn}"), "experts": experts}
        json.dump(meta, open(mp + ".part", "w")); os.replace(mp + ".part", mp)
        print(f"sidecar layer {l}: {len(experts)} experts in {a.fmt}", flush=True)


def cmd_verify(a):
    meta = json.load(open(f"{a.dir}/meta.json"))
    ck = Ckpt()
    for l in (int(x) for x in a.layers.split(",")):
        m = meta["layers"][str(l)]
        s = torch.tensor(m["s"], dtype=torch.uint8)
        f = safe_open(f"{a.dir}/{m['file']}", "pt", device="cpu")
        gu, dn = ck.get(prefix(l) + "mlp.experts.gate_up_proj"), ck.get(prefix(l) + "mlp.experts.down_proj")
        err = {}
        for e in range(0, NE, 64):
            fmt = FMT_IN[m["f"][e]]
            for name, W in (("downT", dn[e].T), ("up", gu[e])):
                Wr = rotate(W.to(DEV), s)
                q = torch.gather(f.get_tensor(tname(e, name, "codebook")).to(DEV).float(), 1,
                                 unpack_bits(f.get_tensor(tname(e, name, "codes")).to(DEV), fmt).long())
                rel = ((Wr - q).pow(2).sum() / Wr.pow(2).sum()).sqrt().item()
                err.setdefault(fmt, []).append(rel)
                # the rotation is orthonormal: the rotated rows have the BF16 rows' norms
                assert abs(Wr.pow(2).sum().sqrt().item() / W.float().pow(2).sum().sqrt().item() - 1) < 1e-3
        print(f"layer {l}: relative weight error by format", {k: round(float(np.mean(v)), 4) for k, v in err.items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--profile", required=True); p.add_argument("--hot", type=float, default=0.5)
    p.add_argument("--hot4", type=float, default=0.0); p.add_argument("--out", required=True)
    p.add_argument("--layer-weights", help="comma-separated weights of the layers (or of equal runs of them): cb4 by mass times weight over all the layers")
    p = sub.add_parser("pack"); p.add_argument("--plan", required=True); p.add_argument("--out", required=True); p.add_argument("--layers", default="0:49")
    p = sub.add_parser("verify"); p.add_argument("--dir", required=True); p.add_argument("--layers", default="0,24,47")
    p = sub.add_parser("sidecar"); p.add_argument("--pack", required=True); p.add_argument("--only", default="cb4"); p.add_argument("--fmt", default="cb3")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "pack": cmd_pack, "verify": cmd_verify, "sidecar": cmd_sidecar}[a.cmd](a)


if __name__ == "__main__":
    main()
