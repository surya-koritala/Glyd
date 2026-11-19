"""How far below its stored size each weight format can go, losslessly: estimates and achieved sizes on real tensors read by HTTP
range (a pod's CPU; nothing here needs a GPU).

    cc -O2 -shared -fPIC -o libcm.so cm.c -lm
    python measure.py --selftest
    python measure.py PLAN_DIR OUT_DIR          (plan.py's samples; xz, zstd and zpaq on PATH)

Phase 1, a sample at a time: the sample's stored bytes (codes, scales and zero points as stored; GGUF blocks as stored), unpacked
by research/quant-2026's quant.load (gguf.unpack for GGUF), and the codes of its matching tensor (plan.py). For every part of the
sample (the weights or codes; each side stream of scales, zero points, block minimums), in bits a value of that part:
  H0          order-0 entropy, a table a tensor (plug-in with the Miller-Madow correction)
  fields      floats: sign, exponent and mantissa each on its own, and the exponent with the rest given it
  ctx         conditional entropies H(key | context) + H(value | key), the key being the value's exponent for floats, its magnitude
              for FP4, its bit length for INT8, the whole value otherwise; contexts: row, column, 64 x 128 tile, the block's scale,
              the group's zero point, the left and upper neighbours (each and both), the matching tensor's value at the same place
              (a context counts only with 10 values a nonzero cell of its histogram: quant.MIN_PER_CELL)
  quant       research/quant-2026's numbers for the same values: its contexts, and its fixed-rate layout ("fixed": for FP8 the format
              as built, fp8ref, packed and unpacked bit for bit; for bf16 the tiered code over the exponent)
Arrays for phase 2 go to $ARRAYS (default OUT_DIR/arrays/).
Phase 2: the context-mixing code length of each part (cm.c: every context model above but the zero point, learned as it goes,
mixed; up to CAP_BITS bits a part, whole rows from the top: cm_values in the record says how many; the CM is a code length a coder
with that model would spend, not a lower bound), and for the first COMP_SAMPLES samples of a checkpoint also xz -9e, zstd --ultra -22
--long=27 and zpaq -m5 on the sample's stored bytes, and the CM ladder (order 0; + position and scale; + neighbours; + the
matching tensor) and a second mixer rate. Records: OUT_DIR/phase1/<tag>.json, OUT_DIR/phase2.jsonl (a line a task, as it ends).
"""
import concurrent.futures as cf, ctypes, json, math, os, subprocess, sys, tempfile, time, traceback

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "..", "quant-2026"))
import quant  # research/quant-2026
from ent import H
import gguf

CAP_BITS = 48 << 20
COMP_SAMPLES = 3
LN2 = math.log(2)
O0, LEFT, ABOVE, LEFT2, LA, NBHD, COL, ROW, TILE, AUX1, AUX2 = (1 << i for i in range(11))
LADDER = {"order 0": O0, "+ position and scale": O0 | COL | ROW | TILE | AUX1}
LADDER["+ neighbours"] = LADDER["+ position and scale"] | LEFT | ABOVE | LEFT2 | LA | NBHD
LADDER["+ matching tensor"] = LADDER["+ neighbours"] | AUX2
SIDE = O0 | LEFT | ABOVE | LA | NBHD | COL | ROW
LR = (0.002, 0.02)

# a part's encoding: bits a value, the order its bits are coded in (most informative first), its float fields (sign bit, exponent
# bits, mantissa bits) if a float
ENC = {"bf16": (16, list(range(14, 6, -1)) + [15] + list(range(6, -1, -1)), (15, (14, 7), (6, 0))),
       "f16": (16, list(range(14, 9, -1)) + [15] + list(range(9, -1, -1)), (15, (14, 10), (9, 0))),
       "e4m3": (8, [6, 5, 4, 3, 7, 2, 1, 0], (7, (6, 3), (2, 0))),
       "e2m1": (4, [2, 1, 0, 3], (3, (2, 1), (0, 0))),
       "i8": (8, list(range(7, -1, -1)), None)}
for _k in (4, 5, 6, 8):
    ENC[f"u{_k}"] = (_k, list(range(_k - 1, -1, -1)), None)


def key_of(enc, x):
    """(key, alphabet): what a neighbour or context is compared on."""
    x = x.astype(np.int32)
    if enc == "bf16":
        return (x >> 7) & 255, 256
    if enc == "f16":
        return (x >> 10) & 31, 32
    if enc == "e4m3":
        return (x >> 3) & 15, 16
    if enc == "e2m1":
        return x & 7, 8
    if enc == "i8":  # zigzag value: its bit length
        return sum(((x >> b) > 0).astype(np.int32) for b in range(8)), 9
    return x, 1 << ENC[enc][0]


def zigzag(q):
    v = q.view(np.int8).astype(np.int16)
    return (((v << 1) ^ (v >> 7)) & 255).astype(np.uint16)


def Hmm(counts):
    c = np.asarray(counts).ravel()
    n = c.sum()
    return H(c) + ((c > 0).sum() - 1) / (2 * n * LN2) if n else 0.0


def hc(ctx, nctx, key, A):
    j = np.bincount((ctx.astype(np.int64) * A + key).ravel(), minlength=nctx * A).reshape(nctx, A)  # (int64: no overflow)
    return quant.Hc_dense(j)


def static(x, enc, sctx=None, zctx=None, prev=None):
    """A part's order-0, field and context estimates, bits a value."""
    nb = ENC[enc][0]
    R, C = x.shape
    o = {"enc": enc, "n": int(x.size), "width": nb}
    h0 = Hmm(np.bincount(x.ravel().astype(np.int64), minlength=1 << nb))
    k, A = key_of(enc, x)
    kh = np.bincount(k.ravel(), minlength=A)
    rest = h0 - Hmm(kh)
    o["H0"], o["H_value_given_key"], o["key_hist"] = h0, rest, kh.tolist()  # (key_hist: tables shared by kind or layer, in report.py)
    f = ENC[enc][2]
    if f:
        xi = x.astype(np.int64)
        s, (eh, el), (mh, ml) = f
        e = (xi >> el) & ((1 << (eh - el + 1)) - 1)
        m = (xi >> ml) & ((1 << (mh - ml + 1)) - 1) if mh >= ml else np.zeros_like(xi)
        sg = (xi >> s) & 1
        hs, he, hm = (Hmm(np.bincount(a.ravel())) for a in (sg, e, m))
        o["fields"] = {"sign": hs, "exponent": he, "mantissa": hm, "fields apart": hs + he + hm, "exponent, then the rest given it": h0}
    ctx = {}

    def add(name, c, nc):
        v = hc(c, nc, k, A)
        if v is not None:
            ctx[name] = v + rest

    rr = np.broadcast_to(np.arange(R)[:, None], (R, C))
    cc = np.broadcast_to(np.arange(C)[None, :], (R, C))
    add("row", rr, R)
    add("column", cc, C)
    nt = -(-C // 128)
    add("tile 64x128", (rr // 64) * nt + cc // 128, (-(-R // 64)) * nt)
    L = np.full((R, C), A, np.int32)
    L[:, 1:] = k[:, :-1]
    U = np.full((R, C), A, np.int32)
    U[1:] = k[:-1]
    add("left", L, A + 1)
    add("above", U, A + 1)
    if (A + 1) ** 2 * A <= 1 << 25:
        add("left and above", L * (A + 1) + U, (A + 1) ** 2)
    if sctx is not None:
        add("block scale", sctx, int(sctx.max()) + 1)
    if zctx is not None:
        add("zero point", zctx, int(zctx.max()) + 1)
    if prev is not None:
        add("matching tensor", key_of(enc, prev)[0], A)
    o["ctx"] = ctx
    best = min([("order 0", h0)] + list(ctx.items()), key=lambda t: t[1])
    o["best_static"], o["best_static_ctx"] = best[1], best[0]
    return o


# ---------------- phase 1 ----------------

CAP = []


def capture(url, rng, tries=8):
    b = GET(url, rng, tries)
    CAP.append(b)
    return b


GET = quant.get
quant.get = capture


def cm_rows(R, C, nb):
    return max(1, min(R, CAP_BITS // (nb * C)))


def stream(name, a, enc, width):
    """A side stream: values as uint16 [rows, cols], its encoding and stored bits a value."""
    return {"name": name, "x": np.ascontiguousarray(a.astype(np.uint16)), "enc": enc, "width": width}


def load_st(p, s):
    hdr = p["hdr"]
    ts = quant.tensors(hdr)
    cls = quant.classify(ts, p["qm"])
    quant.REPO[0] = p["repo"]
    t = (s["fmt"], s["cat"], s["name"], s["r0"], s["r1"], s["e"])
    CAP.clear()
    d = quant.load(ts, cls, p["qm"], t)
    raw = b"".join(CAP)
    prev = None
    if s["prev"]:
        dp = quant.load(ts, cls, p["qm"], (s["fmt"], s["cat"], s["prev"][0], s["r0"], s["r1"], s["prev"][1]))
        prev = dp.get("W16", dp.get("Q"))
    out = {"raw": raw, "streams": [], "quant": {}}
    if s["fmt"] == "bf16":
        dt = ts[s["name"]]["dtype"]
        enc = "bf16" if dt == "BF16" else "f16"
        out["main"] = {"name": "weights", "x": d["W16"].astype(np.uint16), "enc": enc, "width": 16, "prev": prev}
        out["quant"]["weights"] = quant.bf16_stats(d["W16"], dt)
        return out
    d["fmt"] = s["fmt"]
    Q = d["Q"]
    out["quant"]["codes"] = quant.code_stats(d)
    if s["fmt"] in ("fp8",):
        enc, x, pv = "e4m3", Q.astype(np.uint16), prev
    elif s["fmt"] in ("nvfp4", "mxfp4"):
        enc, x, pv = "e2m1", Q.astype(np.uint16), prev
    elif s["fmt"] == "int8":
        enc, x, pv = "i8", zigzag(Q), (zigzag(prev) if prev is not None else None)
    else:
        enc, x, pv = "u4", Q.astype(np.uint16), prev
    R, C = Q.shape
    sctx = zctx = None
    if d.get("S") is not None and d["S"].size > 1:
        sb, _ = quant.scale_bucket(d["S"], d["sdt"])
        sctx = quant.expand(sb, d["bh"], d["bw"], R, C)
        if d["S"].size >= 64:
            out["quant"]["scales"] = quant.scale_stats(d["S"], d["sdt"], d["sw"])
            sw = d["sw"]
            if sw == 1:
                out["streams"].append(stream("scales", d["S"], "u8", 8))
            elif sw == 2:
                out["streams"].append(stream("scales", d["S"], "bf16" if d["sdt"] == "BF16" else "f16", 16))
            else:  # 32-bit scales: the quant-2026 numbers only (sign and exponent modelled, the mantissa raw)
                out["wide_scales"] = {"n": int(d["S"].size), "width": 8 * sw}
    if d.get("Z") is not None:
        zctx = quant.expand(d["Z"].astype(np.int32), d["zbh"], d["zbw"], R, C)
        out["quant"]["zero points"] = quant.zero_stats(d["Z"])  # named after its stream: one part, not two
        out["streams"].append(stream("zero points", d["Z"], "u4", 4))
    out["main"] = {"name": "codes", "x": x, "enc": enc, "width": d["nb"], "prev": pv, "sctx": sctx, "zctx": zctx}
    return out


def load_gguf(p, s):
    url = f"{quant.HF}/{p['repo']}/resolve/main/{p['file']}"
    n = (s["r1"] - s["r0"]) * s["row_bytes"]
    raw = GET(url, (s["start"], s["start"] + n - 1))
    Q, st, width = gguf.unpack(raw, s["type"], s["cols"])
    prev = None
    if s["prev"]:
        prev = gguf.unpack(GET(url, (s["prev"]["start"], s["prev"]["start"] + n - 1)), s["type"], s["cols"])[0]
    name = gguf.TYPES[s["type"]][0]
    enc = {"Q4_K": "u4", "Q5_K": "u5", "Q6_K": "u6", "Q8_0": "i8"}[name]
    x = zigzag(Q) if enc == "i8" else Q.astype(np.uint16)
    pv = (zigzag(prev) if enc == "i8" else prev.astype(np.uint16)) if prev is not None else None
    R, C = Q.shape
    blk = gguf.block_of(s["type"])
    if name in ("Q4_K", "Q5_K"):
        sc = (st["sc"].astype(np.int64) << 6) | st["mn"]
        sctx = np.repeat(sc, blk["sc"], axis=1)[:, :C]
    elif name == "Q6_K":
        sctx = np.repeat(st["sc"].astype(np.int64), blk["sc"], axis=1)[:, :C]
    else:
        sctx = np.repeat((st["d"].astype(np.int64) >> 10) & 31, blk["d"], axis=1)[:, :C]
    streams = []
    for k, a in st.items():
        enc_s = {16: "f16", 6: "u6", 8: "u8"}[width[k]]
        streams.append(stream(k, a, enc_s, width[k]))
    out = {"raw": raw, "streams": streams, "quant": {},
           "main": {"name": "codes", "x": x, "enc": enc, "width": ENC[enc][0], "prev": pv, "sctx": sctx, "zctx": None}}
    h = np.bincount(x.ravel().astype(np.int64), minlength=1 << ENC[enc][0])
    if enc == "i8":
        b, _ = quant.int8_float_like(Q)
        out["quant"]["codes"] = {"fixed": b, "fixed_name": "sign + class by tiers + low bits raw"}
    else:
        b, nm = quant.fixed_codes(h, ENC[enc][0])
        out["quant"]["codes"] = {"fixed": b, "fixed_name": nm}
    for sm in streams:
        nbs = ENC[sm["enc"]][0]
        if nbs <= 8:
            b, nm = quant.fixed_codes(np.bincount(sm["x"].ravel().astype(np.int64), minlength=1 << nbs), nbs)
            out["quant"][sm["name"]] = {"fixed": b, "fixed_name": nm}
    return out


def phase1(args):
    p, s, adir = args
    t0 = time.time()
    rec = {k: s[k] for k in ("id", "fmt", "cat", "name", "r0", "r1")}
    rec["expert"] = s.get("e")
    rec["prev"] = s["prev"]
    try:
        o = load_gguf(p, s) if p["kind"] == "gguf" else load_st(p, s)
        m = o["main"]
        rec["n"] = int(m["x"].size)
        rec["stored_bytes"] = len(o["raw"])
        parts = {m["name"]: static(m["x"], m["enc"], m.get("sctx"), m.get("zctx"), m.get("prev"))}
        parts[m["name"]]["width"] = m["width"]
        for sm in o["streams"]:
            if sm["x"].size >= 64:
                parts[sm["name"]] = static(sm["x"], sm["enc"])
                parts[sm["name"]]["width"] = sm["width"]
        for k, v in o["quant"].items():
            parts.setdefault(k, {})["quant"] = v
        if "wide_scales" in o:
            parts.setdefault("scales", {}).update(o["wide_scales"])
        rec["parts"] = parts
        base = os.path.join(adir, f"{p['tag']}.{s['id']}")
        R, C = m["x"].shape
        rc = cm_rows(R, C, ENC[m["enc"]][0])
        arr = {"main": m["x"][:rc]}
        if m.get("prev") is not None:
            arr["prev"] = m["prev"][:rc].astype(np.uint16)
        if m.get("sctx") is not None:
            arr["sctx"] = m["sctx"][:rc].astype(np.uint32)
        meta = {"main": [m["name"], m["enc"]], "streams": []}
        for i, sm in enumerate(o["streams"]):
            if sm["x"].size >= 64:
                arr[f"s{i}"] = sm["x"][:cm_rows(*sm["x"].shape, ENC[sm["enc"]][0])]
                meta["streams"].append([f"s{i}", sm["name"], sm["enc"]])
        np.savez(base + ".npz", **arr)
        json.dump(meta, open(base + ".json", "w"))
        open(base + ".raw", "wb").write(o["raw"])
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]
    rec["seconds"] = round(time.time() - t0, 1)
    return p["tag"], rec


# ---------------- phase 2 ----------------

_LIB = []


def lib():
    if not _LIB:
        L = ctypes.CDLL(os.path.join(D, "libcm.so"))
        L.cm_bits.restype = ctypes.c_double
        L.cm_bits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
        _LIB.append(L)
    return _LIB[0]


def cm(x, enc, mask, lr=LR[0], aux1=None, aux2=None, limit=1023):
    """Bits a value (and a list: bits a value at each coding position) of the context-mixing model `mask` over x."""
    nb, order, _ = ENC[enc]
    x = np.ascontiguousarray(x, np.uint16)
    key = np.ascontiguousarray(key_of(enc, x)[0], np.uint16)
    o = np.array(order, np.int8)
    pb = np.zeros(nb)
    a1 = None if aux1 is None else np.ascontiguousarray(aux1, np.uint32)
    a2 = None if aux2 is None else np.ascontiguousarray(key_of(enc, aux2)[0], np.uint32)
    bits = lib().cm_bits(x.ctypes.data, key.ctypes.data, x.shape[0], x.shape[1], nb, o.ctypes.data,
                         None if a1 is None else a1.ctypes.data, None if a2 is None else a2.ctypes.data, mask, lr, limit, pb.ctypes.data)
    if bits < 0:
        raise MemoryError("cm_bits")
    return bits / x.size, (pb / x.size).tolist()


def comp(path):
    out = {}
    for name, cmd in (("xz -9e", ["xz", "-9e", "-T1", "-c", path]), ("zstd -22 --long", ["zstd", "--ultra", "-22", "--long=27", "-T1", "-c", "-q", path])):
        t0 = time.time()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out[name] = {"bytes": len(r.stdout) if r.returncode == 0 else None, "seconds": round(time.time() - t0, 1)}
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.zpaq")
        t0 = time.time()
        r = subprocess.run(["zpaq", "a", a, path, "-m5", "-t1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out["zpaq -m5"] = {"bytes": os.path.getsize(a) if r.returncode == 0 and os.path.exists(a) else None, "seconds": round(time.time() - t0, 1)}
    return out


def phase2(task):
    tag, sid, kind, adir = task
    base = os.path.join(adir, f"{tag}.{sid}")
    t0 = time.time()
    res = {"tag": tag, "id": sid, "task": kind}
    try:
        if kind == "comp":
            res["comp"] = comp(base + ".raw")
        else:
            a = np.load(base + ".npz")
            meta = json.load(open(base + ".json"))
            name, enc = meta["main"]
            x, prev, sctx = a["main"], a["prev"] if "prev" in a else None, a["sctx"] if "sctx" in a else None
            out = {}
            if kind == "cm":
                b, pb = cm(x, enc, LADDER["+ matching tensor"], LR[0], sctx, prev)
                out[name] = {"cm": b, "cm_per_bit": pb, "cm_values": int(x.size)}
                for key, sname, senc in meta["streams"]:
                    out[sname] = {"cm": cm(a[key], senc, SIDE)[0], "cm_values": int(a[key].size)}
            else:  # the ladder and the second mixer rate, on the weights or codes
                out[name] = {"ladder": {k: cm(x, enc, m, LR[0], sctx, prev)[0] for k, m in LADDER.items() if k != "+ matching tensor"},
                             "cm_lr2": cm(x, enc, LADDER["+ matching tensor"], LR[1], sctx, prev)[0]}
                for key, sname, senc in meta["streams"]:
                    out[sname] = {"cm_lr2": cm(a[key], senc, SIDE, LR[1])[0]}
            res["parts"] = out
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()[-2000:]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def cpus():
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(int(q) / int(per))))
    except Exception:
        try:
            q, per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()), int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, q // per))
        except Exception:
            pass
    return n


def main():
    pd, od = sys.argv[1], sys.argv[2]
    adir = os.environ.get("ARRAYS") or os.path.join(od, "arrays")
    for x in (adir, os.path.join(od, "phase1")):
        os.makedirs(x, exist_ok=True)
    ncpu = int(os.environ.get("PROCS", cpus()))
    plans = [json.load(open(os.path.join(pd, f))) for f in sorted(os.listdir(pd)) if f.endswith(".json")]
    print(f"{len(plans)} checkpoints, {sum(len(p['samples']) for p in plans)} samples, {ncpu} processes", flush=True)
    t0 = time.time()
    recs = {p["tag"]: [] for p in plans}
    jobs = [(p, s, adir) for p in plans for s in p["samples"]]
    with cf.ProcessPoolExecutor(min(len(jobs), max(2, min(8, 2 * ncpu)))) as ex:
        for tag, rec in ex.map(phase1, jobs):
            recs[tag].append(rec)
            if "error" in rec:
                print(f"  phase 1 {tag} {rec['id']}: {rec['error']}", flush=True)
    for p in plans:
        out = {k: p[k] for k in ("repo", "spec", "tag", "kind", "census", "stored")}
        out["qm"] = p.get("qm")
        out["samples"] = sorted(recs[p["tag"]], key=lambda r: r["id"])
        json.dump(out, open(os.path.join(od, "phase1", p["tag"] + ".json"), "w"), indent=1)
    errs = sum(1 for v in recs.values() for r in v if "error" in r)
    print(f"phase 1 done in {time.time() - t0:.0f} s, {errs} errors", flush=True)
    ok = {(p["tag"], r["id"]) for p in plans for r in recs[p["tag"]] if "error" not in r}
    tasks = [(p["tag"], s["id"], "cm", adir) for p in plans for s in p["samples"] if (p["tag"], s["id"]) in ok]
    for kind in ("ladder", "comp"):  # the ladder first: the compressors are the slow part and the least needed
        tasks += [(p["tag"], s["id"], kind, adir) for p in plans for s in p["samples"][:COMP_SAMPLES] if (p["tag"], s["id"]) in ok]
    done = 0
    with open(os.path.join(od, "phase2.jsonl"), "a") as f, cf.ProcessPoolExecutor(ncpu) as ex:
        futs = [ex.submit(phase2, t) for t in tasks]
        for fu in cf.as_completed(futs):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            if "error" in r or done % 25 == 0:
                print(f"  phase 2 {done}/{len(tasks)} at {time.time() - t0:.0f} s" + (f": {r['tag']} {r['id']} {r['task']} {r['error']}" if "error" in r else ""), flush=True)
    print(f"all done in {time.time() - t0:.0f} s", flush=True)


def selftest():
    """The CM against sources of known entropy, the static estimates, the GGUF unpackers (gguf.selftest) and quant-2026's own."""
    gguf.selftest()
    quant.selftest()
    rng = np.random.default_rng(0)
    p = np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.005, 0.002, 0.002, 0.0005, 0.0005])
    p /= p.sum()
    h = float(-(p * np.log2(p)).sum())
    x = rng.choice(16, size=(256, 1024), p=p).astype(np.uint16)
    b0, pb = cm(x, "u4", O0)
    ball = cm(x, "u4", LADDER["+ matching tensor"], aux1=np.zeros(x.shape, np.uint32), aux2=x[::-1].copy())[0]
    assert abs(b0 - h) < 0.01 and abs(sum(pb) - b0) < 1e-9, (b0, h)
    assert ball < h + 0.1, (ball, h)  # contexts that know nothing cost little once the mixer has learnt to ignore them
    # a copy of the left neighbour 90% of the time: the left model takes it
    y = x.copy()
    keep = rng.random(y.shape) < 0.9
    for c in range(1, y.shape[1]):
        y[:, c] = np.where(keep[:, c], y[:, c - 1], y[:, c])
    assert cm(y, "u4", O0 | LEFT)[0] < 0.5 * cm(y, "u4", O0)[0]
    # what the matching tensor says, it knows
    assert cm(x, "u4", O0 | AUX2, aux2=x)[0] < 0.05
    assert cm(np.zeros((64, 1024), np.uint16), "bf16", O0)[0] < 0.01
    # static: a context that fixes the key leaves only the rest
    w = (rng.integers(100, 140, (512, 512)) << 7 | rng.integers(0, 128, (512, 512))).astype(np.uint16)
    st = static(w, "bf16", prev=w)
    assert st["ctx"]["matching tensor"] < st["H_value_given_key"] + 0.01 < st["H0"], st
    assert abs(st["fields"]["fields apart"] - st["H0"]) < 0.03  # independent fields: the sum is the whole
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
