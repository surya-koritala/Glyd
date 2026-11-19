"""How much smaller 2026 models get losslessly, in the formats they are released in (BF16, FP8, FP4), from tensors
sampled across each checkpoint and read by HTTP range from Hugging Face (no whole checkpoint is downloaded).

    python fmt_measure.py plan REPO...            # headers only: what each checkpoint holds, by class and kind
    python fmt_measure.py run OUT_DIR REPO...     # sample, measure, write OUT_DIR/<repo>.json and summary.md

Per class:
- bf16: Glyd's packed size as released (glyd_gpu.kernels.pack_mma and pack_mma12, each unpacked and compared bit for bit on the GPU),
  and the order-0 floor of the sign and exponent with the 7 mantissa bits as they are.
- fp8 (e4m3): the order-0 entropy of the byte, and of the sign and exponent with the 3 mantissa bits as they are.
- fp4 (e2m1, two to a byte): the order-0 entropy of the 4-bit value, and of the sign and exponent with the mantissa bit as it is.
- scale: the order-0 entropy of the byte (one-byte scales), else zstd's.
- everything else (norms, biases, small tensors): unchanged.
Practical lossless coders on the same bytes, for reference: zstd -19 and Glyd's codec (glyd --max), on up to 8 MB of each sample.
A kind (a tensor name with its layer and expert numbers taken out) is sampled in up to PER_KIND tensors across its layers; its
result is applied to all of the kind's bytes."""
import collections, concurrent.futures as cf, json, math, os, random, re, shutil, struct, subprocess, sys, tempfile, time, urllib.error, urllib.request

import numpy as np

HF = "https://huggingface.co"
PER_KIND = 3
SAMPLE_MAX = 32 << 20  # bytes a sample
MODEL_CAP = int(os.environ.get("FMT_CAP", 3 << 30))  # sampled bytes a model at the most
CODER_BYTES = 8 << 20
ZSTD_BYTES = 2 << 20  # zstd -19 is slow: on the first 2 MB of a sample
BIG = 1 << 20  # elements: a smaller tensor is "other"


def get(url, rng=None, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise IOError(f"short read {len(data)} of {rng[1] - rng[0] + 1}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(min(30, 2 ** i))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(30, 2 ** i))


def tensors(repo):
    """Every tensor of the checkpoint: name -> (dtype, shape, shard, absolute start, end)."""
    try:
        shards = sorted(set(json.loads(get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]

    def head(fn):
        n = struct.unpack("<Q", get(f"{HF}/{repo}/resolve/main/{fn}", (0, 7)))[0]
        return fn, n, json.loads(get(f"{HF}/{repo}/resolve/main/{fn}", (8, 8 + n - 1)))

    out = {}
    with cf.ThreadPoolExecutor(12) as ex:
        for fn, n, hdr in ex.map(head, shards):
            for k, v in hdr.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    out[k] = (v["dtype"], v["shape"], fn, 8 + n + a, 8 + n + b)
    return out


def scale_of(name, ts):
    """The scale tensor beside a weight, if any."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name[: -len(".weight_packed")] if name.endswith(".weight_packed") else name
    for s in (".weight_scale", ".weight_scale_inv", ".scale", ".weight_scales"):
        if stem + s in ts:
            return stem + s
    return None


def classify(ts):
    """name -> class: bf16, fp8, fp4 (with its group), scale or other."""
    cls = {}
    for k, (dt, shape, *_) in ts.items():
        numel = math.prod(shape) if shape else 1
        if "scale" in k.rsplit(".", 1)[-1] or k.endswith("_scale_2") or k.endswith("input_scale"):
            cls[k] = "scale" if numel >= 4096 and len(shape) >= 2 else "other"
        elif numel < BIG or len(shape) < 2:
            cls[k] = "other"
        elif dt == "BF16":
            cls[k] = "bf16"
        elif dt in ("F8_E4M3", "F8_E5M2"):
            cls[k] = "fp8"
        elif dt in ("U8", "I8"):
            s = scale_of(k, ts)
            cols = shape[-1]
            if s is not None:
                scols = ts[s][1][-1] if ts[s][1] else 1
                if scols * 32 == 2 * cols or scols * 16 == 2 * cols:
                    cls[k] = "fp4"  # two e2m1 values a byte, a scale for every 32 (MXFP4) or 16 (NVFP4) of them
                elif scols * 32 == cols:
                    cls[k] = "fp8"  # e4m3 kept as bytes, a scale for every 32
                else:
                    cls[k] = "other"
            else:
                cls[k] = "other"
        else:
            cls[k] = "other"
    return cls


def kind(name):
    return re.sub(r"\.\d+(?=\.)", ".N", name)


def plan(repo, ts, cls, seed=0):
    """By kind: (class, total bytes, the samples: (name, start, end, shape of the sample))."""
    groups = collections.defaultdict(list)
    for k in sorted(ts):
        groups[(cls[k], kind(k))].append(k)
    kinds = {}
    for (c, kd), names in groups.items():
        total = sum(ts[k][4] - ts[k][3] for k in names)
        kinds[(c, kd)] = [c, total, names]
    def count(total, names):  # more picks for a big kind: 3, and one more for every 10 GB, up to 16
        return min(len(names), PER_KIND + min(13, int(total // 10e9)))
    sampled = [v for v in kinds.values() if v[0] in ("bf16", "fp8", "fp4", "scale")]
    budget = sum(count(v[1], v[2]) for v in sampled)
    per = max(1 << 20, min(SAMPLE_MAX, MODEL_CAP // max(1, budget)))
    rnd = random.Random(seed)
    out = {}
    for key, (c, total, names) in kinds.items():
        if c == "other":
            out[key] = (c, total, [])
            continue
        picks = rnd.sample(names, count(total, names))  # random across every layer and expert, not the names' order
        samples = []
        for k in picks:
            dt, shape, fn, a, b = ts[k]
            esize = {"BF16": 2, "F32": 4, "F16": 2}.get(dt, 1)
            row = shape[-1] * esize
            rows_all = (b - a) // row
            if c == "bf16":
                take = max(64, min(rows_all, per // row) // 64 * 64)
                take = min(take, rows_all // 64 * 64)
                if take < 64:
                    continue
                r0 = rnd.randrange(0, rows_all - take + 1) // 64 * 64 if rows_all > take else 0
                if len(shape) == 3:  # (experts, out, in): stay within one expert
                    o = shape[1]
                    take = min(take, o // 64 * 64)
                    if take < 64:
                        continue
                    e = rnd.randrange(shape[0])
                    r0 = e * o + (rnd.randrange(0, o - take + 1) // 64 * 64 if o > take else 0)
                samples.append((k, fn, a + r0 * row, a + (r0 + take) * row - 1, [take, shape[-1]], dt))
            else:
                take = max(1, min(rows_all, per // max(1, row)))
                r0 = rnd.randrange(0, rows_all - take + 1) if rows_all > take else 0
                samples.append((k, fn, a + r0 * row, a + (r0 + take) * row - 1, [take, shape[-1]], dt))
        out[key] = (c, total, samples)
    return out


def entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def measure_bytes(c, raw, dt):
    """bits a stored element at the floor (order-0), and the split floor (sign and exponent coded, mantissa as is)."""
    b = np.frombuffer(raw, dtype=np.uint8)
    if c == "fp8":
        h8 = entropy(np.bincount(b, minlength=256))
        hse = entropy(np.bincount(b >> 3, minlength=32)) + 3  # e4m3: sign and 4 exponent bits, 3 mantissa bits
        return {"native_bits": 8, "floor_bits": h8, "split_bits": hse}
    if c == "fp4":
        nib = np.concatenate([b & 15, b >> 4])
        h4 = entropy(np.bincount(nib, minlength=16))
        hse = entropy(np.bincount(nib >> 1, minlength=8)) + 1  # e2m1: sign and 2 exponent bits, 1 mantissa bit
        return {"native_bits": 4, "floor_bits": h4, "split_bits": hse}
    if c == "scale":
        if dt in ("U8", "I8", "F8_E4M3", "F8_E8M0", "F8_E5M2"):
            h8 = entropy(np.bincount(b, minlength=256))
            return {"native_bits": 8, "floor_bits": h8, "split_bits": h8}
        return None
    if c == "bf16":
        w = np.frombuffer(raw, dtype=np.uint16)
        hse = entropy(np.bincount(w >> 7, minlength=512)) + 7  # sign and 8 exponent bits, 7 mantissa bits
        return {"native_bits": 16, "split_bits": hse, "floor_bits": hse}


def coders(raw, glyd_bin, tmp):
    import zstandard
    part = raw[:CODER_BYTES]
    zp = part[:ZSTD_BYTES]
    out = {"bytes": len(part), "zstd_bytes": len(zp), "zstd19": len(zstandard.ZstdCompressor(level=19).compress(zp))}
    if glyd_bin:
        src, dst = os.path.join(tmp, "s.bin"), os.path.join(tmp, "s.glyd")
        with open(src, "wb") as f:
            f.write(part)
        r = subprocess.run([glyd_bin, "--max", src, "-o", dst], capture_output=True)
        if r.returncode == 0:
            out["glyd_max"] = os.path.getsize(dst)
            back = os.path.join(tmp, "b.bin")
            subprocess.run([glyd_bin, "-d", dst, "-o", back], check=True, capture_output=True)
            assert open(back, "rb").read() == part, "glyd round trip"
    return out


def glyd_pack(raw, shape):
    """Glyd's packed bytes of a bf16 matrix in both layouts, each unpacked and compared bit for bit."""
    try:
        import torch
        from glyd_gpu import kernels as g
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None
    w = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*shape).cuda()
    if w.shape[0] % 64 or w.shape[1] % 16:
        return None
    out = {}
    for name, fn in (("mma", g.pack_mma), ("mma12", g.pack_mma12)):
        q = fn(w)
        assert torch.equal(g.mma_unpack(q).view(torch.int16), w.view(torch.int16)), name
        out[name] = q.nbytes()
        del q
    return out


def run(out_dir, repo, glyd_bin):
    t0 = time.time()
    ts = tensors(repo)
    cls = classify(ts)
    pl = plan(repo, ts, cls)
    jobs = [(key, s) for key, (c, total, samples) in pl.items() for s in samples]
    data = {}
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(get, f"{HF}/{repo}/resolve/main/{s[1]}", (s[2], s[3])): (key, s) for key, s in jobs}
        for f in cf.as_completed(futs):
            key, s = futs[f]
            data[(key, s[0])] = f.result()
    t_dl = time.time() - t0
    res = {"repo": repo, "tensors": len(ts), "kinds": [], "download_s": round(t_dl), "sampled_bytes": sum(len(v) for v in data.values())}
    tmp = tempfile.mkdtemp()
    for key, (c, total, samples) in sorted(pl.items(), key=lambda kv: -kv[1][1]):
        row = {"class": c, "kind": key[1], "bytes": total, "tensors": sum(1 for k in ts if (cls[k], kind(k)) == key), "samples": []}
        for s in samples:
            raw = data[(key, s[0])]
            m = {"name": s[0], "bytes": len(raw)}
            fl = measure_bytes(c, raw, s[5])
            if fl:
                m.update(fl)
            if c == "bf16":
                p = glyd_pack(raw, s[4])
                if p:
                    m.update({k: v / len(raw) for k, v in p.items()})  # packed bytes per bf16 byte
            m.update({k: v for k, v in coders(raw, glyd_bin, tmp).items()})
            row["samples"].append(m)
        res["kinds"].append(row)
    shutil.rmtree(tmp, ignore_errors=True)
    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(out_dir, repo.replace("/", "__") + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def summarize(res):
    """Model-level bytes: as released, at the order-0 floor, at the split floor, with Glyd's bf16 layouts, and with zstd -19 and glyd --max on disk."""
    tot = collections.Counter()
    for k in res["kinds"]:
        c, b = k["class"], k["bytes"]
        tot["native"] += b
        tot["native_" + c] += b
        ss = k["samples"]

        def avg(f, default):
            vals = [(s["bytes"], f(s)) for s in ss if f(s) is not None]
            return sum(w * v for w, v in vals) / sum(w for w, _ in vals) if vals else default

        floor = avg(lambda s: s.get("floor_bits") / s["native_bits"] if "floor_bits" in s else None, 1.0)
        split = avg(lambda s: s.get("split_bits") / s["native_bits"] if "split_bits" in s else None, 1.0)
        mma = avg(lambda s: s.get("mma"), None) if c == "bf16" else 1.0
        mma12 = avg(lambda s: s.get("mma12"), None) if c == "bf16" else 1.0
        z = avg(lambda s: s["zstd19"] / s["zstd_bytes"] if "zstd19" in s else None, 1.0)
        gm = avg(lambda s: s["glyd_max"] / s["bytes"] if "glyd_max" in s else None, 1.0)
        tot["floor"] += b * floor
        tot["split"] += b * split
        tot["glyd_mma"] += b * (mma if mma is not None else 1.0)
        tot["glyd_mma12"] += b * (mma12 if mma12 is not None else 1.0)
        tot["zstd19"] += b * z
        tot["glyd_max"] += b * gm
        tot[c + "_floor"] += b * floor
        tot[c + "_split"] += b * split
    return tot


def main():
    mode = sys.argv[1]
    if mode == "plan":
        for repo in sys.argv[2:]:
            ts = tensors(repo)
            cls = classify(ts)
            by = collections.Counter()
            for k, (dt, shape, fn, a, b) in ts.items():
                by[(cls[k], dt)] += b - a
            pl = plan(repo, ts, cls)
            ns = sum(len(v[2]) for v in pl.values())
            sb = sum(s[3] - s[2] + 1 for v in pl.values() for s in v[2])
            print(repo, {f"{c}/{d}": round(v / 1e9, 1) for (c, d), v in by.most_common()}, f"kinds {len(pl)}, samples {ns}, {sb / 1e9:.2f} GB")
        return
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    glyd_bin = shutil.which("glyd")
    lines = []
    for repo in sys.argv[3:]:
        try:
            res = run(out_dir, repo, glyd_bin)
        except Exception as e:
            print(f"{repo}: FAILED {e!r}", flush=True)
            lines.append(f"| {repo} | failed: {str(e)[:80]} |")
            continue
        t = summarize(res)
        n = t["native"]
        pct = lambda x: f"{100 * (1 - t[x] / n):.1f}%"
        lines.append(f"| {repo} | {n / 1e9:.0f} | " + " | ".join(f"{100 * t['native_' + c] / n:.0f}%" for c in ("bf16", "fp8", "fp4", "scale", "other")) +
                     f" | {pct('floor')} | {pct('split')} | {pct('glyd_mma')} | {pct('glyd_mma12')} | {pct('zstd19')} | {pct('glyd_max')} | {res['sampled_bytes'] / 1e9:.2f} GB, {res['seconds']} s |")
        print(lines[-1], flush=True)
    head = ("| Model | GB as released | bf16 | fp8 | fp4 | scales | other | saved at the order-0 floor | saved, split floor | Glyd today (smallest layout, bf16 only) | "
            "Glyd today (12-bit, bf16 only) | zstd -19 on disk | glyd --max on disk | sampled |\n| :--- |" + " ---: |" * 13)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(head + "\n" + "\n".join(lines) + "\n")
    print(head)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
