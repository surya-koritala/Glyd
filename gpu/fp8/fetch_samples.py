"""FP8 (e4m3) weight samples from the 2026 checkpoints, read by HTTP range from the public Hugging Face repos
(no token; no whole checkpoint is downloaded), for the format study (fmt_study.py) and the reference's tests.

    python fetch_samples.py OUT_DIR [REPO...]

Per repo: 10 e4m3 weight tensors (a Linear's or a fused expert's, 8 MB or more each), chosen at random (seed 0) across the
checkpoint's layers and experts: attention, MLP and, where there are any, experts (4 / 3 / 3; else 5 / 5). Each is a slice of whole
rows (a multiple of 128 of them, 8-16 MB, within one expert of a fused tensor). Writes OUT_DIR/<repo>__NN.npy (uint8 [rows, K])
and OUT_DIR/manifest.json (repo, tensor, shape, expert, slice rows, scale tensor). Run again: files already there are kept."""
import collections, concurrent.futures as cf, json, math, os, random, re, struct, sys, time, urllib.error, urllib.request

import numpy as np

HF = "https://huggingface.co"
REPOS = ["zai-org/GLM-5.3-Flash", "Qwen/Qwen3.8-27B-FP8", "MiniMaxAI/MiniMax-M3-MXFP8", "mistralai/Mistral-Small-4-119B-2603", "deepseek-ai/DeepSeek-V4.1-Flash"]
MIN_BYTES, TARGET_BYTES, MAX_BYTES, ROWS_MULT = 8 << 20, 12 << 20, 16 << 20, 128
BIG = 1 << 20  # elements: a smaller tensor is "other"


# ---- the checkpoint's headers (as fmt_measure.py, which measured the 13 checkpoints) ----

def get(url, rng=None, tries=8):
    """The bytes of url (rng: an inclusive byte range), retried."""
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
    for s in (".weight_scale", ".weight_scale_inv", ".scale", ".weight_scales", "_scale", "_scale_inv"):
        if stem + s in ts:
            return stem + s
    return None


def classify(ts):
    """name -> class: bf16, fp8, fp4, scale or other."""
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
                    cls[k] = "fp4"
                elif scols * 32 == cols:
                    cls[k] = "fp8"
                else:
                    cls[k] = "other"
            else:
                cls[k] = "other"
        else:
            cls[k] = "other"
    return cls


# ---- the choice of samples ----

def category(name):
    if ".experts." in name and "shared_experts" not in name or name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
        return "expert"
    if "attn" in name:
        return "attn"
    return "mlp"


def kind(name):
    return re.sub(r"\.\d+(?=\.)", ".N", name)


def plan(repo, ts):
    """The ~10 samples of a repo: [(category, name, row0 within the expert or tensor, rows, expert or None)]."""
    cls = classify(ts)
    rnd = random.Random(0)
    groups = {"expert": collections.defaultdict(list), "attn": collections.defaultdict(list), "mlp": collections.defaultdict(list)}
    for k in sorted(ts):
        dt, shape, _, a, b = ts[k]
        # e4m3 weights of the model's own layers only: not e5m2, not the multi-token-prediction head (mtp.*) or DeepSeek's engram tables
        if cls[k] == "fp8" and dt == "F8_E4M3" and not k.startswith("mtp") and "engram" not in k and len(shape) >= 2 and b - a >= MIN_BYTES:
            groups[category(k)][kind(k)].append(k)
    quotas = {"expert": 4, "attn": 3, "mlp": 3} if groups["expert"] else {"attn": 5, "mlp": 5}
    out = []
    for cat, quota in quotas.items():
        kinds = sorted(groups[cat].items(), key=lambda kv: -sum(ts[k][4] - ts[k][3] for k in kv[1]))
        left = {kd: list(names) for kd, names in kinds}
        for i in range(quota):
            if not kinds:
                break
            kd = kinds[i % len(kinds)][0]
            name = left[kd].pop(rnd.randrange(len(left[kd])))
            dt, shape, fn, a, b = ts[name]
            experts, o, K = (shape[0], shape[1], shape[2]) if len(shape) == 3 else (None, shape[0], shape[1])
            take = max(ROWS_MULT, TARGET_BYTES // K // ROWS_MULT * ROWS_MULT)
            take = min(take, o // ROWS_MULT * ROWS_MULT) if o >= ROWS_MULT else o
            e = rnd.randrange(experts) if experts else None
            r0 = rnd.randrange(0, o - take + 1) // ROWS_MULT * ROWS_MULT
            out.append((cat, name, r0, take, e))
    return out


def fetch(repo, out_dir):
    ts = tensors(repo)
    cls_names = plan(repo, ts)
    jobs = []
    for i, (cat, name, r0, take, e) in enumerate(cls_names):
        dt, shape, fn, a, b = ts[name]
        K, o = shape[-1], shape[-2]
        base = a + ((e * o) if e is not None else 0) * K
        start, end = base + r0 * K, base + (r0 + take) * K - 1
        assert end < b and MIN_BYTES // 2 <= end - start + 1 <= MAX_BYTES, (name, end - start + 1)
        sc = scale_of(name, ts)
        path = os.path.join(out_dir, f"{repo.replace('/', '__')}__{i:02d}.npy")
        jobs.append((path, f"{HF}/{repo}/resolve/main/{fn}", (start, end), {
            "file": os.path.basename(path), "repo": repo, "category": cat, "tensor": name, "dtype": dt, "tensor_shape": shape, "expert": e,
            "slice_rows": [r0, r0 + take], "shape": [take, K], "scale_tensor": sc, "scale_shape": ts[sc][1] if sc else None}))

    def one(job):
        path, url, rng, meta = job
        if not os.path.exists(path):
            raw = get(url, rng)
            a = np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"])
            np.save(path + ".part.npy", a)
            os.replace(path + ".part.npy", path)
        return meta

    with cf.ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, jobs))


def main():
    out_dir = sys.argv[1]
    repos = sys.argv[2:] or REPOS
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for repo in repos:
        t0 = time.time()
        metas = fetch(repo, out_dir)
        manifest += metas
        print(f"{repo}: {len(metas)} samples, {sum(math.prod(m['shape']) for m in metas) / 1e6:.0f} MB, {time.time() - t0:.0f} s", flush=True)
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
