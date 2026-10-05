"""Corpus of real model-file fragments: valid safetensors and GGUF files made of real tensors of public, ungated
Hugging Face repos, read by HTTP range (no token, at most 2 requests a second), headers rewritten to match what was kept.

    python build_corpus.py list                 # the corpus plan
    python build_corpus.py build NAME...        # (or all)

Selection: tensors are grouped by (dtype, kind) (kind = the name with layer and expert numbers taken out); each kind gets
a share of the file's byte budget in proportion to its share of the source's bytes; a tensor too big for its share is cut
to its first whole rows (a 3-D expert tensor: its first whole experts), so every kept byte is a real byte at its real
place in a real row. Small tensors (norms, biases, scales) are kept whole. The GGUF files keep the first shard's key-value
section byte for byte (tokenizer included); only the tensor count and the tensor infos are rewritten.
"""
import collections, json, math, os, random, re, struct, sys, time, concurrent.futures as cf

SCRATCH = os.environ.get("CODECW_SCRATCH", ".")  # holds gguf/gguflib.py (a GGUF header reader over range reads) and codecw/
sys.path.insert(0, SCRATCH + "/gguf")
import gguflib  # noqa: E402  (its get() is fmt_measure.get with a global 2 requests a second limit and the 429's own wait)
from gguflib import get, HF, TYPES  # noqa: E402

OUT = SCRATCH + "/codecw/corpus"
HDR = SCRATCH + "/codecw/hdr_st"
MB = 1 << 20
CHUNK = 24 * MB  # one range request

ESIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4,
         "U32": 4, "I64": 8, "U64": 8, "BOOL": 1}

# name, kind, repo, selection (safetensors: shard budget; gguf: file or directory), budget MB
ST = [
    ("st-qwen3.8-27b-bf16", "st", "Qwen/Qwen3.8-27B", None, 128),
    ("st-gemma-4-26b-a4b-bf16", "st", "google/gemma-4-26B-A4B-it", None, 128),
    ("st-glm-5.3-flash-bf16", "st", "zai-org/GLM-5.3-Flash-BF16", None, 192),
    ("st-minimax-m3-bf16", "st", "MiniMaxAI/MiniMax-M3", None, 192),
    ("st-glm-5.3-flash-fp8", "st", "zai-org/GLM-5.3-Flash", None, 128),
    ("st-qwen3.8-27b-fp8", "st", "Qwen/Qwen3.8-27B-FP8", None, 128),
    ("st-mistral-small-4-fp8", "st", "mistralai/Mistral-Small-4-119B-2603", None, 128),
]
Q = "ggml-org/Qwen3.8-27B-GGUF"
G = "ggml-org/gemma-4-26B-A4B-it-GGUF"
GU = "unsloth/gemma-4-26B-A4B-it-GGUF"
L = "unsloth/GLM-5.3-Flash-GGUF"
M = "bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF"
MU = "unsloth/Mistral-Small-4-119B-2603-GGUF"
X = "unsloth/MiniMax-M3-GGUF"
XB = "bartowski/MiniMax-M3-GGUF"
GG = [
    ("gguf-qwen3.8-27b-bf16", "gguf", Q, "Qwen3.8-27B-BF16.gguf", 192),
    ("gguf-qwen3.8-27b-q8_0", "gguf", Q, "Qwen3.8-27B-Q8_0.gguf", 160),
    ("gguf-qwen3.8-27b-q4_k_m", "gguf", Q, "Qwen3.8-27B-Q4_K_M.gguf", 160),
    ("gguf-gemma-4-26b-a4b-bf16", "gguf", G, "gemma-4-26B-A4B-it-BF16.gguf", 192),
    ("gguf-gemma-4-26b-a4b-q8_0", "gguf", G, "gemma-4-26B-A4B-it-Q8_0.gguf", 160),
    ("gguf-gemma-4-26b-a4b-ud-q4_k_m", "gguf", GU, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 160),
    ("gguf-glm-5.3-flash-bf16", "gguf", L, "BF16", 192),
    ("gguf-glm-5.3-flash-q8_0", "gguf", L, "Q8_0", 160),
    ("gguf-mistral-small-4-bf16", "gguf", MU, "BF16", 192),
    ("gguf-mistral-small-4-q8_0", "gguf", M, "mistralai_Mistral-Small-4-119B-2603-Q8_0", 160),
    ("gguf-mistral-small-4-q4_k_m", "gguf", M, "mistralai_Mistral-Small-4-119B-2603-Q4_K_M", 160),
    ("gguf-minimax-m3-bf16", "gguf", X, "BF16", 192),
    ("gguf-minimax-m3-q8_0", "gguf", X, "Q8_0", 160),
    ("gguf-minimax-m3-q4_k_m", "gguf", XB, "MiniMax-M3-Q4_K_M", 160),
    ("gguf-gpt-oss-20b-f16", "gguf", "unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-F16.gguf", 160),
]
PLAN = {n: (k, r, s, b) for n, k, r, s, b in ST + GG}


def kind_of(name):
    return re.sub(r"\.\d+(?=\.)", ".N", re.sub(r"(?<=\.)\d+$", "N", name))


def layer_key(name):
    return [int(x) for x in re.findall(r"\.(\d+)(?=\.)", name)]


# ---------------------------------------------------------------- safetensors


def st_shards(repo):
    try:
        idx = json.loads(get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))
        return sorted(set(idx["weight_map"].values()))
    except Exception:
        return ["model.safetensors"]


def st_header(repo, fn):
    f = f"{HDR}/{(repo + '__' + fn).replace('/', '_')}.json"
    if os.path.exists(f):
        return json.load(open(f))
    url = f"{HF}/{repo}/resolve/main/{fn}"
    first = get(url, (0, 2 * MB - 1))
    n = struct.unpack("<Q", first[:8])[0]
    raw = first[8:8 + n] if 8 + n <= len(first) else get(url, (8, 8 + n - 1))
    h = json.loads(raw)
    meta = h.pop("__metadata__", None)
    out = {"n": n, "meta": meta, "tensors": {k: [v["dtype"], v["shape"], v["data_offsets"][0], v["data_offsets"][1]] for k, v in h.items()}}
    json.dump(out, open(f, "w"))
    return out


def pick_shards(shards, k=6):
    if len(shards) <= k:
        return shards
    idx = sorted({round(i * (len(shards) - 1) / (k - 1)) for i in range(k)})
    return [shards[i] for i in idx]


# ---------------------------------------------------------------- selection


class T:
    """A source tensor: name, type tag, dims as stored (safetensors: shape, outermost first; gguf: ne, innermost first),
    where it is (file, absolute start), bytes, bytes in a row (the unit it is cut at), rows in a slice, and the row
    count of the contiguous unit."""

    def __init__(self, name, ty, dims, file, start, nbytes, row_bytes, order):
        self.name, self.ty, self.dims, self.file, self.start, self.nbytes, self.row_bytes, self.order = name, ty, dims, file, start, nbytes, row_bytes, order


def select(ts, budget, rnd, per_tensor_cap=12 * MB, small=256 * 1024, tail=False):
    """-> [(T, kept_bytes)] in source order. kept_bytes is a multiple of row_bytes (the whole tensor when it fits)."""
    total = sum(t.nbytes for t in ts)
    groups = collections.defaultdict(list)
    for t in ts:
        groups[(t.ty, kind_of(t.name))].append(t)
    picks = []
    for (ty, kd), items in groups.items():
        gbytes = sum(t.nbytes for t in items)
        alloc = budget * gbytes / total
        items = sorted(items, key=lambda t: (layer_key(t.name), t.name))
        if max(t.nbytes for t in items) <= small:
            # norms, biases, scales: whole, a few layers of them
            n = min(len(items), 4)
        else:
            n = min(len(items), max(1, math.ceil(alloc / per_tensor_cap)))
        step = len(items) / n
        off = rnd.random() * step
        chosen = [items[min(len(items) - 1, int(off + i * step))] for i in range(n)]
        chosen = list(dict.fromkeys(chosen))
        each = alloc / max(1, len(chosen))
        for t in chosen:
            if t.nbytes <= max(each, small):
                keep = t.nbytes
            else:
                rows = max(1, int(each // t.row_bytes), -(-65536 // t.row_bytes))
                keep = min(t.nbytes, rows * t.row_bytes)
            picks.append((t, keep))
    picks.sort(key=lambda p: p[0].order)
    return picks


def fetch(url, a, b):
    """bytes [a, b) of url, in CHUNK-sized range requests."""
    out = bytearray()
    pos = a
    while pos < b:
        end = min(b, pos + CHUNK)
        out += get(url, (pos, end - 1))
        pos = end
    return bytes(out)


def fetch_all(reqs, label):
    """reqs: [(url, a, b)] -> [bytes]; adjacent ranges of one url (gap under 192 KB) are one request, the gap dropped."""
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i][0], reqs[i][1]))
    runs = []  # (url, a, b, [indices])
    for i in order:
        url, a, b = reqs[i]
        if runs and runs[-1][0] == url and a - runs[-1][2] <= 192 * 1024 and a >= runs[-1][2]:
            runs[-1][2] = b
            runs[-1][3].append(i)
        else:
            runs.append([url, a, b, [i]])
    res = [None] * len(reqs)
    t0 = time.time()
    done = 0
    nbytes = sum(r[2] - r[1] for r in runs)

    def go(run):
        url, a, b, idx = run
        data = fetch(url, a, b)
        for i in idx:
            _, x, y = reqs[i]
            res[i] = data[x - a:y - a]
        return len(data)

    last = 0.0
    with cf.ThreadPoolExecutor(4) as ex:
        for n in ex.map(go, runs):
            done += n
            if time.time() - last > 15:
                last = time.time()
                print(f"  {label}: {done / MB:.0f} of {nbytes / MB:.0f} MB read, {len(runs)} runs, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    print(f"  {label}: {done / MB:.0f} MB read in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return res


# ---------------------------------------------------------------- build: safetensors


def build_st(name, repo, budget_mb):
    rnd = random.Random(name)
    shards = pick_shards(st_shards(repo))
    print(f"{name}: {repo}, shards {len(shards)}: {shards[0]} .. {shards[-1]}", file=sys.stderr)
    ts, order = [], 0
    allt = {}
    for fn in shards:
        h = st_header(repo, fn)
        base = 8 + h["n"]
        for k, (dt, shape, a, b) in sorted(h["tensors"].items(), key=lambda kv: kv[1][2]):
            esz = ESIZE[dt]
            row = (shape[-1] if shape else 1) * esz
            t = T(k, dt, shape, fn, base + a, b - a, row, order)
            order += 1
            ts.append(t)
            allt[k] = t
    picks = select(ts, budget_mb * MB, rnd)
    # companions of a cut or whole 8-bit weight: its scales (rows cut to match a 128-block when the weight was cut)
    chosen = {p[0].name: p for p in picks}
    extra = []
    for t, keep in list(picks):
        if t.ty in ("F8_E4M3", "F8_E5M2", "U8", "I8") and t.name.endswith(".weight"):
            stem = t.name[:-len("weight")]
            for k2, t2 in allt.items():
                if k2.startswith(stem) and k2 not in chosen and k2 != t.name and t2.nbytes <= 64 * MB:
                    k2keep = t2.nbytes
                    if keep < t.nbytes and t2.dims and t2.dims[0] * 128 >= t.dims[0] > (t2.dims[0] - 1) * 128:
                        rows_keep = math.ceil((keep // t.row_bytes) / 128)
                        k2keep = rows_keep * t2.row_bytes
                    extra.append((t2, min(k2keep, t2.nbytes)))
                    chosen[k2] = (t2, k2keep)
    picks = sorted(picks + extra, key=lambda p: p[0].order)
    reqs = [(f"{HF}/{repo}/resolve/main/{t.file}", t.start, t.start + keep) for t, keep in picks]
    data = fetch_all(reqs, name)
    hdr, off, parts = {"__metadata__": {"format": "pt"}}, 0, []
    for (t, keep), d in zip(picks, data):
        assert len(d) == keep
        shape = list(t.dims)
        if keep < t.nbytes:
            rows_all = t.nbytes // t.row_bytes
            rows = keep // t.row_bytes
            # shape[0] counts the outermost units: cut it, and keep the rest unless the cut is inside a unit
            inner = math.prod(shape[1:]) if len(shape) > 1 else 1
            esz = ESIZE[t.ty]
            if len(shape) >= 2:
                unit_rows = rows_all // shape[0]  # rows per outermost index
                if rows % unit_rows == 0:
                    shape[0] = rows // unit_rows
                else:  # a cut inside the first slice: its first rows, as a matrix
                    shape = [rows, shape[-1]]
            else:
                shape = [keep // esz]
            assert math.prod(shape) * esz == keep, (t.name, shape, keep)
        hdr[t.name] = {"dtype": t.ty, "shape": shape, "data_offsets": [off, off + keep]}
        off += keep
        parts.append(d)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    path = f"{OUT}/{name}.safetensors"
    with open(path + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for d in parts:
            f.write(d)
    os.replace(path + ".part", path)
    summarize(name, path, len(hj) + 8, picks)


# ---------------------------------------------------------------- build: gguf


def gguf_skip_kv(buf, nkv, pos):
    sc = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def string(p):
        n = struct.unpack_from("<Q", buf, p)[0]
        return p + 8 + n

    def value(t, p):
        if t in sc:
            return p + sc[t]
        if t == 8:
            return string(p)
        if t == 9:
            et, n = struct.unpack_from("<IQ", buf, p)
            p += 12
            if et in sc:
                return p + n * sc[et]
            assert et == 8, et
            for _ in range(n):
                p = string(p)
            return p
        raise ValueError(t)

    for _ in range(nkv):
        pos = string(pos)
        t = struct.unpack_from("<I", buf, pos)[0]
        pos = value(t, pos + 4)
    return pos


def build_gguf(name, repo, sel, budget_mb):
    rnd = random.Random(name)
    ts_all, kv0, sh = gguflib.file_tensors(repo, sel)
    ts, order = [], 0
    for tname, dims, ty, path, start, nb in ts_all:
        tn, blck, tsz = TYPES[ty]
        row = dims[0] // blck * tsz
        ts.append(T(tname, ty, dims, path, start, nb, row, order))
        order += 1
    print(f"{name}: {repo} {sel}: {len(ts)} tensors in {len(sh)} shards", file=sys.stderr)
    picks = select(ts, budget_mb * MB, rnd)
    # the key-value section: the first shard's, byte for byte
    first = sh[0][0]
    h0 = gguflib.header(repo, first)
    hb = fetch(f"{HF}/{repo}/resolve/main/{first}", 0, h0["data_start"])
    assert hb[:4] == b"GGUF"
    ver, nt0, nkv = struct.unpack_from("<IQQ", hb, 4)
    kv_end = gguf_skip_kv(hb, nkv, 24)
    align = int(h0["kv"].get("general.alignment", 32))
    reqs = [(f"{HF}/{repo}/resolve/main/{t.file}", t.start, t.start + keep) for t, keep in picks]
    data = fetch_all(reqs, name)
    infos, off, body = bytearray(), 0, []
    for (t, keep), d in zip(picks, data):
        assert len(d) == keep
        dims = list(t.dims)
        if keep < t.nbytes:
            rows_all = t.nbytes // t.row_bytes
            rows = keep // t.row_bytes
            unit = math.prod(dims[1:-1]) if len(dims) > 2 else 1  # rows per outermost slice
            slices = rows // unit if len(dims) >= 3 else 0
            if len(dims) >= 3 and slices >= 1 and rows % unit == 0:
                dims[-1] = slices
            elif len(dims) >= 3:  # inside the first slice
                dims = [dims[0], rows] + [1] * (len(dims) - 2)
            else:
                dims[-1] = rows if len(dims) >= 2 else dims[0]
            assert math.prod(dims) // TYPES[t.ty][1] * TYPES[t.ty][2] == keep, (t.name, dims, keep)
        nm = t.name.encode()
        infos += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims) + struct.pack("<IQ", t.ty, off)
        pad = -keep % align
        body.append(d + b"\0" * pad)
        off += keep + pad
    head = b"GGUF" + struct.pack("<IQQ", ver, len(picks), nkv) + hb[24:kv_end] + bytes(infos)
    head += b"\0" * (-len(head) % align)
    path = f"{OUT}/{name}.gguf"
    with open(path + ".part", "wb") as f:
        f.write(head)
        for d in body:
            f.write(d)
    os.replace(path + ".part", path)
    summarize(name, path, len(head), picks)


def summarize(name, path, head, picks):
    by = collections.Counter()
    for t, keep in picks:
        by[t.ty if isinstance(t.ty, str) else TYPES[t.ty][0]] += keep
    size = os.path.getsize(path)
    print(f"{name}: {size / MB:.1f} MB, header {head / MB:.2f} MB, {len(picks)} tensors, types {dict((k, round(v / MB, 1)) for k, v in by.items())}", file=sys.stderr)
    with open(f"{OUT}/MANIFEST.tsv", "a") as f:
        f.write(f"{name}\t{os.path.basename(path)}\t{size}\t{head}\t{len(picks)}\t{json.dumps(dict(by))}\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    names = sys.argv[2:] or list(PLAN)
    if cmd == "list":
        for n, (k, r, s, b) in PLAN.items():
            print(n, k, r, s, b)
    elif cmd == "build":
        os.makedirs(OUT, exist_ok=True)
        for n in names:
            k, r, s, b = PLAN[n]
            ext = "safetensors" if k == "st" else "gguf"
            if os.path.exists(f"{OUT}/{n}.{ext}"):
                print(f"{n}: exists", file=sys.stderr)
                continue
            (build_st(n, r, b) if k == "st" else build_gguf(n, r, s, b))
