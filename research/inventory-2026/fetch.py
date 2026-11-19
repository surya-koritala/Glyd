"""Safetensors headers of 2026 checkpoints by HTTP range (no weights). Resumable: one JSON per model in headers/."""
import json, os, struct, sys, time, urllib.request, urllib.error
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headers")
UA = {"User-Agent": "glyd-inventory/1"}
last = [0.0]

def get(url, rng=None, gap=1.0, tries=8):
    for t in range(tries):
        wait = gap - (time.time() - last[0])
        if wait > 0: time.sleep(wait)
        h = dict(UA)
        if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        try:
            last[0] = time.time()
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(30 * (t + 1)); continue
            raise
        except Exception:
            time.sleep(10 * (t + 1))
    raise RuntimeError(f"gave up {url}")

def header(repo, f):
    url = f"https://huggingface.co/{repo}/resolve/main/{f}"
    b = get(url, (0, 262143))
    n = struct.unpack("<Q", b[:8])[0]
    if n + 8 > len(b):
        b += get(url, (len(b), n + 7))
    h = json.loads(b[8:8 + n])
    h.pop("__metadata__", None)
    return {k: [v["dtype"], v["shape"]] for k, v in h.items()}

def model(repo):
    path = os.path.join(OUT, repo.replace("/", "__") + ".json")
    if os.path.exists(path):
        return
    try:
        info = json.loads(get(f"https://huggingface.co/api/models/{repo}", gap=2.5))
    except urllib.error.HTTPError as e:
        json.dump({"repo": repo, "err": f"HTTP {e.code}"}, open(path, "w")); print(repo, "ERR", e.code, flush=True); return
    files = [s["rfilename"] for s in info.get("siblings", [])]
    rec = {"repo": repo, "created": info.get("createdAt"), "gated": info.get("gated"), "license": (info.get("cardData") or {}).get("license"),
           "st_params": (info.get("safetensors") or {}).get("parameters"), "files": files}
    try:
        rec["config"] = json.loads(get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
    except Exception as e:
        rec["config_err"] = str(e)
    st = sorted(f for f in files if f.endswith(".safetensors") and "/" not in f)
    if not st:  # (a subfolder layout)
        st = sorted(f for f in files if f.endswith(".safetensors"))
    rec["shards"] = {}
    for f in st:
        try:
            rec["shards"][f] = header(repo, f)
        except Exception as e:
            rec.setdefault("shard_err", {})[f] = str(e)
    json.dump(rec, open(path, "w"))
    print(repo, len(st), "shards", sum(len(v) for v in rec["shards"].values()), "tensors", flush=True)

if __name__ == "__main__":
    for r in [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]:
        try:
            model(r)
        except Exception as e:
            print(r, "FAILED", e, flush=True)
