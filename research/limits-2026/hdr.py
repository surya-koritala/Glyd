"""Safetensors headers + config.json of each repo by HTTP range reads (no weights), 2 s apart, as research/quant-2026/hdr.py
writes them, and the head of a GGUF file (its metadata and tensor table). Resumable.

    python hdr.py OUT_DIR REPO ... [REPO:FILE.gguf ...]
"""
import json, os, struct, sys, time, urllib.error, urllib.request

last = [0.0]


def get(url, rng=None):
    for t in range(8):
        w = 2.0 - (time.time() - last[0])
        if w > 0:
            time.sleep(w)
        h = {"User-Agent": "glyd-limits/1"}
        if rng:
            h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        try:
            last[0] = time.time()
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(30 * (t + 1))
                continue
            raise
        except Exception:
            time.sleep(10 * (t + 1))
    raise RuntimeError(url)


def safetensors(repo, out):
    base = f"https://huggingface.co/{repo}/resolve/main/"
    o = {"repo": repo, "config": json.loads(get(base + "config.json")), "files": {}}
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    for fn, size in [(x["rfilename"], x.get("size")) for x in info["siblings"] if x["rfilename"].endswith(".safetensors")]:
        b = get(base + fn, (0, 262143))
        n = struct.unpack("<Q", b[:8])[0]
        if n + 8 > len(b):
            b += get(base + fn, (len(b), n + 7))
        o["files"][fn] = {"size": size, "hlen": n, "header": json.loads(b[8:8 + n])}
    o["created"], o["downloads"] = info.get("createdAt"), info.get("downloads")
    json.dump(o, open(out, "w"))


def gguf(repo, fn, out):
    """The raw head of a GGUF file, grown until the tensor table parses (gguf.py parses it)."""
    from gguf import parse
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    head = b""
    while True:
        head += get(url, (len(head), len(head) + (16 << 20) - 1))
        try:
            meta = parse(head)
            break
        except IndexError:
            continue
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    size = {x["rfilename"]: x.get("size") for x in info["siblings"]}[fn]
    json.dump({"repo": repo, "file": fn, "size": size, "created": info.get("createdAt"), "downloads": info.get("downloads"),
               **meta}, open(out, "w"))


if __name__ == "__main__":
    od = sys.argv[1]
    os.makedirs(od, exist_ok=True)
    for spec in sys.argv[2:]:
        repo, _, fn = spec.partition(":")
        p = os.path.join(od, repo.replace("/", "__") + (("__" + fn) if fn else "") + ".json")
        if os.path.exists(p):
            continue
        gguf(repo, fn, p) if fn else safetensors(repo, p)
        print(spec, flush=True)
