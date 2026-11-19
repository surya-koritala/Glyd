"""For each repo: exists, gated, its safetensors index or single file, and the dtypes and sizes of its tensors (from the shards' headers, read by HTTP range)."""
import json, sys, struct, urllib.request, collections, concurrent.futures as cf
H = "https://huggingface.co"
import time
def get(url, rng=None):
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404): raise
            time.sleep(2 * (i + 1))
        except Exception:
            if i == 5: raise
            time.sleep(2 * (i + 1))
def header(repo, fn):
    n = struct.unpack("<Q", get(f"{H}/{repo}/resolve/main/{fn}", (0, 7)))[0]
    return n, json.loads(get(f"{H}/{repo}/resolve/main/{fn}", (8, 8 + n - 1)))
def probe(repo):
    out = {"repo": repo}
    try:
        info = json.loads(get(f"{H}/api/models/{repo}"))
    except Exception as e:
        return {**out, "error": f"api: {e}"}
    out["gated"] = info.get("gated"); files = [s["rfilename"] for s in info.get("siblings", [])]
    st = [f for f in files if f.endswith(".safetensors")]
    out["shards"] = len(st)
    if out["gated"]:
        return out
    try:
        cfg = json.loads(get(f"{H}/{repo}/resolve/main/config.json"))
        q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
        out["quant"] = {k: q.get(k) for k in ("quant_method", "fmt", "format", "weight_block_size", "bits", "group_size") if k in (q or {})} if q else None
        out["arch"] = (cfg.get("architectures") or ["?"])[0]
    except Exception as e:
        out["config_error"] = str(e)
    dt = collections.Counter(); nbytes = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for n, hdr in ex.map(lambda f: header(repo, f), st[:400]):
            for k, v in hdr.items():
                if k == "__metadata__": continue
                b = v["data_offsets"][1] - v["data_offsets"][0]
                dt[v["dtype"]] += b; nbytes += b
    out["bytes_by_dtype_GB"] = {k: round(v / 1e9, 2) for k, v in dt.most_common()}
    out["total_GB"] = round(nbytes / 1e9, 1)
    return out
repos = sys.argv[1:] if __name__ == "__main__" else []
def safe(r):
    try: return probe(r)
    except Exception as e: return {'repo': r, 'error': str(e)[:200]}
with cf.ThreadPoolExecutor(3) as ex:
    for r in ex.map(safe, repos):
        print(json.dumps(r))
