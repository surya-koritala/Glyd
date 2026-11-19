"""What a checkpoint stores, by tensor kind, from the safetensors headers alone (HTTP range reads of each
shard's header: a few hundred KB a shard, no weights).

    python census.py REPO OUT.json [--prefix DIR/]       # e.g. moonshotai/Kimi-K3 k3_census.json

Each tensor is classified by its role (routed experts, shared experts, attention, dense MLP, router, embeddings,
lm_head, norms and the like, vision) and its payload (fp8 codes, fp4 codes packed two a byte, block scales,
bf16, other). The totals are what the "fit" arithmetic in README.md starts from.
"""
import collections, json, re, struct, sys, concurrent.futures as cf
import requests

DTSIZE = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
          "U8": 1, "BOOL": 1, "U16": 2, "U32": 4, "U64": 8}


def files_of(repo, prefix=""):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True, token=False)
    return sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(prefix) and s.rfilename.endswith(".safetensors"))


def header(repo, fn):
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    for attempt in range(5):
        try:
            n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
            body = requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=120).content
            h = json.loads(body)
            h.pop("__metadata__", None)
            return fn, h
        except Exception as e:  # network hiccup: retry
            err = e
    raise err


def role(name):
    n = name.lower()
    if "vision" in n or "mm_projector" in n or "visual" in n:
        return "vision"
    if "embed_tokens" in n or n.endswith("embedding.weight") or "embedding" in n.split(".")[-2:-1]:
        return "embed"
    if "lm_head" in n or "unembedding" in n:
        return "lm_head"
    if "shared_expert" in n:
        return "shared_experts"
    if re.search(r"experts?[._]?\d|\.experts\.|expert_[a-z_]*proj", n) and "routed_expert_" not in n and "e_score" not in n:
        return "routed_experts"
    if "routed_expert_" in n:  # K3's latent up/down projections around the experts
        return "expert_latent_proj"
    if re.search(r"(router|\.gate\.|\.gate$|e_score_correction)", n):
        return "router"
    if "self_attn" in n or "attention" in n or "attn" in n:
        return "attention"
    if ".mlp." in n:
        return "dense_mlp"
    return "other"


def payload(name, dtype):
    n = name.lower()
    if n.endswith(("weight_scale_inv", "weight_scale", "_scales", ".scales")):
        return "scale"
    if n.endswith(("weight_packed", "_blocks")):
        return "fp4_packed"
    if dtype == "F8_E4M3":
        return "fp8"
    if dtype == "BF16":
        return "bf16"
    return "other_" + dtype


def census(repo, prefix=""):
    fns = files_of(repo, prefix)
    tot = collections.defaultdict(lambda: [0, 0, 0])  # (role, payload) -> tensors, elements, bytes
    per_file = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for fn, h in ex.map(lambda f: header(repo, f), fns):
            sz = 0
            for name, t in h.items():
                n = 1
                for d in t["shape"]:
                    n *= d
                b = t["data_offsets"][1] - t["data_offsets"][0]
                assert b == n * DTSIZE[t["dtype"]], (name, t)
                k = (role(name), payload(name, t["dtype"]))
                tot[k][0] += 1
                tot[k][1] += n
                tot[k][2] += b
                sz += b
            per_file[fn] = sz
    rows = [{"role": r, "payload": p, "tensors": v[0], "elements": v[1], "bytes": v[2]} for (r, p), v in sorted(tot.items())]
    return {"repo": repo, "files": len(fns), "total_bytes": sum(r["bytes"] for r in rows), "kinds": rows}


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
    c = census(repo, prefix)
    json.dump(c, open(out, "w"), indent=1)
    print(f"{repo}: {c['files']} files, {c['total_bytes'] / 1e9:.2f} GB")
    for r in c["kinds"]:
        print(f"  {r['role']:20s} {r['payload']:12s} {r['tensors']:8d} tensors {r['elements'] / 1e9:10.3f} G elements {r['bytes'] / 1e9:10.3f} GB  {100 * r['bytes'] / c['total_bytes']:5.1f}%")
