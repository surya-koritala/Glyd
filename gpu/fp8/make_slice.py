"""A few-layer model directory from a real public Hugging Face FP8 checkpoint, for loading in vLLM (gpu/fp8/check_fp8_vllm.py): the checkpoint's own
FP8 weights and scales of its layers 0..N-1 and its embedding, final norm and head, read by HTTP range (no shard is downloaded whole), written as
safetensors beside its config (the layer count set to N, what is listed per layer cut to N) and its tokenizer files. Standard library only.

    python make_slice.py REPO OUT_DIR --layers N [--max-gb 12] [--plan] [--weights consolidated]
    python make_slice.py Qwen/Qwen3.8-27B-FP8 /root/slice-qwen --layers 4
    python make_slice.py zai-org/GLM-5.3-Flash /root/slice-glm --layers 3        (its first_k_dense_replace = 3 dense layers: the MoE layers are routed experts, 7 GB a layer, and stay out)
    python make_slice.py stepfun-ai/Step-3.5-Flash /root/slice-step --pick 0,3,43 --max-gb 30     (layers 0, 3 and 43 as the slice's 0, 1 and 2: a layer past the first ones, with what the config says of it)

--pick I,J,...: those layers in that order in place of the first N, renumbered from 0; what the config lists a layer (a list at least as long as the layer
count, a nested config's layer numbers, a string of layer numbers as Step's moe_layers_enum) taken for them.
--weights consolidated: a Mistral-format checkpoint (consolidated*.safetensors and params.json: n_layers set, no vision encoder), which vLLM serves as its own MoE language model.
--plan: what would be fetched, no more. A tensor is kept if it is in a layer below N (any `layers.<i>.` in its name), or outside the layers and
not the vision tower's, the multi-token-prediction head's or a projector's (the embedding, the final norm, the LM head). Layers at or past the
model's first_k_dense_replace hold routed experts: the plan says so and the size cap (--max-gb) stops it where they would not fit. The source's
tokenizer and other small files (json, txt, jinja, tiktoken, tokenizer.model: up to 20 MB each) are copied; config.json is edited, not copied.
The slice's logits are not the model's (the layers after N are gone): it is for checking one vLLM FP8 path against another on the same weights.
A multimodal checkpoint's slice has no vision tower: serve it with --language-model-only. No token is read or sent (public repos)."""
import argparse
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

HF = "https://huggingface.co"
CHUNK = 32 << 20  # bytes a range request
SHARD = 4 << 30  # bytes a written shard, about
SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vit|multi_modal_projector|mm_projector|patch_merge_mlp|vision_encoder|vision_language_adapter|patch_merger|pre_mm_projector_norm|mtp|nextn)(\.|$)|^mtp\b")
SMALL = re.compile(r"\.(json|txt|jinja|tiktoken|model)$")
NOT_SMALL = re.compile(r"(model\.safetensors\.index\.json|^config\.json$|^params\.json$|consolidated|\.bin$|\.pt$|\.gguf$)")
MAX_SMALL = 20 << 20


def get(url, rng=None, tries=12):
    """url's bytes (the byte range rng, [a, b] inclusive), retried (the Hub answers a burst of requests with 429 and 5xx now and then); 401, 403 and 404 raised, and the last error where the tries run out."""
    err = None
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
            err = e
        except Exception as e:
            err = e
        time.sleep(min(30, 2**i))
    raise IOError(f"{url} {rng}: {err}")


def tensors(repo, prefix="model"):
    """Every tensor of the checkpoint (its model*.safetensors, or consolidated*.safetensors, Mistral's own format): name -> (dtype, shape, shard, absolute start, end), from the shards' headers."""
    try:
        shards = sorted(set(json.loads(get(f"{HF}/{repo}/resolve/main/{prefix}.safetensors.index.json"))["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = [f"{prefix}.safetensors"]

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


def layer_of(name):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def select(ts, pick):
    """The tensors of the slice by their names in it: the layers of pick (layer pick[j] as layer j), and what is outside the layers but not the
    vision tower's or the MTP head's."""
    keep, new = {}, {i: j for j, i in enumerate(pick)}
    for k, v in ts.items():
        if SKIP.search(k):
            continue
        i = layer_of(k)
        if i is None:
            keep[k] = v
        elif i in new:
            keep[re.sub(rf"(^|\.)layers\.{i}\.", rf"\g<1>layers.{new[i]}.", k, count=1)] = v
    return keep


def nbytes(v):
    return v[4] - v[3]


def edit_config(cfg, pick):
    """config.json with the layers of pick (range(n): the first n): num_hidden_layers (the text config's and the top's), the lists that hold a value a
    layer (layer_types, indexer_types, mlp_layer_types, moe_layer_freq, Step's swiglu_limits ...: any list at least as long as the layer count, in a
    nested config too) taken for them, the layer numbers a nested config names (linear_attn_config's) or a string of them (Step's moe_layers_enum)
    renumbered, no multi-token-prediction layers."""
    cfg = json.loads(json.dumps(cfg))
    tc = cfg.get("text_config") or cfg
    old, n, new = tc.get("num_hidden_layers", cfg.get("num_hidden_layers")), len(pick), {i: j for j, i in enumerate(pick)}
    if not isinstance(old, int) or max(pick) >= old:
        raise SystemExit(f"config.json: num_hidden_layers {old!r}: layer {max(pick)} is past it")
    for d in {id(tc): tc, id(cfg): cfg}.values():
        if "num_hidden_layers" in d:
            d["num_hidden_layers"] = n
        for k, v in list(d.items()):
            if isinstance(v, list) and len(v) >= old and k != "modules_to_not_convert":
                d[k] = [v[i] for i in pick]
            elif isinstance(v, str) and re.fullmatch(r"\d+(,\d+)+", v):
                d[k] = ",".join(str(new[int(x)]) for x in v.split(",") if int(x) in new)
            elif isinstance(v, dict) and k not in ("quantization_config", "rope_parameters", "rope_scaling"):
                for kk, vv in list(v.items()):
                    if isinstance(vv, list) and vv and all(isinstance(x, int) for x in vv) and kk.endswith("_layers"):
                        v[kk] = [new[x] for x in vv if x in new]
                    elif isinstance(vv, list) and len(vv) >= old:  # (a value a layer, in a nested config: MiniMax's sparse_attention_config)
                        v[kk] = [vv[i] for i in pick]
        if isinstance(d.get("first_k_dense_replace"), int):  # (the dense layers picked, which come first)
            d["first_k_dense_replace"] = sum(i < d["first_k_dense_replace"] for i in pick)
        for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "num_mtp_modules"):
            if k in d:
                d[k] = 0
    return cfg


def edit_params(params, n):
    """params.json (Mistral's own config) with n layers and no vision encoder (a text model: vLLM takes it as the MoE language model alone)."""
    params = json.loads(json.dumps(params))
    if not isinstance(params.get("n_layers"), int) or n > params["n_layers"]:
        raise SystemExit(f"params.json: n_layers {params.get('n_layers')!r}: --layers {n} is past it")
    params["n_layers"] = n
    params.pop("vision_encoder", None)
    return params


def small_files(repo):
    """The repo's small text files (tokenizer, generation config, chat template ...) by name and size, from the Hub's file list."""
    info = json.loads(get(f"{HF}/api/models/{repo}?blobs=true"))
    return [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", []) if SMALL.search(s["rfilename"]) and not NOT_SMALL.search(s["rfilename"]) and (s.get("size") or 0) <= MAX_SMALL]


def write_shard(path, repo, items):
    """One safetensors file of items [(name, (dtype, shape, shard, start, end))], fetched by range in parallel chunks, written in place."""
    header, off = {}, 0
    for k, v in items:
        header[k] = {"dtype": v[0], "shape": v[1], "data_offsets": [off, off + nbytes(v)]}
        off += nbytes(v)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.pwrite(fd, struct.pack("<Q", len(hb)) + hb, 0)
        base, jobs = 8 + len(hb), []
        for k, v in items:
            a = header[k]["data_offsets"][0]
            for c in range(0, nbytes(v), CHUNK):
                jobs.append((f"{HF}/{repo}/resolve/main/{v[2]}", v[3] + c, v[3] + min(nbytes(v), c + CHUNK) - 1, base + a + c))

        def one(j):
            url, s, e, at = j
            os.pwrite(fd, get(url, (s, e)), at)
            return e - s + 1

        done = 0
        with cf.ThreadPoolExecutor(10) as ex:
            for n in ex.map(one, jobs):
                done += n
        os.ftruncate(fd, base + off)
    finally:
        os.close(fd)
    return off


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--pick", default="", help="I,J,...: these layers, renumbered from 0, in place of the first --layers")
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--weights", default="model", choices=("model", "consolidated"), help="the weights' files: model*.safetensors with config.json, or Mistral's consolidated*.safetensors with params.json")
    a = ap.parse_args()
    t0 = time.time()
    own = a.weights == "consolidated"
    pick = [int(x) for x in a.pick.split(",")] if a.pick else list(range(a.layers or 0))
    if not pick or (own and a.pick):
        raise SystemExit("--layers N, or --pick I,J,... (not with --weights consolidated)")
    a.layers = len(pick)
    cfg = json.loads(get(f"{HF}/{a.repo}/resolve/main/{'params' if own else 'config'}.json"))
    new_cfg = edit_params(cfg, a.layers) if own else edit_config(cfg, pick)
    ts = tensors(a.repo, a.weights)
    keep = select(ts, pick)
    total = sum(nbytes(v) for v in keep.values())
    kinds = {}
    for v in keep.values():
        kinds[v[0]] = kinds.get(v[0], 0) + nbytes(v)
    tc = cfg.get("text_config") or cfg
    dense = tc.get("first_k_dense_replace", (cfg.get("moe") or {}).get("first_k_dense_replace"))
    layers_ = sorted({layer_of(k) for k in keep if layer_of(k) is not None})
    experts = sum(nbytes(v) for k, v in keep.items() if re.search(r"\.experts?\.", k))
    print(f"{a.repo}: {len(ts)} tensors; the slice: {len(keep)} tensors of layers {layers_[:1]}..{layers_[-1:]} and {sum(layer_of(k) is None for k in keep)} outside the layers, {total / 2**30:.2f} GiB ({', '.join(f'{k} {b / 2**30:.2f}' for k, b in sorted(kinds.items()))})")
    if dense is not None and a.layers > dense:
        print(f"NOTE: first_k_dense_replace is {dense}: layers {dense}.. are mixture-of-experts layers (routed experts in the slice: {experts / 2**30:.2f} GiB)")
    if total > a.max_gb * 2**30:
        raise SystemExit(f"the slice is {total / 2**30:.1f} GiB, over --max-gb {a.max_gb}")
    if a.plan:
        for k in list(keep)[:12] + ["..."]:
            print("  ", k, keep[k][:2] if k in keep else "")
        return
    os.makedirs(a.out, exist_ok=True)
    smalls = small_files(a.repo)
    for fn, _ in smalls:
        os.makedirs(os.path.dirname(os.path.join(a.out, fn)) or a.out, exist_ok=True)
        with open(os.path.join(a.out, fn), "wb") as f:
            f.write(get(f"{HF}/{a.repo}/resolve/main/{fn}"))
    with open(os.path.join(a.out, "params.json" if own else "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)
    names = sorted(keep, key=lambda k: (keep[k][2], keep[k][3]))  # (the source's order: its shards' reads stay sequential)
    shards, cur, size = [], [], 0
    for k in names:
        if cur and size + nbytes(keep[k]) > SHARD:
            shards.append(cur)
            cur, size = [], 0
        cur.append((k, keep[k]))
        size += nbytes(keep[k])
    shards.append(cur)
    weight_map, written = {}, 0
    for i, items in enumerate(shards, 1):
        fn = f"{a.weights}-{i:05d}-of-{len(shards):05d}.safetensors"
        written += write_shard(os.path.join(a.out, fn), a.repo, items)
        weight_map.update({k: fn for k, _ in items})
        print(f"{fn}: {len(items)} tensors, {written / 2**30:.2f} GiB so far, {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(a.out, f"{a.weights}.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": written}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(a.out, "SLICE.json"), "w") as f:
        json.dump({"source": a.repo, "layers": a.layers, "picked": pick if a.pick else None, "tensors": len(keep), "bytes": written, "small_files": [s for s, _ in smalls]}, f, indent=2)
    # a check: the first and last bytes of up to 48 tensors a shard (the largest and every k-th), against the source's, by the header the slice was written with
    bad, probes = 0, []
    for i, items in enumerate(shards, 1):
        path = os.path.join(a.out, f"{a.weights}-{i:05d}-of-{len(shards):05d}.safetensors")
        with open(path, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hn))
            step = max(1, len(items) // 48)
            for k, v in sorted(items, key=lambda kv: -nbytes(kv[1]))[:8] + items[::step]:
                s_, e_ = hdr[k]["data_offsets"]
                m = min(4096, e_ - s_)
                for lo in (0, e_ - s_ - m):
                    f.seek(8 + hn + s_ + lo)
                    probes.append((k, lo, f.read(m), f"{HF}/{a.repo}/resolve/main/{v[2]}", (v[3] + lo, v[3] + lo + m - 1)))

    def probe(p):
        return p[0], p[1], p[2] == get(p[3], p[4])

    with cf.ThreadPoolExecutor(16) as ex:
        for k, lo, ok in ex.map(probe, probes):
            if not ok:
                bad += 1
                print("MISMATCH", k, lo)
    print(f"{a.out}: {len(keep)} tensors, {written / 2**30:.2f} GiB, {len(smalls)} small files, ends of a sample checked against the source: {'FAIL' if bad else 'ok'}, {time.time() - t0:.0f} s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
