"""A Glyd checkpoint's model card (README.md) from its save: the source's repo, commit and license (the Hub's card data),
the packed and bf16 sizes measured from the save and its plan, the model's parameters (the Hub's "Model size" counts each
packed byte as one), and how to load and check it.

    python model_card.py [--numbers NUMBERS.json] SAVE_DIR PLAN.json OUT/README.md
    python model_card.py [--numbers NUMBERS.json] hub:ORG/NAME PLAN.json OUT/README.md      (a published save: its glyd.json and safetensors headers, from the Hub)

The level (penguin, kestrel or swift) is the save's glyd.json "level", or from its "layout" (i6, i5, or mma for penguin). A "calibration" there (glyd-gpu pack --codes) is on the card's first lines. NUMBERS.json holds what was
measured, and only that is on the card: {"size_gb", "gpus": [{"name", "tok_s", "first_token_ms", "mem_gib_4k", "fits_32k"}], "kl_bf16", "kl_fp8_ref", "date", "versions": {...},
"evals": [{"dataset", "task_id", "value", "n", "harness": {"inspect_ai", "inspect_evals"}, "notes", "date"}], "evals_bf16": [the same on the bf16 model]} (tools/cards/evals.sh)
With "evals" the card has a Benchmarks table and OUT/.eval_results/glyd.yaml is written beside the README (the Hub's model-index replacement).
"""
import json
import math
import sys
from pathlib import Path


def gb(n):
    """Decimal GB: one decimal from 1 GB, two below."""
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e9:.2f} GB"


def billions(n):
    return f"{n / 1e9:.2f}B"


def hub_bytes(api, repo):
    return sum(f.size for f in api.list_repo_tree(repo, recursive=True) if getattr(f, "size", None) and f.path.endswith(".safetensors"))


def save_facts(save):
    """(glyd.json, the safetensors files' bytes, [(tensor name, dtype, shape)] of every tensor in them) of a save on disk or, as hub:ORG/NAME, on the Hub."""
    if str(save).startswith("hub:"):
        from huggingface_hub import HfApi, hf_hub_download
        repo = str(save)[4:]
        manifest = json.loads(Path(hf_hub_download(repo, "glyd.json")).read_text())
        api = HfApi()
        saved = hub_bytes(api, repo)
        meta = api.get_safetensors_metadata(repo)
        tensors = [(n, t.dtype, list(t.shape)) for fm in meta.files_metadata.values() for n, t in fm.tensors.items()]
        return manifest, saved, tensors
    save = Path(save)
    from safetensors import safe_open
    tensors = []
    for f in sorted(save.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as st:
            tensors += [(k, st.get_slice(k).get_dtype(), list(st.get_slice(k).get_shape())) for k in st.keys()]
    return json.loads((save / "glyd.json").read_text()), sum(f.stat().st_size for f in save.glob("*.safetensors")), tensors


LEVELS = ("penguin", "kestrel", "swift")
LAYOUT_LEVEL = {"i6": "kestrel", "i5": "swift"}
CALIBRATION = {"gptq-act": "GPTQ"}  # glyd.json's "calibration": {"method": ...}


def level_of(manifest):
    return manifest.get("level") or LAYOUT_LEVEL.get(manifest.get("layout"), "penguin")


TITLES = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}


def benchmarks(n):
    """The Benchmarks section: each eval's score at this level and, where the numbers file has evals_bf16, on bf16 (the same harness), and the harness line."""
    ev = n.get("evals")
    if not ev:
        return []
    bf = {e["task_id"]: e for e in n.get("evals_bf16", [])}
    pct = lambda e: f"{100 * e['value']:.1f}%"
    rows = ["", "## Benchmarks", "", "| Benchmark | this level |" + (" bf16 |" if bf else ""), "| :-- | --: |" + (" --: |" if bf else "")]
    for e in ev:
        rows.append(f"| {TITLES.get(e['task_id'], e['task_id'])} | {pct(e)} |" + (f" {pct(bf[e['task_id']]) if e['task_id'] in bf else '–'} |" if bf else ""))
    h = ev[0]["harness"]
    rows += ["", f"Run through `glyd serve` with Inspect AI {h['inspect_ai']} (inspect_evals {h['inspect_evals']})" + ("; the bf16 column is the source model in bf16 served by vLLM, with the same harness and settings" if bf else "") + ". "
             + "; ".join(f"{TITLES.get(e['task_id'], e['task_id'])}: {e['notes']}, {e['n']} questions" for e in ev) + "."]
    return rows


def eval_results(n, level):
    """.eval_results/glyd.yaml (the Hub's format: a list of entries, one per eval), or None when the numbers have no evals."""
    if not n.get("evals"):
        return None
    q = json.dumps
    return "".join(f"- dataset:\n    id: {e['dataset']}\n    task_id: {e['task_id']}\n  value: {e['value']}\n  date: {q(e.get('date') or n.get('date') or '')}\n"
                   f"  notes: {q(f'glyd serve, {level}, inspect-ai {e['harness']['inspect_ai']}, {e['notes']}')}\n" for e in n["evals"])


def short_gpu(name):
    return name.replace("NVIDIA ", "").replace("GeForce ", "")


def pct_smaller(n, bf16):
    return f"{round(100 * (1 - n / bf16))}%"


def kl(v):
    return "–" if v is None else f"{v:.3g}"


def compared(level, name, saved, bf16, numbers, levels, fp8):
    """The Compared table: bf16, the base org's FP8 (fp8 = (org, bytes) or None), and every Glyd level on the Hub (levels = {level: {"size", "kl", "tok_s"}}; this one from the save)."""
    n = numbers or {}
    rows = [("bf16 (original)", gb(bf16), "–", "0", "–")]
    if fp8:
        rows.append((f"{fp8[0]} FP8", gb(fp8[1]), pct_smaller(fp8[1], bf16), kl(n.get("kl_fp8_ref")), "–"))
    here = {"size": saved, "kl": 0 if level == "penguin" else n.get("kl_bf16"), "tok_s": next((g["tok_s"] for g in n.get("gpus", []) if "RTX 4090" in g["name"]), None)}
    for lv in LEVELS:
        d = here if lv == level else levels.get(lv)
        if d is None:
            continue
        label = f"[{lv}](https://huggingface.co/glyd/{name}-{lv})" + (" (lossless)" if lv == "penguin" else "")
        cells = [label, gb(d["size"]), pct_smaller(d["size"], bf16), kl(d["kl"]), "–" if d["tok_s"] is None else f"{d['tok_s']:.0f} tok/s"]
        if lv == level:
            cells[0] = f"{lv} (lossless, this repo)" if lv == "penguin" else f"{lv} (this repo)"
            cells = [f"**{c}**" for c in cells]
        rows.append(tuple(cells))
    return ["| | Size | Smaller than bf16 | KL vs bf16 | RTX 4090 |", "| :-- | --: | --: | --: | --: |"] + ["| " + " | ".join(r) + " |" for r in rows]


def speed(n):
    gpus = n.get("gpus")
    if not gpus:
        return []
    rows = ["## Speed and memory", "", "| GPU | Tokens/s | First token | Memory at 4k | 32k context |", "| :-- | --: | --: | --: | :-- |"]
    gpus = sorted(gpus, key=lambda g: -g["tok_s"])
    for g in gpus:
        rows.append(f"| {short_gpu(g['name'])} | {g['tok_s']:.0f} | {g['first_token_ms']:.0f} ms | {g['mem_gib_4k'] * 1.073741824:.1f} GB | {'yes' if g['fits_32k'] else 'no'} |")
    v = (n.get("versions") or {}).get("glyd")
    rows += ["", (f"Measured {n['date']}" if n.get("date") else "Measured") + (f" with glyd {v}" if v else "") + ": one GPU each, a 4k-token context; memory is the GPU memory in use."]
    fits = [short_gpu(g["name"]) for g in gpus if g["fits_32k"]]
    if fits:
        rows += ["", "Fits a 32k context on: " + ", ".join(fits) + "."]
    return rows


def render(manifest, saved, bf16, hub_count, plan, license_, numbers=None, levels=None, fp8=None):
    """The card's text: what it is, how to run it, how it compares, what was measured, the fine print. Pure: the Hub's answers come in as levels ({level: {"size", "kl", "tok_s"}} of the other Glyd repos) and fp8 ((org, bytes) or None)."""
    numbers = {k: v for k, v in (numbers or {}).items() if v is not None}   # (a null is a number not measured)
    repo, revision = manifest["source"]["repo"], manifest["source"]["revision"]
    level = level_of(manifest)
    lossless = level == "penguin"
    params = bf16 // 2
    name = repo.split("/")[-1]
    bpw = saved * 8 / params
    source = f"[{repo}](https://huggingface.co/{repo}) (commit [`{revision[:8]}`](https://huggingface.co/{repo}/tree/{revision}))"
    hero = f"**{gb(saved)}, {pct_smaller(saved, bf16)} smaller than bf16 ({gb(bf16)}).**"
    if lossless:
        summary = f"Glyd's lossless compression of {source}: every weight decodes to its bf16 value, bit for bit."
        python = f"""```python
import glyd
model = glyd.from_pretrained("glyd/{name}-{level}")   # pip install "glyd[gpu]"
```

"""
        runs = f"Linux with an NVIDIA GPU (Ampere or later), driver 580 or newer; glyd {manifest['glyd']} or later."
        stores = "stores"
    else:
        cal = manifest.get("calibration")
        how = f" with calibration ({CALIBRATION.get(cal['method'], cal['method'])})" if cal else ""
        summary = (f"Glyd's {level} level of {source}: about {bpw:.1f} bits per weight, quantized once from the original bf16 weights{how}. Not lossless"
                   + (f": KL divergence from bf16 is {numbers['kl_bf16']:.3g} (lower is closer)." if "kl_bf16" in numbers else "."))
        python = ""
        runs = "Runs on Glyd's engine (`glyd run`): Linux with an NVIDIA GPU, driver 580 or newer. Not for vLLM or transformers yet."
        stores = "is"
    text_only = any(k.startswith("model.visual") for k in plan.get("not_loaded", {}))
    table = "\n".join(compared(level, name, saved, bf16, numbers, levels or {}, fp8))
    optional = "".join("\n".join(x).strip() + "\n\n" for x in (speed(numbers), benchmarks(numbers)) if x)
    good = [runs] + (["Text only: the model's vision part is not included."] if text_only else []) + [
        f"The Hub's \"8-bit precision\" and \"Model size\" labels count packed bytes: this checkpoint {stores} about {bpw:.1f} bits per weight, from a model with {params:,} parameters."]
    license_text = (f"Weights: {license_.replace('apache-2.0', 'Apache-2.0')}, from [{repo}](https://huggingface.co/{repo}). "
                    "Glyd's engine, which runs them: open source, Apache-2.0 ([GitHub](https://github.com/surya-koritala/Glyd)).")
    return f"""---
license: {license_}
base_model: {repo}
base_model_relation: quantized
library_name: glyd
pipeline_tag: text-generation
tags:
- glyd
- {level}
- {"lossless" if lossless else "lossy"}
---

# {name} · Glyd {level}

{hero} {summary}

```sh
curl -LsSf https://getglyd.com/install.sh | sh
glyd run {repo}:{level}
```

{python}## Compared

{table}

Size: the files you download. KL: how far the model's next-token probabilities move from bf16, measured on WikiText-2.

{optional}## Good to know

{chr(10).join("- " + g for g in good)}

## License

{license_text}
"""


def hub_context(api, repo, name, level, numbers_path):
    """(levels, fp8) from the Hub: the other Glyd repos' sizes (and their numbers files beside --numbers), and the base org's -FP8 size of the same text weights."""
    levels = {}
    for lv in LEVELS:
        if lv == level:
            continue
        if not api.repo_exists(f"glyd/{name}-{lv}"):
            continue
        size = hub_bytes(api, f"glyd/{name}-{lv}")
        nf = (Path(numbers_path).parent if numbers_path else Path("cards")) / f"{name}-{lv}.json"   # (a penguin card has no numbers file: cards/ beside it)
        n = json.loads(nf.read_text()) if nf.exists() else {}
        levels[lv] = {"size": size, "kl": 0 if lv == "penguin" else n.get("kl_bf16"), "tok_s": next((g["tok_s"] for g in n.get("gpus", []) if "RTX 4090" in g["name"]), None)}
    fp8 = None
    if api.repo_exists(f"{repo}-FP8"):
        meta = api.get_safetensors_metadata(f"{repo}-FP8")
        fp8 = (repo.split("/")[0], sum(t.data_offsets[1] - t.data_offsets[0] for fm in meta.files_metadata.values() for k, t in fm.tensors.items() if "visual" not in k and not k.startswith("mtp")))
    return levels, fp8


def main():
    argv = sys.argv[1:]
    numbers = numbers_path = None
    if "--numbers" in argv:
        i = argv.index("--numbers")
        numbers_path = argv[i + 1]
        numbers = json.loads(Path(numbers_path).read_text())
        del argv[i:i + 2]
    save, plan_path, out = argv[0], Path(argv[1]), Path(argv[2])
    from huggingface_hub import HfApi
    manifest, saved, tensors = save_facts(save)
    plan = json.loads(plan_path.read_text())
    repo, revision = manifest["source"]["repo"], manifest["source"]["revision"]
    info = HfApi().model_info(repo, revision=revision)
    card = info.card_data.to_dict() if info.card_data else {}
    license_ = card.get("license") or "other"
    # the bf16 bytes of what the save holds: each packed tensor's shape, and each tensor saved as it is (bf16)
    bf16 = sum(2 * math.prod(t["shape"]) for e in manifest["packs"].values() for t in e["tensors"])
    bf16 += sum(2 * math.prod(shape) for name, _, shape in tensors if ".glyd_" not in name)
    hub_count = sum(math.prod(shape) for _, _, shape in tensors)  # (what the Hub's "Model size" adds up: a packed byte is one element of a U8 tensor)
    levels, fp8 = hub_context(HfApi(), repo, repo.split("/")[-1], level_of(manifest), numbers_path)
    text = render(manifest, saved, bf16, hub_count, plan, license_, numbers, levels, fp8)
    out.write_text(text)
    yaml = eval_results(numbers or {}, level_of(manifest))
    if yaml:
        (out.parent / ".eval_results").mkdir(exist_ok=True)
        (out.parent / ".eval_results" / "glyd.yaml").write_text(yaml)
    print(text)


if __name__ == "__main__":
    main()
