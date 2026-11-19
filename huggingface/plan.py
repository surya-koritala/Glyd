"""A bf16 checkpoint's glyd-v4 save plan, made on the CPU without its weights: what format.save_pretrained writes,
in its order, for the model hf.py's from_pretrained loads, for the Rust packer (`glyd-gpu pack DIR OUT --plan PLAN`).

The model is built on the meta device as transformers builds it for from_pretrained (a causal LM, else an
image-text-to-text one), and glyd_gpu.model's own rules pick what is packed: every plain Linear of rows a multiple of
64 and columns of 16 (`plain`), each decoder layer's q, k, v and gate, up as one (`groups`, saved after their
module's other children, where `Merged` is attached), the embeddings that lookup takes and that pack_fast holds
(columns a multiple of 128: saved as they are, in their module's place), and the rest in state_dict's order.

Names: the model's (what save_pretrained writes), each read from the checkpoint tensor transformers' own renaming
(get_model_conversion_mapping, rename_source_key: from_pretrained's) puts there; tensors the model does not take
(transformers drops them: a vision tower a causal LM has not, a multi-token-prediction head) are not saved, as there.

Refused rather than guessed: a tensor the model has that no checkpoint tensor becomes by a rename alone (missing, or
made by a converter), or that the checkpoint holds in another shape or dtype; a mixture of experts; weights tied to
others (but the output layer to the input embedding: it is left out); an embedding with a scale or norm of its own.

    python plan.py MODEL OUT.json [--revision R]     (MODEL: a directory, or a Hugging Face repo id)
"""
import argparse
import json
import os
import sys
from pathlib import Path


def checkpoint_tensors(model, revision):
    """{name: (dtype, shape)} of the checkpoint's safetensors (headers only), and the revision they are at."""
    if os.path.isdir(model):
        from safetensors import safe_open
        out = {}
        for f in sorted(Path(model).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as s:
                for n in s.keys():
                    sl = s.get_slice(n)
                    out[n] = (sl.get_dtype(), list(sl.get_shape()))
        return out, revision
    from huggingface_hub import HfApi
    api = HfApi()
    sha = api.model_info(model, revision=revision).sha
    meta = api.get_safetensors_metadata(model, revision=sha)
    out = {}
    for f in meta.files_metadata.values():
        for n, t in f.tensors.items():
            out[n] = (t.dtype, list(t.shape))
    return out, sha


def build(model, revision):
    """The model from_pretrained builds (hf.py's classes, bf16), on the meta device."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model, revision=revision)
    with torch.device("meta"):
        try:
            return AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)


def checkpoint_names(m, tensors):
    """{model key: checkpoint key} as from_pretrained renames them, and the checkpoint keys a converter takes (a tensor
    made from others, not renamed: not planned)."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    transforms = get_model_conversion_mapping(m)
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    meta = m.state_dict()
    names, converted = {}, []
    for ck in sorted(tensors):
        mk, pattern = rename_source_key(ck, renamings, converters, getattr(m, "base_model_prefix", None), meta)
        if pattern is not None:
            converted.append(ck)
        elif mk in meta:
            if mk in names:
                raise SystemExit(f"{mk}: two checkpoint tensors become it ({names[mk]}, {ck})")
            names[mk] = ck
    return names, converted


def plan(m, tensors, lossy=False):
    """The puts save_pretrained makes for model m (glyd_gpu.model's rules), checked against the checkpoint's tensors. lossy: for the
    block layouts (i6, i5), an embedding of rows a multiple of 64 is a pack as a Linear's matrix is, not a copy."""
    from glyd_gpu import model as gm, moe
    if moe.targets(m):
        raise SystemExit("a mixture of experts: not planned here (python -m glyd_gpu pack)")
    tied = getattr(m, "all_tied_weights_keys", None) or {}
    emb = next((f"{n}.weight" for n, mod in m.named_modules() if mod is m.get_input_embeddings()), None)
    if tied and tied != {"lm_head.weight": emb}:  # (only the output layer tied to the input embedding: not saved, tied again as it loads)
        raise SystemExit(f"weights tied to others ({', '.join(sorted(tied))}): not planned here (python -m glyd_gpu pack)")
    packable = lambda w: w.dim() == 2 and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0
    linear = {name for name, mod in m.named_modules() if gm.plain(mod) and packable(mod.weight) and f"{name}.weight" not in tied}  # each a GLinear
    embedding = set()  # each a GEmbedding: saved as it is, in its module's place
    for name, mod in m.named_modules():
        if gm.lookup(mod) and mod.weight.dim() == 2 and mod.weight.shape[1] % 128 == 0:
            if hasattr(mod, "embed_scale") or hasattr(mod, "embed_norm"):
                raise SystemExit(f"{name}: an embedding with a scale or norm of its own: not planned here")
            embedding.add(name)
    paths = {id(mod): name for name, mod in m.named_modules()}
    groups = {}  # id(module): its group's member paths (merged: q, k, v; gate, up)
    for mod, names in gm.groups(m):
        members = [paths[id(getattr(mod, c))] for c in names]
        if all(p in linear for p in members) and len({getattr(mod, c).bias is None for c in names}) == 1:
            groups[id(mod)] = members
    grouped = {p for ms in groups.values() for p in ms}
    puts = []

    def walk(mod, prefix):  # named_modules' order, a group's Merged after its module's other children
        if prefix in linear and prefix not in grouped:
            puts.append({"linear": [prefix]})
        elif prefix in embedding:
            puts.append({"linear": [prefix]} if lossy and m.get_submodule(prefix).weight.shape[0] % 64 == 0 else {"copy": f"{prefix}.weight"})
        for n, c in mod.named_children():
            walk(c, f"{prefix}.{n}" if prefix else n)
        if id(mod) in groups:
            puts.append({"linear": groups[id(mod)]})

    walk(m, "")
    gone = {f"{p}.{w}" for p in linear for w in ("weight", "bias")} | {f"{p}.weight" for p in embedding} | set(tied)
    puts += [{"copy": k} for k in m.state_dict() if k not in gone]
    shapes = m.state_dict()  # (meta tensors: their shapes, the packed ones' too)
    names, converted = checkpoint_names(m, tensors)
    import torch
    keep = m._get_dtype_plan(torch.bfloat16)  # tensors transformers keeps in f32 in a bf16 model (none: each f32 one is made bf16)
    packed = {f"{x}.{w}" for p in puts if "linear" in p for x in p["linear"] for w in ("weight", "bias")}
    named = [n for p in puts for n in ([f"{x}.weight" for x in p["linear"]] + [f"{x}.bias" for x in p["linear"] if getattr(m.get_submodule(x), "bias", None) is not None] if "linear" in p else [p["copy"]])]
    problems = []
    for n in named:
        c = names.get(n)
        if c is None:
            problems.append(f"{n}: the model has it, no checkpoint tensor becomes it by a rename")
        elif tensors[c][0] != "BF16" and not (tensors[c][0] == "F32" and n not in packed and not keep):
            problems.append(f"{n}: {tensors[c][0]} in the checkpoint ({c}), where glyd packs a bf16 checkpoint")
        elif list(shapes[n].shape) != tensors[c][1]:
            problems.append(f"{n}: {tensors[c][1]} in the checkpoint ({c}), {list(shapes[n].shape)} in the model")
    if len(set(named)) != len(named):
        problems.append("a tensor planned twice")
    for p in puts:
        if "linear" in p and not problems:
            p["from"] = [names[f"{x}.weight"][: -len(".weight")] for x in p["linear"]]
            for x, f in zip(p["linear"], p["from"]):
                if f"{x}.bias" in names and names[f"{x}.bias"] != f"{f}.bias":
                    problems.append(f"{x}: its weight and bias come from two modules ({f}.weight, {names[x + '.bias']})")
        elif "copy" in p and not problems:
            p["from"] = names[p["copy"]]
            if tensors[p["from"]][0] == "F32":
                p["cast"] = "BF16"  # as transformers loads it into the bf16 model
    if problems:
        raise SystemExit("not planned:\n  " + "\n  ".join(problems[:40]) + (f"\n  ... {len(problems) - 40} more" if len(problems) > 40 else ""))
    used = {names[n] for n in named}
    unused = sorted(set(tensors) - used)
    return puts, unused, converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out", type=Path)
    ap.add_argument("--revision")
    ap.add_argument("--lossy", action="store_true", help="for a save in the i6 or i5 layout: the embedding packed too")
    args = ap.parse_args()
    import transformers
    tensors, revision = checkpoint_tensors(args.model, args.revision)
    m = build(args.model, revision)
    puts, unused, converted = plan(m, tensors, args.lossy)
    packs = sum(1 for p in puts if "linear" in p)
    dropped = {}
    for n in unused:  # by their first two name parts
        k = ".".join(n.split(".")[:2])
        dropped[k] = dropped.get(k, 0) + 1
    out = {"model": args.model, "revision": revision, "class": type(m).__name__, "transformers": transformers.__version__,
           "packs": packs, "copies": len(puts) - packs, "not_loaded": dropped, "converted_not_loaded": len(converted), "puts": puts}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "puts"}))


if __name__ == "__main__":
    sys.exit(main())
