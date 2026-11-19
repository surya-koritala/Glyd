"""A family read automatically from vLLM 0.30.0's model file for an architecture classify.py has no hand-read entry for: whether
its VocabParallelEmbedding and ParallelLMHead calls pass a quant config, whether it maps mtp. to None, and its towers' prefixes.
Marked "auto" in the inventory (not hand-checked)."""
import os, re

V = os.environ.get("VLLM_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm030", "vllm-0.30.0")  # vLLM 0.30.0 sdist, unpacked
REG = open(os.path.join(V, "vllm/model_executor/models/registry.py")).read()


def module_file(arch):
    m = re.search(r'"%s":\s*\(\s*"([^"]+)",' % re.escape(arch), REG)
    if not m:
        return None
    mod = m.group(1)
    if mod.startswith("vllm."):
        base = os.path.join(V, *mod.split("."))
        for p in (base + ".py", os.path.join(base, "nvidia", "model.py"), os.path.join(base, "__init__.py")):
            if os.path.exists(p):
                return p
        return None
    p = os.path.join(V, "vllm/model_executor/models", mod + ".py")
    return p if os.path.exists(p) else None


def calls(src, name):
    """The argument text of each call name( ... ) in src."""
    out = []
    for m in re.finditer(r"\b%s\(" % name, src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.end(): i - 1])
    return out


def sources(path, depth=1):
    src = open(path).read()
    yield path, src
    if depth:
        for m in re.finditer(r"^from \.(\w+) import", src, re.M):
            p = os.path.join(os.path.dirname(path), m.group(1) + ".py")
            if os.path.exists(p):
                yield from sources(p, depth - 1)


def family(arch, text_arch=None):
    m = re.search(r'"%s":\s*\(\s*"transformers",' % re.escape(arch), REG)
    if m:  # the Transformers backend (transformers/utils.py replace_embedding_class, causal.py): both with the quant config
        return dict(embed_qc=True, head_qc=True, drop=[], drop_extra_layers=True, towers=[], auto="vllm/model_executor/models/transformers/ (Transformers backend)")
    path = module_file(arch)
    if path is None:
        return None
    emb = head = None
    if text_arch and text_arch != arch:
        t = family(text_arch)
        if t:
            emb, head = t["embed_qc"], t["head_qc"]
    for p, src in sources(path):
        e = calls(src, "VocabParallelEmbedding") + calls(src, "make_input_embedding")
        h = calls(src, "ParallelLMHead")
        if emb is None and e:
            emb = "quant_config" in e[0]
        if head is None and h:
            head = "quant_config" in h[0]
    src = open(path).read()
    drop = [r"^mtp\.", r"(^|\.)mtp\."] if re.search(r'"mtp\.":\s*None|mtp\.\"?\s*in name|startswith\("mtp', src) else []
    towers = [(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder|vit|multi_modal_projector|mm_projector|vision_projection)\.", "vision", "quant_config" in src),
              (r"(^|\.)(audio_tower|audio_encoder|audio|sound_encoder)\.", "audio", "quant_config" in src)]
    return dict(embed_qc=bool(emb), head_qc=bool(head) if head is not None else True, drop=drop, drop_extra_layers=True,
                towers=towers, tower_plain=[r"pos_embed", r"patch_embed", r"position_embedding"], auto=os.path.relpath(path, V))


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(a, family(a))
