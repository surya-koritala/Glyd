"""The sweep's table from one or more sweep.tsv files (one row a model): Glyd's and bf16's weights on the GPU, the saving, the KV cache
each left, the answers, and why a server did not start.
    python3 sweep_table.py LIST.txt sweep.tsv [sweep.tsv ...]      (models in LIST's order; models with no row are listed as not run)"""
import csv
import json
import sys


def rows(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                out.setdefault(r["model"], {})[r["server"]] = r
    return out


def ans(r):
    try:
        a = json.loads(r["answer"])
    except (ValueError, KeyError, TypeError):
        return "?"
    if not a.get("ok"):
        return "no answer"
    return ("Paris" if a.get("paris") else "answered") + ("" if a.get("how") == "chat" else f" ({a['how']})")


def short(err):
    e = err or ""
    if "ValueError: glyd:" in e:
        return "Glyd's plugin refused: " + e.split("ValueError: glyd:", 1)[1].split("(APIServer")[0].strip()[:200]
    for key, say in (("AmbiguousGlobalPerLayerAttributeError", "Glyd's plugin stops on Gemma 4's per-layer config (transformers' AmbiguousGlobalPerLayerAttributeError)"),
                     ("are not supported for now", "vLLM 0.30.0 has no class for this architecture"),
                     ("trust_remote_code", "needs the maker's remote code (not run)"),
                     ("out of memory", "out of GPU memory"), ("OutOfMemory", "out of GPU memory"),
                     ("larger than the GPU", "the checkpoint is larger than the GPU"), ("gated", "gated on Hugging Face")):
        if key in e:
            return say
    return e[-160:].replace("|", "/")


def main():
    order = [l.split()[0] for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    data = rows(sys.argv[2:])
    print("| Model | Glyd weights GiB | bf16 weights GiB | Saved | Glyd KV tokens | bf16 KV tokens | Glyd answer | bf16 answer | Note |")
    print("| :-- | --: | --: | --: | --: | --: | :-- | :-- | :-- |")
    for m in order:
        d = data.get(m)
        if not d:
            print(f"| {m} | | | | | | | | not run |")
            continue
        g = d.get("glyd-2k") if d.get("glyd-2k", {}).get("started") == "up" else d.get("glyd", {})
        b = d.get("bf16-2k") if d.get("bf16-2k", {}).get("started") == "up" else d.get("bf16", {})
        gw, bw = g.get("weights_gib") or "", b.get("weights_gib") or ""
        saved = f"{100 * (1 - float(gw) / float(bw)):.1f}%" if gw and bw and g.get("started") == "up" and b.get("started") == "up" else ""
        notes = []
        if g and g.get("started") != "up":
            notes.append("Glyd: " + short(g.get("error")))
        if b and b.get("started") != "up":
            notes.append("bf16: " + short(b.get("error")))
        if "glyd-2k" in d or "bf16-2k" in d:
            notes.append("2,048-token window (the model's own is shorter than 8,192)")
        if d.get("download"):
            notes.append("download failed: " + short(d["download"].get("error")))
        gl = g.get("glyd_lines") or ""
        if "experts kept bf16" in gl:
            notes.append("experts kept bf16")
        print(f"| {m} | {gw if g.get('started') == 'up' else ''} | {bw if b.get('started') == 'up' else ''} | {saved} | {g.get('kv_tokens') or ''} | "
              f"{b.get('kv_tokens') or ''} | {ans(g) if g.get('started') == 'up' else ''} | {ans(b) if b.get('started') == 'up' else ''} | {'; '.join(notes)} |")


if __name__ == "__main__":
    main()
