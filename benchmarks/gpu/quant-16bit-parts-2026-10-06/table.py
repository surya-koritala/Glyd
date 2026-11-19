"""The record's table from quant_parts.py's summary.json files: python table.py GPU_LABEL RESULTS_DIR [GPU_LABEL RESULTS_DIR ...]"""
import glob
import json
import os
import sys


def tps(r, n):
    return ((r.get("tps") or {}).get(n) or {}).get("median")


def exact_cell(ex):
    if not ex:
        return "not run"
    if ex["equal"]:
        return f"bit for bit ({ex['verified']} of {ex['packed']} packs verified{', + image' if ex.get('image') else ''})"
    selfd = ex.get("stock_self_diffs")
    if isinstance(selfd, dict) and any(selfd.values()):
        return "undecided: stock eager differs from itself"
    return "DIFFERS"


rows = []
args = sys.argv[1:]
for gpu, d in zip(args[::2], args[1::2]):
    for f in sorted(glob.glob(os.path.join(d, "*", "summary.json"))):
        s = json.load(open(f))
        st, g, inv = s.get("stock") or {}, s.get("glyd") or {}, s.get("inventory") or {}
        w0, w1 = st.get("weights_gib"), g.get("weights_gib")
        saved = f"{w0 - w1:.2f} ({(w0 - w1) / w0 * 100:.1f}%)" if w0 and w1 else "-"
        cell = lambda n: (f"{tps(st, n) or '-'} -> {tps(g, n) or '-'}" + (f" ({tps(g, n) / tps(st, n):.2f}x)" if tps(st, n) and tps(g, n) else ""))
        users = f"{st.get('concurrency', '-')} -> {g.get('concurrency', '-')}"
        rows.append([s["model"], gpu, st.get("quant_config") or g.get("quant_config") or "-", str(inv.get("packed_count", "-")),
                     f"{inv['packed_bf16_bytes'] / 2**30:.2f} -> {inv['packed_pack_bytes'] / 2**30:.2f}" if inv else "-",
                     f"{w0 or ('did not start' if st.get('error') else '-')} -> {w1 or ('did not start' if g.get('error') else '-')}", saved,
                     f"{st.get('kv_tokens', '-'):,} -> {g.get('kv_tokens', '-'):,}" if st.get("kv_tokens") and g.get("kv_tokens") else f"{st.get('kv_tokens', '-')} -> {g.get('kv_tokens', '-')}",
                     users, exact_cell(s.get("exact")), cell("1"), cell("32")])
hdr = ["Checkpoint", "GPU", "vLLM's config", "Layers packed", "Packed GiB (bf16 -> Glyd)", "Weights GiB (stock -> Glyd)", "Saved GiB",
       "KV tokens (stock -> Glyd)", "8,192-token users (stock -> Glyd)", "Glyd exact vs stock eager", "1 user tok/s", "32 users tok/s"]
print("| " + " | ".join(hdr) + " |")
print("| " + " | ".join([":--"] * 3 + ["--:"] * 6 + [":--"] + ["--:"] * 2) + " |")
for r in rows:
    print("| " + " | ".join(str(x) for x in r) + " |")
