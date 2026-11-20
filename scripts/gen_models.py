#!/usr/bin/env python3
"""Write the site's models page from the engine's capability table.

    python3 scripts/gen_models.py [OUT.md]        (default: docs/guide/src/models.md)

Reads glyd-engine/src/support.rs (the table of families), glyd-engine/src/levels.rs (the levels) and
glyd-engine/src/machine.rs (the machine side). Nothing on the page is typed by hand except the column
words. It stops with an error where a source no longer has the shape it reads, so a refactor cannot
leave a page that is quietly wrong: fix this script in the same change.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "glyd-engine" / "src"


def die(msg):
    sys.exit(f"gen_models.py: {msg}")


def read(name):
    return (SRC / name).read_text()


def const_int(text, name):
    m = re.search(rf"pub const {name}: usize = ([0-9_]+);", text)
    if not m:
        die(f"no `pub const {name}: usize` in the source")
    return int(m.group(1).replace("_", ""))


def str_list(expr, consts):
    """`&["a", "b"]` or the name of a const slice."""
    expr = expr.strip()
    if expr in consts:
        return consts[expr]
    m = re.fullmatch(r"&\[(.*)\]", expr, re.S)
    if not m:
        die(f"cannot read the list `{expr}`")
    return re.findall(r'"([^"]*)"', m.group(1))


def parse_support(text):
    consts = {}
    for m in re.finditer(r"^const (\w+): &\[&str\] = (.+?);$", text, re.M):
        consts[m.group(1)] = re.findall(r'"([^"]*)"', m.group(2))
    default_ctx = const_int(text, "DEFAULT_CTX")
    lossy_ctx = const_int(text, "LOSSY_CTX")
    m = re.search(r"pub static TABLE: \[Support; (\d+)\] = \[(.*?)\n\];", text, re.S)
    if not m:
        die("no `pub static TABLE: [Support; N]` in support.rs")
    want, body = int(m.group(1)), m.group(2)
    rows = []
    for blk in re.findall(r"Support \{(.*?)\n    \},", body, re.S):
        def field(name):
            f = re.search(rf"\b{name}: (.+?),\n", blk + "\n")
            if not f:
                die(f"a row has no `{name}`")
            return f.group(1).strip()

        rows.append(
            {
                "name": field("name").strip('"'),
                "levels": str_list(field("levels"), consts),
                "wren": field("wren").split("::")[-1],
                "hb_packed": field("hummingbird_packed_only") == "true",
                "default_ctx": field("default_ctx"),
                "max_ctx": field("max_ctx"),
                "split": str_list(field("split_levels"), consts),
                "ram": str_list(field("experts_in_ram_levels"), consts),
            }
        )
    if len(rows) != want:
        die(f"TABLE says {want} rows, {len(rows)} were read")
    return rows, default_ctx, lossy_ctx


def parse_levels(text):
    levels = re.findall(r'Level \{ name: "(\w+)", layout: [^,]+(?:\([^)]*\))?, bits: "([^"]+)", lossless: (true|false) \}', text)
    m = re.search(r"pub static LEVELS: \[Level; (\d+)\]", text)
    if not m or int(m.group(1)) != len(levels):
        die("LEVELS in levels.rs was not read in full")
    return [(n, b, l == "true") for n, b, l in levels]


def n(x):
    return f"{x:,}"


def ctx_words(r, default_ctx, lossy_ctx, max_ctx):
    d = r["default_ctx"]
    if d.startswith("Some("):
        inner = d[5:-1]
        d = n({"DEFAULT_CTX": default_ctx, "MAX_CONTEXT": max_ctx}.get(inner) or int(inner))
        return f"{d} tokens"
    if not r["levels"]:
        return "-"
    return f"what fits, up to {n(lossy_ctx)}"


def levels_words(r):
    out = []
    for l in r["levels"]:
        if l == "wren":
            out.append("wren" + {"No": "", "DevBuild": " (dev build)", "QatOrDevBuild": " (QAT checkpoint, or dev build)"}[r["wren"]])
        elif l == "hummingbird" and r["hb_packed"]:
            out.append("hummingbird (pre-packed)")
        else:
            out.append(l)
    return ", ".join(out) if out else "none: [Condor](condor.md) runs it"


def where(ls):
    return ", ".join(ls) if ls else "no"


def commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "guide" / "src" / "models.md"
    sup = read("support.rs")
    rows, default_ctx, lossy_ctx = parse_support(sup)
    levels = parse_levels(read("levels.rs"))
    mach = read("machine.rs")
    drv = const_int(mach.replace("pub const DRIVER_MIN: i32", "pub const DRIVER_MIN: usize"), "DRIVER_MIN")
    archs = re.search(r"pub const ARCHS: &\[&str\] = &\[(.*?)\];", mach)
    gpus = re.search(r"pub const GPUS: &\[\(&str, \(i32, i32\)\)\] = &\[(.*?)\n\];", mach, re.S)
    if not archs or not gpus:
        die("ARCHS or GPUS not found in machine.rs")
    arch_list = re.findall(r'"([^"]+)"', archs.group(1))
    gpu_list = re.findall(r'\("([^"]+)", \((\d+), (\d+)\)\)', gpus.group(1))
    kv = read("kvcal.rs")
    tiers = re.search(r"pub const TIERS: \[usize; \d+\] = \[([0-9_, ]+)\]", kv)
    if not tiers:
        die("TIERS not found in kvcal.rs")
    max_ctx = int(tiers.group(1).split(",")[-1].replace("_", "").strip())
    archs_of = {}
    for pat, fam in re.findall(r'^\s*((?:"\w+" \| )*"\w+") => Ok\(Family::(\w+)\)', sup, re.M):
        for a in re.findall(r'"(\w+)"', pat):
            archs_of[a] = fam
    if "Qwen3" not in archs_of.values() or "Qwen35" not in archs_of.values():
        die("the architecture names of Qwen3 and Qwen3.5 were not found in support.rs `family`")

    o = []
    o.append("# Models\n")
    o.append(
        "This page is generated from `glyd-engine/src/support.rs`, `levels.rs` and `machine.rs` by "
        f"`scripts/gen_models.py` at commit `{commit()}` of main. It is the table `glyd-engine fit`, `run` and "
        "`serve` read, so it says what they do. Version `" + version() + "`.\n"
    )
    o.append("## Families the engine runs\n")
    o.append("| Family | Levels | Context where none is asked | Weights split over GPUs at | Experts in RAM at |")
    o.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        o.append(f"| {r['name']} | {levels_words(r)} | {ctx_words(r, default_ctx, lossy_ctx, max_ctx)} | {where(r['split'])} | {where(r['ram'])} |")
    o.append("")
    o.append(
        "- The engine tells a family apart by its `config.json`, not by its name. Qwen3 dense is "
        + ", ".join(f"`{a}`" for a, f in archs_of.items() if f == "Qwen3")
        + "; Qwen3.5, 3.6 and 3.8 dense are "
        + ", ".join(f"`{a}`" for a, f in archs_of.items() if f == "Qwen35")
        + ". Any other kind is refused with exit 3 and the reason."
    )
    o.append("- `wren` runs in a development build, which is what `cargo build --release -p glyd-engine` makes. A release build leaves it out, except for a quantization-aware checkpoint, where it is exact.")
    o.append("- A level a family does not list is refused: `glyd-engine run Qwen/Qwen3-8B:swift` stops and says Qwen3 runs at penguin.")
    o.append(f"- The context is at most {n(max_ctx)} tokens (the KV tables' largest tier), and never more than the model's own limit.")
    o.append("- `glyd-engine supports MODEL` answers for one model: exit 0 where the engine runs its kind, 3 where it does not.")
    o.append("")
    o.append("## Levels\n")
    o.append("| Level | Bits a weight | Lossless |")
    o.append("| :--- | :--- | :--- |")
    for name, bits, lossless in levels:
        o.append(f"| {name} | {bits} | {'yes' if lossless else 'no'} |")
    o.append("")
    o.append("What each level costs in quality and memory: [Choosing a level](levels.md).\n")
    o.append("## The machine\n")
    o.append(f"- NVIDIA, compute capability 8.0 or newer (Ampere and later), a driver for CUDA {drv // 1000} (driver 580 or newer).")
    o.append("- Computer architecture: " + ", ".join(arch_list) + ".")
    o.append("- `glyd run` hands a model to the engine by default only on the GPUs below, which are the ones the engine was measured on against vLLM (`machine.rs`, `GPUS`). `glyd-engine` itself runs on any GPU that fits the first rule.")
    for name, a, b in gpu_list:
        o.append(f"  - {name} (compute capability {a}.{b})")
    o.append("- AMD GPUs run in development builds only (`GLYD_HIP=1`). Apple GPUs and a machine with no GPU are refused with the reason. Nothing is built or tested for Windows or macOS.")
    o.append("")
    text = "\n".join(o) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(rows)} families, {len(levels)} levels)")


def version():
    m = re.search(r'^version = "([^"]+)"', (ROOT / "glyd-engine" / "Cargo.toml").read_text(), re.M)
    return m.group(1) if m else "unknown"


if __name__ == "__main__":
    main()
