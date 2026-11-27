#!/usr/bin/env python3
"""Write the site's models page from the engine's capability table.

    python3 scripts/gen_models.py [OUT.md]        (default: docs/guide/src/models.md)

Reads glyd-engine/src/support.rs (the table of families), levels.rs (the levels), machine.rs (the machine side), kvcal.rs and plan.rs
(two constants), runtime/config.rs (`family_by`: which config.json is which family) and runtime/describe.rs (`LLAMA`). Nothing on the page is
typed by hand except the column words. The Rust is read as a literal: comments are dropped, a struct's fields are read by name (so any
order, any extra field, any line breaks), and the script stops with an error where a source no longer has the shape it reads, so a refactor
cannot leave a page that is quietly wrong: fix this script in the same change (scripts/check_tree.sh runs it on every pull request).
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "glyd-engine" / "src"
OPEN, CLOSE = "([{", ")]}"


def die(msg):
    sys.exit(f"gen_models.py: {msg}")


def read(name):
    return (SRC / name).read_text()


class Rust:
    """One source file as text with its comments blanked (`text`) and, of that, a copy with the inside of every string blanked too (`mask`), the same length:
    a span found by brackets and commas in `mask` is the span in `text`."""

    def __init__(self, source, start=0):
        # (the text before `start` is not lexed: a quote of an earlier item's cannot confuse the item that is read)
        src, n, i, text, mask = source[start:], len(source) - start, 0, [], []
        while i < n:
            if src.startswith("//", i):
                j = src.find("\n", i)
                j = n if j < 0 else j
                text.append(" " * (j - i))
                mask.append(" " * (j - i))
            elif src.startswith("/*", i):
                j = src.find("*/", i + 2)
                j = n if j < 0 else j + 2
                text.append(re.sub(r"[^\n]", " ", src[i:j]))
                mask.append(text[-1])
            elif src[i] == '"':
                j = i + 1
                while j < n and src[j] != '"':
                    j += 2 if src[j] == "\\" else 1
                j = min(j + 1, n)
                text.append(src[i:j])
                mask.append('"' + "_" * (j - i - 2) + '"')
            elif m := re.match(r"'(?:\\.|[^\\'])'", src[i : i + 4]):
                j = i + m.end()
                text.append(src[i:j])
                mask.append("'_'")
            else:
                j = i + 1
                text.append(src[i])
                mask.append(src[i])
            i = j
        self.text, self.mask = "".join(text), "".join(mask)

    @classmethod
    def after(cls, source, header, what):
        """The source from the first match of the regex `header` on, lexed; (Rust, end of the match in it)."""
        m = re.search(header, source)
        if not m:
            die(f"no {what} in the source")
        r = cls(source, m.start())
        return r, m.end() - m.start()

    def close(self, i):
        """The index of the bracket that closes the one at `i`."""
        depth = 0
        for j in range(i, len(self.mask)):
            c = self.mask[j]
            depth += (c in OPEN) - (c in CLOSE)
            if depth == 0:
                return j
        die("a bracket is never closed")

    def items(self, lo, hi):
        """The comma-separated items between `lo` and `hi`, as (start, end) spans at bracket depth 0, none empty."""
        spans, depth, a = [], 0, lo
        for j in range(lo, hi + 1):
            c = self.mask[j] if j < hi else ","
            depth += (c in OPEN) - (c in CLOSE)
            if c == "," and depth == 0:
                if self.text[a:j].strip():
                    spans.append((a, j))
                a = j + 1
        return spans

    def fields(self, lo, hi):
        """`name: value` items of a struct literal's body: {name: value text}; a `..base` item is left out."""
        out = {}
        for a, b in self.items(lo, hi):
            k, colon, v = self.text[a:b].strip().partition(":")
            if colon and not k.startswith(".."):
                out[k.strip()] = v.strip()
        return out

    def struct(self, a, b):
        """The fields of the struct literal `Name { ... }` in the span (a, b)."""
        i = self.mask.find("{", a, b)
        if i < 0:
            die(f"no struct literal in `{self.text[a:b].strip()[:60]}`")
        return self.fields(i + 1, self.close(i))


def array(source, name):
    """(type text, Rust, spans of the items) of `static|const NAME: TYPE = [&][ ITEMS ];`."""
    r, k = Rust.after(source, rf"(?:pub )?(?:static|const) {name}: ([^=]+?) = &?\[", f"`static|const {name}: [...] = [...]`")
    typ = re.match(r"(?:pub )?(?:static|const) \w+: ([^=]+?) = ", r.text).group(1)
    return typ, r, r.items(k, r.close(k - 1))


def int_of(text, what):
    t = text.strip().replace("_", "")
    if not re.fullmatch(r"\d+", t):
        die(f"cannot read {what}: `{text.strip()}`")
    return int(t)


def const_int(source, name):
    m = re.search(rf"pub const {name}: \w+ = ([0-9_]+);", source)
    if not m:
        die(f"no `pub const {name}` of a number in the source")
    return int_of(m.group(1), name)


def strings(text):
    return re.findall(r'"([^"]*)"', text)


def str_consts(source):
    """The `const NAME: &[&str] = &[...]` lists of a file: {name: [strings]}."""
    out = {}
    for m in re.finditer(r"^(?:pub )?const (\w+): &\[&str\] = &?\[", source, re.M):
        _, r, spans = array(source, m.group(1))
        out[m.group(1)] = [s for a, b in spans for s in strings(r.text[a:b])]
    return out


def str_list(expr, consts, what):
    """`&["a", "b"]` or the name of a const list."""
    expr = expr.strip()
    if expr in consts:
        return consts[expr]
    if re.fullmatch(r"&\[[^\]]*\]", expr, re.S):
        return strings(expr)
    die(f"cannot read the list `{expr}` ({what})")


def option(expr):
    """None -> None; `Some(x)` -> x."""
    expr = expr.strip()
    if expr == "None":
        return None
    m = re.fullmatch(r"Some\((.*)\)", expr, re.S)
    if not m:
        die(f"cannot read the option `{expr}`")
    return m.group(1).strip()


def parse_support(text, plan_text, max_ctx):
    consts = str_consts(text)
    default_ctx = const_int(text, "DEFAULT_CTX")
    # (the lossy paths' most context is the planner's constant, which support.rs re-exports)
    lossy_ctx = const_int(plan_text, "LOSSY_CTX")
    typ, r, spans = array(text, "TABLE")
    m = re.fullmatch(r"\[Support; (\d+)\]", typ)
    if not m:
        die("the table is no longer `pub static TABLE: [Support; N]` in support.rs")
    rows = []
    for a, b in spans:
        f = r.struct(a, b)
        for key in ("family", "name", "levels", "hummingbird_packed_only", "default_ctx", "max_ctx", "split_levels", "experts_in_ram_levels"):
            if key not in f:
                die(f"a row of the table in support.rs has no `{key}`")
        row = {
            "family": f["family"].split("::")[-1],
            "name": f["name"].strip('"'),
            "levels": str_list(f["levels"], consts, "levels"),
            "hb_packed": f["hummingbird_packed_only"] == "true",
            "split": str_list(f["split_levels"], consts, "split_levels"),
            "ram": str_list(f["experts_in_ram_levels"], consts, "experts_in_ram_levels"),
        }
        for key in ("default_ctx", "max_ctx"):
            v = option(f[key])
            row[key] = None if v is None else {"DEFAULT_CTX": default_ctx, "MAX_CONTEXT": max_ctx, "LOSSY_CTX": lossy_ctx}.get(v) or int_of(v, f"{row['name']}'s {key}")
        rows.append(row)
    if len(rows) != int(m.group(1)):
        die(f"the table says {m.group(1)} rows, {len(rows)} were read")
    return rows, lossy_ctx


def parse_levels(text):
    typ, r, spans = array(text, "LEVELS")
    levels = []
    for a, b in spans:
        f = r.struct(a, b)
        for key in ("name", "bits", "lossless"):
            if key not in f:
                die(f"a level in levels.rs has no `{key}`")
        levels.append((f["name"].strip('"'), f["bits"].strip('"'), f["lossless"] == "true"))
    m = re.fullmatch(r"\[Level; (\d+)\]", typ)
    if not m or int(m.group(1)) != len(levels):
        die("LEVELS in levels.rs was not read in full")
    return levels


def parse_family_by(config_text, llama):
    """{family variant: (kind, [names])} of runtime/config.rs `family_by`: the architectures (`a`) or model types (`m`) a config.json names."""
    r, k = Rust.after(config_text, r"fn family_by\(", "`fn family_by` (runtime/config.rs)")
    body = r.mask.find("match () {")
    if body < 0:
        die("`family_by` in runtime/config.rs is no longer a `match () {` of guards")
    lo = body + len("match () ")
    hi, out, a, i = r.close(lo), {}, lo + 1, lo + 1
    while True:
        j = r.mask.find("=>", a, hi)
        if j < 0:
            break
        guard = r.text[a:j]
        i = j + 2
        while r.mask[i] in " \n":
            i += 1
        end = r.close(i) + 1 if r.mask[i] == "{" else next(iter(r.items(i, hi)), (i, hi))[1]
        fam = re.search(r"Family::(\w+)", r.text[i:end])
        if fam:
            if "LLAMA" in guard:
                out[fam.group(1)] = ("model type", llama)
            elif m := re.search(r"\b([am])\b\s*==|matches!\(\s*([am])\s*,", guard):
                out[fam.group(1)] = ("architecture" if (m.group(1) or m.group(2)) == "a" else "model type", strings(guard))
        a = end
        while a < hi and r.mask[a] in ", \n":
            a += 1
    return out


def n(x):
    return f"{x:,}"


def ctx_words(r, lossy_ctx):
    d = r["default_ctx"]
    if d is not None:
        return f"{n(d)} tokens"
    if not r["levels"]:
        return "-"
    return f"what fits, up to {n(lossy_ctx)}"


def levels_words(r):
    out = []
    for l in r["levels"]:
        if l == "hummingbird" and r["hb_packed"]:
            out.append("hummingbird (pre-packed)")
        else:
            out.append(l)
    return ", ".join(out) if out else "none"


def where(ls):
    return ", ".join(ls) if ls else "no"


def levels_phrase(ls):
    """`the penguin and hummingbird levels`: how a refusal (support.rs `Support::refusal`) names the levels a family runs."""
    return f"the {ls[0]} level" if len(ls) == 1 else f"the {', '.join(ls[:-1])} and {ls[-1]} levels"


def drivers(mach):
    """(CUDA of the kernels of CUDA 13 alone, [(CUDA of a build with CUDA 12's too, its driver series, what it is for)]) of machine.rs `driver_min_for`, `driver_series`."""
    series = [(int_of(a, "driver_series"), int_of(b, "driver_series")) for a, b in re.findall(r"(\d+)\.\.\s*=>\s*(\d+)", re.search(r"fn driver_series\(.*?\n\}", mach, re.S).group(0))]
    of = lambda cuda: next(s for c, s in series if cuda >= c)
    body = re.search(r"fn driver_min_for\(.*?\n\}", mach, re.S)
    versions = [int_of(v, "driver_min_for") for v in re.findall(r"\{\s*(\d{5})\s*\}", body.group(0))] if body else []
    if len(versions) != 2:
        die("`driver_min_for` in machine.rs does not give two CUDA 12 versions (Blackwell's, the others') any more")
    v13 = const_int(mach, "DRIVER_MIN")
    return (v13, of(v13)), [(v, of(v)) for v in versions]


def commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def version():
    m = re.search(r'^version = "([^"]+)"', (ROOT / "glyd-engine" / "Cargo.toml").read_text(), re.M)
    return m.group(1) if m else "unknown"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "guide" / "src" / "models.md"
    kv = read("kvcal.rs")
    tiers = re.search(r"pub const TIERS: \[usize; \d+\] = \[([0-9_, ]+)\]", kv)
    if not tiers:
        die("TIERS not found in kvcal.rs")
    max_ctx = int_of(tiers.group(1).strip().rstrip(",").split(",")[-1], "the last of TIERS")
    rows, lossy_ctx = parse_support(read("support.rs"), read("plan.rs"), max_ctx)
    levels = parse_levels(read("levels.rs"))
    names = {r["family"]: r["name"] for r in rows}

    mach = read("machine.rs")
    archs = array(mach, "ARCHS")
    arch_list = [s for a, b in archs[2] for s in strings(archs[1].text[a:b])]
    (v13, s13), cuda12 = drivers(mach)
    gpus = array(mach, "GPUS")
    gpu_list = []
    for a, b in gpus[2]:
        tup = gpus[1].text[a:b]
        m = re.fullmatch(r'\s*\(\s*"([^"]+)"\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*&\[([^\]]*)\]\s*\)\s*', tup)
        if not m:
            die(f"cannot read the line of GPUS `{tup.strip()}`")
        gpu_list.append((m.group(1), m.group(2), m.group(3), strings(m.group(4))))

    llama = str_consts(read("runtime/describe.rs")).get("LLAMA")
    told = parse_family_by(read("runtime/config.rs"), llama or [])
    for fam in ("Qwen3", "Qwen35"):
        if not told.get(fam, (0, []))[1] or fam not in names:
            die(f"the architecture or model type names of Family::{fam} were not found in runtime/config.rs `family_by` (and the table's rows)")
    what = lambda fam: ", ".join(f"`{x}`" for x in told[fam][1])

    flash = next((r for r in rows if r["name"] == "GLM-5.3-Flash"), None)
    unlisted = [l for l, _, _ in levels if flash and l not in flash["levels"]]
    refused = "swift" if "swift" in unlisted else next(iter(unlisted), None)
    if not flash or not refused:
        die("no row GLM-5.3-Flash, or it runs every level: the example refusal of the page has no case")
    capped = [r for r in rows if r["max_ctx"] is not None]

    o = []
    o.append("# Models\n")
    o.append(
        "This page is generated from `glyd-engine/src/support.rs`, `levels.rs` and `machine.rs` by "
        f"`scripts/gen_models.py` at commit `{commit()}`. It is the table `glyd fit`, `run` and "
        "`serve` read, so it says what they do. Version `" + version() + "`.\n"
    )
    o.append("## Families the engine runs\n")
    o.append("| Family | Levels | Context where none is asked | Weights split over GPUs at | Experts in RAM at |")
    o.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        o.append(f"| {r['name']} | {levels_words(r)} | {ctx_words(r, lossy_ctx)} | {where(r['split'])} | {where(r['ram'])} |")
    o.append("")
    o.append(
        f"- The engine tells a family apart by its `config.json`, not by its name. The {names['Qwen3']} kind is every model whose {told['Qwen3'][0]} is {what('Qwen3')}"
        f" (Qwen3 dense among them), whatever the release calls its architecture; the {names['Qwen35']} layout (3.5, 3.6 and 3.8, dense or with experts) is the {told['Qwen35'][0]}s "
        f"{what('Qwen35')}. Any other kind is refused with exit 3 and the reason."
    )
    o.append("- Every build runs every level a family lists, wren included (since v0.34.0). On a quantization-aware checkpoint (Google's `-qat-` releases) wren is exact: the release's own four-bit weights.")
    o.append(
        f"- A level a family does not list is refused: `glyd run zai-org/GLM-5.3-Flash:{refused}` stops and says GLM-5.3-Flash runs at {levels_phrase(flash['levels'])}."
    )
    o.append(
        "- The context is never more than the model's own limit"
        + (
            "; "
            + ", ".join(f"{r['name']} takes at most {n(r['max_ctx'])} tokens" + (" (the KV tables' largest tier)" if r["max_ctx"] == max_ctx else "") for r in capped)
            + ", and no other family has a limit of its own."
            if capped
            else ", and no family has a limit of its own."
        )
    )
    o.append("- `glyd supports MODEL` answers for one model: exit 0 where the engine runs its kind, 3 where it does not.")
    o.append("")
    o.append("## Levels\n")
    o.append("| Level | Bits a weight | Lossless |")
    o.append("| :--- | :--- | :--- |")
    for name, bits, lossless in levels:
        o.append(f"| {name} | {bits} | {'yes' if lossless else 'no'} |")
    o.append("")
    o.append("What each level costs in quality and memory: [Choosing a level](levels.md).\n")
    o.append("## The machine\n")
    blackwell, other = sorted(cuda12, reverse=True)[0], sorted(cuda12)[0]
    o.append(
        f"- NVIDIA, compute capability 8.0 or newer (Ampere and later), a driver for CUDA {v13 // 1000} (driver {s13} or newer); "
        f"a build that carries CUDA 12's kernels too runs on CUDA {other[0] // 1000}.{other[0] % 1000 // 10} (driver {other[1]} or newer) and, on Blackwell, "
        f"CUDA {blackwell[0] // 1000}.{blackwell[0] % 1000 // 10} (driver {blackwell[1]})."
    )
    o.append("- Computer architecture: " + ", ".join(arch_list) + ".")
    o.append(
        "- Where vLLM is installed, `glyd run` hands a model to the engine by default only on the GPUs below, which are the (GPU, model family) pairs the engine was measured on against vLLM "
        "(`machine.rs`, `GPUS`; the lossy levels are the engine's wherever it runs). With the Rust install alone, or when the engine is asked for by name, it runs on every GPU "
        "that fits the first rule."
    )
    for name, a, b, fams in gpu_list:
        o.append(f"  - {name} (compute capability {a}.{b}): " + ", ".join(fams) + " models")
    o.append("- AMD GPUs run in development builds only (`GLYD_HIP=1`). Apple GPUs and a machine with no GPU are refused with the reason. Windows x86_64 is a preview ([Install](install.md)).")
    o.append("")
    text = "\n".join(o) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(rows)} families, {len(levels)} levels)")


if __name__ == "__main__":
    main()
