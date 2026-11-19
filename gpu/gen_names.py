#!/usr/bin/env python3
"""The rename header of a release library: every kernel, device function, device variable, type and namespace of
glyd_gpu.cu as a #define to a neutral name (gk_N kernels, gd_N device functions, gv_N device variables, gt_N types and
namespaces), so that the library's device code and the names its host code registers for it say nothing of what they do.
A development build does not include the header and keeps the readable names.

    python3 gen_names.py glyd_gpu.cu OUT.h [MAP]

MAP (a private file, never shipped: where it is kept is the caller's choice) has a line for each name, `new old kind`, for
reading a profile of a release build. The numbering is by kind and name, so the same source gives the same header.

What it reads is the source's own declarations (comments and strings taken out): `__global__` functions, `__device__`
and `__host__ __device__` functions that are not members of a type (also where a macro defined as `__device__` qualifies
them), device and shared variables of a namespace, and every struct, class, union, enum, alias and namespace. KEEP names
what stays (a name a CUDA header or the C++ library owns, or that is the C API's). A kernel made by a function-like macro
that pastes its name (`#define M(N) __global__ void k_##N(...)`, then `M(1) M(2)`) is read from each use of the macro, written out
(expand_pasted): the header's #define of the pasted name then renames it, since a pasted token is rescanned for macros. The release
gate (check_wheel.py) looks at the names the machine code has."""
import re
import sys

KEEP = {"main", "std", "cuda", "nvcuda", "kernel", "int", "T", "St", "Fmt", "x", "y", "n"}  # (Fmt, St: template parameter names, not entities)
ID = re.compile(r"[A-Za-z_]\w*")


def expand_pasted(src):
    """Every use of a function-like macro whose body declares a `__global__` with a pasted (##) name, written out: the macro's
    parameters replaced by the use's arguments and the ## joins made. Only what names() needs to find the kernels; nvcc still
    expands the macros itself."""
    out = []
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)((?:[^\n]*\\\n)*[^\n]*)", src, re.M):
        name, params, body = m.group(1), [p.strip() for p in m.group(2).split(",")], m.group(3).replace("\\\n", "\n")
        if "##" not in body or "__global__" not in body:
            continue
        for use in re.finditer(r"^[ \t]*" + name + r"\(([^()\n]*)\)", src, re.M):
            args = [a.strip() for a in use.group(1).split(",")]
            if len(args) != len(params):
                continue
            text = body
            for p, a in zip(params, args):
                text = re.sub(r"\b" + re.escape(p) + r"\b", a, text)
            out.append(re.sub(r"\s*##\s*", "", text))
    return "\n" + "\n".join(out) + "\n"


def blank(src):
    """src with comments, string and character literals replaced by spaces (newlines kept)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(re.sub(r"[^\n]", " ", src[i : j + 1]))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def skip_parens(s, i):
    """The index after the parenthesis group that starts at s[i]."""
    depth = 0
    while i < len(s):
        depth += (s[i] == "(") - (s[i] == ")")
        i += 1
        if depth == 0:
            return i
    return i


def block_kinds(s):
    """For each index of s that is an `{`: whether it opens a type's body ('type'), a namespace ('ns') or anything else ('other')."""
    kinds, stack, last = {}, [], 0
    for m in re.finditer(r"[{};]", s):
        i = m.start()
        if s[i] == "{":
            head = s[last:i]
            if re.search(r"\b(?:struct|class|union)\b[^;(){}]*$", head) and "=" not in head.rsplit("struct", 1)[-1].rsplit("class", 1)[-1]:
                kind = "type"
            elif re.search(r"\bnamespace\b[^;(){}]*$", head) or re.search(r'extern\s*"?"?\s*$', head):
                kind = "ns"
            elif re.search(r"\benum\b[^;(){}]*$", head):
                kind = "enum"
            else:
                kind = "other"
            kinds[i] = kind
            stack.append(kind)
            last = i + 1
        elif s[i] == "}":
            if stack:
                stack.pop()
            last = i + 1
        else:
            last = i + 1
    return kinds


def innermost(s, kinds, pos):
    """The kind of the block that holds position pos ('file' outside every block)."""
    stack = []
    for m in re.finditer(r"[{}]", s[:pos]):
        if m.group() == "{":
            stack.append(kinds.get(m.start(), "other"))
        elif stack:
            stack.pop()
    return stack[-1] if stack else "file"


def names(src):
    """{kind: sorted names} of the entities the header renames."""
    s = blank(src)
    kinds = block_kinds(s)
    found = {"k": set(), "d": set(), "v": set(), "t": set()}
    # kernels
    def skip_ws(i):
        while s[i].isspace():
            i += 1
        return i

    for m in re.finditer(r"__global__\b", s):
        i = m.end()
        while True:
            i = skip_ws(i)
            w = ID.match(s, i)
            if w and w.group() == "void":
                i = w.end()
            elif w and w.group() in ("__launch_bounds__", "__maxnreg__"):
                i = skip_parens(s, skip_ws(w.end()))
            else:
                break
        w = ID.match(s, i)
        if w and s[w.end() :].lstrip().startswith("("):
            found["k"].add(w.group())
    # device functions and variables (a macro that stands for __device__, `#define GEMM_HD __host__ __device__ __forceinline__`, qualifies as it does)
    qualifiers = [r"__device__\b", r"__constant__\b", r"__managed__\b", r"__shared__\b"] + [rf"\b{q}\b" for q in re.findall(r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+[^\n]*\b__device__\b", s, re.M)]
    for m in re.finditer("|".join(qualifiers), s):
        if s[s.rfind("\n", 0, m.start()) + 1 : m.start()].lstrip().startswith("#"):
            continue  # (the definition of the macro is no declaration)
        where = innermost(s, kinds, m.start())
        if where in ("type", "other", "enum"):
            continue  # (a member, or a local: `__shared__` arrays inside a kernel stay)
        if m.group() == "__shared__" and where != "ns" and where != "file":
            continue
        j = m.end()
        stop = re.compile(r"[(;={]").search(s, j)
        if not stop:
            continue
        decl = s[j : stop.start()]
        # the name: the last identifier before the delimiter (and before any `[`)
        decl = re.sub(r"__align__\s*\([^)]*\)", " ", decl)
        decl = decl.split("[")[0]
        ids = ID.findall(decl)
        if not ids:
            continue
        name = ids[-1]
        if stop.group() == "(":
            found["d"].add(name)
        elif m.group() != "__shared__" or where in ("ns", "file"):
            found["v"].add(name)
    # types and namespaces (not an alias a function declares for itself)
    for m in re.finditer(r"\b(?:struct|class|union)\s+(?:alignas\s*\([^)]*\)\s*)?(\w+)\s*(?=[:{;<])|\benum\s+(?:class\s+|struct\s+)?(\w+)|\bnamespace\s+(\w+)|\busing\s+(\w+)\s*=|\btypedef\b[^;{}]*?\b(\w+)\s*;", s):
        if innermost(s, kinds, m.start()) in ("other", "enum"):
            continue
        found["t"].add(next(g for g in m.groups() if g))
    out = {}
    for kind, vals in found.items():
        out[kind] = sorted(v for v in vals if len(v) > 1 and v not in KEEP and not v.startswith(("__", "glyd_gpu_", "GLYD_")) and not re.fullmatch(r"u?int\d+_t|size_t|cudaStream_t", v))
    return out


def header(found, prefixes=("gk", "gd", "gv", "gt")):
    lines = ["/* Generated by gen_names.py: the release library's names. Not in a development build; never shipped as source. */", "#ifndef GLYD_GPU_NAMES_H", "#define GLYD_GPU_NAMES_H"]
    table = []
    for kind, pre in zip("kdvt", prefixes):
        for n, name in enumerate(found[kind]):
            new = f"{pre}_{n}"
            lines.append(f"#define {name} {new}")
            table.append((new, name, kind))
    lines.append("#endif")
    return "\n".join(lines) + "\n", table


if __name__ == "__main__":
    src = open(sys.argv[1], encoding="utf-8").read()
    found = names(src + expand_pasted(src))
    text, table = header(found)
    open(sys.argv[2], "w").write(text)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write("".join(f"{new} {old} {kind}\n" for new, old, kind in table))
    print({k: len(v) for k, v in found.items()}, file=sys.stderr)
