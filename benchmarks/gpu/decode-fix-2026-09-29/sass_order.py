"""The 12-bit decode kernels' memory order in SASS (cuobjdump -sass of one architecture), with no GPU.

    python sass_order.py TREE.sass [TREE.res] ...    each mma_unpack_kernel<Nib, ...>: its registers, then its global
                                                      loads, shuffles, calls and stores by instruction index
    python sass_order.py --same REL.sass FIX.sass    every kernel of REL the same instructions in FIX (a decode kernel
                                                      of REL as FIX's ORDER 0), and FIX's kernels REL has not

In the sequences: LDG.128 the step's codes, LDG.128+0x200 and +0x400 its low bytes, the first LDG and LDG+0x4 its
exception bounds (exc_base; in the mixture-of-experts kernel, after its plan's two), later LDGs the exception entries;
SHFL a load's wait (Nib::load_decode's after()); CALL the 64-bit division's slow path (and on sm_86 the shuffle's,
where the warp is not converged)."""
import re, subprocess, sys

KIND = {"Lb0ELb0E": "whole", "Lb0ELb1E": "ahead", "Lb1ELb0E": "moe"}


def kernels(path):
    out = {}
    for chunk in open(path).read().split("Function : ")[1:]:
        name = chunk.split()[0]
        ops = [re.sub(r"\s+", " ", l.strip()) for l in chunk.splitlines() if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
        out[name] = [re.sub(r"^/\*[0-9a-f]+\*/ ", "", o).split(" ;")[0] for o in ops]
    return out


def regs(path):
    if not path:
        return {}
    text = open(path).read()
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"Function (\S+):\s*\n?\s*REG:(\d+)", text)}


def seq(ops):
    s, st = [], []
    for i, o in enumerate(ops):
        w = o.split()
        op = w[1] if w[0].startswith("@") else w[0]
        if op.startswith("STG"):
            st.append(i)
        elif op.startswith(("LDG", "SHFL", "CALL")):
            m = re.search(r"\[(R\d+)\.64(\+0x[0-9a-f]+)?\]", o)
            s.append(f"{i}:{op.split('.')[0]}{'.128' if '.128' in op else ''}{(m.group(2) or '') if m and op.startswith('LDG') else ''}")
    return s + ([f"STG x{len(st)} {st[0]}-{st[-1]}"] if st else [])


def name_of(n):
    m = re.match(r"_Z17mma_unpack_kernelI3Nib(Lb[01]ELb[01]E)(?:Li(\d)E)?E", n)
    return f"{KIND[m.group(1)]} " + (f"ORDER {m.group(2)}" if m.group(2) else "as built") if m else n


if sys.argv[1] == "--same":
    rel, fix = kernels(sys.argv[2]), kernels(sys.argv[3])
    as_fix = lambda n: re.sub(r"^(_Z17mma_unpack_kernelI(?:3Nib|6Tiered)Lb[01]ELb[01]E)(E)", r"\1Li0E\2", n)
    same = [n for n in rel if fix.get(as_fix(n), fix.get(n)) == rel[n]]
    new = sorted(set(fix) - {as_fix(n) for n in rel} - set(rel))
    print(f"{sys.argv[2]} -> {sys.argv[3]}: {len(same)} of {len(rel)} kernels the same instructions"
          + "".join(f"\n   NOT: {n}" for n in rel if n not in same) + f"; {len(new)} added:")
    for n in new:
        print(f"   {name_of(n)}: {len(fix[n])} instructions")
    sys.exit(0 if len(same) == len(rel) else 1)
args = sys.argv[1:]
while args:
    sass, res = args[0], args[1] if len(args) > 1 and args[1].endswith(".res") else None
    args = args[2:] if res else args[1:]
    r = regs(res)
    for n, ops in kernels(sass).items():
        if n.startswith("_Z17mma_unpack_kernelI3Nib"):
            print(f"{sass} {name_of(n)} ({len(ops)} instructions{f', {r[n]} registers' if n in r else ''}):")
            print("   " + " ".join(seq(ops)))
