#!/usr/bin/env python3
"""A kernel that uses more than 64 registers a thread names its block size (__launch_bounds__), or a launch of 1,024 threads fails.

    python3 scripts/check_kernel_regs.py PTXAS_OUTPUT KERNELS_DIR

PTXAS_OUTPUT is nvcc's `-Xptxas -v` output for one architecture (kernels.yml compiles sm_120 for it: consumer Blackwell's compiler
gives some kernels more registers than Ada's). A block of N threads needs N times the registers a thread uses, and an SM's register
file holds 65,536: a kernel with no __launch_bounds__ may be launched with 1,024 threads, so it keeps to 64. g4_add_norm used 80 on
sm_120 and could not be launched on an RTX 5060 Ti ("too many resources requested for launch", the M1 gate, 2026-11-21).
The bounds are read in KERNELS_DIR's .cu files and the files they include ("..."): kernels/fold.cu's include of the GPU library's
gpu/glyd_gpu_kernels.h puts the library's own kernels in the module, by their C++ names (_ZN4glib3kvc12write_kernelE...: write_kernel).
"""
import pathlib, re, sys

out, kdir = sys.argv[1], pathlib.Path(sys.argv[2])
files, todo = [], sorted(kdir.glob("*.cu"))
while todo:
    p = todo.pop().resolve()
    if p.exists() and p not in files:
        files.append(p)
        todo += [p.parent / i for i in re.findall(r'#include\s+"([^"]+)"', p.read_text())]
src = "".join(p.read_text() for p in files)
# (a bound's arguments may hold parentheses of their own: `32 * (CW + PW)`, `(Tma12<NT, 2>::THREADS)`)
BOUND = r"__launch_bounds__\((?:[^()]|\([^()]*\))*\)"
bounded = set(re.findall(r"__global__\s+void\s+" + BOUND + r"\s+(\w+)\s*\(", src))
# (kernels made by a macro: KERNEL_##N names; a macro's launch bounds cover every kernel it makes)
bounded |= set(re.findall(BOUND + r"\s+(\w+)##", src))


def plain(k):
    """A kernel's name as its source writes it: a C++ kernel's last name (_ZN4glib3kvc12write_kernelENS0_11WriteParamsE: write_kernel), an extern "C" one's own."""
    if not k.startswith("_Z"):
        return k
    i, name = 3 if k.startswith("_ZN") else 2, k
    while i < len(k) and k[i].isdigit():
        n = re.match(r"\d+", k[i:]).group()
        i += len(n)
        name, i = k[i:i + int(n)], i + int(n)
    return name


regs, cur = {}, None
for line in open(out):
    m = re.search(r"Compiling entry function '([^']+)'", line)
    if m:
        cur = m.group(1)
        continue
    m = re.search(r"Used (\d+) registers", line)
    if m and cur:
        regs[cur] = int(m.group(1))
        cur = None
bad = [(k, n) for k, n in sorted(regs.items()) if n > 64 and not any(plain(k).startswith(b) for b in bounded)]
for k, n in bad:
    print(f"{k}: {n} registers a thread with no __launch_bounds__ (at most 64: a block of 1,024 threads needs 65,536)")
print(f"{len(regs)} kernels, {len(bad)} past 64 registers without __launch_bounds__")
sys.exit(1 if bad else 0)
