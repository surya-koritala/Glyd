#!/usr/bin/env python3
"""A kernel that uses more than 64 registers a thread names its block size (__launch_bounds__), or a launch of 1,024 threads fails.

    python3 scripts/check_kernel_regs.py PTXAS_OUTPUT KERNELS_DIR

PTXAS_OUTPUT is nvcc's `-Xptxas -v` output for one architecture (kernels.yml compiles sm_120 for it: consumer Blackwell's compiler
gives some kernels more registers than Ada's). A block of N threads needs N times the registers a thread uses, and an SM's register
file holds 65,536: a kernel with no __launch_bounds__ may be launched with 1,024 threads, so it keeps to 64. g4_add_norm used 80 on
sm_120 and could not be launched on an RTX 5060 Ti ("too many resources requested for launch", the M1 gate, 2026-11-21).
"""
import pathlib, re, sys

out, kdir = sys.argv[1], pathlib.Path(sys.argv[2])
src = "".join(p.read_text() for p in sorted(kdir.glob("*.cu")))
bounded = set(re.findall(r"__global__\s+void\s+__launch_bounds__\([^)]*\)\s+(\w+)\s*\(", src))
# (kernels made by a macro: KERNEL_##N names; a macro's launch bounds cover every kernel it makes)
bounded |= {m for m in re.findall(r"__launch_bounds__\([^)]*\)\s+(\w+)##", src)}
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
bad = [(k, n) for k, n in sorted(regs.items()) if n > 64 and k not in bounded and not any(k.startswith(b) for b in bounded)]
for k, n in bad:
    print(f"{k}: {n} registers a thread with no __launch_bounds__ (at most 64: a block of 1,024 threads needs 65,536)")
print(f"{len(regs)} kernels, {len(bad)} past 64 registers without __launch_bounds__")
sys.exit(1 if bad else 0)
