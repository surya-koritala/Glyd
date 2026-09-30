# The route SPLIT's decode, measured alone (2026-09-29)

Excerpts of the research branch's logs (research-kernels, 10c3142) that the route SPLIT's decode kernel and docs cite:
the decode alone, weights a clock an SM on s SMs, its configuration `sb ldg D1 NT1 w4` (split byte, loads a unit ahead,
4 warps: glyd_gpu.cu's mma12_split_kernel).

| file | what |
| :--- | :--- |
| `round1/results/a100-sxm4-40gb-followup/` | A100-SXM4-40GB: 9.58 on 16 SMs (summary.txt, a.txt) |
| `round1/results/h100-sxm-followup/` | H100 SXM: 10.24 on 16 SMs |
| `round1/results/h100-pcie/` | H100 PCIe: 10.39 on 16 SMs |
| `box/dec2.txt` | RTX 4080 SUPER (box/machine.txt): main's whole-matrix decode (mma12_unpack) on 2-4 SMs, 2.70-2.72 |
