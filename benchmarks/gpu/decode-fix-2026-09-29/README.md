# The 12-bit decode kernel's load order, 2026-09-29

v0.25.0's decode of a whole matrix (for cuBLAS and exact mode: `mma_unpack_kernel<Nib>`) takes 3.0-5.8% longer than the
12-bit layout's before split byte on an H100 SXM and 1.0-1.6% on an A10 (benchmarks/gpu/splitbyte-2026-09-29, rc/). In
SASS for sm_90a its kernel issues a step's three loads together (the codes, the low bytes, the exception bounds),
where main's issued the low bytes once the codes were in. This tree makes the order a choice, `GLYD_DEC_ORDER` (read
at each call of the decode; unset: 1), each order the same bytes decoded:

- 0: v0.25.0's kernel.
- 1: the codes and the bounds, then the low bytes once those are in: main's order on Hopper.
- 2: the codes and the bounds, then the low bytes once the high bytes are patched (as late as can be).
- 3: the low bytes and the bounds, then the codes once those are in.

A load waits on words through a shuffle of their sum from its own lane less the sum (Nib::load_decode's after()),
and the row block's division follows the step's last load, as before split byte.

| file | what |
| :--- | :--- |
| `sass.sh`, `sass.txt`, `sass_order.py` | with no GPU (NVIDIA's `cuda:13.0.3-devel-rockylinux8` image, arm64, GCC 13): main's, v0.25.0's and this tree's glyd_gpu.cu for sm_90a and sm_86 with build_lib.sh's flags; v0.25.0's kernels against this tree's, and each 12-bit decode kernel's loads, shuffles, calls and stores in order, with its registers |
| `dec_time.py MAIN_TREE MAIN_LIB REL_LIB MODEL` | layer 10's matrices (q, k, v and gate, up merged) decoded by main's library, v0.25.0's and this tree's orders in one process: each variant's output the weights' bits, then each call timed alone after an L2 flush, the median of 21 (whole: a warp a step; ahead: 2 warps an SM) |
| `dec_e2e.py MODEL` | a model loaded as a user loads it, eager: exact mode's steps and long prompts (exact, then fused), the orders in turn in one process (or a library as it is), each output's sha256 |
| `dec_prof.py` | for Nsight Compute: one matrix decoded once by each variant |
| `h100_dec.sh`, `dec_summary.py` | the unattended job (Hopper; an A10 by the same script): builds, the self-test and xcheck.py with each order, dec_time.py, the profile, dec_e2e.py, and the summary |

## SASS (sass.txt; nvcc 13.0, 2026-09-29)

- v0.25.0's 87 kernels: the same instructions in this tree's build, on sm_90a and sm_86; 9 kernels added (orders 1-3
  of the whole-matrix, decode-ahead and mixture-of-experts decodes).
- The whole-matrix decode, sm_90a, by instruction index (codes: LDG.128; low bytes: LDG.128+0x200, +0x400; bounds: the
  first LDG, LDG+0x4; the 64-bit division: CALL):

  | kernel | loads in order | registers |
  | :-- | :-- | :-- |
  | main | codes 47, bounds 51-52, low bytes 85-86 (the codes' first use at 53), division 120 | 40 |
  | v0.25.0, order 0 | codes 46, low bytes 50-52, bounds 53-54, division 79 | 40 |
  | order 1 | bounds 48-50, codes 51, wait 54, low bytes 61-62, division 88 | 40 |
  | order 2 | codes 49, bounds 50-51, exception entries 89-140, wait 271, low bytes 280-281, division 287 | 32 |
  | order 3 | bounds 48-50, low bytes 51-52, wait 55, codes 63, division 89 | 38 |

  sm_86 alike, but main's own decode there issues its loads together (44-52), as v0.25.0's does: on an A10 the load
  order does not tell the two apart. The decode-ahead and mixture-of-experts kernels follow the same orders. Order 2's
  32 registers let an H100 SM hold 64 warps of it against 48 of the others (256-thread blocks): its time carries that.
