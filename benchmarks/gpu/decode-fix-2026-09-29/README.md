# The 12-bit decode kernel's load order, 2026-09-29

v0.25.0's decode of a whole matrix (for cuBLAS and exact mode: `mma_unpack_kernel<Nib>`) takes 3.0-5.8% longer than the
12-bit layout's before split byte on an H100 SXM and 1.0-1.6% on an A10 (benchmarks/gpu/splitbyte-2026-09-29, rc/). In
SASS for sm_90a its kernel issues a step's three loads together (the codes, the low bytes, the exception bounds),
where main's issued the low bytes once the codes were in. 73b9560 made the order a choice, `GLYD_DEC_ORDER` (read at
each call of the decode; unset: 1), each order the same bytes decoded:

- 0: v0.25.0's kernel.
- 1: the codes and the bounds, then the low bytes once those are in: main's order on Hopper.
- 2: the codes and the bounds, then the low bytes once the high bytes are patched (as late as can be).
- 3: the low bytes and the bounds, then the codes once those are in.

A load waits on words through a shuffle of their sum from its own lane less the sum (Nib::load_decode's after()),
and the row block's division follows the step's last load, as before split byte.

On a GH200 (gh200/) order 3 was the fastest whole-matrix decode and order 0 the fastest decode ahead; on an L4 (l4/)
order 0 was the fastest of both. So v0.25.1 fixes the order in the code and drops the choice (no GLYD_DEC_ORDER, no
orders 1 and 2): the whole-matrix decode (a warp a step: for cuBLAS, exact mode, and the experts' decode) in order 3's
loads on Hopper (f83fae3 on every GPU, then dc490e4 on Hopper alone), the decode ahead (a few warps an SM) and every
other GPU's decode in v0.25.0's. Its build (sass-final.txt) has l4-routes' kernels but two on sm_90a, the whole and
experts' 12-bit decodes, each 73b9560's order 3's instructions; on sm_80, 86, 89, 100 and 120 all 87 are l4-routes'.

| file | what |
| :--- | :--- |
| `sass.sh`, `sass.txt`, `sass_order.py` | with no GPU (NVIDIA's `cuda:13.0.3-devel-rockylinux8` image, arm64, GCC 13): main's, v0.25.0's and 73b9560's glyd_gpu.cu for sm_90a and sm_86 with build_lib.sh's flags; v0.25.0's kernels against 73b9560's, and each 12-bit decode kernel's loads, shuffles, calls and stores in order, with its registers |
| `sass_final.sh`, `sass-final.txt` | v0.25.1's glyd_gpu.cu against l4-routes' (1f4343b) and 73b9560's for sm_80, 86, 89, 90a, 100 and 120: `sass_order.py --final` (f83fae3's, then dc490e4's) |
| `dec_time.py MAIN_TREE MAIN_LIB REL_LIB MODEL [--final LIB]` | layer 10's matrices (q, k, v and gate, up merged) decoded by main's library, v0.25.0's, 73b9560's orders and (--final) v0.25.1's in one process: each variant's output the weights' bits, then each call timed alone after an L2 flush, the median of 21 (whole: a warp a step; ahead: 2 warps an SM) |
| `dec_e2e.py MODEL` | a model loaded as a user loads it, eager: exact mode's steps and long prompts (exact, then fused), the orders in turn in one process (or a library as it is), each output's sha256 |
| `dec_prof.py` | for Nsight Compute: one matrix decoded once by each variant |
| `h100_dec.sh`, `dec_summary.py` | the unattended job (Hopper; an A10 by the same script): builds, the self-test and xcheck.py with each order, dec_time.py, the profile, dec_e2e.py, and the summary |
| `gh200/` | h100_dec.sh on a GH200 (Lambda, 132 SMs, aarch64 Grace host; 73b9560, bfb2c4e and db8e7b0): summary.txt; respond/: bf16 Qwen3-8B, eager and compiled, in the same session (respond.py, branch respond-bench) |
| `l4_dec.sh`, `l4/`, `l4-head/` | the AWS dev L4 (sm_89). l4/ (fd24666: order 3 on every GPU): the self-test, xcheck.py, check_capi.py, test_gpu.py and cargo test; dec_time.py with --final; dec_e2e.py, Qwen3-8B exact (the GPU the bound) and Qwen3-0.6B exact steps (the host the bound): l4/summary.txt, l4/host/summary.txt. l4-head/ (dc490e4, the release's code): the library as build_lib.sh builds it, the same checks, dec_time.py |

## A GH200 (gh200/)

Every order's decodes the weights' bits and main's (the self-test and xcheck.py with each order; dec_time.py's checks;
dec_e2e.py's 49 sha256 lines, 7 outputs, the same in every variant and tree). The decode of layer 10 (q, k, v and gate,
up merged), each over main's (the 12-bit layout before split byte), the median of 21, two runs:

| model | decode | main, us | v0.25.0 | order 0 | order 1 | order 2 | order 3 |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| Qwen3-8B | whole | 283-285 | 1.035-1.037 | 1.035-1.040 | 1.037-1.041 | 1.070-1.074 | 0.964-0.975 |
| Qwen3-14B | whole | 513-514 | 1.051-1.052 | 1.051-1.052 | 1.013-1.014 | 1.060-1.061 | 0.970 |
| Qwen3-32B | whole | 723 | 1.059-1.060 | 1.060 | 1.016-1.017 | 1.063-1.065 | 0.966 |
| Qwen3-8B | ahead | 905-906 | 0.725 | 0.725 | 0.963-0.965 | 0.998 | 1.067-1.068 |
| Qwen3-14B | ahead | 1517-1518 | 0.707 | 0.707 | 0.971-0.972 | 0.990-0.991 | 1.059 |
| Qwen3-32B | ahead | 2221-2222 | 0.703-0.704 | 0.704 | 0.973-0.974 | 0.988 | 1.058 |

Order 3's whole decode moves 2.32-2.46 TB/s (the pack read and the bf16 matrix written), v0.25.0's 2.14-2.30. The
decode ahead (two warps an SM) is the reverse: all three loads at once the fastest, order 3 the slowest.

End to end (e2e-*.txt; Qwen3-8B loaded as a user loads it, eager), exact mode's steps did not tell the decodes apart:
73b9560's four orders, in one process, took 52.84-53.02 ms a token though their whole decodes differ by 10% (orders 2
and 3), main's library 54.87 and 54.75 (two processes, before and after), v0.25.0's 56.31. Eager `generate()` on the
GH200 is bound by its host (the Grace CPU): the GPU's work a token is about 15 ms (the decode's 36 × 0.28 ms as timed
above, the products' about 5 by their bytes), and in the same session bf16 itself, eager, took 42 ms a token at batch 1
against 8.3 compiled (respond.py: gh200/respond/). So a step's time is the host's time for the same Python calls, and
that differed between processes: order 0 runs v0.25.0's kernel (the same instructions) through v0.25.0's Python (the
same files), yet 53.02 ms against 56.31. The same offsets show in the 1280-token exact prompts (0.965-0.976 of main's
in 73b9560's process, 1.032 in v0.25.0's), where the host's and the GPU's times are about even, and none at 2048-4096
tokens or in fused prompts, where the GPU is the bound (within 1% throughout). The decode's own time is the kernel
table's.

## An L4 (l4/, l4-head/)

The AWS dev L4 (g6.4xlarge: sm_89, 58 SMs, 72 W; AMD EPYC 7R13, 16 vCPUs). l4/: fd24666 (order 3 in every GPU's
whole-matrix decode) beside 73b9560's orders, v0.25.0 and main; l4-head/: dc490e4 (order 3 on Hopper alone), the
release's code. Every decode the weights' bits at both: the self-test, and xcheck.py's 3205 calls the same bits as
main's library (the whole and experts' decodes among them); every variant's and tree's sha256 lines the same (21 lines,
3 outputs; 14 and 1 in host/). check_capi.py (also with GLYD_DEC_MIN 1000 and 3000), test_gpu.py and the crate's tests
pass at both. (l4/'s job, as first written, waited on its own nvidia-smi after the builds; the rest ran with
L4DEC_REUSE=1, and job.txt holds both. l4-head/: L4DEC_TREE=final2, the steps builds to kernel.)

The decode of layer 10, each over main's, the median of 21, two runs:

| model | decode | main, us | v0.25.0 | order 0 | order 1 | order 2 | order 3 |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| Qwen3-8B | whole | 2923-2925 | 1.001-1.002 | 1.001-1.002 | 1.015-1.016 | 1.014-1.015 | 1.020-1.021 |
| Qwen3-4B-Instruct-2507 | whole | 1561-1562 | 1.003-1.004 | 1.003 | 1.017-1.020 | 1.018-1.019 | 1.019-1.022 |
| Qwen3-8B | ahead | 3012-3015 | 0.994 | 0.994 | 1.101-1.102 | 1.099 | 1.106 |
| Qwen3-4B-Instruct-2507 | ahead | 1637-1638 | 0.992-0.994 | 0.992-0.994 | 1.124-1.125 | 1.128-1.129 | 1.132-1.134 |

Here the whole decode runs at 74-77% of the L4's 300 GB/s, and order 3 took 1.9-2.2% longer than all three at once (fd24666's library as built: 1.020-1.022; its decode ahead, v0.25.0's loads,
0.994-0.995). So v0.25.1 loads the low bytes first on Hopper alone: dc490e4's kernels for sm_89 are l4-routes' and
v0.25.0's instructions (sass-final.txt): on the L4 its whole decode took 1.002-1.003 of main's (v0.25.0's 1.001,
order 3's 1.019-1.021 in the same run), its decode ahead 0.993-0.994 (l4-head/).

End to end, exact mode on Qwen3-8B, the GPU the bound (about 155 ms a token): the four orders in one process took
155.3-156.5 ms a token (order 0 155.3, order 3 155.8), 1024-token prompts 423.8-426.1 ms and 4096-token 1555-1559;
main's, v0.25.0's and fd24666's libraries in their own processes 155.0, 154.9 and 155.5 ms a token, 421.6-424.1 and
1544-1548 ms: all within about 1%, order 3 against order 0 +0.3% in the steps.

The host the bound (host/): Qwen3-0.6B's exact steps, 64 tokens, the median of 5, in eight processes in turn. 73b9560's
four orders took 39.39-39.44 ms a token in the first process and 40.51-40.57 in the fifth; main's library 40.09 and
40.53, v0.25.0's 40.50 and 40.34, fd24666's 40.45 and 40.08. Within a process the orders' steps are within 0.15%, their
decodes 2% apart; the same code in two processes 2.8% apart. The GH200's exact steps are this case, their spread wider
there (6%): 0.966 against 1.026 compares processes, not kernels.

## SASS (sass.txt, sass-final.txt; nvcc 13.0, 2026-09-29)

- sass-final.txt: v0.25.1's glyd_gpu.cu against l4-routes' (1f4343b) and 73b9560's, for sm_80, 86, 89, 90a, 100 and
  120. dc490e4: on sm_90a the whole and experts' 12-bit decodes are 73b9560's order 3's instructions and the other 85
  kernels l4-routes'; on the other five architectures all 87 kernels are l4-routes'. (f83fae3, order 3 on every GPU:
  those two changed on every architecture, each order 3's.)

- v0.25.0's 87 kernels: the same instructions in 73b9560's build, on sm_90a and sm_86; 9 kernels added (orders 1-3
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
