# The revised 12-bit layout against the one before it, 2026-09-29

The checks for the revised 12-bit layout: main's 12-bit layout (origin/main, its package and library) and this
tree's, on any GPU. Main's tree and library sit beside this tree's (`git archive origin/main`, each built by
`gpu/build_lib.sh`); the scripts load both libraries into one process where they compare (`both.py`: main's through a
second copy of `glyd.gpu._lib` at main's C API version, 4, with the same functions and arguments as this tree's 5, and
main's `pack_mma12` taken from its `kernels.py`).

| script | what |
| :--- | :--- |
| `xcheck.py MAIN_TREE MAIN_LIB [MODEL ...]` (`SYNTHETIC=0`: the models alone) | every 12-bit entry point the GPU takes, each output the same bits as main's (a call refused by one refused by both), and checked against the weights or an fp32 product: the self-test's and check_capi's matrices, every bf16 bit pattern, mixtures of experts, and every Linear weight of each MODEL unpacked bit for bit in both layouts (their sizes) with layers 0, 1, 2, the middle one and the last multiplied at 1-1100 tokens |
| `e2e12.py MODEL ...` | a model loaded in the 12-bit layout (default mode, then exact), its logits for prompts of 1, 17, 64, 300 and 2100 tokens and 32 greedy tokens, as sha256 lines: run with main's package and library and with this tree's, the lines compared |
| `layer.py MAIN_TREE MAIN_LIB MODEL` | one layer's products (q, k, v and gate, up merged) and unpacks, main's 12-bit layout against this tree's, each call alone after an L2 flush, the median of 15 |
| `sass_diff.py MAIN.sass TREE.sass [MAIN.res TREE.res]` | the two libraries' disassembly for one architecture (`cuobjdump -sass -arch sm_XX`) compared |
| `emulate.py`, `emulate.txt` | on the CPU (numpy 2.0.2, Python 3.9), no GPU: `pack_mma12` ported line by line and the kernels' unpacking written out as they run, checked against the operands built straight from the weights, bit for bit: 10 matrices, every bf16 bit pattern among them |
| `container.sh`, `container.txt`, `sass-main-to-branch.txt` | with no GPU (Docker on a Mac, NVIDIA's `cuda:13.0.3-devel-rockylinux8` image for arm64, the release's build image, with GCC 13 as the release builds): main's library and this tree's built for sm_80, 86, 89 and 90a with build_lib.sh's flags and their disassembly compared by `sass_diff.py`; and this tree's JIT source compiled as PyTorch 2.14's extension build compiles it (the headers of its CUDA 13 wheel) and linked with every symbol resolved |
| `run.sh` | all of the above on the RTX 4080 SUPER box, one step at a time, with the self-test, check_capi, test_gpu.py and check_api dense and MoE, and xcheck.py again through builds for sm_89 that take the prompt paths an A10 and an L40S take |
| `jobs/` | the unattended jobs that took round1/ and rc/: `r1_job.sh` (the steps; `r1_summary.py` writes summary.txt), a wrapper for each GPU class (`r1_ada.sh`, `r1_a10.sh`, `r1_a100.sh`, `r1_hopper.sh`; `r1_ada_rerun.sh` the L4's rerun; the `_rc.sh` ones the release candidate's) and `run-e2e.sh` (rc/l4-e2e/). The tree and main's went with them as tars (`git archive` of the release's tree and of origin/main, with a COMMIT file), every command as it ran |

## With no GPU (container.txt, sass-main-to-branch.txt; nvcc 13.0, aa7a6fc, 2026-09-29)

- Both libraries compile and link with no warning; the JIT source compiles with none of its own (only PyTorch's
  headers' `module` remarks) and links with every symbol resolved; its code is the library's, kernel for kernel (87 on
  each of the four).
- The 48 kernels not of the 12-bit layout: the same instructions in the same order, on each architecture.
- Fewer instructions than main in every kernel of the 12-bit layout built for its architecture.

## Round 1 (round1/; 2026-09-29, the release's tree before aa7a6fc: 975bbb1 on the L4, 8a7769e elsewhere)

The 12-bit layout steps of the release's round-1 jobs (jobs/r1_job.sh), their logs as they came back: an L4 on AWS (l4/,
and l4-rerun/: the MoE compare fixed in 8270a46 and the attribution), an A10 (a10/), an A100 SXM4 40 GB
(a100-sxm4-40gb/) and an H100 PCIe (h100-pcie/). Each dir's summary.txt holds its layer.py ratios. A summary's
"decoder test" line (A10, A100, H100 PCIe: INCOMPLETE; in rc/, the A100 and H100 SXM: FAIL) is another experiment's
gate, run in the same session; its logs are not here and it is not part of these checks.

- Every xcheck.py all the same bits (3052-4817 calls a run; the L4's first granite run failed on the harness's MoE
  compare, fixed in 8270a46), every self-test and e2e12.py's lines main's (20 logits and 4 generations each).
- This tree's time over main's for one layer: the H100 0.942-0.991 at 32-1024 tokens, the A100 0.923-1.000 at 32-128
  and 0.975-0.989 at 256-768 tokens, and 0.989-1.006 at 1-16 tokens on all four; the A10's and L4's prompts
  1.007-1.032, their unpack for cuBLAS 0.998-1.023. The A10 and L4 prompt rows are of a build that aa7a6fc changed (the
  attribution runs are `attr-*`); the release candidate's run has the final ones.

## The release candidate (rc/; 85ae892, 2026-09-29)

The same steps on the final tree: an L4 on AWS (l4/), an A10 (a10/), an A100 SXM4 40 GB (a100-sxm4-40gb/) and an H100
SXM (h100-sxm/), each summary.txt with its ratios; xcheck and the self-test on the A100 and the H100 (all the same
bits: 3052-5517 calls a run).

- Faster: the H100 SXM 0.933-0.969 at 32-1024 tokens and 0.986-0.992 at 1-16 tokens; the A100 0.923-0.987 at 64-128
  tokens (0.985-1.003 at 32), 0.969-0.982 at 256-768 and 0.976-0.998 at 1-16.
- The same: the A10 and the L4 at 1-1024 tokens, 0.989-1.010. The A10's rows at 640 and 1024 tokens time a kernel that
  glyd.gpu does not run there, so those prompts' time is not measured.
- Slower: unpacking a whole matrix (for cuBLAS and exact mode) 1.030-1.058 on the H100 SXM and 1.010-1.016 on the A10;
  the A100 0.998-1.012, the L4 0.999-1.002. Round 1's 1.021-1.023 on the A10 and 1.013-1.017 on the A100 were of the
  earlier build. Counting the packed weights read and the bf16 matrix written, the unpack rows move 1.87-1.99 TB/s
  before this layout and 1.79-1.93 after on the H100 SXM (53-59% of 3.35), 226-231 GB/s on the L4 either way (75-77% of
  300).

## Logits and greedy tokens on the release candidate (rc/l4-e2e/; 85ae892 against main db8e7b0, 2026-09-29)

jobs/run-e2e.sh on an L4 (AWS g6.4xlarge): r1_job.sh's step (g) alone, e2e12.py with main's package and library, then the
release candidate's, then main's again, on Qwen3-1.7B, Qwen3-4B-Instruct-2507 and granite-3.1-3b-a800m-instruct from
the machine's cache, offline (job.txt's "no Qwen/..." lines: the job's download check wants each snapshot's README and
LICENSE too, which that cache lacks; e2e12.py loads the weights by name). Main twice the same lines, and the release
candidate's main's: 30 logits (1, 17, 64, 300 and 2100 tokens, default and exact) and 6 generations of 32 tokens.
