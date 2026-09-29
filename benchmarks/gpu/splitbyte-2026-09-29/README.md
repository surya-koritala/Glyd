# Split byte against the 12-bit layout before it, 2026-09-29

The checks for the 12-bit layout in split byte: main's 12-bit layout (origin/main, its package and library) and this
tree's in the same kernels, on any GPU. Main's tree and library sit beside this tree's (`git archive origin/main`, each
built by `gpu/build_lib.sh`); the scripts load both libraries into one process where they compare (`both.py`: main's
through a second copy of `glyd.gpu._lib`, the C API being version 4 in both, and main's `pack_mma12` taken from its
`kernels.py`).

| script | what |
| :--- | :--- |
| `xcheck.py MAIN_TREE MAIN_LIB [MODEL ...]` (`SYNTHETIC=0`: the models alone) | every 12-bit entry point the GPU takes, each output of split byte the same bits as main's (a call refused by one refused by both), and checked against the weights or an fp32 product: the self-test's and check_capi's matrices, every bf16 bit pattern, mixtures of experts, and every Linear weight of each MODEL decoded bit for bit in both layouts (their sizes) with layers 0, 1, 2, the middle one and the last multiplied at 1-1100 tokens |
| `e2e12.py MODEL ...` | a model loaded in the 12-bit layout (fused, then exact), its logits for prompts of 1, 17, 64, 300 and 2100 tokens and 32 greedy tokens, as sha256 lines: run with main's package and library and with this tree's, the lines compared |
| `layer.py MAIN_TREE MAIN_LIB MODEL` | one layer's products (q, k, v and gate, up merged) and decodes, main's 12-bit layout against split byte, by the library's route and by each kernel, each call alone after an L2 flush, the median of 15 |
| `sass_diff.py MAIN.sass TREE.sass [MAIN.res TREE.res]` | the two libraries' SASS for one architecture (`cuobjdump -sass -arch sm_XX`): the 12-bit layout's kernels by kind of instruction, registers and spills, and whether their memory, tensor-core, barrier and branch instructions are the same (each opcode with its modifiers, as many of each; ptxas orders them around the arithmetic); every other kernel the same instructions in the same order |
| `emulate.py`, `emulate.txt` | on the CPU (numpy 2.0.2, Python 3.9), no GPU: `pack_mma12` ported line by line, its bytes against the layout `glyd_gpu.h` writes out, and the kernels' decodes written out as they run (`Nib::decode`: the step, prompt, mixture-of-experts, unpack and A100 mid kernels; `decode12_rows`: the mid and TMA kernels; `stage12_exc` and `step12`: Hopper's wgp kernel) against the mma fragments built straight from W, bit for bit: 10 matrices, every bf16 bit pattern among them, the base at 0 and at 120, 0-94% of the weights exceptions |
| `container.sh`, `container.txt`, `sass-*.txt` | with no GPU (Docker on a Mac, NVIDIA's `cuda:13.0.3-devel-rockylinux8` image for arm64, the release's build image, with GCC 13 as the release builds): main's library, this tree's, and main's again with its exception loop kept to an entry a pass as this tree's is (`main-once`, the baseline for the schedule), each for sm_80, 86, 89 and 90a with build_lib.sh's flags; their SASS compared by `sass_diff.py` (main -> this tree, main -> main-once, main-once -> this tree); and this tree's JIT source compiled as PyTorch 2.14's extension build compiles it (the headers of its CUDA 13 wheel) and linked with every symbol resolved |
| `run.sh` | all of the above on the RTX 4080 SUPER box, one step at a time under its lock, with the self-test, check_capi, test_gpu.py and check_api dense and MoE, and xcheck.py again through builds for sm_89 whose test for GeForce by name never matches (the grid prompt kernels, as an A10 and an L40S take them) |

## With no GPU (container.txt, sass-*.txt; nvcc 13.0, 2026-09-29)

Run on 7bbfb94, 873b7ed before its message was reworded (the same tree).

- The three libraries compile and link with no warning; the JIT source compiles with none of its own (only PyTorch's
  headers' `module` remarks) and links with every symbol resolved; its SASS is the library's, kernel for kernel (87 on
  each of the four).
- The 48 kernels not of the 12-bit layout: the same instructions in the same order, on each architecture.
- The 39 of the 12-bit layout against main-once (`sass-main-once-to-branch.txt`): the same memory, tensor-core, barrier
  and branch instructions in 34, 35, 35 and 32 (sm_80, 86, 89, 90a), the Hopper TMA and wgp kernels, the A100's mid
  kernel and the step, prompt, stream-K and decode kernels among them (registers: the same in 112 of the 156, fewer in
  40, by up to 17, and more in 4, by up to 4: sm_90a's mid kernel for 32 tokens, which Hopper does not run); the rest
  differ by one call (sm_80, 86, 89: mma_moe_kernel's 2-token tiles and mma_gemm_kernel<Nib, 2>) or one BSSY/BSYNC
  pair (sm_90a: mma_moe_kernel), or in spills where no GPU of the architecture runs the kernel (mma12_mid_kernel<64,
  4> on sm_80, 13 to 20 spill instructions, where the A100 takes mma12_ws_kernel; on sm_90a the mid kernel 38 to 36
  and the A100's prompt kernel 19 to 0). Fewer instructions in each of the 132 built for their architecture (the TMA
  and wgp kernels are sm_90a's alone, stubs elsewhere): in the TMA kernel for 128 tokens 4981 to 4809, the wgp
  kernel 6979 to 6929, the step kernel 2254-2279 to 2116-2190, the decode kernel 421-425 to 357-364.
- main -> main-once alone (`sass-main-to-main-once.txt`) changes the memory and branch instructions of 25 of the 39:
  nvcc had unrolled the exception loop by 2 or 4 a pass; main -> this tree (`sass-main-to-branch.txt`) likewise.

The GPU scripts have not run yet: the box reset (2026-09-29 08:48 EDT) before the first step, and GPU runs wait for
another GPU (the jobs for Ada, an A10, an A100 and Hopper are prepared apart).
