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
| `container.sh`, `container.txt`, `sass-main-to-branch.txt` | with no GPU (Docker on a Mac, NVIDIA's `cuda:13.0.3-devel-rockylinux8` image for arm64, the release's build image, with GCC 13 as the release builds): main's library and this tree's for sm_80, 86, 89 and 90a with build_lib.sh's flags; their SASS compared by `sass_diff.py`, and each kernel's exception-loop loads, branches and calls and its spills against main's; and this tree's JIT source compiled as PyTorch 2.14's extension build compiles it (the headers of its CUDA 13 wheel) and linked with every symbol resolved |
| `run.sh` | all of the above on the RTX 4080 SUPER box, one step at a time under its lock, with the self-test, check_capi, test_gpu.py and check_api dense and MoE, and xcheck.py again through builds for sm_89 whose test for GeForce by name never matches (the grid prompt kernels, as an A10 and an L40S take them) |

## With no GPU (container.txt, sass-main-to-branch.txt; nvcc 13.0, aa7a6fc, 2026-09-29)

- Both libraries compile and link with no warning; the JIT source compiles with none of its own (only PyTorch's
  headers' `module` remarks) and links with every symbol resolved; its SASS is the library's, kernel for kernel (87 on
  each of the four).
- The 48 kernels not of the 12-bit layout: the same instructions in the same order, on each architecture.
- The exception loop (Nib::patch) as main has it: every kernel of the 12-bit layout with main's loads, branches and
  calls (LDG, LDS, BRA, CALL) on sm_86, 89 and 90a, and all but mma_moe_kernel<Nib, 4, 1 and 2> on sm_80, which keep an
  entry a pass (unrolled there, the loop took main's 4 spill instructions to 12; so, none). An entry a pass everywhere
  (873b7ed) had cost the L4's prompt kernel 1.0-2.3% at 256-1024 tokens (round 1: main's kernels with that loop
  against main), split byte alone being 0.2-1.0% faster; mma12_ws_kernel's loop is unrolled twice, as nvcc unrolled
  it in main (split byte's lighter body it unrolls 4 times).
- Spill instructions, main -> this tree: 4 -> 0 in those two sm_80 kernels, 2 -> 0 in mma_moe_kernel<Nib, 4, 0> on
  sm_86 and 89, 38 -> 36 in sm_90a's mma12_mid_kernel<64, 4>; more only in sm_80's mma12_mid_kernel<64, 4>, 13 -> 20
  (its decode12_rows, which has no such loop; no GPU runs it: the A100 takes mma12_ws_kernel). Fewer instructions
  than main in every kernel of the 12-bit layout built for its architecture (the TMA and wgp kernels are sm_90a's
  alone, stubs elsewhere).
- sass_diff.py's count of kernels with main's memory, tensor-core, barrier and branch instructions (36, 38, 38 and 32
  of 39) takes the spills and those two loops in, and on sm_90a one or two BSSY/BSYNC pairs fewer or more in
  mma_moe_kernel's 16- and 64-token products: split byte's decode, the same with either loop.

The GPU scripts run in round 1's jobs (Ada, an A10, an A100, Hopper), whose logs are kept with the release's checks;
the first, on an L4, is summed up above for the exception loop.
