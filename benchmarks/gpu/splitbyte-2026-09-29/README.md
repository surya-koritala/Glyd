# Split byte against the 12-bit layout before it, 2026-09-29

The checks for the 12-bit layout in split byte: main's 12-bit layout (origin/main, its package and library) and this
tree's in the same kernels, on any GPU. Main's tree and library sit beside this tree's (`git archive origin/main`, each
built by `gpu/build_lib.sh`); the scripts load both libraries into one process where they compare (`both.py`: main's
through a second copy of `glyd.gpu._lib`, the C API being version 4 in both, and main's `pack_mma12` taken from its
`kernels.py`).

| script | what |
| :--- | :--- |
| `xcheck.py MAIN_TREE MAIN_LIB [MODEL ...]` | every 12-bit entry point the GPU takes, each output of split byte the same bits as main's (a call refused by one refused by both), and checked against the weights or an fp32 product: the self-test's and check_capi's matrices, every bf16 bit pattern, mixtures of experts, and every Linear weight of each MODEL decoded bit for bit in both layouts (their sizes) with layers 0, 1, 2, the middle one and the last multiplied at 1-1100 tokens |
| `e2e12.py MODEL ...` | a model loaded in the 12-bit layout (fused, then exact), its logits for prompts of 1, 17, 64, 300 and 2100 tokens and 32 greedy tokens, as sha256 lines: run with main's package and library and with this tree's, the lines compared |
| `layer.py MAIN_TREE MAIN_LIB MODEL` | one layer's products (q, k, v and gate, up merged) and decodes, main's 12-bit layout against split byte, by the library's route and by each kernel, each call alone after an L2 flush, the median of 15 |
| `sass_diff.py MAIN.sass TREE.sass [MAIN.res TREE.res]` | the two libraries' SASS for one architecture (`cuobjdump -sass -arch sm_XX`): the 12-bit layout's kernels by kind of instruction, and whether their memory, tensor-core, barrier and branch instructions are the same ones in the same order; every other kernel the same instructions |
| `emulate.py`, `emulate.txt` | on the CPU (numpy 2.0.2, Python 3.9), no GPU: `pack_mma12` ported line by line, its bytes against the layout `glyd_gpu.h` writes out, and the kernels' decodes written out as they run (`Nib::decode`: the step, prompt, mixture-of-experts, unpack and A100 mid kernels; `decode12_rows`: the mid and TMA kernels; `stage12_exc` and `step12`: Hopper's wgp kernel) against the mma fragments built straight from W, bit for bit: 10 matrices, every bf16 bit pattern among them, the base at 0 and at 120, 0-94% of the weights exceptions |
| `run.sh` | all of the above on the RTX 4080 SUPER box, one step at a time under its lock, with the self-test, check_capi, test_gpu.py and check_api dense and MoE, and xcheck.py again through builds for sm_89 whose test for GeForce by name never matches (the grid prompt kernels, as an A10 and an L40S take them) |

But for emulate.py, none of these has run yet; their logs land here.
