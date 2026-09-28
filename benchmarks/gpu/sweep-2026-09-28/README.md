# v0.22.0's wheel on six GPUs, 2026-09-28

The release workflow's build of main at 4fd327d (`glyd-0.21.0-py3-none-manylinux_2_28_x86_64.whl` by its version then;
the same code as v0.22.0 but check_capi.py's Hopper check, #44), installed in a fresh venv as a user installs it, then
the checks and bf16 against Glyd on the same GPU: `sweep_job.sh` (Lambda: H100 PCIe, A100 SXM4 40 GB, A10; AWS: g5
A10G, g7e RTX PRO 6000 Blackwell), `box_final.sh` (RTX 4080 SUPER). The first version of the job ran check_capi and the
self-test without ninja on PATH, test_gpu.py apart from the source tree and check_api's models in one process (16 GB):
`sweep_fix_job.sh` (on the same Lambda machines, H100, A100 and A10) and `box_final2.sh` (the RTX 4080 SUPER) ran those
steps again, their files replacing the first ones; `g7e_fix_job.sh` ran them on a second g7e instance
(`rtx-pro-6000-aws-g7e-checks`, the same wheel). The A10G's were not run again. `gpu/gen_eager.py` gives the eager and
compiled tokens/s.

- `a10-ecc-fault`: the first A10 stopped at its first check with CUDA_ERROR_ECC_UNCORRECTABLE (nvidia-smi: 0 ECC errors
  at the start); a second A10 ran the job (`a10`).
- `h100-pcie-prompts-same-machine` (`h100_bisect_job.sh`): prompts of Qwen3-8B and Qwen3-32B through the library built
  at ebc8f91, bc146c3, 53b1659 and 4fd327d on one H100 PCIe, twice each: within 1% of one another. bf16's own prompts
  differ by up to 32% between H100 PCIe instances; compare within one machine.
- The RTX PRO 6000's Qwen3-32B run was killed for host memory (62 GB, free -g) and its Qwen3-30B-A3B run ran out of disk; its
  test_gpu.py stopped at a bf16 model's convolution (cuDNN's runtime-compiled engines missing on that image).
