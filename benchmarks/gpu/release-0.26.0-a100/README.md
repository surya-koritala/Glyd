# The vLLM plugin's check on an A100 SXM4 40 GB, 2026-09-30

`gpu/vllm/check_vllm.py` on Qwen3-8B (vLLM 0.30, torch 2.13 with CUDA 13.0), on vllm-plugin's tree at 0b217d3 (v0.25.1
with the plugin; its library C API 5, the same kernels as this release's, whose C API 7 adds the route SPLIT, which
the plugin never asks for): all 15 checks passed, both layouts, exact eager and compiled in inductor's deterministic
mode bf16's bits (vllm_check_summary.txt, vllm_check-Qwen3-8B.txt). The same session's response-speed run:
benchmarks/gpu/respond-2026-09-29/a100-v0.25.1.
