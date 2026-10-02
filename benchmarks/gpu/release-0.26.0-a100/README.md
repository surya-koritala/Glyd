# The vLLM plugin's check on an A100 SXM4 40 GB, 2026-09-30

`gpu/vllm/check_vllm.py` on Qwen3-8B (vLLM 0.30, torch 2.13 with CUDA 13.0), on vllm-plugin's tree at 0b217d3 (v0.25.1
with the plugin; library C API 5; this release's C API 7 adds an opt-in long-prompt mode, which the plugin does not
ask for): all 15 checks passed, both layouts; exact mode gave bf16's bits, eager and compiled in inductor's
deterministic mode (vllm_check_summary.txt, vllm_check-Qwen3-8B.txt). The same session's response-speed run:
benchmarks/gpu/respond-2026-09-29/a100-v0.25.1.
