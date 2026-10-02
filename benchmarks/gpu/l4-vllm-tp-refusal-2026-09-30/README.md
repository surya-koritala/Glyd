# Refusals over several processes, on the L4 (2026-09-30)

Over two GPUs, M4 (`../vllm-m4-2xa6000-2026-09-30`) found a refusal raised in the workers: the engine saw only vLLM's
"WorkerProc initialization failed". The fix refuses in the engine's process, before any worker starts. The workers
refuse the same again, with what only they know. This dry run is on one L4, with vLLM's workers in processes of their
own, as over several GPUs (`--distributed-executor-backend mp`; `check_vllm.py --mp`).

- **`check_vllm.py --quick --mp` on Qwen3-1.7B:** all 15 passed (`check-Qwen3-1.7B.txt`).
  - Exact under torch.compile was refused with Glyd's message, from the engine's process: pydantic's "1 validation
    error for VllmConfig ... glyd: exact mode gives vLLM's bf16 logits bit for bit eager ...".
  - The default-mode, compiled, deterministic check wrote its line, the layout from a worker's config.
- **`vllm serve Qwen/Qwen3-1.7B --quantization glyd --distributed-executor-backend mp` with `GLYD_EXACT=1`:** the
  server stopped at start, with Glyd's message the last error line (`serve-refused.txt`).

`m7_mp.sh` is the run, `m7_mp.log` its console. The plugin and `check_vllm.py` are the tree committed with this
directory.
