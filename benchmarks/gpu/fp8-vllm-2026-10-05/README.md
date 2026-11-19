# FP8 checkpoints under vLLM with the Glyd wrapper: bit-for-bit checks (Runpod, 2026-10-05)

The GPU runs behind the FP8 vLLM wrapper's exactness claims (branch `fp8-vllm`, Task 3 of the FP8 plan; `glyd_gpu/vllm_fp8.py`
with the identity codec): stock vLLM 0.30.0 against the wrapped model, tokens, token logprobs and top-20 prompt logprobs, on slices of
Qwen3.8-27B-FP8 (layers 0-3, block-scaled and a per-tensor rescale) and GLM-5.3-Flash (its 3 dense layers), and MiniMax-M3-MXFP8
in the fix round. RTX 4090 (Ada) unless named; vLLM 0.30.0, torch 2.13.0+cu130. Report: `report.md` (the worker's, with the fix rounds).

| run (final tree `f8a5`) | exact | weights vLLM logs (stock -> Glyd, identity codec) |
| :--- | :--- | :--- |
| Qwen4 eager / graph / graphonly | PASS bit for bit | 6.32 -> 6.48 GiB (+ one scratch) |
| GLM3 eager / compile | PASS | 3.58 -> 3.67 GiB |
| Qwen4 per-tensor eager / graphq (`+quant_fp8`) | PASS | 6.29 -> 6.45 GiB |
| Qwen4 per-tensor graph (default compile) | FAIL (stock vs stock again identical) | |

Fix round (`report.md`, second half):

| run | GPU | result |
| :--- | :--- | :--- |
| `f8h1` | H100 80GB HBM3 | stock and Glyd alike stopped at FlashInfer's JIT link (`-lnvrtc`); the job's fix links the pip toolkit's `lib*.so.N` as `lib*.so` |
| `f8h2` | H100 | Qwen4 eager and default graphs **PASS bit for bit**; GLM3 eager and compile **PASS**; bf16 smoke ok. The same runs also report `kernel[...]` FAIL in `steps.txt`: the harness's comparison of the two servers' kernel log lines (stock's include DeepGEMM's lines); the report does not discuss it, so it is open. |
| `f8a6` | RTX 4090 | MiniMax-M3-MXFP8, 3 dense layers (Marlin MXFP8), eager and graph **PASS bit for bit**; weights 5.72 -> 5.84 GiB |
| `f8a7` | RTX 4090 | Qwen4 seed sweep (`PYTHONHASHSEED` 0..4, fresh compile caches, scratch overwritten after every product) **PASS** 5/5; the Mistral slice crashed on a throttled download (fixed after) |

```sh
# f8src.tar = git archive HEAD glyd-gpu-py gpu/fp8 + COMMIT; f8env = the plan of the run (F8_PLAN=...)
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=45 NAME=f8a5 UPLOAD="f8src.tar f8env" bash rp_run.sh gpu/fp8/fp8_vllm_job.sh
GPU="NVIDIA H100 80GB HBM3"   MAX_MIN=45 NAME=f8h2 UPLOAD="f8src.tar f8env" bash rp_run.sh gpu/fp8/fp8_vllm_job.sh
```

Files: one folder a pod (`job.log`, `tests.txt`, `steps.txt`, `env.txt`, `COMMIT`, `log/`, and per model and mode the JSON dumps
compared, gzipped where large; the compile caches the job saved are left out), `job/` (the three versions of the job used, the
`f8env` files, the launcher's output `rp_run-*.log`).
