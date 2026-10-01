# vLLM with `--quantization glyd` over two GPUs, and mixtures of experts (M4, 2026-09-30)

M4's unattended job (`vllm_job.sh`) on one instance with two NVIDIA RTX A6000s, every run tensor parallel over both:
the checks on a dense model and two mixtures of experts, then `vllm bench serve` bf16 against Glyd on Qwen3-30B-A3B.

## Setup

- **Machine:** Lambda, 2x RTX A6000 (48 GB each, 300 W), driver 580.126.20, x86_64, 28 CPUs.
- **Software:** the job's tarball at 5513d94, the vllm-plugin branch after review 1 (v0.25.1's library with the
  plugin). vLLM 0.30.0 came from PyPI in the job (torch 2.13.0+cu130, transformers 5.17.0), and the library was built
  there for sm_86 (C API 5; GPU code 86).
- **Run:** 4 jobs in 59 minutes (`steps.txt`): `tp`, `moe-granite`, `moe-30b`, `moebench`. Qwen3-30B-A3B (61.1 GB)
  downloaded in 81 s.
- **Glyd's layout:** the one `best_layout` picks, tiered for the mixtures of experts on this GDDR Ampere GPU (`glyd:
  mma layout` in the servers' logs).

## Checks, tensor parallel over the two GPUs

| Model (`check_vllm.py`) | Result |
| :--- | :--- |
| Qwen3-8B, `--quick --tp 2` (`tp/`) | 13 passed; 1 failed (the check's, below); the last check computed from its logs |
| granite-3.1-3b-a800m-instruct, `--quick --tp 2` (`moe-granite/`) | the same: 13 passed, 1 failed, the last from its logs |
| Qwen3-30B-A3B, `--brief --tp 2` (`moe-30b/`) | all 5 passed |

- **Passed over two GPUs:**
  - every pack decoded to its weights bit for bit: 288 Linears for Qwen3-8B, 128 and 64 layers' experts for granite,
    288 and 96 layers' experts for Qwen3-30B-A3B, each rank's own;
  - every product within 3.64e-3, 3.78e-3 and 3.76e-3 of the same product on its matrix decoded;
  - top-1 on bf16's continuation 0.9942 and 0.9948 for Qwen3-8B (bf16 eager's 0.9942), 0.9857 for granite (0.9818)
    and 0.9870 for Qwen3-30B-A3B (0.9857);
  - exact eager bf16 eager's bits, for all three;
  - exact compiled in inductor's deterministic mode, compiled bf16's bits;
  - a compile cache for each layout, each rank's own, each loaded again.
- **Failed: "exact under torch.compile: refused".** The refusal happened, but in the two workers. The check saw
  vLLM's "WorkerProc initialization failed", not Glyd's message, which stood in the workers' log
  (`*-glyd-exact-compiled.log`: "glyd: exact mode gives vLLM's bf16 logits bit for bit eager ...").
- **The last check ("fused, compiled, deterministic: the same bits from one run to the next") did not report.**
  - `check_vllm.py` stopped on a TypeError while writing its line: over several GPUs the plugin's options stand in the
    workers' config, not the engine's.
  - Its two runs' JSONs give 8 of 8 prompts bit for bit and the continuation the same, for Qwen3-8B and for granite.

## Serving Qwen3-30B-A3B over two GPUs (`moebench/`)

`vllm bench serve` bf16 against Glyd, warm with the cold start noted, `--gpu-memory-utilization 0.9`, 1,024 tokens in
and 256 out: 64 prompts at 1 request a second, 128 at 4, 256 at once.

| | Weights, a GPU | KV cache | Requests of 1,280 tokens at once |
| :--- | ---: | ---: | ---: |
| bf16 | 28.46 GiB | 283,904 tokens | 221 |
| Glyd tiered | 19.75 GiB | 473,232 tokens (1.67x) | 369 |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.94 | 239.7 | 156 / 531 | 16.4 / 21.6 | 15.1 / 94.4 |
| 1 | glyd | 0.94 | 239.8 | 204 / 399 | 15.3 / 21.8 | 13.2 / 152.1 |
| 4 | bf16 | 3.20 | 818.1 | 198 / 515 | 44.7 / 61.5 | 37.2 / 176.7 |
| 4 | glyd | 2.83 | 723.6 | 378 / 1,151 | 70.9 / 97.5 | 58.5 / 306.6 |
| inf | bf16 | 5.68 | 1453.6 | 5,840 / 19,990 | 126.8 / 139.0 | 76.7 / 319.4 |
| inf | glyd | 4.92 | 1260.6 | 7,851 / 27,691 | 162.6 / 184.8 | 109.6 / 420.1 |

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1.00x (0.94 to 0.94) | +30% (156 to 204 ms) | -7% (16.4 to 15.3 ms) | -5% |
| 4 | 0.88x (3.20 to 2.83) | +91% (198 to 378 ms) | +59% (44.7 to 70.9 ms) | +59% |
| inf | 0.87x (5.68 to 4.92) | +34% (5,840 to 7,851 ms) | +28% (126.8 to 162.6 ms) | +29% |

- **Wins:**
  - 1.67x the KV cache: 369 requests of 1,280 tokens at once against 221;
  - at 1 request a second, each token 7% sooner, and each request 5% sooner.
- **Losses:**
  - From 4 requests a second, Glyd served fewer requests a second: 0.88x at 4, and 0.87x saturated.
  - Each token took 59% longer at 4 a second, and 28% longer saturated. The first token came 30-91% later.
- **Why saturation doesn't turn on the KV cache here.** At once, 256 requests (vLLM's default for sequences at once)
  are more than bf16's KV cache holds (221), but the servers' logs show both modes running about 200 at once with
  requests waiting (bf16 a median of 210, Glyd 199). So both ran the same batches, and the steps' time decided.
- **The steps' time:** Glyd's ran longer (the ITL median 58.5 against 37.2 ms at 4 a second, 109.6 against 76.7
  saturated). The mixture of experts' grouped products at these batch sizes are the next work: measured against the
  routed experts decoded and vLLM's Triton kernel, on the dev L4.

## Files

- `vllm_job.sh`: the job (M4's, byte for byte). `steps.txt` and `gpus.txt`: the instance's run.
- `tp/`, `moe-granite/`, `moe-30b/`, `moebench/`: each job's results, as M3's: `summary.txt`, `steps.txt`, `job.log`,
  `machine.txt`, `env.txt`, `log/`, and a check's JSONs and vLLM logs, or the bench's result JSONs, servers' logs,
  KV cache and nvidia-smi's samples. `moebench/bench-Qwen3-30B-A3B/summary.txt` is regenerated with the clock's median
  over loaded samples (`gpu/vllm/bench_summary.py`); the job's own `moebench/summary.txt` kept as printed.
