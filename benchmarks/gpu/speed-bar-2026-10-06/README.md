# The speed bar: Glyd 0.29.0 against vLLM bf16, 1 to 256 users (Runpod, 2026-10-06)

Glyd as released (`glyd serve MODEL`, its defaults) against vLLM's own bf16 (`vllm serve MODEL`), the same vLLM 0.30.0 (the one in Glyd's tool
environment), one server after the other on the same pod. `vllm bench serve`: random prompts of 1,000 tokens in and 256 out (`--ignore-eos`),
`--max-concurrency` 1, 2, 4, 8, 16, 32, 64, 128 and 256 with every request sent at once, max(8, 3 x users) prompts a level, a seed of its own
a level (the same prompts for both servers), a warm-up of 64 prompts first. The prefix cache's hits from `/metrics` around each level: 0.0% on
every level of every server (prefix caching on, vLLM's default).

- **Glyd:** `install.sh` of tag v0.29.0 as a user runs it. glyd-gpu 0.29.0 was not on PyPI when the pods started, so its wheel was built on each pod
  from `glyd-gpu-py` at commit 63625822ef6a72abb76b8809783c385ba714a63b (where tag v0.29.0 pointed then) with the GPU libraries the release run
  built. Those libraries are byte for byte the ones in the PyPI wheel published later (glyd_gpu-0.29.0-cp310-abi3-manylinux_2_28_x86_64.whl,
  sha256 4d46a35c35bef7aea620526ac91e6b3279a8a511eb55d3305f50d8babec7caab, as PyPI lists it); sha256 of the files uploaded to the pods
  (whose first 16 characters each pod's `glyd-gpu-source.txt` records) and of the same files extracted from that wheel:
  - `libglyd_gpu_cuda13.so` 6878cf67b9be5f405739be814ec18be66e02fed3a522e42de00aa73535e19655 (both)
  - `libglyd_gpu_cuda12.so` 539361898e5ae662753e9ae2dc3ffb9d4226d8ebd62d9241f151407d7eea8f2d (both)

  The tag later moved to d7f53f4, which differs from the measured commit only in `check_wheel.py`, `tests/test_gpu.py` and `tests/test_onboard.py`. The engine program was left out (`glyd serve` uses vLLM). `env.txt`: glyd 0.29.0, glyd-gpu 0.29.0, vllm 0.30.0,
  torch 2.13.0, transformers 5.18.0.
- **Memory budget:** `glyd serve` runs vLLM with `--gpu-memory-utilization 0.92` (its default, in its logs); bf16 ran at vLLM's default 0.90.
  The two servers were not given the same share of the GPU.
- **bf16:** vLLM's defaults, with one change: `VLLM_USE_FLASHINFER_SAMPLER=0`, the PyTorch sampler that Glyd's default sets, so both sample alike
  (FlashInfer's sampler is otherwise compiled at the first request and needs `nvcc`).
- **Where a server did not start at its defaults** (`speed_bar_job.sh`'s ladder): the most `--max-num-seqs` vLLM's own error names, else fewer
  sequences, then `--max-model-len 4096`. Each server's `started.txt` names the try that started; the failed tries' logs are kept beside it.
- Files: `speed_bar_job.sh` (the job), `job-*.sh` (the models a pod ran), `speed_bar_summary.py` (the tables), and per GPU `summary.txt`, each
  server's `cN.json` (vllm bench serve's result), `bench-cN.txt`, `smi-cN.csv` (nvidia-smi each second), `prefix.txt`, `serve-try*.txt` and
  Glyd's own log under `state-try*/glyd/logs/` (including the tries that did not start), and the 4090's `job.log`.
- Each cell is a single, unrepeated run, and 1 to 16 users use 8 to 48 prompts, so p99 there is close to the largest of a few samples.
  A change within 3% is treated as a tie, not a win or a loss (`speed_bar_summary.py` marks these "(tie)").

## What was measured

| GPU | Model | bf16 started with | Glyd started with | Weights bf16 / Glyd (GiB) | KV cache bf16 / Glyd (tokens) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| RTX 4090 24 GB | Qwen3.5-9B | `--max-model-len 4096 --max-num-seqs 95` (did not start at defaults, 256 or 128 sequences) | defaults (189,440-token context) | 17.66 / 12.69 (-28.1%) | 29,257 / 197,225 |
| H100 80GB HBM3 | Qwen3.8-27B | `--max-num-seqs 256` (out of memory at the default 1,024) | `-- --max-num-seqs 595` (out of memory at the default 1,024) | 51.1 / 39.52 (-22.7%) | 287,512 / 464,325 |
| H100 80GB HBM3 | Qwen3.5-35B-A3B | `--max-model-len 4096 --max-num-seqs 128` | `-- --max-num-seqs 760` | 65.53 / 53.21 (-18.8%) | 85,606 / 794,656 |
| L40S 48 GB | Qwen3.8-27B | does not fit (out of memory loading the weights) | `-- --max-num-seqs 48` | - / 35.6 | - / 34,588 |

**The released defaults do not start these 2026 models on the H100 and the L40S.** `glyd serve` 0.29.0 at its defaults ran out of memory
(vLLM's 1,024 sequences) on the H100 for Qwen3.8-27B and Qwen3.5-35B-A3B ("Engine core initialization failed"; vLLM's message: at most 595 / 760
sequences), and on the L40S for Qwen3.8-27B at the defaults, 256 and 128 sequences (it started at 48). So the H100 and L40S rows are not Glyd at
its defaults; the default has to choose the number of sequences itself. This is a release defect, reported to the owner, not fixed here.

**GPU memory in use (nvidia-smi), Glyd minus bf16:** Glyd uses more, not less, so the memory half of the bar is not met by nvidia-smi either.

| GPU, model | After start | Idle after warm-up | Peak per level |
| :-- | --: | --: | :-- |
| RTX 4090, Qwen3.5-9B | +164 MiB (20,862 / 20,698) | +46 MiB (21,222 / 21,176) | +46 MiB at every level except 64 users (-345) |
| H100, Qwen3.8-27B | +244 MiB (73,271 / 73,027) | +700 MiB (76,807 / 76,107) | +700 MiB at 1-64 users, -370 at 128, +1,230 at 256 |

vLLM takes its fraction of the GPU whatever the weights, so the smaller weights show as KV cache, not as free memory. Part of Glyd's larger KV
cache, and so part of its throughput at high load, comes from its 2% larger share of the GPU (about 1.6 GiB on the H100), not from the weights
alone. A weights-only comparison needs a bf16 run at `--gpu-memory-utilization 0.92`.

## Glyd against bf16 (Glyd / bf16; latency as a change, negative is faster)

**RTX 4090, Qwen3.5-9B** (median SM clock per level: bf16 2,685-2,700 MHz; Glyd 2,565-2,700 MHz, about 2,565-2,580 from 32 users up, which favours bf16 slightly):

| Users | Requests/s | Output tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | :-- | :-- |
| 1 | 1.36x | 1.36x | +13% / +14% | -28% / -28% |
| 2 | 1.36x | 1.36x | +16% / +11% | -28% / -28% |
| 4 | 1.32x | 1.32x | +19% / +15% | -27% / -26% |
| 8 | 1.27x | 1.27x | +31% / +13% | -27% / -23% |
| 16 | 1.22x | 1.22x | +23% / +13% | -21% / -21% |
| 32 | 1.80x | 1.80x | -88% / -60% | +3% / -35% |
| 64 | 1.95x | 1.95x | -94% / -69% | +78% / +45% |
| 128 | 1.83x | 1.83x | -65% / -58% | +121% / +85% |
| 256 | 1.82x | 1.82x | -54% / -51% | +118% / +85% |

**H100, Qwen3.8-27B** (median SM clock per level: bf16 1,980 MHz; Glyd 1,965-1,980 MHz):

| Users | Requests/s | Output tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
| ---: | ---: | ---: | :-- | :-- |
| 1 | 1.14x | 1.14x | +18% / +18% | -13% / -13% |
| 2 | 1.13x | 1.13x | +14% / +11% | -13% / -13% |
| 4 | 1.11x | 1.11x | +14% / +13% | -12% / -11% |
| 8 | 1.07x | 1.07x | +18% / +18% | -10% / -10% |
| 16 | 1.04x | 1.04x | +11% / +10% | -7% / -6% |
| 32 | 1.02x | 1.02x | +14% / +8% | -7% / -4% |
| 64 | 0.99x | 0.99x | +7% / +7% | -1% / +0% |
| 128 | 1.08x | 1.08x | -76% / +30% | +49% / +56% |
| 256 | 1.11x | 1.11x | -25% / -7% | +45% / +43% |

The absolute numbers are in each GPU's `summary.txt`.

**Every cell where Glyd is slower** (the verdict column of `summary.txt`, from `speed_bar_summary.py`; "tie" = within 3%):

| GPU | Users | Slower cells (change against bf16) |
| :-- | ---: | :-- |
| RTX 4090 | 1 | TTFT p50 +13%, TTFT p99 +14% |
| RTX 4090 | 2 | TTFT p50 +16%, TTFT p99 +11% |
| RTX 4090 | 4 | TTFT p50 +19%, TTFT p99 +15% |
| RTX 4090 | 8 | TTFT p50 +31%, TTFT p99 +13% |
| RTX 4090 | 16 | TTFT p50 +23%, TTFT p99 +13% |
| RTX 4090 | 32 | TPOT p50 +3% |
| RTX 4090 | 64 | TPOT p50 +78%, TPOT p99 +45% |
| RTX 4090 | 128 | TPOT p50 +121%, TPOT p99 +85% |
| RTX 4090 | 256 | TPOT p50 +118%, TPOT p99 +85% |
| H100 | 1 | TTFT p50 +18%, TTFT p99 +18% |
| H100 | 2 | TTFT p50 +14%, TTFT p99 +11% |
| H100 | 4 | TTFT p50 +14%, TTFT p99 +13% |
| H100 | 8 | TTFT p50 +18%, TTFT p99 +18% |
| H100 | 16 | TTFT p50 +11%, TTFT p99 +10% |
| H100 | 32 | TTFT p50 +14%, TTFT p99 +8% |
| H100 | 64 | requests/s and output tok/s 0.99x (tie), TTFT p50 +7%, TTFT p99 +7%, TPOT p99 +0% (tie) |
| H100 | 128 | TTFT p99 +30%, TPOT p50 +49%, TPOT p99 +56% |
| H100 | 256 | TPOT p50 +45%, TPOT p99 +43% |

Every other cell Glyd is faster or equal: requests/s and output tok/s at every load except the H100's 64 users; first token on the 4090 from
32 users up and on the H100 at 128 (p50) and 256 users, where bf16 queues requests it has no room for; per-token time at 1-16 users on the
4090 and 1-64 (p50) on the H100. So no GPU/model is green on every cell, and the memory half fails by nvidia-smi: the bar (at least as fast on
every metric at every load, and less memory) is not met anywhere measured. The cells the engine must win are exactly the table above.

## Not measured, and why

- **All Runpod pods on the account were terminated at about 05:17 UTC** (this job's two and the other workflow's), not by this job and
  not at its time cap; the account showed $0/h and no pods afterwards. Nothing was relaunched. Lost with it:
  - H100: Glyd on Qwen3.5-35B-A3B (the server had started; no level ran), and gemma-4-26B-A4B-it (neither server).
  - L40S: Qwen3.8-27B at 128 and 256 users, and gemma-4-31B-it (neither server).
- **The L40S numbers are not usable:** Glyd's SM clock swung between 210 and 2,520 MHz, up to 89 C and 327 W (`smi-c*.csv`; a host that cannot cool it);
  Glyd's one-user TTFT was 1,091 ms. They are kept in `l40s/` as measured and need a run on another L40S. bf16 does not fit there at all.
- The H100 and L40S `job.log` copies were lost when the pods went (the per-level files and every server log are complete).
- A first attempt of each pod (04:07-04:27 UTC) was stopped by hand after its start ladder ran out of memory and FlashInfer's sampler could not
  find `ninja`; the job was fixed (`--max-num-seqs` from vLLM's message, the tool environment's bin on PATH, the PyTorch sampler) and run again.
