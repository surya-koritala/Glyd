# v0.29.0 pre-release run: Qwen3.5-9B on an RTX 4090 and an RTX 5090 (Runpod, 2026-10-05)

The source of the v0.29.0 CHANGELOG's Qwen3.5 numbers (weights 12.68 GiB on the 4090, 13.89 on the 5090, against bf16's 17.66; bf16
out of memory on the 4090 at vLLM's defaults; a 189,440-token context; compiled 115.9 / 77.3 tokens a second against 28.7 / 26.6 eager;
up in 135 s on the 5090). Two pods at once, one job each, 2026-10-06 01:31-02:05 UTC (the evening of 2026-10-05 in New York).

What ran: the v0.29.0 pre-release `glyd-gpu` wheel CI built from `engine-v029` (named 0.28.0, the version PyPI had, so uv took it
from `UV_FIND_LINKS`), installed by `install.sh` as `curl | sh`; `glyd run` and `glyd serve` on Qwen3.5-9B (Qwen3.5-0.8B for the
`GLYD_ENGINE=0` line), vLLM 0.30.0, one user's speed by `job/speed1.py` (prompt about 1,000 tokens, 256 out, median of 3), plain vLLM bf16
from the same environment. Three blocks were appended to the running job (`job/v029_append*.sh`): the baselines again off port 8001,
`glyd serve` compiled (`-- --no-enforce-eager`), and plain vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`.

```sh
D=engine/v029up; UP="$D/install.sh $D/glyd_gpu-0.28.0-cp310-abi3-manylinux_2_28_x86_64.whl $D/api_features_check.py $D/speed1.py"
for spec in "v29-4090|NVIDIA GeForce RTX 4090" "v29-5090|NVIDIA GeForce RTX 5090"; do n=${spec%%|*}; g=${spec#*|}
  GPU="$g" MAX_MIN=95 NAME=$n MIN_DRIVER=580 RP_CUDA=13.0 RP_DISK=200 UPLOAD="$UP" nohup bash rp_run.sh v029_job.sh > $n.run.log 2>&1 &
done
```

## Qwen3.5-9B, vLLM 0.30.0, one user

| | RTX 4090 (24 GB, tiered layout) | RTX 5090 (32 GB, 12-bit layout) |
| :--- | ---: | ---: |
| weights, Glyd (vLLM's "Model loading took") | 12.69 GiB (12.68 in the load line) | 13.89 GiB |
| weights, bf16 | did not start: out of memory at `--max-model-len 4096` | 17.66 GiB |
| KV cache, Glyd compiled (`glyd serve`'s own context) | 197,225 tokens (`--max-model-len 189440`) | 389,307 tokens |
| KV cache, bf16 (`--max-model-len 4096`) | - | 152,137 tokens |
| decode, Glyd eager (v0.28's mode) | 26.59 tok/s | 28.71 tok/s |
| decode, Glyd compiled (v0.29's mode) | 77.30 tok/s | 115.85 tok/s |
| decode, vLLM bf16 | - | 94.98 tok/s |
| first token, Glyd compiled / eager / bf16 | 196.4 / 196.1 / - ms | 131.2 / 148.0 / 112.7 ms |
| `glyd serve` compiled, up in | 221 s | 135 s (eager 72 s) |

Checks (`*/summary3.txt`): 18 passed and 3 failed on each GPU. The failures: the engine's API checks (`api_check.out`); `glyd serve
(default)` on port 8001, which is Runpod's nginx (the same check passed on port 8005 in the appended block); and a traceback in
the user commands' output (the list is in the check's line of `job.log`). The rows in `speed.txt` and `numbers.txt` for the engine's
own model family (pre-2026) are the engine-vs-vLLM comparison for its GPU list: pass/fail only, not for publication.

## Files

- `rtx4090/`, `rtx5090/`: everything the job wrote (`summary*.txt`, `speed.txt`, `numbers.txt`, `times.txt`, `sizes.txt`,
  `machine.txt`, each step's `.out`/`.err`, the servers' logs `serve-*.log`/`run-*.log`, `speed-*.json`).
- `job/`: `v029_job.sh` and the three appended blocks, `speed1.py`, `api_features_check.py`, `v029_table.py` (the table from the
  results: `python3 v029_table.py RUNS` with `RUNS/ceiling/results-v29-*`), and the launcher's own output (`rp_run-*.log`).
- The launcher and the job template: `tools/runpod/`.
