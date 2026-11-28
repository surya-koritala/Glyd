# Runpod tooling

How every GPU result since 2026-10-04 was made: one Runpod pod per batched job, launched from a laptop (no GPU, no torch, no CUDA
there), the pod deleted when the job ends or a time cap passes. Private. Results go under `GLYD_RUNS` (default `$HOME/glyd-runs`),
never into the repository by themselves: a run worth keeping becomes a record under `benchmarks/gpu/<name>-<date>/` (README with
the command and the table, the job script, the logs; `git add -f` for `.log` files; no weights, tarballs or wheels).

Needs on the launching machine: `runpodctl` (the current Runpod CLI), `ssh`, `ssh-keygen`, `python3`, `tar`. The API key is a file,
`~/.runpod/api_key`; every script reads it into one `runpodctl` call's environment and never prints or stores it.

## The scripts

| file | what it does |
| :--- | :--- |
| `rp_run.sh JOB.sh` | Creates one pod (`glyd-bench-$NAME`) with an ssh key made for this run, checks the driver, uploads `JOB.sh` (as `~/job.sh`) and `UPLOAD` files to the pod's home, starts the job with `nohup`, pulls `~/results` into `$GLYD_RUNS/results-$NAME` and the tail of `~/job.log` every minute, and stops when `results/DONE` appears. The exit trap deletes the pod and lists the pods left; a watchdog ends the script at `MAX_MIN`. Exit codes under Gotchas. |
| `rp_try.sh JOB.sh` | `rp_run.sh` up to `TRIES` times (default 4) while it exits 3 or 4 (an old host, no capacity, a stopped pod); same environment. Exit 5 (the budget) is not retried. |
| `classes.sh`, `rp_class.sh CLASS JOB.sh` | GPU classes (`4090 5090 3090 16gb l4 a6000 l40 a100 h200`, and the 4-GPU `glm glmh nemo` of `tools/cards/glm_skew.sh`, `tools/cards/glm_quality.sh`, `tools/cards/nemotron_quality.sh` and `tools/cards/nemotron_probe.sh`: `class_gpus` is a class's GPU count, passed on as `RP_GPUS` unless the environment sets it: `RP_GPUS=2 bash tools/runpod/rp_class.sh 4090 JOB.sh` asks for a pod of two for its memory) as lists of Runpod types in fallback order, with approximate prices; `rp_class.sh` runs `rp_run.sh` on each type of a class in turn (`ROUNDS`, default 2) while there is no capacity or the host is old. Used by `tools/cards/numbers.sh` (model card numbers) and `tools/matrix/run.sh` (release matrix). |
| `rp_guard.sh` | An independent watchdog for the case the launcher dies: every 2 minutes it deletes any pod named `glyd-*` that this guard has seen for `MAX_AGE` minutes (default 60), or that has stopped (a stopped pod still bills its disk). Run it in the background for the whole session: `MAX_AGE=160 bash tools/runpod/rp_guard.sh &`. |
| `rp_budget.sh` | The owner's spend cap: every minute it adds up what the `glyd-*` pods have cost (per pod id, so deleted pods still count) into `$GLYD_RUNS/rp_budget/spent.txt`; at `SOFT` (default $92) it writes `rp_budget/STOP` and `rp_run.sh` refuses new pods (exit 5); at `CAP` (default $100) it deletes every `glyd-*` pod. One per machine, in the background, for the day: `bash tools/runpod/rp_budget.sh &`. For a new day's budget, stop it and move `rp_budget/` aside. |
| `speed1.py URL MODEL OUT.json` | One user's speed of an OpenAI-compatible server: time to first token and decode tokens a second, streamed, the median of 3 rounds (a prompt of about 1,000 tokens, exactly 256 new, greedy, thinking off). |
| `jobs/v029_job.sh` | The v0.29.0 pre-release pod job: install.sh as `curl \| sh` with the CI wheel, `glyd run` / `glyd serve` / the engine's checks, one user's speed of Glyd and plain vLLM bf16. Record: `benchmarks/gpu/v029-qwen35-9b-2026-10-05`. |
| `jobs/e1_pod_job.sh` | (`ENGINE_STEPS=1` only) The engine hand-off's user flow (install.sh, `glyd run`, `glyd serve`, the Rust `glyd`, the fallback lines). Record: `benchmarks/gpu/engine-e1-rtx4090-2026-10-05`. |
| `jobs/f8k_job.sh`, `jobs/f8f_job.sh` | The lossless FP8 kernels' checks (`gpu/fp8/check_f8.py`) and the fused FP8 product's checks and bench (`check_f8.py`, `bench_f8.py`), on a tar of the worktree's `gpu/` and `glyd-gpu-py/` (`UPLOAD="f8k_job.sh f8k_src.tar"`). Records: `fp8-kernels-2026-10-05`, `fp8-linear-2026-10-05`. The FP8 under vLLM job is in the tree: `gpu/fp8/fp8_vllm_job.sh`. These need `gpu/fp8/` (branch `fp8-lossless` / `merge-pending` until merged). |
| `jobs/multigpu_job.sh` | Qwen3.8-Flash-Next split over a pod's two GPUs (`qwenserve --gpus 0,1`) against one: the logits byte for byte, places, cache hits and tok/s of each, tests/test_server.py on the split. Its header has the command and the files to upload (built on the PC from the branch). |
| `jobs/amd_job.sh` | The AMD bring-up on a rented RX 7900 XTX (Vast.ai or TensorDock: Runpod has no Radeon cards), launched by hand with `scp` and `nohup`, the same `~/results` contract: the HIP builds of the engine and the library from a tar of the branch, fast's download, one Strata chunk's logits for a KL against the CUDA run's, tests/test_server.py and bench/compare.py's prompt times. Its header has the uploads and the box it needs. |
| `jobs/fmt_job.sh`, `jobs/fmt_measure.py` | The lossless room in 2026 checkpoints' released formats (bf16, FP8, MXFP4, NVFP4), sampled by HTTP range. Record: `formats-2026-2026-10-05`. |

**New runs use 2026 models only.** The templates default to Qwen3.5-9B and Qwen3.5-0.8B. The steps that need the Rust engine to
run a model family (`ENGINE_STEPS=1`) are off by default (the engine runs Qwen3.5 now; the job scripts were not changed); with them off, the engine's fallback to
vLLM is checked on Qwen3.5. A release acceptance (published wheels, old saves, a clean pip resolve) follows the scripts kept in
`benchmarks/gpu/acceptance-v0.28.0rc1-l4-2026-10-04/scripts/` (written for an AWS box that is gone: port them to a pod job, with
2026 models, before reuse).

## A job's contract

`rp_run.sh` runs `bash job.sh` in the pod's home (`/root`) with the `UPLOAD` files beside it. The job writes into `~/results` and
**must** create `~/results/DONE` when it ends however it ends (`trap 'touch "$R/DONE"' EXIT`), or the launcher waits until
`MAX_MIN`. Keep a `MAX_MIN` of its own inside the job (the templates skip a step that would not end in time) so the results come
back before the launcher's cap kills the pod.

## Launching

```sh
export GLYD_RUNS=$HOME/glyd-runs        # results land in $GLYD_RUNS/results-$NAME
MAX_AGE=160 bash tools/runpod/rp_guard.sh &   # once per machine (MAX_AGE above the 150-min cap)
bash tools/runpod/rp_budget.sh &               # once per machine: the spend cap (exit 5 once it says STOP)

# RTX 4090 (Ada; the cheapest, ~$0.6-0.75/h; for Ada work and the tiered layout)
GPU="NVIDIA GeForce RTX 4090" MAX_MIN=90 NAME=q35-4090 UPLOAD="install.sh speed1.py" \
  bash tools/runpod/rp_try.sh JOB.sh

# RTX 5090 (Blackwell, 32 GB)
GPU="NVIDIA GeForce RTX 5090" MAX_MIN=90 NAME=q35-5090 RP_DISK=200 UPLOAD="..." bash tools/runpod/rp_try.sh JOB.sh

# H100 80GB (Hopper)
GPU="NVIDIA H100 80GB HBM3" MAX_MIN=60 NAME=q38-h100 UPLOAD="..." bash tools/runpod/rp_try.sh JOB.sh

# several GPUs in one pod (tensor parallel, big models): RP_GPUS, and the disk for the weights
GPU="NVIDIA H100 80GB HBM3" RP_GPUS=4 RP_DISK=1000 MAX_MIN=120 NAME=big-h100x4 UPLOAD="..." bash tools/runpod/rp_try.sh JOB.sh
```

Run a launcher with the shell tool's background mode (it lasts the whole job) and read `$GLYD_RUNS/results-$NAME/job.log` while
it runs. One batched job per pod: everything a session needs on that GPU goes into one job script.

| variable | default | |
| :--- | :--- | :--- |
| `GPU` | `NVIDIA H100 80GB HBM3` | Runpod's GPU id (`runpodctl` lists them) |
| `NAME` | `rp` | unique per run: the pod is `glyd-bench-$NAME`, the results `results-$NAME` (an earlier `results-$NAME` is reused) |
| `MAX_MIN` | 50 | hard cap; the pod is deleted at it; at most 150 (larger values are cut to 150) |
| `UPLOAD` | none | files copied to the pod's home (absolute paths) |
| `RP_GPUS` | 1 | GPUs in the pod; at most 4 (more exits 5) |
| `RP_DISK` | 150 | container disk, GB (models, the tool's environment) |
| `RP_TEMPLATE` | `runpod-torch-v280` | Runpod template (Ubuntu 24.04, Python, a CUDA toolkit in `/usr/local/cuda`) |
| `RP_CUDA` | 13.0 | lowest CUDA version the host's driver must offer |
| `MIN_DRIVER` | 580 | lowest driver major; under it the run exits 3 (`rp_try.sh` tries another host) |
| `TRIES` | 4 | `rp_try.sh` only |
| `GLYD_RUNS` | `$HOME/glyd-runs` | where results and the run's temporary key go |

A loop over GPU ids when one has no capacity (as the formats study did):
`for G in "NVIDIA GeForce RTX 4090" "NVIDIA L40S" "NVIDIA RTX 6000 Ada Generation"; do GPU="$G" ... bash tools/runpod/rp_run.sh JOB.sh && break; done`.

## Gotchas

- **Port 8001 is Runpod's nginx** on its images: a server started on 8001 never answers and the check fails as if Glyd had (the
  v0.29 run's `glyd serve (default)` failure). Use 8000, 8002 and up.
- **Plain `vllm serve` from the glyd tool's environment** (`$(uv tool dir)/glyd/bin/vllm`, the bf16 baseline) needs that
  environment's `bin` on `PATH` (vLLM's compile calls `ninja` from there) and `VLLM_USE_FLASHINFER_SAMPLER=0` (glyd sets it; without
  it plain vLLM does not start on a pip-only setup): `env VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$TOOL/bin:$PATH" "$TOOL/bin/vllm" serve ...`.
- **`MIN_DRIVER` 580**: the CUDA 13 wheels need a 580+ driver; Runpod hosts vary (595.91 seen on 4090/5090 hosts). An older host
  exits 3 and `rp_try.sh` asks for another. For a job on CUDA 12.8 wheels set `MIN_DRIVER=550 RP_CUDA=12.8`.
- **The no-toolkit user flow**: the pod images ship a CUDA toolkit, users mostly do not. For the install.sh / `glyd run` flow, hide
  it first (`mkdir -p /opt/hidden && mv /usr/local/cuda* /opt/hidden/`) and run the flow as a user that is not root
  (`gpu/vllm/acceptance.sh --host` does the glyd and vllm flows without Docker).
- **Exit codes**: 3 = driver too old, 4 = no capacity (no pod id came back) or the pod stopped (both retried by `rp_try.sh`),
  5 = refused by the budget or the 4-GPU limit (not retried), 1 = ssh never answered or an upload failed; 0 = `DONE` came back, whatever the job's checks said (read `summary.txt` / `job.log`). Capacity comes and
  goes in minutes; `rp_try.sh` sleeps 20 s between tries.
- **FlashInfer JIT on Hopper** with pip's CUDA: `ld: cannot find -lnvrtc` (the pip toolkit has `libnvrtc.so.13` without the
  `libnvrtc.so` link). `gpu/fp8/fp8_vllm_job.sh` links every `lib*.so.N` of the toolkit as `lib*.so`.
- **`ncu` is not permitted** on Runpod pods (no profiling counters); `nsys` and timing loops work.
- **`rp_guard.sh` deletes every `glyd-*` pod** past its age cap, other agents' too: run one guard per machine, with `MAX_AGE`
  above the longest job's `MAX_MIN`.
- **Pip installs are slow on some hosts** (8 to 14 minutes of a run were the install): budget it in `MAX_MIN`.
- Hugging Face downloads need no token for the public 2026 models; never put a token in a job or an upload.
- `nohup ... &` jobs: a job that backgrounds a server must kill it before the end (`pkill -f vllm.entrypoints`), or the GPU is not
  idle for the next step.

## Spend

Runpod bills per second while a pod exists (a stopped pod bills its disk). The launcher prints the pod's minutes at the end
(`N min`); multiply by the GPU's hourly price shown when the pod is created (`runpodctl` or the console). Report the spend of
every session.
