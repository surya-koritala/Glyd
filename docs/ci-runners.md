# CI runners

CI here runs on GitHub's free runners (`ubuntu-latest`, `ubuntu-24.04`, `ubuntu-24.04-arm`, `windows-latest`, `macos-latest`) and on the owner's PC
`glyd-pc` (self-hosted). The repository is public, so both cost nothing. **Keep it that way: never use a larger, paid or GPU-hosted runner label**
(anything with cores, xlarge or GPU in its name), and add no workflow to a private repository. Keep the Actions caches under the repository's 10 GB:
rust-cache keys carry the lock file's hash, so every `Cargo.lock` change adds a copy of each job's cache (about 2 GB in all); delete the old ones
(`gh api -X DELETE repos/OWNER/REPO/actions/caches/ID`, older duplicates of a key and caches of merged pull requests first) before evictions slow the runs.

## What can land on glyd-pc

| Workflow | Jobs | Routed by | Container |
|---|---|---|---|
| `gpu.yml` (calls `gpu-tests.yml`) | the GPU tests | `vars.GPU_RUNNER` (`["self-hosted","linux","gpu"]`) | CUDA devel image, `--gpus all`, memory capped |
| `gpu-bench.yml` | the engine's benchmark gate | `vars.GPU_RUNNER` | the same, capped |
| `ci.yml` | the Linux jobs (`codec`, `lint`, `versions`, `rust`, `engine`, `engine-release`) | `vars.CI_RUNNER`, same-repository pull requests and pushes only (never a fork); unset: GitHub's | none (they run on the host: `.github/actions/pc-job` gives each its own home and limits cargo to 8 jobs) |
| `starlings-ci.yml` | Linux x86_64 core jobs | `vars.CI_RUNNER`, same guard | none (`CARGO_BUILD_JOBS=8`) |
| `kernels.yml` | nvcc compile of the engine's kernels | `vars.CI_RUNNER`, same guard | CUDA 13.0.3 devel image, `--memory=28g --memory-swap=28g`, nvcc processes limited to 2 |

Everything else (`gpu-check-bundle.yml`, `release*.yml`, `python.yml`, the CUDA 12.8 release jobs, Windows, macOS, arm64) runs on GitHub's machines and cannot
land on glyd-pc. `CI_RUNNER` is not set today, so the first three rows' Linux jobs are on GitHub's runners; only `GPU_RUNNER` points at glyd-pc.

## Rules for a job that can run there

- **A memory cap on every container** (`options: --memory=28g --memory-swap=28g`): on Nov 20 `ptxas` in an uncapped CUDA container, with five CUDA containers up at
  once, took the machine down with a global out-of-memory. The PC has 60 GB; 28 GB leaves room for a second job and the system.
- **Bound the parallelism inside it** (`CARGO_BUILD_JOBS`, `xargs -P`, nvcc's `--threads`): one job must not take all 32 threads.
- **One job at a time.** A runner service runs one job at a time, so one runner service per kind of job is the rule. Nothing in a workflow may start build
  containers in parallel (no `docker run &`, no matrix that fans out on `glyd-pc`). Check that only one is registered:
  `systemctl --user list-units 'actions.runner*'` and `ls ~/actions-runners`.
- **Benchmarks.** CI load disturbs them: a benchmark reads `uptime` first and waits for a quiet machine; GPU measurements by hand take `~/glyd-scratch/gpu.lock`.
- **Never for a fork.** Every route above has the same-repository guard.

## Stopping and starting the runners

On glyd-pc, as the owner (the runners are user services; lingering keeps them up after logout):

```
systemctl --user list-units 'actions.runner*'      # which are running
systemctl --user stop 'actions.runner*'            # stop them all (a job in progress is cancelled)
systemctl --user start 'actions.runner*'
```

To take the PC out of CI without stopping anything, unset the variables: `gh variable delete CI_RUNNER` (the Linux jobs go back to GitHub's runners on
the next run) and `gh variable delete GPU_RUNNER` (the GPU workflows then do not run at all).

## The disk

A job on glyd-pc leaves its caches there (`glyd-ci-cache`, the Docker volume the GPU jobs mount, and the runner's work folder), and the disk filled once
("No space left on device", Nov 19). Check it before a long run: `df -h /` and `docker system df`. Cleaning is the owner's to name; the usual candidates,
in order of size, are `docker system df -v` (unused images of the old CUDA versions), the runner's `_work` folders and the Hub cache under `~/.cache/huggingface`.

## GPU jobs after a kernel change

A kernel update on glyd-pc needs the NVIDIA module built for it before the GPU jobs can run: until then they fail at the driver (`nvidia-smi`), which is the
machine's state and not a code failure.
