# Contributing

Two rules above all: nothing may fail to read back what an earlier version wrote, and every claim about speed, size or
quality is a measurement someone can repeat. Open an issue before anything beyond a small fix. Security problems go to
[SECURITY.md](SECURITY.md), not a public issue.

New here? Issues labeled [good first issue](https://github.com/surya-koritala/Glyd/labels/good%20first%20issue) are small and say
which files to start in. The labels are listed in [docs/labels.md](docs/labels.md). Questions about an issue go in the issue.

## Build

```bash
cargo build --release --workspace                      # codec, store, packer, engine (no nvcc: engine without CUDA kernels)
NVCC=/usr/local/cuda/bin/nvcc GLYD_ENGINE_ARCHS="89" \
  cargo build --release -p glyd-engine                 # engine with kernels for one GPU arch (CUDA 12.8+)
bash gpu/build_lib.sh target/gpu                       # GPU library (libglyd_gpu_cudaN.so)
GLYD_HIP=1 cargo build --release -p glyd-engine        # AMD RDNA 3/4 (ROCm hipcc)
(cd starlings && cargo build --release)                # cluster: own workspace and pinned toolchain
```

## Test

| What | Command | Needs |
| :--- | :--- | :--- |
| Codec, store, packer, engine host code | `cargo test --release --workspace` | CPU |
| Engine release feature set | `cargo test --release -p glyd-engine --no-default-features` | CPU |
| Starlings | `cd starlings && cargo test --workspace --locked` | CPU; IPv6 loopback and `systemd-analyze` for a few tests |
| Engine on a GPU | `glyd-engine check`, `kcheck`, `kvtest`, `stress`, `bench` (development build) | NVIDIA GPU |
| GPU jobs on rented machines | `glyd-engine/jobs/*.sh`, `tools/runpod/` | see the scripts' headers |

## What CI runs today

On every pull request and push to main (`.github/workflows/`):

- `ci.yml` on every pull request and push to main. Its `changes` job lists the files a pull request touches; a pull request that changes
  only docs, benchmarks, research, condor, starlings, cards, tools or other workflows skips the jobs below, so it reports in seconds.
  The jobs: Build & Test (the codec and store tests, regression floors, C ABI, a CLI round trip), Lint (`cargo fmt --check`,
  `cargo clippy -D warnings` over the workspace and the engine's release feature set, cargo-deny), Versions agree, the `glyd-gpu`
  crate's tests, and the engine's host tests as a development build and as the release build, with no nvcc and no GPU. When Rust
  code changes: the workspace compiled and unit-tested on Windows x86_64 and macOS arm64 (six jobs: the codec, store and CLI, the
  engine, the engine's release build, on each), a Minimum Rust version job (`rust-version` 1.89), and, when the kernels or the
  engine's sources change, `HIP kernels compile (AMD)` (ROCm's compiler, no GPU). The tests build with `--profile ci`
  (`Cargo.toml`: release code, no fat LTO) and keep their build in a cache saved from main.
  **`CI result` is the one check to require** on main: it waits for all of them and passes when each succeeded or was skipped.
  Main has a merge queue: `gh pr merge <N> --merge` (or the button) puts a pull request in the queue, where `CI result`, `Tree size` and `signoff` run
  again on the pull request on top of the latest main (and the entries ahead of it) before it lands. Never merge with `--admin`: it skips the queue.
- Only when their paths change: `python.yml` (the Python GPU package and vLLM plugin), `kernels.yml` (the engine's kernels
  compile with nvcc 13 for sm_80 to sm_120, no GPU), `starlings-ci.yml` (Linux on a pull request and a push; the full matrix, macOS and Windows included, nightly), `condor-ci.yml`, `cpu-pack.yml`.
- Always: `tree.yml` (no large data files, `scripts/check_tree.sh`) and `dco.yml` (sign-off on every commit).

GPU tests (`gpu.yml`) run on one self-hosted machine, the one `vars.GPU_RUNNER` names (`tools/ci/setup-gpu-runner.sh` registers
an Ubuntu machine's GPU). They run on pushes to main that touch `glyd-engine/`, `gpu/`, `glyd-gpu/` or the lockfile, on a
pull request from this repository labeled `gpu`, and by hand, never for a fork. They build the engine for sm_89 (set
`vars.GPU_RUNNER_ARCHS` for another GPU), run its kernel checks and Qwen/Qwen3-0.6B end to end. That covers one GPU
architecture and one small dense model: the other families, other GPUs and multi-GPU runs are tested by hand (the table
above). `gpu-cloud.yml` starts a cloud instance for a run you ask for; it never runs by itself.

## What a change must show

- **Correctness.** Round trips byte-exact; format changes keep old files readable (a test reading files the released
  version wrote, in `tests/data/legacy/`, and the format in `docs/spec.md`). A new model or kernel comes with a logits
  test against the CPU reference.
- **Speed, size, quality.** Before and after on the same machine, with the command: `glyd-engine bench` (docs/bench.md), `costs`, `rates`
  for the engine; `examples/bench_suite.rs`, `scripts/bench_versions.sh` for the codec. Results go in `benchmarks/`.
- **Scope.** One change per pull request; code that reads like the code around it.

## Adding a model

1. Describe the architecture (layer kinds, weight names) next to the existing families in `glyd-engine/src/`.
2. Add or reuse a CPU reference for each layer kind (`*_ref.rs`).
3. Test logits against the reference and greedy tokens against the original checkpoint.
4. Add the model to `glyd-engine supports` and measure it with `costs` and `bench`.

## Results

A benchmark or measurement goes in `benchmarks/<run>/` as a summary (`README.md`) and the scripts that made it. Raw outputs
(logs, per-tensor JSON, CSV) stay out of git; CI refuses data files over 64 KB (`scripts/check_tree.sh`, `benchmarks/README.md`).

## Commits and licensing

Every commit carries `Signed-off-by: Your Name <you@example.com>` (`git commit -s`), certifying the
[Developer Certificate of Origin 1.1](DCO): you wrote the change or have the right to submit it under the project's
license. Contributions are licensed under the Apache License 2.0 ([LICENSE](LICENSE)), the license the project ships under;
there is no separate contributor agreement. Contributions made before 2026-11-19 were made under an earlier contributor
license agreement (in the git history). See [NOTICE](NOTICE) for earlier releases.

## Conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
