# Contributing

Two rules above all: nothing may fail to read back what an earlier version wrote, and every claim about speed, size or
quality is a measurement someone can repeat. Open an issue before anything beyond a small fix. Security problems go to
[SECURITY.md](SECURITY.md), not a public issue.

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

CI runs the CPU suites on every pull request. The GPU suites (`.github/workflows/gpu.yml`) run on `main` and on request once
a GPU runner is registered (`vars.GPU_RUNNER`); until then, run them locally as below.

## What a change must show

- **Correctness.** Round trips byte-exact; format changes keep old files readable (a test reading files the released
  version wrote, in `tests/data/legacy/`, and the format in `docs/spec.md`). A new model or kernel comes with a logits
  test against the CPU reference.
- **Speed, size, quality.** Before and after on the same machine, with the command: `glyd-engine bench`, `costs`, `rates`
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
