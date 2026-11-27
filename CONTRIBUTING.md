# Contributing

Two rules above all: nothing may fail to read back what an earlier version wrote, and every claim about speed, size or
quality is a measurement someone can repeat. Open an issue before anything beyond a small fix. Security problems go to
[SECURITY.md](SECURITY.md), not a public issue.

New here? Issues labeled [good first issue](https://github.com/surya-koritala/Glyd/labels/good%20first%20issue) are small and say
which files to start in. The labels are listed in [docs/labels.md](docs/labels.md). Questions about an issue go in the issue; other questions, ideas and
help with using Glyd go in [Discussions](https://github.com/surya-koritala/Glyd/discussions).

## Build

```bash
cargo build --release --workspace                      # codec, store, packer, engine (no nvcc: engine without CUDA kernels), Starlings
NVCC=/usr/local/cuda/bin/nvcc GLYD_ENGINE_ARCHS="89" \
  cargo build --release -p glyd-engine                 # engine with kernels for one GPU arch (CUDA 12.8+)
bash gpu/build_lib.sh target/gpu                       # GPU library (libglyd_gpu_cudaN.so): the Python package's, and what `foldcheck` compares the engine's kernels with (GLYD_GPU_LIB=PATH); a CUDA build of the engine loads none
GLYD_HIP=1 cargo build --release -p glyd-engine        # AMD RDNA 3/4 (ROCm hipcc)
cargo build --release -p starlings-cli                  # cluster: target/release/starlings (Starlings is in the workspace; edition 2024, rustfmt defaults)
```

## Test

| What | Command | Needs |
| :--- | :--- | :--- |
| Codec, store, packer, engine host code, Starlings | `cargo test --release --workspace` | CPU; IPv6 loopback and `systemd-analyze` for a few of Starlings' tests |
| Engine release feature set | `cargo test --release -p glyd-engine --no-default-features` | CPU |
| Starlings alone | `cargo test --locked -p starlings-protocol -p starlings-core -p starlings-adapters -p starlings-cli -p starlings-security` | CPU; IPv6 loopback and `systemd-analyze` for a few tests |
| Engine on a GPU | `glyd-engine check`, `kcheck`, `kindcheck`, `logitscheck`, `usercheck`, `bench` (development build) | NVIDIA GPU |
| GPU jobs on rented machines | `glyd-engine/jobs/*.sh`, `tools/runpod/` | see the scripts' headers |

## What CI runs today

On every pull request and push to main (`.github/workflows/`):

- `ci.yml` on every pull request and push to main. Its `changes` job lists the files a pull request touches; a pull request that changes
  only docs, benchmarks, research, condor, starlings, cards, tools or other workflows skips the code jobs below, so it reports in seconds; it still runs the
  Starlings job (tests, lint, dependency policy) where it changes `starlings/`, actionlint where it changes a workflow or an action, and Versions agree where it changes `tools/report/`.
  The jobs: Build & Test (the codec and store tests, regression floors, C ABI, a CLI round trip), Lint (`cargo fmt --check`,
  `cargo clippy -D warnings` over the workspace and the engine's release feature set, cargo-deny), Versions agree, the `glyd-gpu`
  crate's tests, and the engine's host tests as a development build and as the release build, with no nvcc and no GPU. When Rust
  code changes: a Minimum Rust version job (`rust-version` 1.89), the workspace compiled and unit-tested on Windows x86_64, Linux aarch64 and macOS arm64
  (six jobs: the codec, store and CLI, and the engine, on each) where the pull request carries the `platforms` label, changes a manifest, a build script, the checkout rules or `ci.yml`, or
  changes Rust lines near code for one system (`cfg` on windows, unix or target_os; the engine's release build on the three systems runs on main, the nightly and by hand only), and, when the kernels or the
  engine's sources change, `HIP kernels compile (AMD)` (ROCm's compiler, no GPU). The tests build with `--profile ci`
  (`Cargo.toml`: release code, no fat LTO) and keep their build in a cache saved from main.
  **`CI result` is the one check to require** on main: it waits for all of them and passes when each succeeded or was skipped.
  Main is protected: `CI result`, `Tree size` and `signoff` must pass, and the rules apply to admins too (no `--admin` bypass). Merge with
  `gh pr merge <N> --merge --auto`: it lands when the required checks pass, with no need to bring the branch up to date first. If main breaks after a merge,
  fixing it comes first. (`ci.yml`, `tree.yml` and `dco.yml` also run for a GitHub merge queue, `merge_group`, which main would use if the repository moved to an
  organization; a queue needs one.)
- Only when their paths change: `python.yml` (the Python GPU package and vLLM plugin), `kernels.yml` (the engine's kernels
  compile with nvcc 13 for sm_80 to sm_120, no GPU), `starlings-ci.yml` (the full matrix, macOS and Windows included: nightly, by hand, and on a pull request labeled `full-ci`; the Linux run of a pull request or push is a job of `ci.yml`), `condor-ci.yml`, `cpu-pack.yml`.
- Always: `tree.yml` (no large data files, `scripts/check_tree.sh`) and `dco.yml` (sign-off on every commit).

GPU tests (`gpu.yml`) run on one self-hosted machine, the one `vars.GPU_RUNNER` names (`tools/ci/setup-gpu-runner.sh` registers
an Ubuntu machine's GPU). They run on pushes to main that touch `glyd-engine/`, `gpu/`, `glyd-gpu/`, `tests/clients/`, the lockfile or the job's own scripts and workflows, on a
pull request from this repository labeled `gpu`, and by hand, never for a fork. They build the engine for sm_89 (set
`vars.GPU_RUNNER_ARCHS` for another GPU), run its kernel checks, google/gemma-4-12B-it end to end, and every family's checks on small random models of its
layout made in the job (`gpu-tests.yml`: logits, several users at once, the lossy levels, the expert cache). That covers one GPU
architecture and one released model: other releases, other GPUs and multi-GPU runs are tested by hand (the table above). `gpu-cloud.yml` starts a cloud instance for a run you ask for; it never runs by itself.

## What a change must show

- **Correctness.** Round trips byte-exact; format changes keep old files readable (a test reading files the released
  version wrote, in `tests/data/legacy/`, and the format in `docs/codec/spec.md`). A new model or kernel comes with a logits
  test against the CPU reference.
- **Speed, size, quality.** Before and after on the same machine, with the command: `glyd-engine bench` (docs/bench.md), `costs`, `rates`
  for the engine; `examples/bench_suite.rs`, `scripts/bench_versions.sh` for the codec. Results go in `benchmarks/`.
- **Scope.** One change per pull request; code that reads like the code around it.
- **Changelog.** A user-facing change adds its entry as a new file, `changelog.d/<slug>.md`, not an edit of CHANGELOG.md
  ([changelog.d/README.md](changelog.d/README.md)); the release puts the entries together.

## Adding a model

A model is added by its kind and its family, not one by one: a model of a family that is there runs with no code of its own.

1. Describe the architecture from its `config.json` as layer kinds and weight names (`glyd-engine/src/runtime/describe.rs`), and lower it to the
   executor's operations (`glyd-engine/src/dynamic/lower.rs`). The executor's one loader (`Dyn::new_placed`) reads the weights from that description, so a family adds no loader.
2. Add or reuse the CPU reference of each layer kind (`glyd-engine/src/runtime/reference/`; `glyd-engine kindcheck` holds the kernels to it).
3. Test logits against the reference and greedy tokens against the original checkpoint.
4. Give the family its row in `glyd-engine/src/support.rs` (its levels, where its weights may go, how the planner sizes it) and its architecture names in `glyd-engine/src/runtime/config.rs`; `glyd-engine supports`
   then runs it, and `costs` and `bench` measure it.

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
