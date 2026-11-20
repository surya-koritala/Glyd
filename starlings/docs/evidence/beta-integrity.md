# Beta integrity and local regression coverage

Related work: GLY-363 and GLY-366. This follows the CI safeguards in
[PR #154](https://github.com/Sigbound/starlings/pull/154), source
`476e9d5af8323bdf62e71a345d37c11992bf4399`. Its
[draft workflow](https://github.com/Sigbound/starlings/actions/runs/41212760273)
was skipped: every job has no assigned runner. Hosted execution remains deferred
until included compute resets; the repository's $0 paid-overage budget is active.

Implementation source: `3f274f9cf26b1275fff449c43bbce35fdd44faff`, including installer
commit `14ce7e7bd099d815314a96c3aa4fe5ea6b61b874`. This evidence-only follow-up does
not change executable code.

## Implementation

- Checkpoint approval now covers file names, sizes and contents, reusing the
  catalog's streaming SHA-256 reader. It includes root configuration, tokenizer,
  templates and safetensors shards, independent of the absolute model path.
  Stages verify the identity before engine launch and after model/tokenizer load.
  Join codes carry the expected identity before enrollment.
- Cached Hub revisions are reused without fetching a moving `main`. Saved setups
  reject changed contents, incompatible workload/topology and invalid layer cuts.
  Invalid workloads are refused before downloading or launching an engine.
  Old saved records and stage files require a new explicit setup; there is no
  silent migration from description-only approval.
- Split estimates reject malformed tensor extents, missing layer weights,
  unsupported layer counts and integer overflow. Tied head embeddings are counted
  on the head stage. The estimate remains a model, not a measured GPU capacity.
- Installation retains notices, checksums and removal tooling. Staging validates
  the executable before replacing an install; activation failure restores the
  previous version. Foreign directories and command symlinks are preserved.
- The development engine exposes a bounded phase barrier for explicit fault
  qualification. It is absent from the default build. The full matrix is ignored
  by ordinary CI and adds no automatic platform jobs.

## Local validation

Validation uses Rust 1.97.1 on macOS ARM64, with rustup's shims first on PATH,
locked offline dependencies, synthetic files/engines and loopback sockets. Native
model opt-ins are unset. No real model, GPU, paid runner or published artifact is
involved. Both default and feature-mode workspace lint configurations pass.

- Default workspace: 312 passed, two explicitly ignored tests.
- CLI/adapter feature-mode libraries, binary units and CLI integration: 111 passed,
  one explicitly ignored native check.
- Installer regression: install, version upgrade, checksum rejection, broken
  executable, failed activation, foreign path preservation and uninstall retaining
  state. The GNU-tar beta packaging branch awaits Linux execution.
- Actual `up`/`join` entry flows with a stand-in engine: one-machine and two-machine
  enrollment, pasted invitation/approval codes and fingerprint confirmation,
  missing state parents, cached Hub revision reuse, READY, streamed chat, orderly
  stop, saved restart and rejection of changed weights and out-of-range saved cuts.
  A separate pipeline case verifies matching checkpoint files at different paths.
- Existing pipeline regressions retain UI/API, sampled/drafted responses,
  concurrency, link delay, hung calls and restart coverage.
- Eleven distinct pipeline/onboarding tests passed across the local batches,
  including the opt-in sixteen-case fault matrix and two-machine onboarding.

Commands (with `CARGO_TARGET_DIR` outside the checkout):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo test -p starlings-cli -p starlings-adapters --all-features --lib --bins --test cli --locked --offline
cargo test -p starlings-cli --all-features --test daemon pipeline_fake_engine --locked --offline -- --nocapture
python3 -I scripts/test-beta-install.py
```

For deliberate fault and two-machine onboarding qualification, run individually:

```sh
cargo test -p starlings-cli --all-features --test daemon pipeline_fake_engine_phase_fault_matrix --locked --offline -- --ignored --exact --nocapture
cargo test -p starlings-cli --all-features --test daemon pipeline_fake_engine_two_machine_up_join_and_saved_restart --locked --offline -- --ignored --exact --nocapture
```

The fault matrix passed all sixteen cases locally. It crosses both stages,
daemon/engine death, prefill/decode and one/two-session
configurations. It checks bounded request failure, refusal of dead engines,
restart success and departure of the old stand-in process. It does not prove
GPU allocation reclamation. A blocked stand-in is released after daemon death
to let it observe closed pipes; native hung-child cleanup remains a separate gate.
Do not rebuild the default executable in the same target directory during these
tests: their subprocesses require the feature-mode binary throughout the run.

## Remaining qualification and implementation

- Run exact-head Linux CI after the quota reset, followed by deliberate platform
  qualification. Do not equate a skipped draft workflow with tested code.
- The retained Windows failure occurred in the `two` production scenario: a
  prepared service lost its registry fence after a receive timeout; coordinator
  registry update took 3.631 seconds. Reproduce that case on Windows before
  changing deadlines or repeatedly rerunning successful platforms. No Windows
  root-cause fix is claimed by this patch. [Failure log](https://github.com/Sigbound/starlings/actions/runs/39883821036/job/120187213715).
- Native gates: a checkpoint that cannot fit on one selected device, measured
  peak GPU/host memory, current startup hashing cost, sustained concurrency,
  sampled/drafted equivalence, cancellation, slow readers, network loss and
  process/GPU cleanup for each fault phase. Existing two-4090 records do not
  establish the cannot-fit-on-one-device case. Device, runtime and spending caps
  must be approved before these workloads.
- Qualify the final installer and distributable on supported clean Linux systems,
  including the interactive two-machine join. Synthetic checks do not qualify a
  native engine/library pair. Signatures/provenance, license/distribution decision
  and release acceptance remain open (GLY-205/206).
- Apple GPU adapter/conformance work remains open (GLY-191/192); current beta
  coverage is Linux/NVIDIA. No cross-platform GPU or comparison claim is added.

Complexity review: no new dependencies, cryptographic implementation, workflow or
service. Existing engine supervision and pipeline fixtures are reused. The content
inventory is intentionally limited to assets read by the supported flat checkpoint
layout. Independent release review is not supplied by these local checks.
