# GLY-185 delivered range-worker wire mapping

Issue: [GLY-185](https://linear.app/getglyd/issue/GLY-185).
Provider contract: [glyd-range/1 API documentation at d0e389c](https://github.com/Sigbound/glyd-gpu-private/blob/d0e389cbcedbc177935a8b19c6ae78cae03e6127/docs/range-worker.md).
PR/commit/CI: recorded with the reviewed source and automatic main result in Linear.

`partial::glyd::Worker` maps the delivered pipe framing and
describe/prepare/commit/begin/run/status/cancel/end/unload operations through
exclusive caller-supplied `Read`/`Write` halves. It reuses bounded duplicate-aware
JSON, immutable `BoundaryContract`, `RunChunk`, `StepIdentity` and `DataScope`.
Package/asset approval, process deadlines and live StageGrant/lease/output fencing
remain the existing supervisor/coordinator's responsibility. No launch, daemon
activation, backend implementation, dependency or workflow is added.

The pure in-memory control exercises the public adapter API and complete wire
replies, including independent status/cancel replies before chunk completion,
complete ordered raw bodies, owner/epoch/lease/session/generation/plan/sequence/
position mismatches, duplicate invocation refusal, cancelled-success withdrawal,
first/last u32 token and BF16 logits mapping, final-prefill token/output limits,
foreign IDs, duplicate JSON, invalid/truncated lengths, restarted incarnation,
partial-write failure and 301 sequential steps without a 256-request ceiling.
After ambiguous I/O/protocol failure the mapping cannot dispatch another chunk.

Commands use the pinned 1.97.1 rustup toolchain, offline/current lockfile and
`CARGO_BUILD_JOBS=2` with the existing debug compilation cache:

```sh
cargo test --offline --locked -p compute-adapters --lib partial::glyd::tests::owned_range_wire_maps_interleaving_cancellation_fences_and_bounded_failures -- --exact
cargo clippy --offline --locked -p compute-adapters --all-targets -- -D warnings
cargo clippy --offline --locked -p compute-adapters --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
git diff --check
```

The selected test opens no file, socket, service, subprocess or model and inspects
no host/device process. It is a software control, not an installed-worker probe,
GPU allocation measurement, numerical conformance, native completion/drain,
multi-machine pipeline or performance result. The provider's own native record
was received as documentation and was not rerun. Engine counters/zero retained
engine bytes cannot release a physical GPU reservation or catalog quarantine.
GLY-185 and milestone qualification remain open for actual installed integration,
live authority activation, numerical and resource/cleanup evidence and review.
