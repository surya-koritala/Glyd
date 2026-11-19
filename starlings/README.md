# Starlings

**Beta:** run one model across two GPUs, in one machine or two on a LAN, with one command per
machine. See [the beta guide](docs/beta.md).

Start with the [operator entry guide](docs/getting-started.md) for current commands,
evidence, support limits and failure actions.

Independent local distributed inference project. The current foundation contains
five Rust crates, strict versioned contracts, bounded lifecycle/admission logic,
an opt-in three-process fault simulator, and a daemon with fingerprint-approved
enrollment, protected local administration and authenticated registry heartbeats. Follow the [daemon guide](docs/daemon.md)
for init/join/status/nodes/drain/stop/leave. [Discovery and topology](docs/discovery.md)
adds explicit multi-interface routes and optional untrusted mDNS hints. The [registry contract](docs/registry.md)
covers durable epochs, freshness and restart fencing. The daemon routes whole
requests through explicitly approved external CPU services; see
[qualification and activation](docs/production-routing.md). It does not shard real
models or automatically qualify an installed backend.

The [model catalog](docs/catalog.md) hashes protected source files, registers exact
manifests and records explicit local approval separately from artifact readiness.
Use `model inspect`, `seal`, `register`, `approve`, `show`, `check` and `revoke`;
these commands do not download weights or run inference.

[`benchmark inspect RUN_JSON`](docs/benchmark-inspection.md) checks supplied run
records and evidence hashes offline. It preserves failures and never qualifies a
backend or starts a workload.

[`cluster diagnostics`](docs/diagnostics.md) prints a redacted JSON support bundle
from three existing authenticated local queries, with no probes or backend start.

M2 also provides verified artifact storage, authenticated host-memory diagnostics,
a supervised external service adapter and bounded placement logic. The first
[native API smoke](docs/evidence/native/2026-10-12-m1max-cpu-api/README.md) passed
on an M1 Max CPU. Client/coordinator/worker routing, JSON/SSE, bounded placement
and failure handling are implemented and tested with synthetic services. A bounded
[native daemon acceptance](docs/evidence/native/2026-10-12-m1max-cpu-daemon-acceptance/README.md)
also exercises a fixed-input Mac CPU route. General installed Glyd capability,
precision, memory/drain qualification and matched real-worker benchmarks remain
M2 gates; none is inferred from synthetic CI.

| Crate | Responsibility |
| --- | --- |
| `starlings-protocol` | Shared protocol definitions |
| `starlings-core` | Control-plane state and admission |
| `starlings-adapters` | External backends; test-only fake backend |
| `starlings-cli` | The `starlings` executable |
| `starlings-security` | Protected identities, verified TLS, live authorization, revocation and bounded listeners |

The workspace uses established Rust libraries for JSON/schema validation, UUIDs,
timestamps, canonicalization and hashing; see [dependency purposes](docs/dependencies.md).
It has no GPU, model, private source or external inference runtime dependency.
The [enrollment contract](docs/enrollment.md) covers GLY-164 APIs, trust boundaries
and remaining M1 gates. [Live authorization](docs/authorization.md) covers GLY-165
permissions, renewal and worker cancellation. Tests use loopback TCP with real
certificates and TLS. [Listener limits](docs/listeners.md) covers GLY-166 loopback HTTP,
TLS admission, local administrative IPC and remaining integration gates.
The [security boundary inventory](docs/security-boundaries.md) maps enforcement
entry points and remaining release-review obligations without claiming an audit.

## Build and check

Install Rust through rustup. The repository pins Rust 1.97.1, including rustfmt
and clippy. Use the pinned toolchain for every command:

```sh
rustup run 1.97.1 cargo fetch --locked
rustup run 1.97.1 cargo build --workspace --locked --offline
rustup run 1.97.1 cargo fmt --all -- --check
rustup run 1.97.1 cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
rustup run 1.97.1 cargo test --workspace --locked --offline
rustup run 1.97.1 cargo test --workspace --all-features --locked --offline
rustup run 1.97.1 cargo run -p starlings-cli --locked --offline -- doctor
```

CI runs default and development-feature checks on Linux x64, Linux ARM64,
macOS and Windows.
On Linux, the full CLI test suite also needs `/usr/bin/systemd-analyze` >=250
for [inactive unit-file validation](docs/packaging.md); it starts no service.
Windows caches the pinned MSVC toolchain and update hash separately from Cargo
builds. The backend partition is the sole cache writer after successful toolchain
installation; all three partitions still run rustup to validate the pin/components.
Default Windows priority and remaining non-CLI tests use the same Cargo package
selection, avoiding separate shared-crate build graphs. Only the priority test
runs in the first pass; the remainder skips that exact case, and CLI tests remain
serialized separately. Empty filtered harnesses do not count as extra coverage.
Each PR update runs one matrix; pushes to `main` also run CI. New updates cancel
older PR runs; a failed platform leaves sibling diagnostics running. Formatting
and both lint configurations precede tests; tests also build the binaries.
The dependency/build cache is keyed by OS, architecture, lockfile, manifests,
toolchain and workflow, with an OS/architecture fallback. It stores registry
archives/index and compiled artifacts, following [Cargo's CI cache guidance](https://doc.rust-lang.org/cargo/guide/cargo-home.html#caching-the-cargo-home-in-ci).
Unpacked dependency sources are recreated by the locked fetch instead of being
stored twice; legacy fallback sources are removed before that fetch. Cargo revalidates source,
compiler, feature and dependency fingerprints on reuse; every hit/miss runs the
same complete checks. CI disables
incremental/debug artifacts to reduce compile and cache size; local profiles stay
unchanged. Compiler, documentation and Clippy warnings fail the checks. Jobs have a
ten-minute limit and emit a platform-specific error summary on failure.
Default and catalog lifecycle checks print failure details immediately. Windows
runs its known whole-request security lifecycle case first, then the disjoint
remaining default suite, without omitting or duplicating it. Windows default runs
the default suite; Windows backend runs the development lint and production route;
Windows lifecycle runs development tests, short catalog controls and recovery.
Backend and CLI tests are serialized;
the remaining non-CLI default tests retain Rust's normal harness scheduling.
The three jobs run independently and their test groups are disjoint; the aggregate
Windows check requires all three to pass. This adds one required lifecycle check
while retaining the seven existing check names and every test command.
Optional cache writes start only during the first seven minutes after the initial
step, reserving roughly three minutes of the unchanged cap. Late validation
skips cache preparation/upload; every test and warning check remains required.
Toolchain installation and the initial locked dependency fetch require network
access. Subsequent checks use offline mode. Keep rustup's shims ahead of other Rust installations in PATH so Cargo's
formatting and lint subcommands use the same pinned tools.

## Current hardware scope

The [private candidate builder](docs/packaging.md) stages checksummed default CLI
artifacts for review. Desktop installation, service templates and release
qualification remain open; no license or publication approval is implied.
Prepared source checkouts can also exercise a
[private Cargo prefix](docs/packaging.md#private-source-staging-with-cargo),
including static CLI output and tracked removal.

`starlings scoreboard report examples/scoreboard-plan.json` validates the
offline comparison plan and shows every missing measurement. It runs no backend;
see the [versioned record contract and comparison rules](docs/scoreboard.md).

The current scoreboard scope is NVIDIA-only and Mac-only single machines and
same-type clusters. Mixed NVIDIA/Apple, Raspberry Pi and mobile work is parked;
existing ARM64/CPU policy code remains. See [scope and gates](docs/architecture.md).
Linux ARM64 CI validates software, not Raspberry Pi inference or thermal behavior.

The available inventory is one M1 Max Mac and one Ubuntu RTX 4080 SUPER machine.
It cannot supply a two-node same-type comparison. Native/device workloads require
separate approval; missing second NVIDIA/Mac nodes remain unmeasured. No hardware
purchase, rental or new workload is implied.

## Development backend

The fake backend is excluded from the default build. `simulate` starts a
coordinator and two worker subprocesses over bounded local pipes. Eight scenarios
check whole requests, ordered stages, duplicate delivery, rollback, expiry,
cancellation, disconnect, crash and stale telemetry. Byte transforms and injected
time are synthetic. No model, network listener or throughput measurement is involved:

```sh
rustup run 1.97.1 cargo run -p starlings-cli --features test-backend --locked --offline -- simulate
```

## Tracking and license

M0 implementation is tracked by GLY-158 through GLY-163. Read the
[architecture](docs/architecture.md), [decisions](docs/decisions.md),
[development rules](CONTRIBUTING.md), [contract boundary](spec/README.md) and
[M0 review evidence](docs/evidence/M0.md) and [integrated M1 review](docs/evidence/M1.md).
The [project index](https://linear.app/getglyd/document/glyd-compute-project-index-and-execution-tracking-6c19de2ab4f4)
links the architecture, decisions and milestone gates. Review and validate each
milestone before advancing to the next.

The project license is undecided. All packages have `publish = false`; no license
grant is implied. Existing Glyd licensing remains separate. No Exo or Glyd engine
code is included.
