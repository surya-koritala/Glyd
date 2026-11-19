# GLY-197 opt-in redacted diagnostic bundle

Issue: [GLY-197](https://linear.app/getglyd/issue/GLY-197).
Base before PR: main `625f233298256ff6daf96cc43970aa6c36f8fd64` (PR #97).
The initial implementation/checks used `3537c52` before that integration.

## Implemented software

`cluster diagnostics` reuses the existing local authenticated status/nodes/service
calls, exactly once each. The library projects replies into a small typed schema
and writes JSON to an explicit stdout writer. It starts no daemon/service/model,
performs no host/process/network inspection, reads no log/config/environment bundle
and sends no telemetry or support message. There is no new daemon operation,
public listener, dependency, retry, per-node fan-out or CI job.

Static labels, counts, known service flags and nullable reported reservations are
the only exported runtime fields. Unavailable/invalid sections retain explicit
actions; raw messages/errors never enter the output. No section is a new
measurement, atomic authority snapshot, grant, native readiness or cleanup proof.
See [the exact contract](../diagnostics.md).

## Validation and review

Two bounded in-process controls pass in each feature mode (0.02s/0.01s observed
execution). They use synthetic AdminReply values, never open sockets or inspect
the host. They exercise all11 known service states, fixed counters/member counts,
nullable reservations/no service, unavailable/refused/wrong-code sections, malformed
and duplicate JSON, repeated status fields, unknown states, overflow, size and
full64/65-member limits/duplicate nodes, retained good sections, no fan-out and redaction of sentinel
credentials/errors/identifiers/addresses/paths. An injected BrokenPipe writer
proves the actual write-error path returns a static failure without panic; closing
stdout before a Rust process starts would not establish that behavior.

The first test draft falsely matched the ordinary word endpoint in a static action
as if it were an exported endpoint field. The corrected assertion checks the JSON
key and actual sentinel addresses. No raw value was exported in that initial run.
An added membership-bound control also exposed a test helper-name shadow, fixed
before the final checks. Before integration, final warning-denied lints pass
1.34s/1.17s; after merging current main, both unit modes pass0.02s/0.02s and both
warning-denied workspace/all-target lints pass2.97s/3.14s. Both formatting and
diff checks pass. Mac work uses at most2 build jobs, the installed SDKROOT and
cached/offline dependencies only.

One assertion block extends the existing real-process daemon lifecycle case in
CI to exercise the actual CLI command, valid JSON and coordinator/no-service
projection plus identifier/address omission. It adds no server/job and retains
all original lifecycle/security assertions and time limits. This IPC/network
fixture is not run locally under the basic-Mac-only limit. All seven exact-source
checks, current-main review and authorized merge remain required.

Correctness/security review traces CLI selection, only the three fixed commands,
existing authenticated IPC, strict bounded input, typed whitelist projection,
static error actions, JSON output and writer failure. Separate Ponytail complexity
review keeps three direct existing calls, serde/standard-library parsing and one
small shared projection guard; no collector framework or diagnostic daemon.
Independent milestone/release review is deferred, not approved.

Full GLY-197 remains open for actual model/profile/readiness/allocation views,
per-node exclusion and fit explanations and their native integration/review gates.
Packaging/clean-machine/release support and all M2/M3 qualification remain open.

## Catalog verification and stopping states

At base `d64e90ceb3076c2a5430c3f2b80b4e0259e74319`, the service owner already
reports `verifying_catalog` while preparing catalog assets and `stopping` while
stopping the owned service. The diagnostic projection omitted both fixed labels,
so valid reports from the same build became `invalid_report` with a misleading
CLI/daemon compatibility action. The projection now accepts those two labels
without adding a query or exporting any additional raw field.

The existing pure state matrix now checks all 13 producer states and requires an
available section with the exact state. Its shared privacy checks still exclude
sentinel errors, credentials, identifiers, addresses and paths. The existing
refusal control still rejects unknown states, malformed/duplicate reports and
oversized values. Before the enum fix, the expanded matrix failed at
`verifying_catalog` in 0.01 seconds; the refusal control still passed.

Checks use pinned Rust 1.97.1, cached offline dependencies and two Cargo jobs:

```sh
cargo test --offline --locked -p compute-cli --lib diagnostics::tests::
cargo test --offline --locked -p compute-cli --lib --all-features diagnostics::tests::
cargo clippy --offline --locked -p compute-cli --all-targets -- -D warnings
cargo clippy --offline --locked -p compute-cli --all-targets --all-features -- -D warnings
cargo fmt --all --check
git diff --check
```

These controls use synthetic replies in memory; they run no CLI, daemon, service,
socket, process fixture, device probe or model. Existing redaction, unknown-value
refusal and qualification boundaries remain unchanged. Full GLY-197 acceptance
and installed/native/clean-machine qualification remain open; final source/main
CI and signed commit are recorded in the issue/PR.
