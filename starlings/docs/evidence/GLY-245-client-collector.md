# GLY-245 bounded client collector

Issue: https://linear.app/getglyd/issue/GLY-245

## Delivered software

`scoreboard collect MATRIX PLAN NEW_MATRIX` drives already-running numeric-loopback
chat services and regenerates the existing offline scoreboard. It verifies matrix
artifacts/bindings, exact private request bytes, expected prompt usage and supported
metric definitions before requests. Five to eight paired trials alternate system
order, with bounded within-batch concurrency. Client first-content latency and
batch usage throughput have explicit boundaries; JSON duration is never TTFT and
SSE chunks are never counted as tokens.

Raw JSONL contains only hashes, bindings, trial ordering, clock labels, counts,
client timings, HTTP statuses and static outcomes. It records the collector binary
hash and both wire-body hashes. No input/output text, server response IDs, model
aliases, endpoint or credentials are persisted. Existing evidence is not overwritten.
Failure stops subsequent submissions because an ambiguous backend may still own
work; remaining planned slots stay explicitly unattempted. Cancellation is client
disconnect, not proof of drain. No backend launch, automatic retry or redirect.

## Review

Correctness/security review traced CLI dispatch, strict preflight, every network
submission, bounded JSON/SSE reads, finish/usage ordering, failure-stop, raw sync/hash
and final matrix validation/reporting. Fixed relative output filenames to resolve
against the current directory. Scope/limits are documented in
[the collector contract](../scoreboard-collector.md).

Separate Ponytail complexity review: reused existing strict JSON/ChatRequest,
matrix/artifact comparison, binary hashing, stdout error handling, serde and the
already locked ureq client. No new third-party package, framework, backend-launch
abstraction, thread pool or CI job. Removed unnecessary public config types and
lint allowances; a borrowed request context shares data across bounded threads.
Remaining HTTP output parsing is collector-specific and owns no service process.
Independent milestone review remains deferred, not independently approved.

## Checks

Pinned Rust/Clippy 1.97.1 with rustup shims first on PATH:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`
- `cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings`
- `RUSTFLAGS='-D warnings' cargo test -p compute-cli --lib --bins --test cli --locked --offline`
- Same test command with `--all-features`.

Controlled synthetic services cover paired order, JSON, fragmented CRLF SSE,
role-only events, two reported tokens across differently sized chunks, static
redaction, missing/malformed usage, invalid JSON, incomplete/oversized SSE,
HTTP failure and redirect refusal, first-content cancellation, timeout and stopping
all later submissions. Preflight controls cover invalid budget/endpoint/fixture,
unknown/duplicate fields, repeated cells, path escape, changed request bytes,
unsupported metric and exhausted pre-submission deadline. The collected matrix
replays through `scoreboard report`; existing outputs refuse replacement.
CLI controls separately verify fixture defaults, development-only opt-in and
redacted preflight failure before any output files exist.

Final pre-PR checks: default library 11, binary 6 and CLI 6 pass; development
library 19 pass/one existing ignored native acceptance, binary 10 and CLI 6 pass.
Default log SHA-256 `22c5f6a19e3dd9ae99ca96e31d3d3f06138410eefc9c18f03f7d197bee02e0e5`;
development `51db62afb697cc2b85c369d93f86a154476e8bc6d298883c14763cfb7da19664`.
A subsequent test-only timeout control uses 100 ms against a 200 ms synthetic
response delay, without assuming the listener was scheduled before connect expiry;
final source CI covers that adjustment.

Initial local checks caught a reference type error and unused version fields;
these were corrected without warning allowances. The first socket fixture failed
because macOS accepted sockets inherited listener nonblocking state; the fixture
now explicitly returns accepted streams to blocking mode before its bounded I/O.
This is not evidence of the historical Windows daemon cause. One permission-review
timeout occurred before a socket test started; the permitted retry was approved.

## Remaining qualification

GLY-245 stays open. Native best-working Exo/Glyd builds, approved 2026 model/device
workloads and real matched results remain pending. Server decode/prefill, memory,
setup and other scoreboard metrics need their actual collection sources; no client
rate substitutes for them. Backend delivery GLY-238 and M2 acceptance remain open.
The current Windows queue failure stays separately tracked in GLY-267; this
collector makes no claim to fix it. All seven exact-source CI checks, current-main
review and authorized merge must complete before this patch is treated as merged.
