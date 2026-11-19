# GLY-184 receive cursor and bounded replay

Issue: [GLY-184](https://linear.app/getglyd/issue/GLY-184).
Base: `29df7043ce3d12d37789f031690f8e4480e422ea` (PR #94).
PR/source/CI: recorded in Linear after publication and qualification.

## Change and review

The actual FlowReceiver byte counters now live in one private, dependency-free
ReceiveCursor. The normal tests and optional libFuzzer entry point include that
same source. They do not maintain a second implementation of production rules.
The independent oracle instead records each written byte and sums grants/spend
in u128, checking valid/invalid transitions and conserved credit.

Production sequencing stays intact: credit is committed only after a successful
send; bytes are read into the existing reserved buffers and the candidate cursor
is committed only under the post-read live admission check; the tensor advances
only after successful receipt transmission. Error handling still aborts the
immutable local generation and drops unexecuted host assembly. Final handoff,
native ownership, authority, deadlines, record budgets and sender are unchanged.
The extracted cursor also checks positive chunk size and the final flag against
the actual buffer extent; the wire parser already enforces those conditions.

Correctness/security review checked both authority and delegated admission paths,
header normalization, buffer extents and commit/receipt/handoff boundaries; no
new authority, ownership release or retry path was found. Separate complexity
review removed unused derives and kept one private value type and one shared
oracle, no public abstraction, dependency, framework or new CI job. Independent
milestone review remains deferred, not approved.

## Basic checks and limits

Only basic non-instrumented Mac checks are permitted. The two deterministic tests
cover ordered split/two-tensor traces, every truncation and byte mutation of one
valid trace, zero/oversized credit, cumulative overflow, duplicate/gapped/oversize
chunks, exhausted credit, wrong final flag and premature finish. Sixty-four valid
profiles vary one to eight tensors, lengths1/2/255/256 and bulk/per-chunk grants.
Each oracle input owns at most2048 byte-presence flags and handles at most64
actions. There are300 oracle calls across both tests, no networking,
process inspection, backend, model, instrumentation or paid infrastructure.
Normal builds use at most two jobs. Commands and actual outcomes follow below.

Local results: both deterministic tests pass (0.00s test execution), both format
checks and diff whitespace check pass, and both workspace/all-target lint modes
pass with warnings denied. Platform integration remains subject to exact-head
CI and a separate automatic main run after any authorized merge.

```sh
CARGO_BUILD_JOBS=2 cargo test -p compute-security --test receive_cursor --locked --offline
cargo fmt --all -- --check
cargo fmt --manifest-path fuzz/Cargo.toml -- --check
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked --offline -- -D warnings
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
```

The optional receive target is prepared, not an instrumented campaign result.
Real socket/authority/abort and allocation-ledger regression coverage runs in the
existing bounded platform CI. This work does not measure RSS, TLS/host copies,
physical links, native execution, Exo performance or full milestone acceptance.
Those GLY-184 items and independent milestone review remain open. Historical
parser campaign observations are preserved in their original evidence record.

## First source qualification failure

Source `1c7cdb837b3e05c5aeec54d7418b26521b2d072c`,
[CI38899172407](https://github.com/Sigbound/glyd-compute/actions/runs/38899172407),
failed Windows backend job116966885313 in the existing CLI library relay check.
At12:18:14UTC its heartbeat thread returned Transport; the server registry handler
returned Transport at12:18:15, then the enclosing scoped test panicked. Twenty
other CLI library tests passed; that step took75.07s. Raw job log SHA256:
`b09098f88e681f9d1a650687f1f51f80c0ea7314fad5e1425b1bc83491929224`.
These redacted errors do not identify the initiating syscall or exchange phase.
No root cause, cursor regression or successful qualification is inferred.

The correction adds opt-in debug-only failure labels for proof/TLS/frame phases
and I/O kind/numeric OS code, never OS messages, addresses, credentials or payloads.
Windows runs the same relay test first with this flag, then the existing suite
with that exact case excluded. This changes observability/order, not protocol,
assertions, retries, deadlines, platform coverage or job caps. A local list-only
check proves disjoint exact/rest sets (1+21=22 local cases, including the Unix-only
ignored native case; Windows omits that existing Unix-only case). No socket test
was executed locally. Both warning-denied lint modes and YAML/format/diff checks
pass. The source failure is retained; this is not an unchanged-source rerun or a
claim to explain historical GLY-267 failures. New exact-source CI must qualify it.
