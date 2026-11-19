# GLY-182: trusted local stage queue and retained inputs

Issues: [GLY-182](https://linear.app/getglyd/issue/GLY-182),
[GLY-183](https://linear.app/getglyd/issue/GLY-183),
[GLY-181](https://linear.app/getglyd/issue/GLY-181).

The increment connects authentic reassembly receipts to the existing core atomic
local-boundary start and opaque native hold. It uses a bounded nonblocking std
queue, exact retained input ownership and step-bound output admissions over the
actual enrolled output route. No provider process or daemon pipeline starts.

The resource bag is owned by StageWorker even when its public backend/ticket
handles are dropped. Only trusted qualified local step drain or full-instance
cleanup releases retained inputs; provider replies/counters alone do not qualify
that cleanup. No second lifecycle, tombstone registry, dependency, CI job, retry,
deadline increase or security instrumentation is added.

## Controlled checks

The existing security unit-test target includes
`authorization::stage::tests::local_owned_chunks_keep_inputs_until_qualified_drain_and_gate_actual_output_routes`.
Its optional third enrolled worker permits genuine middle-stage output to the
next peer, while the existing fixture remains two workers by default. Preparation
finishes before fresh grants are issued; the test adds no sleep or timeout change.

Cases cover authentic two-tensor BF16 input handoff, exact receiver-lease grant
binding, queue full/disconnection, authority withdrawal, dropped handles with
retained budgets, lost accepted ticket followed by qualified full cleanup,
actual output-route admission, mismatched invocation/phase/position/token count,
local cancellation with a retained bag, and a retired derived grant whose
expired authority cannot withdraw the still-owned lease. Stale output abort and
ticket Drop cannot mutate through old ownership. Cloned derived guards retain
the input reservation until they too are dropped.

These TLS/file/byte controls run in ordinary platform CI only. Locally the pinned
offline Rust 1.97.1 compiler checks all targets with at most two Cargo jobs:

```
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
git diff --check
```

Exact signed commit and source/main CI results belong in the issue/PR. No local
network, process fixture, device probe, model, GPU or native workload is run.

## Remaining acceptance

The actual qualified supervisor, installed package/model/runtime binding,
registered partial-boundary mapping, durable process/cache ownership, first-stage
input and sampled-token next-step feedback, complete daemon pipeline scheduling,
native cleanup proof, physical/numerical/performance evidence and independent
milestone qualification remain open. TokenIds output is refused until GLY-188
defines that feedback mapping. Object destruction and simulated trusted cleanup
calls are not evidence that real work drained.
