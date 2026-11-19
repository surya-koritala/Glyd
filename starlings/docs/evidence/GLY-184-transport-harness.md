# GLY184 independent transport harness

The [test-only harness](../transport-measurements.md) reuses existing authenticated
fixtures, core snapshots and credited GCP1 flow. It adds one mandatory8KiB/two-input
correctness case and one explicit ignored20-sample matrix, without production
APIs, dependencies, jobs, retries, timing thresholds or larger deadlines.

Local mandatory smoke passes0.25s, raw SHA256
1c0d6b1d6d5cef92471b16801704c5390d609df42c7199202554fbe1fcf2d661.
The explicit matrix passes20/20 in10.93s with immutable binary identity, exact bytes,
retained-payload/core-state assertions and completion footer. Raw JSONL and its
hash/limitations are in [the record](../../benchmarks/2026-10-16-loopback-transport/README.md).
Existing-output and relative-path negative checks fail before any fixture starts,
preserve the raw artifact and do not repeat transfers. Both final workspace/
all-target warning-denied lint modes, formatting and diff checks pass.

Correctness/security review covers exact templates/credit/receipts, unchanged core
state, retained input ownership, one absolute flow deadline, bounded matrix and
redacted output. Separate complexity review keeps the fixture/stdlib directly;
no metric framework, generic benchmark runner or new runtime. Stage times are
observed overlapping boundaries, not additive cost decomposition. Explicit nulls
preserve unmeasured costs.

GLY184 stays In Progress: coverage-guided/stateful reassembly fuzzing, allocator/
copy instrumentation, wired physical links, isolated TLS/scheduling/device costs
remain open. GLY183 deployed owner/native cleanup, GLY175 hardware/profile and
GLY241 persistent/cadence qualification remain gates. This is software-only
loopback evidence, not native inference, a device profile or an Exo win.
Independent milestone review remains deferred, not approved. No model, native,
GPU, Ubuntu, SSH, download, spending, quarantine reset, settings or publication.
