# GLY-176 allocator and completion integration

Provider contract: [service PR168](https://github.com/Sigbound/glyd-gpu-private/pull/168),
`8552912684394e49566564f49902997299be15e0`, `glyd-serve/1`.
This is Compute consumer integration; no engine implementation was copied.

The existing owned service now checks actual memory receipts during readiness,
before dispatch and after owned completion/cancellation. It reuses bounded,
duplicate-aware JSON and the existing telemetry TTL. Null parts and declared
overhead remain explicit; same-capture reads do not renew freshness. Contradictory
parts, changed/reset counters, non-idle status or changed approved identity prevent
terminal success/service reuse and stop the owned service. The existing CPU drain
and durable cleanup boundaries remain unchanged: ended requests keep preallocated
KV/state and cannot release GPU reservations or quarantine.

Controlled checks:

- `allocator_receipts_preserve_unknowns_retention_and_capture_age` is a pure
  in-memory check with injected observation time. It checks sums/overflow, nullable
  parts, malformed fields, device/counter changes and expiration without sleeping.
- `glyd_completion_requires_ready_identity_and_retained_memory_before_terminal_output`
  runs only in existing model-free CI child tests. JSON/SSE with reset/inconsistent
  counters or a still-active service must fail before terminal output, stop the
  service and remain ineligible for the CPU drain path.
- Existing owned cancellation/reuse and legacy wire/digest tests retain coverage.

Local validation uses cached offline Rust 1.97.1, at most two Cargo jobs: pure
selected checks, formatting and warning-denied lint/compilation only. Process and
socket tests run in ordinary existing source/main CI, not on the local Mac.
Exact reviewed source, PR and CI results are recorded in Linear after qualification.

Local results: pure selected check passed (0.00s; cached compile 2.67s), default
and all-feature warning-denied Clippy passed (3.97s/4.39s), test-backend test
compilation passed (5.78s), formatting/diff checks passed. Separate correctness
review found a valid same-millisecond/change-free-memory case; the fix retains
the original age while accepting coherent changes and is covered by the pure
check. Subsequent correctness and separate complexity reviews found no blocker.
Independent milestone/release qualification remains deferred, not approved.

Limits: allocator counters are not physical pool inventory, native attestation,
per-process driver/host overhead or memory-budget calibration. The first observed
provider timestamp is self-reported; receiver TTL does not authenticate its age.
No GPU route is enabled, request budget fabricated, package installed, workload
run or native acceptance closed. Installable package/hash/index and approved
installed native resource/drain/recovery qualification remain open.
