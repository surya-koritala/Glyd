# GLY-175 / GLY-176 / GLY-177: bounded native daemon acceptance

The approved corrected daemon experiment passed in 52.033 s. It exercised actual
catalog-backed coordinator/worker routing with one native CPU service: repeated
JSON/SSE semantics and exact usage, context refusal followed by reuse, scoped-token
and field/output refusals, successful CPU request drain, and midstream revocation
with retained failed ownership. No owned group remained; token absent from durable
state, current/prior quarantine retained, spend $0. Approval consumed.

The [complete record](native/2026-10-12-m1max-cpu-daemon-acceptance/README.md) includes
actual runner, immutable identities, full raw result and post-run verification.
It distinguishes the fixed-input local policy from general production, memory,
conversion and service cleanup qualification. It compares daemon JSON/SSE, not a
new direct reference or two physical machines. Independent review remains deferred,
not approved. M2 remains open and M3 unstarted.

Correctness review checks response/refusal/retained-state outcomes, input/driver
hashes, count/generation/time/RSS bounds, durable token withdrawal and quarantine.
Model-free checks cover denied process observation and prompt failure on terminal
service state. Complexity review retains the standard-library runner and reuses
the prior catalog/build/qualification records. No production code, dependency or
CI change is added here; ordinary CI keeps native tests ignored and warnings denied.
Exact final-head PR/CI will be linked in Linear before merge.
