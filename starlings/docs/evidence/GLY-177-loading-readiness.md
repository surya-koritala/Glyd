# GLY-176 / GLY-177: temporary loading responses

The shared readiness probe now classifies an unauthenticated model-listing 503 as
temporary `Unavailable`, preserving the catalog daemon's existing bounded loading
loop. It cannot grant readiness until unauthenticated access is refused, authenticated
health succeeds and the exact model matches. Other unauthenticated responses,
including 200, remain fatal `Unapproved` errors. No precision, credentials,
reservation, drain, retry or quarantine policy changed.

The [native startup record](native/2026-10-12-m1max-cpu-daemon-startup/README.md)
preserves both consumed approvals and failures, exact inputs and runners. The
renewed attempt ended in 71.758 s with zero generations/counts, no owned group
remaining and $0 spend. It retained launch quarantine and established no native
generation acceptance. The pinned runtime's loading-before-auth behavior and a
synthetic regression confirm the software defect; the historical native record
does not retain the actual response, so that causal attribution remains an inference.

The targeted regression fails on the old code (`Unapproved` versus `Unavailable`).
After the fix it checks loading then authenticated readiness and a separate
unprotected backend's refusal. The existing production daemon process test now
starts with the loading fixture, preserving all JSON/SSE, context, revocation,
failure ownership and admin checks without adding another process scenario.
Local checks passed: 119 default tests (one ignored native smoke), 26 development
library/binary/CLI tests (one ignored native relay), six targeted adapter checks,
and two daemon process checks in 48.05 s. Formatting and both workspace/all-target
lint modes passed with warnings denied. Native models are not run by these checks
or CI.

Correctness/security review traces every readiness caller and keeps the 503 path
strictly unavailable. Complexity review uses a single match in the shared probe;
no dependency, retry framework, workflow or configuration field was added. Exact
final-head PR/CI and local outcomes will be recorded in Linear before merge.
Independent review remains deferred, not approved. M2 remains open; M3 unstarted.
