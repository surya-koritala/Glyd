# GLY-177 client/coordinator/worker/service relay

The application integration now forwards one admitted request through the
authenticated generation channel to a supervised service, preserving JSON/SSE
content, reasoning and usage. A single-slot output queue and bounded UTF-8
fragments carry large responses without exceeding the JSON parser's node budget.
Sampling and immutable model identity must match admission. Existing core replay,
leases and registry fencing remain authoritative; native drain is never inferred
from a response or parent exit.

Synthetic tests use real public enrollment, scoped client tokens, TLS and an owned
HTTP fixture subprocess. They compare direct and full-relay JSON/SSE bytes,
including large multibyte Unicode responses; reject wrong models, unresolved
sampling and tools; cancel on disconnect before output, disconnect after partial
output, controller revocation and registry drain; and require one redacted terminal
error without `[DONE]` or retry after backend failure. Request markers assert only
one backend invocation on failure. All paths retain reservations before explicit
synthetic cleanup acknowledgement.

Concurrent maintenance exposed timestamps arriving at the worker out of order.
Enrolled workers now sample their own monotonic clock under the state lock. Tests
cover stale timer samples, lease expiry, and rejection of a foreign connection's
withdrawal without interrupting the owning request. The pure core's injected
clock and rollback checks remain unchanged.

Validation: formatting, default/all-feature all-target Clippy with warnings
denied, default workspace tests, development backend/CLI tests and the lifecycle
regressions. Correctness/security review covers admission, ownership, cancellation,
bounded memory, replay and cleanup. Ponytail review reuses existing transport,
service, fixture subprocess and standard-library channels; no new dependency,
async runtime or scheduling abstraction is added. Independent review remains
deferred under the current merge authorization, not claimed approved.

Exact commit, PR and five-platform/policy-check evidence are recorded in GLY-177.
The daemon does not enable these routes yet. Catalog pin lifecycle, exact native
tokenization, qualified backend/resource/drain profiles, complete HTTP statuses
and native end-to-end/placement acceptance still block M2 completion. No native
workload, Ubuntu access, Pi qualification or paid infrastructure was used here.
