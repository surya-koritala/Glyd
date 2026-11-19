# GLY-173 discrete resource diagnostic

Starting main: `57f485e8d352c43183f6ac8256fa47bf566cf092`. This adds actual NVML
query integration to the explicit probe, not just a parser. The installed native
library must be selected by absolute local path and full SHA-256 approval. It is
never loaded by normal probes or heartbeat/control/service code.

Synthetic regression exercises independent host and GPU reservations, unique
pool identity, malformed/shared/unknown exclusion, five-second expiry, unchanged
CPU qualification identity and production NodeCapabilities encoding with no
backend. The NVML memory-count check accepts reserved bytes already included in
used and rejects overflow/incoherence. Idle and power caps do not imply thermal
throttling. The actual CLI refuses an unapproved temporary file before any loader
or device query. These tests establish software behavior, not hardware readings.

Correctness/security review traces the only call through explicit CLI arguments,
hash verification before loading, canonical IDs and bounded device count, no
setter/process queries, checked pool arithmetic and existing age handling.
Separate complexity review reuses existing observations, pool records, digest
helper and serializers; no subprocess parser, framework, handwritten unsafe FFI
or extra platform SDK. The locked safe wrapper is needed for the native API.

Local validation passed: formatting, default/all-feature workspace lint with warnings
denied, four resource and three CLI checks, 125 default tests (one native ignored),
28 development tests (one native relay ignored), and locked cargo-deny with warnings
denied. The final two test assertions were rerun with the focused resource check.
All five exact-head CI checks remain mandatory before merge. Exact PR/commit/CI
outcomes are linked in Linear.
Independent review is deferred under current authorization, not approved.

No device library was loaded, GPU queried, model executed or infrastructure used
for these checks; spend $0. Windows topology eligibility remains unknown because
the safe wrapper exposes virtualization on Linux only; MIG/virtual/shared/unknown
pools remain diagnostic. Driver calls have no independent deadline and stay out
of serving/control paths. GLY-173 remains open for complete policy/integration and
native qualification. M2 is not complete; no NVIDIA inference, Pi or layer
participation is qualified by this change. See [the contract](../resources.md).
