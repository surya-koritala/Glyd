# GLY-199 diagnostic-output guide

Issues: [GLY-199](https://linear.app/getglyd/issue/GLY-199) and
[GLY-197](https://linear.app/getglyd/issue/GLY-197).

The operator entry guide now includes the implemented directed/profile diagnostic
surface and distinguishes the redacted state/count bundle from private raw link
snapshots. Both are explicit read-only observations of an already-running profile;
neither starts, waits for or retries a workload. The link file's identifiers,
addresses, digests and partial/stale/failure states remain explicit. The scoped
example selects an existing profile and a new private output file.

Correct the production-routing guide's obsolete statement that M3 has not started.
Independent planning/transport/diagnostic software exists; installed/native layer
execution remains unqualified. No parked scope or native capability is restored.

Review traced the documented commands to cluster.rs, links::record,
link_record::project and diagnostics::{collect,write}; it checked profile status,
private-parent/new-file refusal and no-activation/no-admission semantics. Relative
links and heading targets are checked with bounded text/file reads only. No local
Cargo, executable, IPC/network/service, process inspection, model or device test
runs for this documentation-only change. No test suite, dependency, workflow or
job is added. Existing eight exact-source checks and separate automatic main
verification remain required for the reviewed authorized merge.

Separate complexity review reuses the existing guides and commands, with no new
collection or installer workflow. Correctness/privacy review keeps redacted and
raw output distinct and preserves approved-workload and evidence boundaries.
Independent milestone/security/release review remains deferred, not approved.
Full GLY-197/199 native readiness/allocation, packaged clean-machine acceptance,
supported-model/API and claims-review gates remain open. This source-guide fix
is not completion of M2, M3, M6 or the end-to-end goal.
