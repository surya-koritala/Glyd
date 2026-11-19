# GLY-177 enrolled worker lifecycle

`EnrolledWorker` reuses the existing core state machine and registered maintenance
handle instead of copying coordinator policy/CA state to workers. A trusted local
single-service profile constrains whole-request plans, backend build, memory pool,
minimum reservation and context. The existing replay and lease machinery remains
the source of lifecycle state. No new dependency or backend workload is added.

Synthetic tests use real enrollment/TLS but synthetic plans and qualification
values. They require a new start to execute once, exact replay to return the old
acknowledgement, and a replacement connection to be unable to resume execution or
output. Network completion is denied. Disconnect, epoch change and lease expiry
withdraw output while preserving in-flight memory until local completion. Valid
completion followed by release frees the reservation. Wrong profile/build/pool,
insufficient reservation, excessive context, fixture plans and a second lease
are refused without allocating more capacity.

Shared warm-replica validation is reused by both placement and local admission.
The coordinator-authorized worker API remains unchanged. Correctness/security
review checks connection ownership, lock ordering, maintenance and trusted drain;
Ponytail review checks reuse and complexity. No independent approval is claimed.

Checks: formatting, default/all-feature all-target Clippy with warnings denied,
compute-core tests and the full compute-security suite. All four platform CI jobs
and dependency policy must pass on the latest head before merge. Exact PR, commit
and CI evidence are recorded in GLY-177. This does not close GLY-177 or M2: service
execution, client routing, automatic cache/resource lifecycle and native
end-to-end acceptance remain required before M3.
