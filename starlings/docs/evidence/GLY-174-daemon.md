# Directed probe daemon integration

Follows merged PR #43. Exact current-main/head/PR/check identities are in Linear.
Uses the existing session, exclusive profile/IPC ownership and listener budgets;
no model permission, certificate secret copying, framework or dependency added.

Actual model-free coordinator/worker processes enroll, establish a persistent
registry, refuse an unapproved server record, activate a separately budgeted probe
listener and run a worker job in the background. The admin start response stays
within its existing three-second budget. Both directed payloads appear, repeated
status retains the ID while increasing capture age, and the heartbeat stays online.
Worker drain and coordinator revocation hide measurements. Validated digest printing
is offline and matches the approved complete record.

Original server capture time precedes TLS/transfer; completion/publication does not
refresh it. Current server membership/epoch and client fence are checked at view
time. One bounded view avoids response size scaling with cluster size. Shared
budget and immutable route prevent local repeats/config changes from bypassing
traffic policy. Existing stop/drain maintenance stays active through bounded joins.

Local direct daemon test and default/development/lint results are recorded in Linear;
all five final-head checks remain required before merge. Review traces approval,
role checks, background ownership, bounds, credential reload and live publication;
separate complexity review uses native threads/maps/mutex and existing TLS/crypto.
Independent review deferred under current authorization, not approved.

Run 38351668209 passed the probe and default/development checks but failed Windows
service startup with `PROCESS_GROUP_UNCONFIRMED`. Windows group absence is not
qualified, so that later cleanup result alone cannot identify why readiness failed.
Service status now retains a separate redacted `startup_error` (backend probe/process,
registry loss or stop), and the test immediately rejects terminal startup failure.
The synthetic startup-crash case checks that its reason survives cleanup reporting.
No wait, retry, cleanup policy or reservation rule is weakened. This diagnostic
change does not establish the remaining Windows root cause or close GLY-244.

Run 38352861298 instead failed the development cancellation fixture before backend
invocation. Its two relay scenarios used a maintenance-only helper that never
sent heartbeats and stopped ten seconds after fixture creation. The registry's
six-second freshness contract makes that setup invalid when preparation is slow.
Both scenarios now reuse the existing authenticated heartbeat owner through setup,
relay and cleanup, wait for a first heartbeat within three seconds, and stop it
with its scope guard. The sustained test crosses the six-second registry TTL
while requiring Online. Backend-start, cancellation, output and reservation
assertions stay unchanged. This corrects a test lifecycle defect; it does not
establish why the earlier daemon startup failed or qualify Windows job cleanup.

Full GLY-174 is still incomplete for larger hostile-peer/fault matrix and calibrated
asymmetry/congestion planner replay. Software loopback is not physical LAN, native
inference or an Exo scoreboard win. Refocused NVIDIA/Mac-only M0–M6 applies;
parked work and PR #42 code unchanged. No model/device workload, download, paid
infrastructure, license/settings/publication or old quarantine changes.
