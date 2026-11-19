# GLY-253: Windows initialization exit

PR #57 source `a10fa7eee42c87217f31575e89aabf3729c31396`,
run38499207090/job115641877096 passes the production matrix, including registry
loss/reconnect and two-worker overload/release. The controlled adapter barrier
release/expiry check also passes on Windows. The subsequent warm-up case fails
on its first `cluster init`: the complete INITIALIZED response is printed, stderr
is empty, and the process remains alive beyond the existing twelve-second CLI
watchdog. Five other required checks pass; Windows backend/aggregate fail.

Raw `work/m2-gly251-barrier-windows-115641877096.log`, SHA-256
`cc5a707b36c02d0b0d4a8e7f7b697cc754d8f0ae6c290974d4b3e25488b9e4d1`.
This is separate from GLY-251's original idle-capability and startup-admission
events and GLY-252's demonstrated premature barrier. The failed head stays failed.

After its response, initialization drops the authority (SQLite, private files and
cryptographic material), then its state lock, before returning through normal Rust
process exit. Static development-backend-only, test-opted-in stderr markers bracket those
existing drops: authority_drop, state_lock_drop and resources_closed. The drops
stay in their original order and position; there is no extra OS query, destructor
bypass, detached cleanup, resource lifetime extension, output secret/path or timer
change. The existing CLI helper opts in only initialization clients and requires
the exact ordered markers on successful exit; all other stderr still fails.
Production builds and ordinary development commands emit no markers. A marker locates a blocked phase, not
the underlying OS call; absence of a repeated failure is not proof of its cause.

Correctness/security review traces init creation/config commit, the identity
borrow, authority/access/SQLite lifetime, state lock, main return and the existing
CLI watchdog. Ponytail review uses explicit existing drops and static markers,
without a profiler dependency or new CI job. Original cause attribution remains
open until genuine exit evidence or a controlled reproduction supports it.

Exact-source CI and any authorized reviewed merge remain required. M2 stays open
for this investigation and provider/native/current-hardware/performance gates;
M3 has not started. No native/model/GPU/PC action, spend or quarantine reset.

Local validation: formatting and workspace/all-target warnings-as-errors Clippy
in both feature modes pass. The existing synthetic warm-up/restart case passes
with strict opt-in marker ordering in 43.45 seconds:
`cargo test -p compute-cli --all-features --test daemon catalog_service_survives_failure_and_retains_pins_on_drain_and_restart --locked --offline -- --exact --nocapture`.
Raw `work/m2-gly253-local-warm-final.log`, SHA-256
`37cd3db5d3782c8c7b42d768f924b85c805b23a5aff917db8fd1515df3782426`.
