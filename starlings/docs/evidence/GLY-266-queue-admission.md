# Controlled admission boundary and two-worker overload fixture

[GLY-266](https://linear.app/getglyd/issue/GLY-266) tracks a failure distinct from
GLY-263 initialization. [PR65](https://github.com/Sigbound/glyd-compute/pull/65)
source0b8c21437b3c976699d10e7cf8066e67ebc09c87,
[CI38623121627](https://github.com/Sigbound/glyd-compute/actions/runs/38623121627),
Windows backend job116055417230 fails the two-worker overload assertion:
active2, queued1, retained_requests3, retained_bytes711; expected queued8.
Raw log SHA256bb211a8ba7bd63a0cfe0dda96e106da3fe39dcf0ba343720db7b67ff5b250f11.
Both new initialization controls pass; default Windows10cases pass65.59s,
log SHA256d4fcef894aa8dacbbd621277741a65c5bd2681cb671cbfa48d87c447adbc69ee.
The failed run remains failed; no unchanged-source retry.

The fixture previously treated eight completed socket writes as eight admitted
requests. TCP write completion only proves local submission. ClientListener
allows four pending unauthenticated connections per address and eight admissions
per second; prepare_with releases that permit only after authenticated request
validation and response-grant creation. A backlog accepted in a burst or slow
authentication can refuse traffic before Pool::submit, even when writes are paced.
The archived log does not establish which boundary rejected its seven requests:
no request-level admission errors were retained. Exact OS/timing attribution is
unproved; a green replacement run cannot retroactively establish it.

The controlled loopback test completes all five HTTP writes before any server
accept. Four real pending IncomingClient handles occupy the unauthenticated
budget; the fifth accept refuses Capacity. Dropping the four handles restores
connection usage to zero. It needs no credential, model, sleep or injected clock.
This deterministically disproves the fixture's write-equals-admission assumption.

The two-worker fixture now confirms each queue entry through the existing
protected read-only status query before opening the next paced connection. Every
entry requires both active workers, the exact retained request count and body
bytes. The original3-second observation budget is shared across all eight entries;
existing connection-opening time is excluded as before. No socket/request is
retried. Full queue8, excess503/nonretryable refusal, abandoned-client reclamation,
both concurrent worker invocations, exactly three total backend invocations and
successful reuse remain required. Production budgets, runtime, CI workflow,
coverage,12-second child watchdog and10-minute job caps are unchanged.

Local model-free commands, locked/offline and warnings denied:

- cargo test -p compute-security completed_socket_writes_do_not_prove_listener_admission:1pass0.00s; log SHA256545b5841e6cc877f9b2cafe5eaa35c0be86fff5b0468e05821684148eed7e5bc.
- cargo test -p compute-cli --all-features --test daemon catalog_service_two_workers_run_concurrently_through_the_coordinator -- --exact:1pass21.55s; log SHA2564fcd189bd39c5f6ee1e81d6ee8a4a4cae730ceb47574afa8d6680fb0b7feaede.
- Formatting, default and all-feature workspace/all-target Clippy pass with warnings denied.

Correctness/security review traces listener admission, permit release, client
grants, dispatch, queue retention and fixture release; only tests change here.
Separate Ponytail review reuses the existing query/open helpers and stdlib
handles; no new hook, dependency, production policy or observation framework.
Independent milestone review remains deferred, not approved. Latest-source seven
CI checks and reviewed current-main merge are required before closing fixed work.
No native inference/model/kernel/GPU/Ubuntu/PC workload, download, spending,
quarantine reset or M3 work. M2 physical/provider acceptance remains open.
