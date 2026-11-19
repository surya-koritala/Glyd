# Shared unwind context for registry-wait failures

Mainafb8ab1a1131527ba802e5cd67422577e2a02cd9
[CI38655434068](https://github.com/Sigbound/glyd-compute/actions/runs/38655434068)
fails Windows backend job116163950538 in a different case: startup-crash remains
waiting_for_registry at the existing10s warm-up wait. The production two-worker
case passes. Direct API raw log
SHA256779852a0f8fb2e97ca203ce7d08ec5ac1277f53b3feddfacf9744a2683cfcd08
is retained separately from gh-prefixed logs. Five other checks pass; backend and
aggregate fail. This is not evidence of the same queue, start-barrier or init
cause. Neither the genuine failed main nor earlier failures is retried.

The test Daemon's shared Drop now records bounded failure_context only while its
thread is unwinding, before the existing kill/wait. This replaces duplicated
per-barrier context reads and covers warm-up and every other fixture assertion.
Marker booleans and the queue snapshot remain in their original assertions.
Normal cleanup adds no reads. Context may reflect earlier unwind cleanup of
other actors and cannot establish their state before failure. Read/write errors
remain best effort; the original panic and cleanup are preserved.

Development-backend heartbeat failures record only the existing redacted Error
enum before the unchanged fatal/retry policy. No success-path logging, heartbeat
backoff, membership/expiry policy, private value or production API is added.

The existing observation test moves its already stopped daemon through a
controlled unwind, requires the exact original panic payload and exercises
unavailable-context handling without another daemon or test case. The retained
DAEMON_FAILURE_CONTEXT from that deliberate panic is expected test evidence.
Full local catalog lifecycle17cases pass48.36s; log
SHA2569b62fa0b426d10123688649d809d799e5e3b89130653f275b4e099246ead166a.
Formatting and both workspace/all-target warnings-denied Clippy modes pass.
Correctness/security and separate Ponytail review reuse stdlib panicking/Drop,
existing context/error enums and the same bounded admin reads; no per-case fault
framework, dependency, job, filter, deadline relaxation or request retry.
Independent milestone review remains deferred, not approved. Latest-source seven
checks/reviewed merge remain required. Actual causes, M2 provider/hardware/benchmark
gates remain open; M3 unstarted. No native/model/GPU/PC/Ubuntu workload, spending,
downloads or prior quarantine reset.

# Previous post-merge start-barrier failure and bounded context

Merged main8087b2155a9dea46c277abd092a375e2d77f2c4e
[CI38653060370](https://github.com/Sigbound/glyd-compute/actions/runs/38653060370)
fails Windows backend job116156233145 before queue filling: both actual worker
start markers were not observed within the existing3s barrier. Raw log
SHA256ae51389c067ee16b339131c0497393e36c8a4a4013185bb9e29c7a225c19c63a.
Five other checks pass; backend and aggregate fail. PR66 source9b45879's earlier
all-seven pass remains its own record. No run is retried or relabeled.

This failure occurs before the queue-specific diagnostics and does not establish
a matching queue-loss, authorization-expiry or initialization cause. The previous
assertion did not identify which worker started or whether dispatch had refused
before backend invocation.

Development-backend builds now record the final dispatch Error enum as well as
the existing queue-exit phase. The two barrier assertions capture marker presence
and a shared failure-only context from coordinator and both workers: the existing
bounded protected status/service reads and retained daemon logs. Transport errors
are retained as redacted Results, not unwrapped, so an unavailable daemon cannot
replace the original barrier assertion with a diagnostic panic. Snapshots occur
after failure; they do not retroactively establish the earlier state.

Successful lifecycle paths add no such queries. The existing observation test
checks both live replies and, after normal stop, both redacted Storage errors plus
retained log content. These checks follow its paired timing samples and are not
included in them. Two existing bounded reads per actor on failure introduce no
retry, timer/gate extension, private values, job, dependency or new test case.

Full local catalog lifecycle17cases pass48.65s including both-worker overload and
live/stopped context controls, log
SHA256e56746271dba17d86579afa592d8a35dbcb81d27508ffa0eeb903ba7abf965d1.
Formatting and both workspace/all-target warning-denied Clippy modes pass.
Correctness/security review preserves dispatch results, original failure and all
ownership/authorization boundaries. Separate Ponytail review reuses stdlib
Result::inspect_err, existing bounded admin API and one helper for three actors.
Independent milestone review remains deferred, not approved. Exact-source seven
checks and reviewed current-main merge are required; actual causes stay open.
No native/model/GPU/PC/Ubuntu workload, download, spending, quarantine reset or M3.

# Earlier confirmed queue entries disappear before overload admission completes

[GLY-267](https://linear.app/getglyd/issue/GLY-267),
[PR66](https://github.com/Sigbound/glyd-compute/pull/66) source
674c8c1b0563a1c390f68bde250fe30f71aa0887,
[CI38650228668](https://github.com/Sigbound/glyd-compute/actions/runs/38650228668),
Windows backend job116146907123 fails at queue admission4: active2, queued1,
retained_requests3, retained_bytes711. Entries1–3 were already confirmed through
protected status queries. Raw log
SHA2565d400b452b24d42bd16dcd0a5acd4bef0c32705440e3871f739e8f6b43308511.
Windows default10cases pass96.58s, log
SHA2563b501deb002abc877485fbc40904fccb51dabb0db8d53cefa5a7b06746e2b1c5.
All cache/init controls pass. This is distinct from GLY-263's initial empty-output
stall; the failed source remains failed, with no unchanged-source rerun.

Admission confirmation remains necessary but does not explain this loss. Source
tracing finds unclaimed Pending cleanup after a failed client watch or pool poll.
The retained record does not say which guard failed. Authorization maintenance
expiry, transport loss, pool expiry/epoch failure and policy closure must not be
silently conflated. No cause or production correction is claimed yet.

Development-backend queue failures now record watch/pool phase and the existing
redacted Error enum. Debug builds add explicit GLYD_TEST_CLIENT_GATE_TRACE opt-in
flags on an already-failed grant check: closed policy, wall rollback, credential
expiry, monotonic expiry and maintenance gap. The model-free daemon fixture opts
in and includes its existing coordinator log only on queue-acceptance failure.
Flags describe conditions at rejection; later closed checks do not establish the
original cause. No node, token, scope, request body or private path is printed.
Writes are best effort so diagnostic output failure cannot panic this path.

Only failed checks inspect the opt-in or emit flags; default release builds
compile out that block. Queue phase records compile only with test-backend.
The same fail-closed2s maintenance TTL, token lifetime, transport checks, task
deadlines, ownership retention, watchdogs and CI jobs remain. No retry, gate
revival, production API or generic diagnostic framework is introduced.

Local locked/offline model-free controls with warnings denied:

- Existing grant revocation/staleness test:1pass0.08s, exact closed-policy and isolated maintenance-gap flags verified; log SHA2568eb7378a16d2a7590babb93a006e15d3079e550c81d86c35cae76b974b06736e.
- Existing binary-owned routing controls:2pass1.12s, including atomic retention, byte/count limits, abandonment, expiry and exclusions; log SHA25661c0d2ce31227053afd7a76b4801f47d773ff42b5dfb2a86c1df37ebfe49a9e3. An initial library-target selection contained zero tests and is not evidence.
- Full catalog daemon lifecycle:17pass48.20s including two-worker overload, cancellation, revocation, backpressure, ownership and recovery; log SHA2568b55cb9ad0b51024a49280933990599373f9ce8757da23812d5d0ad0c66f997f.
- Formatting and both workspace/all-target warning-denied Clippy modes pass. A locally rejected error branch was replaced with stdlib Result::inspect_err; no lint exception.

Correctness/security review traces every exit and Pending cleanup, diagnostic
redaction, compiled/explicit opt-in gates, unchanged output authorization and
ignored logging errors. Separate Ponytail review reuses existing errors, tests,
log file, stdlib inspection and writes. Independent milestone review remains
deferred, not approved. Final-source seven-check CI/reviewed merge are required;
the genuine queue-loss cause remains open until matching evidence establishes it.
No native/model/kernel/GPU/PC/Ubuntu work, spending, downloads or old quarantine
reset. M2 provider/hardware/benchmark gates stay open; M3 unstarted.
