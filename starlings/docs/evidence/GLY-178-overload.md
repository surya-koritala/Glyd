# GLY-178: production queue overload acceptance

Base: `290e7b31afca651b900c5ceaa7fe4840ad13f6b7`. The existing router checks count,
body-byte, expiry and fairness limits in isolation. This extends the existing
production two-worker daemon scenario to verify the per-client queue boundary
through HTTP admission, authenticated dispatch and supervised synthetic services.
No native model, device memory measurement or physical worker result is claimed.

## Check

`catalog_service_two_workers_run_concurrently_through_the_coordinator` starts two
backend requests behind the existing release barrier, then keeps eight more client
sockets connected. Admin must report two active assignments, eight queued requests,
ten retained bodies and exactly ten serialized request bodies' bytes.

The next request must receive a protocol-valid HTTP 503 CapacityUnavailable error
with retryable=false. A subsequent admin observation must still report those exact
counts and bytes; the refused request owns no retained body or assignment. Dropping
the queued sockets must restore two active assignments, zero queued requests and
exactly two bodies' bytes before the backend barrier is released.

The existing fresh third request and trusted completion assertions remain. Exactly
three backend invocations occur: the two original requests and that fresh request.
No abandoned or refused request reaches a backend. Final admin state requires zero
queued/active/retained requests and bytes. The existing worker/service shutdown is
unchanged. Queued disconnect releases only unclaimed work; this does not weaken
uncertain claimed-work ownership or cleanup quarantine.

Windows executes the same scenario in its existing serialized production matrix.
Other platforms retain the existing separate test. No fixture timer, request/admin
deadline, production scenario/job count, dependency or production limit changes. The decision guide
now describes current bounded replica queues rather than the initial no-queue route.

## Review and verification

Correctness/security review traces IncomingClient admission and its independent
response-grant budget, Pool::submit, TaskRouter's eight-entry client queue, retained
body insertion only after successful enqueue, Pending drop/abort_unclaimed and the
trusted drain path. The next request fails before worker Count/Apply/Generate. The
release barrier prevents normal completion from satisfying the full-queue assertions.
Exact bytes and final invocation counts distinguish refusal/reclamation from replay.

Ponytail review reuses the existing two-worker setup, socket helper, protocol decoder
and status API; no new harness, executor or runtime abstraction. Independent review
is deferred under current green-merge authorization, not claimed approved.

Pinned Rust 1.97.1 with locked offline dependencies and warnings denied:

```sh
cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline
```

Formatting and all-feature workspace/all-target Clippy pass with warnings denied.
The initial overload catalog-service selection passes ten cases in 36.91 seconds. Both
existing CLI binary routing checks pass in 1.12 seconds. The earlier library-target
filter selected zero tests and is not counted as routing validation.
The combined GLY-250 outcome follow-up passes eleven cases in 36.83 seconds with
formatting and all-feature lint still clean.
Source head, PR, CI and merge evidence are recorded in Linear.
All seven latest-head checks and current-main
ancestry remain required before merge. Synthetic checks do not meet GLY-178's
two-real-worker or matched native direct/proxy performance/memory acceptance.

GLY-244/246's verified fixes are Done. GLY-248/249 separately preserve unverified
historical Windows attribution in Backlog without blocking M2. Provider package/API
delivery and approved native/hardware qualification remain open; M3 has not started.
No workload approval, download, spend, license/settings change or quarantine reset.
