# Production selection integration — partial acceptance

The actual coordinator route now calls the existing `TaskRouter`, instead of
keeping a single dispatch mutex. An optional strict `replicas` list adds locally
approved endpoints/plans/qualifications to the legacy primary route. Absent lists
remain omitted from serialization, preserving existing record digests. Validation
requires unique node/endpoint, cluster, sealed manifest/profile, context/output
envelope and fixture mode; the whole record is bounded to16KiB and64 candidates.
Configure only the approved same-type hardware inventory. This interface makes no
hardware-family or native support assertion from an OS/backend name.

Coordinator maintenance refreshes authenticated capability views independently of
requests. Repeated snapshot IDs cannot reset receiver age. Selection/reservation
is atomic under a short router mutex; counting, connections and generation run
outside it. A serial dispatch owner persists per worker/epoch across reconnects.
Exact selected-backend counting precedes mutating controls; selection uses the
entire qualified context envelope rather than estimating prompt tokens.

Each request has one original60s placement deadline across at most two connection
attempts. Only a connection/handshake failure before Count/Apply/Generate may choose
an alternate. A pre-control invalid request releases its unused slot, preserving
context-refusal/reuse. Other uncertain errors retain the reservation and withdraw
the replica; no-output does not establish safe retry. Successful completion uses
the existing qualified idle/drain contract. Capability/deadline withdrawal cancels
the owned channel. Epoch change cannot silently discard outstanding reservations.

Local validation: warning-denied workspace/all-feature lint; existing actual
production-route/refusal/reuse/revocation test passes; a two-candidate synthetic
selection test checks capacity, failed-work retention, persistent owner identity,
original snapshot age and epoch fencing. No model or device workload is executed.

Production queues now retain bounded bodies/requests under existing client/model
limits, oldest-runnable fairness and the original deadline. Per-request connection
exclusions cannot withdraw healthy work for another client. The pending guard
discards only unclaimed work, including assignments allocated by another polling
thread before their dispatcher claims them. Claimed/uncertain work stays reserved.
Tests cover bytes, fairness, exclusions, queue expiry and abandonment.

Actual coordinator and two enrolled worker/service processes pass a controlled
fixture barrier: both backends start concurrently, a third request queues, queued
disconnect releases it without generation, and a later queued request completes.
Three completed requests have coherent usage and final retained bytes/requests are
zero. These are synthetic HTTP services; no model or device workload is performed.
The new fixture-only barrier is bounded and leaves existing scenario timers intact.
Full default/development/lifecycle regressions and the extended process case pass
locally; final-head CI is still required.

This is not full GLY-178 acceptance: qualified two-real-worker comparisons and
physical/native memory records remain required. Synthetic processes do not establish two-machine
routing or an Exo result. M2 and GLY-175–178 remain open; GLY-245 remains first in
M3. Correctness/security and separate complexity review, current-main integration,
PR and all final-head checks are required before merging. Independent review is
deferred, not approved. No new dependency, copied code, parked policy change,
download, spend, license, publication or quarantine reset.
