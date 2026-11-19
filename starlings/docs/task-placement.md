# Bounded whole-request placement

`starlings_core::routing::TaskRouter` is a pure single-owner queue and reservation
manager. It consumes authenticated `NodeCapabilities` through the existing registry
and separately installed `WarmReplica` qualification. That qualification must refer
to a real local native record: exact manifest, execution profile, backend build,
memory pool, measured additional peak request bytes (including scratch/safety),
context bound, session limit and measured request time. An evidence digest is an
identifier, not a cryptographic attestation or an automatic approval. Only trusted
orchestration code may install qualifications. Hardware discovery cannot do so.

M2 initially permits one warm model/profile per worker. Select the oldest runnable
request, then the compatible worker with least measured-time × outstanding-count
cost; ties use node identity. Exact manifest/profile/build/context and generation
capability must match. Atomic selection and reservation occur under one mutable
owner without network/backend calls. Pool accounting includes cancelling work.
Already allocated resident weights are part of loaded idle memory; per-request
cost measures additional capacity. If a report already subtracts pending work,
local subtraction is conservative and can under-admit; it never increases capacity.

Limits: 64 workers/active assignments, 128 queued-plus-undelivered-expiry records,
8 queued and 2 active requests per client, 32 queued and active per model, and a
maximum ten-minute queue/request deadline. An incompatible head cannot block other
runnable work. Expiry results are retained until collected; full result storage
refuses admission. Retry count is zero initially: an ambiguous attempt fails and
is never silently replayed or joined to visible output. Protocol replay fences
remain the worker's responsibility; this queue is not a durable request journal.

Freshness uses receiver time plus reported sample age, with a five-second bound.
Stale/unavailable nodes, changed backend builds and explicit withdrawal stop output
and admission. Cancel/expiry/withdrawal retain all reservations until trusted local
`drained`. Withdraw a hot/throttled/revoked node without cancelling other nodes.
Refresh telemetry does not revive cancelled work. The caller must apply this gate
before output, poll deadlines even during idle periods, and drive backend cancel
and worker lease/authorization gates; this queue alone is not authorization.

Raspberry Pi uses the same CPU host-memory pool and whole-request path. A Linux
ARM64 fixture with `layer_sharding=false` is eligible only with exact qualification
and safe memory. Unqualified, insufficient-memory, incompatible, stale and unavailable
fixtures are refused; a slow fixture does not impose a pipeline wait on faster
workers. Physical sensor policy and measured costs must be supplied by the resource
integration. No native Pi, automatic thermal control, multi-worker daemon routing
or real throughput improvement is claimed by these tests. GLY-178/209 remain open
for integration and physical/native acceptance.
