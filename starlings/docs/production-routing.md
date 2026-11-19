# Explicitly qualified CPU request routing

The daemon connects HTTP admission, coordinator dispatch, worker lifecycle and
the supervised service. A route binds one immutable manifest/profile and one
service slot per worker; explicitly approved replicas support independent requests.
The repository supplies no general production
activation profile. Synthetic CI cannot supply native qualification.

The [sealed-artifact CPU experiment](evidence/native/2026-10-12-m1max-cpu-qualification/README.md)
provides direct count/parity/slot-idle and candidate memory observations. It does
not activate these daemon routes or qualify arbitrary-prompt memory and recovery.

The [bounded native daemon record](evidence/native/2026-10-12-m1max-cpu-daemon-acceptance/README.md)
now exercises these routes with JSON/SSE reuse/refusal and midstream revocation on
one Mac CPU. Its fixed-input local policy establishes no general production profile,
service recovery or two-machine pooling.

## Qualification and route records

`starlings_cli::routes::Qualification` is strict versioned JSON:

| Field | Requirement |
| --- | --- |
| `version` | 1 |
| `fixture` | False for native use; true allowed only in development builds |
| `os`, `architecture`, `device_kind` | Qualified platform and `cpu`; worker checks its own platform |
| `target` | Digest of OS/kernel versions, architecture, host pool totals and available CPU threads; worker requires its current local target |
| `loading_bytes` | Measured peak loading/loaded footprint including safety reserve, as a decimal u64 string; checked before spawn |
| `model_layer_count`, `boundary_profile` | Exact manifest inventory/contract; both endpoints bind the plan and worker checks the catalog |
| `request_drain_profile` | `slots-idle-v1`; exactly one authenticated CPU slot reports `is_processing=false` before success |
| `configuration` | Full canonical digest of the exact `ServiceConfig` |
| `replica` | `WarmReplica`: node, manifest, profile, backend, executable build digest, `host` pool, evidence digest, measured request bytes/time, context, one session and accepting=true |
| `max_output_tokens` | Qualified output bound, positive and smaller than context |
| `memory_qualified`, `exact_count_qualified`, `request_drain_qualified` | All true, supported by reviewed evidence for this exact configuration |

Request bytes and measured milliseconds serialize as decimal u64 strings.
The request budget covers additional peak memory over loaded idle, including
scratch/safety, for the entire admitted context/output envelope. The plan's summed
reservation must equal that budget. Nominal model size or sampled idle RSS is
insufficient. Exact backend counting rejects prompt plus output beyond context.

Qualification is trusted operator policy. `replica.evidence` identifies the reviewed
record; the program does not attest or independently remeasure that archive.
Approval covers all fields, addresses and configuration. Earlier smoke records
leave memory, catalog and request-drain qualification incomplete; they cannot be
promoted. Audit CPU/device arguments, precision, dependencies and model consumption.

`WorkerRoute` contains `version`, `listen`, `service` and `qualification`. Its node
must equal the enrolled worker. The concrete LAN/loopback listener can use port zero
during local preparation; `cluster service` reports its assigned port. Complete
approved catalog assets and exact cache weight arguments are still required.

`CoordinatorRoute` contains `version`, loopback `listen`, concrete local numeric
`worker_endpoint`, an existing whole-request `plan`, and `qualification`. The plan
must belong to this cluster and match the qualified node, manifest/profile,
backend/build, host pool, context, one session and exact request reservation.
Runtime supplies the current durable authority epoch and fresh request IDs/times.

Both protected records are capped at 16 KiB, reject duplicate/unknown fields and
use RFC 8785/SHA-256 approval of the complete record. `model execution-digest FILE`
and `model route-digest FILE` print that digest without granting permission.
`model qualification-target` reports the local target without starting a backend.
The worker rechecks this target before each request.

## Local workflow

1. Prepare and review the manifest/cache and native qualification record.
2. After device/workload approval, start an enrolled worker with
   `daemon --execute PRIVATE_WORKER_RECORD --approve FULL_DIGEST`. The existing
   `daemon --service ... --approve ...` warm-up stays ineligible.
3. Inspect `cluster service`: readiness, endpoint, fixture marker, qualification,
   retained pin and reservation are explicit.
4. `cluster permissions NODE PRIVATE_SCOPES` replaces that node's complete
   allowlist. A scope is `{"model":"<manifest digest>","action":"infer"}`
   or `control`/`artifact`. Grant coordinator Control and worker/controller Infer
   explicitly. No startup grants exist.
5. `cluster token CONTROLLER_NODE PRIVATE_SCOPES SECONDS PRIVATE_OUTPUT` issues a
   narrower token for 1–3600 seconds. SQLite retains only its hash; the secret goes
   in a new protected file. Failed publication revokes the new token.
   `cluster revoke-token PRIVATE_TOKEN` withdraws its live output grants.
6. Enable `cluster route PRIVATE_COORDINATOR_RECORD --approve FULL_DIGEST`, or
   supply the same record to `daemon --route ... --approve ...` at startup.
   Live activation avoids a coordinator restart, which would fence the warm worker.
   A second activation is refused for the process lifetime, preserving sequences.
7. POST strict text-only ChatRequest JSON to the reported loopback endpoint's
   `/v1/chat/completions`, using the manifest digest and explicit temperature/
   top-p/seed. JSON/SSE preserve content/reasoning/usage. Successful SSE has
   `X-Compute-Session-Id` and `[DONE]`.

Admission sends bounded redacted HTTP failures without echoing credentials/input.
Authorization precedes body allocation. Withdrawn output grants cannot publish a
later result/error. Native API and full milestone conformance still need evidence.

## Ownership and remaining gates

Registry/admin maintenance is independent of generation. Worker TLS handshakes use
their own existing bounded budget and background threads; a slow handshake does
not occupy the service owner. One authenticated channel can queue. The coordinator
keeps one serial Dispatch per worker/epoch across connections. The existing pure
TaskRouter now selects/reserves actual configured replicas. An optional `replicas`
array adds strict endpoint/plan/qualification records to the legacy primary route;
absent/empty arrays are omitted so existing digests do not change. The complete
approved route stays capped at16KiB and64 unique nodes/endpoints, with common
cluster/sealed model/profile/context/output envelope and fixture mode. Configure
only the explicitly approved same-type inventory; OS names do not qualify hardware.

Requests queue under existing client/model/count limits and a32MiB serialized-body
retention cap; admission grants independently cap authenticated responses. Oldest
runnable selection prevents a blocked client head from starving another. A request
keeps its original60s placement deadline while waiting and across at most two
connection attempts. Retry exclusions affect only that request. An alternate is
allowed only before Count/Apply/Generate; possible execution or partial output is
terminal. Exact backend counting still precedes mutating controls; selection uses
the whole qualified context envelope, never estimated prompt tokens.

Queued disconnect, authorization loss and expiry discard only unclaimed work.
Once a dispatcher claims an assignment, reservations survive uncertain failure;
live capability/deadline withdrawal cancels its owned channel. Successful native
qualified idle/drain releases its slot. The serial dispatcher's positive proof
that no mutating control was attempted also releases router bookkeeping on any
error; this never frees worker buffers, resets a service quarantine or retries
after Count. Epoch change cannot silently forget outstanding reservations.
There is no automatic backend restart or live KV migration. `cluster status`
reports bounded routing queued/active/retained-request/body-byte counts, no content.

A worker connection lost or malformed before its first Count does not cancel the
loaded service when the runtime has zero reservations. The relay returns an
explicit no-work outcome; the supervisor rechecks process health, zero reserved
bytes and a live registry fence before returning to ready on that same service.
No restart, unpin or recovery occurs. Once Count is accepted, or when any old
reservation remains, the existing conservative cancellation/drain policy applies.

Admission requires a fresh ready capability with exact endpoint/build/profile.
Age is measured from each worker's own registry read. The coordinator captures a
monotonic instant immediately before that read and keeps it paired with the view;
later reads cannot inherit the earlier workers' collection interval. Each view
still accrues its own read, lock wait and subsequent processing time, and the
five-second freshness limit and repeated-snapshot expiry remain unchanged.
Admission also requires enough host memory. The worker checks host memory again
before counting. These are
conservative OS observations, not an OS memory limit. The reviewed native budget
and CPU-only configuration remain required. Ready describes the loaded backend;
the serial owner and worker lifecycle enforce one active slot.

Context refusal precedes generation and keeps the service usable. Other uncertain
failures cancel the process and retain the runtime/reservation. Admin stays
available; shutdown reports incomplete drain for outstanding work. Successful
response permits request-level drain only under approved qualification and live
CPU slot-idle acknowledgement. JSON success and SSE `[DONE]` wait for it. Its
absolute bound is two seconds; unsupported/busy/invalid slot state fails closed
and retains the reservation. Configure the approved backend to expose `/slots`;
readiness or HTTP completion is insufficient.

Unqualified service pins/quarantine survive stop and observed group absence.
Request qualification does not qualify service/device cleanup or escaped descendants.
[Explicit CPU recovery](service-recovery.md) requires policy before launch and a
durable local cleanup receipt; it cannot clear legacy/unknown work. Native catalog/precision/memory/drain checks,
qualified service cleanup/recovery, general native profiles, two-real-worker
benchmarks and the full M2 review remain open. Directed probe software acceptance
is recorded in GLY-174; model-free two-process routing is not native performance
or an Exo win. Independent M3 planning, framing/flow and diagnostic software is
implemented; it does not qualify a deployed native single-model pipeline. See the
[partial-worker boundary](partial-worker-contract.md) and
[token-link diagnostic limits](token-link-profile.md). Parked scope remains excluded.

The [M2 software failure matrix](evidence/GLY-177-m2-failure-matrix.md) distinguishes
production daemon scenarios from shared primitive tests and native gates. Its
active partial-stream disconnect case checks cancellation, one invocation,
responsive admin and retained uncertain reservations/quarantine.
