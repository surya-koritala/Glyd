# Draft contracts and validation boundary

The local [diagnostic bundle v1](../docs/diagnostics.md) projects three existing
authenticated admin replies into a fixed redacted JSON report. It is a CLI output
contract, not a new daemon operation, remote wire API or admission authority.

The [GCP1 framing contracts](../docs/data-framing.md) now define bounded fixed
prefixes and all six header kinds, with synthetic little-endian golden vectors.
The authenticated persistent library requires trusted local lifecycle admission;
child-process TLS checks preserve all four payload dtypes. It adds no daemon
pipeline execution or native drain qualification. The [flow library](../docs/data-flow.md)
adds bounded credit, ordered reassembly and one-use host payload handoff.
The [transport adapter boundary](../docs/transport-adapter-boundary.md) defines
a future RDMA seam and current custom-UDP no-go; no second runtime transport or
registered-memory API is implemented.

The offline [partition preview](../docs/partition-preview.md) uses strict
PreviewInput v1 and existing manifest/capability/profile records. It reports
bounded advisory candidates, not executable plans, reservations or qualification.

The [calibration report](../docs/plan-calibration.md) reuses that PreviewInput and
binds supplied observations to the selected candidate, attempted-plan/producer
receipts and timing definition. It verifies files and reports prediction errors,
not collected measurements, native approval or a provider API.

The internal [mode-cost contract](../docs/mode-costs.md) adds strict ModeInput v1
for offline whole/offload/split ranking. It reuses PreviewInput and profile/pool
checks, with explicit memory-tier rates, separate storage traffic and directed
latency evidence. It defines no provider API or native admission authority.

The proposed [ModelCostProfile v1](../docs/model-cost-profiles.md) is a strict
external InspectModel result with pure per-pool stage accounting. It is not an
installed provider API or a native qualification.

The independent offline scoreboard v1 uses strict typed JSON records documented in
[scoreboard.md](../docs/scoreboard.md). It verifies supplied evidence and never
confers backend qualification; the client benchmark schema below remains distinct.

The first typed `BenchmarkRun` v1 implementation and offline
[`benchmark inspect`](../docs/benchmark-inspection.md) command now validate supplied
run consistency and referenced artifact hashes. The finalized draft requires an
explicit fixture flag. Inspection starts no workload and never qualifies a backend.

GLY-172's protected local cleanup policy/versioned recovery record are specified
in [service-recovery.md](../docs/service-recovery.md). They are local operator/owner
contracts, not remote cleanup instructions or proof inferred from PID/HTTP replies.

The protocol crate implements strict parsing and schema/semantic validation for the M0 node, plan, lease, session, error, manifest, tensor header, chat request and control request records. An explicitly qualified daemon chat route now exists; see [production routing](../docs/production-routing.md). The full HTTP surfaces, benchmark qualification and binary tensor transport remain incomplete. JSON Schema is Draft 2020-12; HTTP uses OpenAPI 3.1. No current Glyd or Exo API compatibility is implied. Register URN schema IDs locally; resolve OpenAPI relative references from `spec/`. Reject duplicate JSON keys before validation and unknown object fields. Formats such as `date-time` must be asserted by the validator, not treated only as annotations.

| File | Contract |
| --- | --- |
| control-request.schema.json | Versioned prepare/commit/release/begin/step/cancel/finish envelope; local URN references |
| chat-request.schema.json | Strict bounded chat input, extracted from the HTTP draft |
| control.schema.json | Node capability, plan, lease, session and error records |
| model-manifest.schema.json | Pinned source, execution profile and complete model inventory |
| tensor-header.schema.json | JSON header for a tensor chunk; the GCP1 binary prefix is in the architecture |
| data-control.schema.json | GCP1 credit, receipt, execution, cancel and error headers with explicit invocation scope |
| model-cost-profile.schema.json | Proposed external per-layer/non-layer memory and timing costs; native evidence/approval remains separate |
| benchmark-run.schema.json | Typed offline run evidence; explicit fixture marking, no fabricated measured example or native qualification |
| openapi.json | Ten paths across client/coordinator/worker HTTP surfaces |

The examples are synthetic. Zero digests, `.invalid` endpoints, fixture memory amounts and dates are placeholders for simulator validation only. They do not identify a real checkpoint, a real device or a measured run. Production must reject fixture manifests/plans and verify actual artifact/build digests. A fixture source is not a pretrained model.

## Required semantic checks beyond schema validation

- Parse decimal strings with checked u64 conversion. Enforce per-message and cumulative memory budgets; reject arithmetic overflow before allocation.
- Certificate identity equals node identity. Roles, current membership, cluster, model permission, plan digest, authority epoch and lease must all agree. Never trust self-reported endpoint addresses without enrollment/address policy.
- Timestamp age comes from receiver-side observation plus a bounded reported sample age, not peer wall clocks alone. Lease expiry uses local monotonic deadlines; renewal requires current authority. Audit timestamps cannot extend a lease.
- A node references each physical memory pool once. Device-to-pool references exist; available plus reserved values are coherent with a fresh report. Shared Apple RAM is not added again as VRAM.
- Layers have unique contiguous indices. Plans cover `[0, layer_count)` exactly once in order; each stage has a nonempty range and a distinct worker node. The worker runtime supports one stage per node, including when device names differ; the same device-name spelling on different nodes is valid. Only stage zero owns embeddings; only the last owns final norm/head. Whole-request mode has exactly one complete stage.
- All stages agree on manifest/config/tokenizer/template, precision, boundary and state profile, context and sampling semantics. Fused residual profiles require proven adapter-specific contracts.
- Recompute plan digest from RFC 8785 canonical JSON excluding `plan_digest`; recompute manifest digest excluding `manifest_digest`. Fixture placeholders are exempt only in explicit simulator mode. No signatures are specified in v1; manifest approval is authenticated local policy plus content hashes. Adding signatures requires a separately versioned signing contract.
- Asset paths must remain relative after normalization. Reject dot segments, absolute paths, symlink escapes, duplicate/colliding names and platform-reserved paths. Schema patterns alone do not provide path safety. Every layer/component references declared assets; pinned source and license approval must be valid.
- Tensor shape product times dtype width equals total bytes, with rank <=4, tensor <=64 MiB, frame payload <=8 MiB and JSON header <=16 KiB. Total bytes cannot exceed the lease/boundary budget. Chunk offset plus bytes cannot overflow or exceed total; offsets are contiguous, aligned to dtype width, with no overlap/gaps. `end_of_tensor` agrees with the final extent. Header chunk length equals binary prefix payload length.
- Session/generation/invocation/sequence/position and peer direction match the active plan. Tensor index is less than tensor count; roles and counts match the boundary profile. The whole invocation's required tensors validate before execution. `u32` is legal only for token IDs in a permitted channel. Token IDs are not a hidden-state boundary.
- Same invocation cannot mutate state twice. Connection loss after possible execution aborts the generation. Cancel waits for safe device completion before freeing buffers. Per-tensor reassembly and per-invocation aggregation both have hard budgets.
- Prompt plus output fits context, and the model's supported limits are narrower where needed. Empty text and special tokens follow the pinned tokenizer/template. Unknown chat fields fail rather than disappearing silently.

## Tensor control frames

Frame kinds1 (TensorChunk),2 (Credit),3 (ReceiptAck),4 (ExecutionResult),5 (Cancel)
and6 (Error) have typed schemas and golden inputs; see [GCP1](../docs/data-framing.md).
Credit counts payload bytes. Aggregate credit, reassembly and metadata-rate
enforcement remain GLY-183. ACK means validated receipt, never completed GPU
execution; an execution report is not local native drain evidence.

## HTTP behavior

Client listener: loopback port 9740 with scoped bearer authentication. Remote client access requires explicit TLS configuration. Peer listeners: port 9741 with mutual TLS; workers register/heartbeat to the coordinator, which calls prepare/commit/release on enrolled worker listeners. Role and epoch checks are mandatory beyond certificate verification. Port 9742 carries tensor frames. Enrollment/admin operations are local privileged socket commands and have no public HTTP routes in this draft.

For chat, reject unknown fields and enforce 1 MiB total HTTP body and aggregate message limits before tokenization. Stream records are the documented `StreamChunk` schema serialized as SSE `data:` records, ending in `[DONE]`. A post-header failure uses `event: error` with the Error body and closes without a success finish marker. Each successful stream includes a stable session ID in the response headers. Stream disconnect cancels work. Session ownership is checked before explicit cancel. Idempotent release/cancel reports already-terminal success only for the authorized caller; an unknown ID must not expose other users' sessions. Prepare/commit use request IDs plus parameter hashes; same ID with different parameters returns conflict.

HTTP status mapping: 400 malformed/unsupported request; 401 absent/invalid client credential; 403 forbidden role/model; 409 stale epoch/incompatible state; 429 caller quota/rate limit; 503 temporary capacity/worker loss; 504 deadline; 500 redacted internal error. Add 404 for authorized missing resources. These status details and all control-frame variants must be covered by generated conformance checks at implementation.


## Implemented M0 API and limits

`decode<T>(bytes, FixtureMode)` and `encode<T>` run duplicate-aware bounded JSON,
Draft 2020-12 validation with asserted formats, typed deserialization and semantic
checks. `ControlRequest::decode` additionally checks the operation's RFC 8785
SHA-256 against `request_hash`. The hash covers the canonical tuple
`[request_id, authority_epoch_decimal_string, request_sequence_decimal_string, operation]`.
The wire epoch and sequence are decimal strings; both are nonzero u64 values.
The coordinator serializes monotonically increasing sequences per worker/epoch. Schema references resolve only from
embedded local resources; network/file retrieval features are disabled.

`Worker::apply(request, monotonic_ms)` returns `Ok(false)` for a new accepted
operation and `Ok(true)` for an exact replay. A replay must not run a backend.
Repeated failed requests retain their original error within the latest 4,096 results.
Older sequences return `ReplayExpired`; they are never executed again. Cleanup
results are recorded too. An out-of-order new request is refused. A retry retains
its original ID, sequence, digest and bytes. A replayed success is a
historical acknowledgement, not proof that its lease is still live. Query the
snapshot for current state. Reusing a request ID with different content fails.
`advance` expires leases; `recovery`/`recover` model restart after shutdown without
KV state. `Registry` rejects duplicate snapshot mutation and never refreshes age
on exact replay. IDs and bookkeeping limits are in docs/decisions.md.

JSON is capped at 1 MiB, depth 32 and 16,384 value nodes; decimal u64 strings are
checked before use. Tensor extents are checked without allocating their claimed
size. M0 accepts one complete tensor per invocation, bounded by the selected
stage's scratch budget. Simulation payloads have a separate 32 KiB cap and the
JSON node budget still applies. Each worker reply has a 10-second wall-clock
watchdog; fake compute delays use injected time. Child processes are reaped on
success, failure and timeout. No backend output survives a cancelled generation, including cached completion
replays.

The semantic list above includes future requirements: certificate/membership
checks (M1), filesystem symlink containment and source/artifact authorization (M2),
actual boundary compatibility, GCP1 framing/reassembly/credits (M3), and numerical
qualification (M4/M5). M0 validates lexical artifact paths only and opens no model
files. Protocol validation alone does not make an endpoint safe to expose.

GLY-167 local admin/config records are documented in [daemon.md](../docs/daemon.md).
They are internal strict typed JSON, separate from the draft OpenAPI contract.

GLY-168 registry/epoch/fence records are strict internal TLS JSON documented in
[registry.md](../docs/registry.md). They reuse the existing production
`NodeCapabilities` validator and 16 KiB security framing. They do not implement
the draft HTTP worker routes or replicate model/token policy. Runtime tests cover
replays, sample expiry, durable restart, epoch fencing and malformed inputs.
Resource-report sequence ranges are durably reserved before publication. Restart
discards the whole range and can return an optional authenticated decimal
`sample_sequence_floor`; only a new probe capture may restore capacity. This
internal change requires coordinated coordinator/worker upgrades; see the registry
document for downgrade limits. It changes no draft backend API.

GLY-169 adds strict protected config fields and local topology records documented
in [discovery.md](../docs/discovery.md). DNS-SD is an untrusted hint protocol, not
a control-plane membership or authorization schema.

## M2 whole-request lifecycle

`start_request` carries one queued Session, one invocation UUID and the bounded
text-only ChatRequest. It references a committed whole-request lease; `model` is
the exact manifest digest resolved from the client alias. Sampling parameters are
fully resolved and must match the admitted Session, including the output limit.
The backend's qualified tokenizer must supply the session's true prompt count
before admission; string length is not a token count.

A single invocation remains in flight for the entire response. It may publish
through the existing live authorization gate; this does not authorize an endpoint
or implement SSE. Exact control replay acknowledges the original operation and
never launches another generation. Cancellation/release/expiry withdraw output
immediately but retain reservations until trusted local backend drain. Successful
local drain completes the whole generation. Network completion instructions are
still denied by the secured worker. Tensor steps require a layer-pipeline plan.

No per-token request identifiers are consumed by this operation. M2 streaming and
backend integration must still enforce output/token/body limits and deadlines,
and terminate on disconnect without retrying visible partial output. Prompt data
is not retained in core replay history (only the request digest and plan are saved).

## Proposed partial-worker metadata

[BoundaryContract and RunChunk](../docs/partial-worker-contract.md) use strict
internal Rust/JSON records in `starlings_adapters::partial`, reusing bounded JSON,
profile keys, scalars and data scopes. They are proposed local backend metadata,
not a delivered native RPC or a new OpenAPI route. The synthetic example requires
explicit fixture mode; layout validation alone grants no execution capability.
