# Node-local stage transport authority

GLY-182 extends the existing enrolled TLS and GCP1 transport to remote stage
workers without replicating the coordinator's authority database, CA signing key
or client token. This is permission and logical lifecycle integration. It does
not load model layers, activate a daemon pipeline listener or qualify a backend.

## Ownership and assignment

The coordinator creates `EnrollmentReader::stage_issuer(token, plan, local_lease)`.
It pins the client token hash, canonical layer plan, local lease and a random public
issuer UUID in an opaque `StageIssuer`. The token/hash stays on the coordinator.
Creating another issuer cannot substitute another client for an existing lease.
The UUID identifies an owner; it is not a bearer credential.

`StageIssuer::assign` sends the plan/issuer on the existing pinned
`GenerationStream`. `StageWorker::accept_assignment` checks its configured model
and execution profile, full registry epoch/incarnation, local membership and the
canonical non-fixture plan. It rejects malformed assignments and closes control.
A trusted local configuration may instead construct `StageWorker::new` with an
explicitly approved plan/issuer. Discovery and data frames cannot create either.

Before assignment and renewal, the coordinator rechecks the owning client's
Infer grant, its own Control grant, every participating worker's current model
permission and credential, and the plan's ready resource/backend eligibility.
The worker uses its existing protected registry fence and independent heartbeat
maintenance. Prepare/Commit/Begin and cleanup reuse the core state machine.
`StartRequest`, `StartStep` and network `CompleteStep` are denied by this actor;
logical Ready is not proof that a native partial loader prepared a model.

## Bounded renewal

The worker calls `refresh_grant(control, local_lease, source_stage, destination_stage)`
for each local directed boundary. Grants are per lease/route and reusable across
sessions. Renewal is independent of tensor/token traffic; no per-token coordinator
round trip is added. The control owner must renew early on its independent loop.
This library does not supply that daemon loop or automatic retry.

The worker sends strict `{version, nonce, scope}`. Scope contains the full
authority epoch, issuer UUID, plan digest, local node/lease and stage direction.
The response is strict `{version, challenge, peer}` and must echo the entire fresh
challenge. `peer` is the coordinator's enrolled worker record/certificate.
Records use the existing duplicate-aware parser with a 16 KiB pre-allocation cap.
Only adjacent forward boundaries and last-to-first token return are permitted.

Permission expires two seconds after the request **started**, never two seconds
after its reply. There is no dependency on clock synchronization or a peer's age
claim. An expired, closed, released or control-orphaned grant cannot revive.
A changed peer certificate requires new reviewed lifecycle ownership; renewal
does not silently replace the peer. Existing generation-channel lifetime,
record and byte limits remain unchanged, as do local lease deadlines.

## Data and withdrawal

`PeerStream::connect_stage` and `IncomingPeer::stage` use only enrolled device
keys, the pinned public CA and the granted exact peer certificate. They preserve
TLS 1.3, identity names, roles, cluster, ALPN, disabled early data/resumption and
listener budgets. Authority-only APIs refuse the delegated stream. Socket waits
are clipped to the current grant deadline as well as the existing record cap.
If the coordinator renews while a raw TCP read/write is waiting, its timed-out
OS wait rechecks current authority and may continue only within that same record
deadline. TLS buffers, offsets and already consumed bytes are retained. No GCP1
frame or backend operation is replayed; expiry without renewal still fails closed.

`StageWorker::data_admission(peer, grant, expected_header)` supplies the same
`DataAdmission` used by `DataStream`. The boundary template is trusted local
input; incoming tensor metadata cannot select it. Every allocation/delivery gate
checks that the grant belongs to this runtime, control owner, actual local session
lease and directed peer, plus current lease/session/epoch, exact boundary template,
receiver position/sequence and invocation state. The wire header names the
receiver's lease; a sender must also prove its own distinct local session lease.

The short enqueue callback holds runtime, registry-fence and grant locks. It must
not perform I/O, invoke a backend, block or reenter these objects. All network I/O
and renewal waits occur outside those locks. Failure poisons the data connection.
Authority withdrawal cancels the affected lease; any in-flight reservation remains
until trusted local drain. Neither a receipt nor an execution frame proves drain.
Input rejection poisons its data connection without withdrawing otherwise valid
route authority or cancelling another lease. Generation-level ambiguous execution
and cancellation ownership remain GLY-183 integration requirements.

## Trusted local queue and resource ownership

`StageWorker::hold_backend(control, lease)` obtains the existing opaque core
`BackendHold` after logical Prepare and before native preparation. It retains
the complete stage reservation across control loss, cancellation and missing
prepare replies. This is local ownership, not installed/native eligibility.
The supervisor must also retain its durable process/catalog owner; no process is
started or adopted by these APIs.

`FlowReceiver::finish` keeps the original immutable admitted header templates
with the complete ordered buffers and `ReceiveBudget` reservation. Templates stay
legal wire chunks even when the reassembled tensor is larger than 8 MiB.
`enqueue_chunk(grant, backend, inputs, sync_sender)` requires that authentic
receipt, exact local lease/route/native hold, complete bytes and the core's
atomic local-boundary admission. The owner separately validates its registered
`BoundaryContract`/`RunChunk` and actual layouts. The only enqueue operation under
runtime/fence/grant locks is standard-library `SyncSender::try_send`; all provider
pipe I/O remains outside those locks. Full/disconnected queues prove no delivery,
so their unsent ticket is failed/drained after leaving the locks. No retry occurs.

`StageChunk::check_dispatch` checks live authority, the current pending invocation
and exact retained resource identity. StageWorker owns a bounded map of native
lease owners; each owner holds only its in-flight input resources, not another
lifecycle or retired-ID history. Dropping both public backend/ticket handles
while StageWorker lives withdraws the affected generation and retains its input
budget and native stage reservation until qualified cleanup.

Native intermediate output uses `output_admission(peer, actual_output_grant,
expected, chunk)`. It binds plan/epoch/session/generation/invocation/sequence,
outgoing stage, phase, position and token count to the current step. A derived
grant carries exact hold and input-resource identity through every admitted
`DataStream`/flow operation, PeerStream read/write and TLS socket deadline check.
Ownership is checked under the same runtime lock before authority-failure cleanup;
pending publication is rechecked after live reconciliation. A retired ticket
cannot withdraw a successor when an old grant expires. Local generation failure
refuses output without withdrawing otherwise valid lease authority. Base route
identity is retained, so a persistent connection may use fresh admissions.

`StageChunk::qualified_drained` is trusted local completion after qualified device
drain; a provider receipt, status or allocation counter is insufficient. Exact
resource identity is required even when wire UUIDs match. Full-instance
`StageBackend::qualified_unloaded` requires the supervisor's qualified proof that
the entire instance stopped; it verifies the original hold before withdrawing
its lease, draining retained items and releasing that hold. Removed resources
drop outside runtime/resource locks. Network StartStep/CompleteStep stay denied.

StageWorker Drop withdraws authority; it is not native cleanup. StageWorker and
the owning supervisor must remain alive while native work is possible. If all
runtime objects disappear, a separate durable process/catalog owner is still
required. No marker mutation or Rust object drop supplies physical cleanup proof.
The daemon/native supervisor is not activated here.

GLY-188 defines the token feedback. On the last-to-first route, a step's TokenIds
output is the first stage's next step: sequence + 1, position = start + rows, decode
phase, one row, and the invocation ID derived from (generation, sequence). Any other
TokenIds output is refused. The first stage starts its own prompt chunks with
`enqueue_prompt`: the same grant, hold and core admission, prefill only, derived IDs.
See [daemon pipeline](daemon-pipeline.md).

## Verification and remaining integration

Controlled tests exercise wrong nonce/scope/issuer, token and any-stage permission
withdrawal, malformed/oversized assignment/reply, local release/expiry/owner loss,
lease isolation and retained in-flight buffers. Two child processes use real
enrollment, registry heartbeats, pinned assignment, Prepare/Commit/Begin, both
stage TLS endpoints and live local data admission. They exchange the unchanged
raw BF16/FP16/FP32/u32 golden payloads and receipt frames, asserting no execution
state changes. Their protected device directories contain no authority database,
CA private key or client-token artifact. This tests API/process integration; it
is not OS sandbox isolation or model inference.

GLY-183 still owns aggregate credit/metadata budgets, reassembly, ordering and
ambiguous-execution handling. GLY-181/185/186/187 and provider GLY-235/236 still own
transactional/native preparation, partial-worker conformance and daemon pipeline
activation. Actual native/device/link/performance and milestone acceptance remain
open. The whole-request daemon path is unchanged; no native layer execution or
hardware qualification follows from these tests.
