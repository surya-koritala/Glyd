# Coordinator reservation transactions

GLY-181 implements `starlings_core::admission::AdmissionBook`, an in-process
coordinator-owned transaction over the existing bounded planner and per-pool
profile budgets. It reserves software accounting before preparation and exposes
the all-stage prepare/commit barrier. It performs no I/O, backend invocation,
device probing or daemon activation. `Ready` means the owner has recorded all
logical observations; it is not a native execution grant or qualification.

The separate [offline calibration report](plan-calibration.md) binds supplied
observations to complete planner/candidate and evidence identities. It collects
no measurements and cannot approve the book's native integration.

## Selection and ownership

Construct one book for an enrolled cluster, nonzero authority epoch and fresh
owner incarnation UUID. Do not recreate a live book to discard reservations or
reuse an incarnation after restart. `begin` recomputes the existing preview from
the immutable input and selects its indexed candidate; a caller cannot submit a
modified Candidate as a reservation. The stored input digest binds the complete
profile/workload/runtime/inventory/link inputs, and each stage retains its
snapshot, complete profile digest, layer range and every physical pool budget.
Trusted local approval/capture clocks must be supplied by the owner, never taken
from an untrusted request. Production mode rejects synthetic inputs.

All mutations require exclusive ownership of the same book. Up to256 live
transactions each have at most8 stages; planner/schema limits bound their pools
and input/search sizes. All sums and deadlines are checked. Handles bind the
incarnation, epoch and a never-reused monotonic serial. Cleanup removes a live
record without resetting the serial; stale handles cannot mutate a newer record.
The book is neither persistent recovery state nor a registry authority fence.

`withdraw` permanently closes this book to new admissions and moves every live
transaction to `Releasing`. It is the local owner's withdrawal operation for
authority withdrawal or shutdown, not a policy check or native cancellation.
Existing handles and partial cleanup remain valid; late prepare/commit replies
cannot restore readiness. Repeating withdrawal is harmless, and draining the last
record does not reopen the book. There is no resume operation. The caller must
withdraw live dispatch/output grants separately and preserve owned cleanup before
constructing another approved owner; this method does not integrate the daemon,
registry fence, durable recovery or provider cancellation.

## Pool accounting and telemetry

Before insertion, every candidate pool must fit with **all** outstanding software
reservations on that node/pool. Shared memory is charged once by physical pool ID;
different node IDs remain separate. Reserved bytes are the entire conservative
profile budget, even when a preparation reports a smaller actual allocation.
For overlapping live reservations, capacity is the minimum of their admitted
free-memory envelopes and the current snapshot's available-minus-reserved value.
A newer, larger report cannot replenish an earlier live reservation. A changed
device or pool-inventory identity is refused while that node has a live record.
These conservative rules can refuse capacity already reflected in telemetry;
they intentionally make no inference about native allocator aliases or caches.

## State transitions

1. `begin` holds every stage/pool and enters `Preparing`, with a monotonic lifetime
   of1..60,000ms. No preparation has yet been invoked by this library.
2. The trusted local execution owner calls `prepared` for the exact handle and
   stage with a complete pool/allocation map. All expected pools, including zero
   allocations, must be present and no budget may be exceeded. Exact repeats
   during preparation are accepted without changing ownership. Conflicting,
   missing, extra or oversized allocations enter `Releasing` and fence new plans
   involving that node until the whole transaction has trusted cleanup. Late
   invalid observations still fence the node after expiry/abort. If that stage
   was already marked drained, its earlier cleanup is invalidated: a fresh trusted
   cleanup observation is required before its hold or transaction can be removed.
3. `start_commit` succeeds only after **every** stage is prepared. It enters
   `Committing`; each matching `committed` observation is recorded. Only the final
   stage acknowledgement enters `Ready`. A partly prepared/committed plan never
   reaches that barrier. No provider commit command is sent by these methods.
4. Abort or `advance(now >= deadline)` enters `Releasing`, including from Ready.
   Backward clocks fail. Timeout, missing reply and cancellation never free bytes.
   Call `advance` before interpreting a snapshot at a new time; a read reflects
   only the last injected time, not a live wall-clock guarantee.
5. Only a trusted local execution owner calls `drained` for each stage, proving
   either no prepare was dispatched or no future native access/allocation remains.
   Each stage's reservation is released then; all-stage cleanup removes the record.
   An IPC/socket exit, remote ACK, wire receipt or parent-process exit is not proof.

## Local loaded-instance ownership

The separate `starlings_core::Worker` now has a trusted local
`hold_backend(lease, now)` prerequisite for loading a native layer stage. An
idle range worker can retain weights, KV, state and scratch after a chunk or
session ends. Invocation completion must therefore not free the stage's complete
lease budget. The owner attaches the hold to a live `Prepared` layer lease,
before dispatching preparation or any potentially ambiguous allocation. This
does not qualify a backend, reserve additional bytes or authorize native work.
Ordinary unheld and whole-request CPU leases keep their existing behavior.

The opaque `BackendHold` binds one local preparation using private `Arc` identity.
It is not serializable, a peer grant, a request result or an allocation receipt.
Neither cloning nor dropping it releases anything. Release, expiry, revocation
and epoch withdrawal close admission/output normally, but keep the lease
`Releasing` with the same reserved bytes until both pending invocations and the
loaded-instance hold are cleared. No native dispatch/output hook is added.

Only the qualified local supervisor may call `backend_unloaded(&hold)` after
withdrawing the lease and proving complete safe instance cleanup. It refuses
pending invocations, foreign ownership and stale tokens for a reclaimed/reused
lease UUID. An exact repeated cleanup of the same retained record is harmless.
A provider chunk/status/end reply, zero engine-buffer count, process exit or
timeout cannot establish this precondition. The existing trusted
`backend_shutdown_drained` acknowledgement clears all instance holds only as
part of confirmed whole-backend cleanup and permanently stops that runtime.
No network operation can mint or complete a hold.

Recovery metadata records nonempty `retained_backends` lease IDs, never the
opaque token. `Worker::recover` refuses those records rather than starting with
unaccounted native allocations. The default empty field is omitted so old CPU
records retain the same serialized representation. Older readers fail closed
on the new nonempty field. A cleared or edited record is not physical cleanup
proof: the actual restart supervisor and durable launch/catalog ownership must
still establish qualified cleanup before creating another owner. That native
restart integration remains open.

This local hold and the coordinator's `AdmissionBook` remain separate layers:
both must retain their respective reservations until the trusted cleanup owner
has sufficient evidence. Provider engine counters alone are not complete
physical pool observations. Actual range dispatch, complete multi-tensor core
start integration, live authority enqueue/output gates and installed qualification remain
required before native activation. See the
[pure control evidence](evidence/GLY-181-native-ownership.md).

## Trusted local complete-boundary start

`Worker::start_local_chunk(&BackendHold, &[TensorHeader], now)` starts exactly one
existing core invocation from a complete ordered boundary. It requires the
current retained local preparation identity; `check_backend_hold` reports its
lease without asserting authority, readiness or native qualification. The
secured owner must hold the live StageGrant/lease fence and retain authentic
fully reassembled bytes, checking their lengths and the registered boundary
layouts before using this core method. The core neither reads buffers nor
launches, queues or retries a backend.

All one-to-eight header templates are validated with the existing strict
protocol validator. Scope, phase, position, token count and profile agree;
indices/counts are complete and ordered; every first-frame offset is zero.
Checked aggregate tensor bytes must fit the admitted local scratch budget.
Embedding input is one token-ID tensor on the embedding stage; non-embedding
stages accept only the approved ordinary boundary roles. Actual dtype/shape/role
agreement with the model's full boundary remains the secured owner's prior
registration check. Every metadata/aggregate check precedes the single existing
session/invocation mutation, so a bad later tensor cannot consume a valid step.

These are immutable original frame templates from the receive owner, not a
claim that one wire frame contains an entire tensor. A completely reassembled
tensor can be larger than the 8 MiB frame ceiling, up to the existing 64 MiB
tensor limit. Its template retains a legitimate first frame; no oversized
`chunk_bytes` or weakened protocol validator is introduced. The owner must keep
the reassembly receipt and its reservation alive through backend use.

The implementation reuses existing pending/sequence/invocation/session/lease
and retired-generation state. `can_publish_step(generation, invocation)` adds an
exact pending-step check to the existing publication gate; it is read-only and
does not replace live authority around synchronous output enqueue. Completion
still uses the existing trusted local path and keeps the loaded-instance hold.
The wire `StartStep` remains single-tensor, peer start/completion remain denied
by StageWorker, and no schema or remote operation expands.

This is a local prerequisite, not an activated range worker or pipeline. Ordinary
local inputs come from the previous stage; incoming token IDs use the existing
authenticated data direction from the final stage to the first stage. This does
not map a sampled output into a new decode invocation, sequence or position.
First-stage text tokenization and outgoing feedback integration, all-stage
transactions, actual process ownership and complete physical-pool/native cleanup
qualification remain open. See the
[pure control](evidence/GLY-181-local-boundary-start.md).

## Software evidence and remaining integration

`tests/admission.rs` uses bounded existing synthetic planner inputs. Seven pure
controls cover the two-stage barrier, competing/multi-pool reservations, inflated
telemetry and changed inventories, over-budget late replies, expiry/rollback,
conflicting preparation, late invalid allocation after partial cleanup, stale
handles, fixture refusal, stale capabilities and irreversible owner withdrawal
across empty, multiple preparing/committing/ready and partly cleaned records.
The existing platform CI runs these through its ordinary core tests; no new job,
dependency, network fixture or timeout is added.

The daemon does not instantiate this book. GLY-235/236 must supply actual range
worker/process/handle/allocation/completion mapping and measured profiles before
integration. Live registry/revocation fences must guard dispatch and output;
durable owned-launch/quarantine recovery must preserve uncertain native ownership
across restart. Reconcile leases, actual provider lifetimes and physical-pool
reservations with those owners before exposing a production pipeline. Native
allocation/exactness, calibration, physical devices and independent reviews remain
open. This software does not complete GLY-181, M2, M3 or a later milestone.
