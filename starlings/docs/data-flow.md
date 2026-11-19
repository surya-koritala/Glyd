# Bounded GCP1 invocation flow

GLY-183 adds exclusive sender/receiver APIs over the existing authenticated
[persistent connection](data-framing.md). It does not activate daemon pipelines,
start a backend, advance execution, qualify a model or release native work.
The existing raw frame API remains a lower-level primitive; it does not provide
these aggregate guarantees. Pipeline integration must use the flow APIs.

## Trusted input and allocation

The local owner supplies one to eight ordered `DataAdmission` values derived from
its approved boundary templates. Incoming headers cannot create admissions.
All templates must have one scope, phase, position, token count, boundary,
binding and authenticated local scratch allowance. Tensor indices cover the
complete ordered list. Full tensor sizes are checked before any payload allocation.
Each template is a legal wire chunk; a reassembled tensor is not coerced into an
oversized wire header or used to bypass the core's existing StartStep checks.

`ReceiveBudget::new(total, per_peer, per_session)` is a shared local allocation
budget. Create it once per host allocation pool and share its clones across
receiving connections, rather than creating a new budget for each connection.
The owner must supply a qualified available host budget; a GPU scratch declaration
does not prove host capacity. The ledger atomically enforces total, per-peer,
per-session and actual local lease scratch limits, including concurrent handoffs.
Session accounting spans peers; lease accounting spans sessions and peers.
An invocation with live buffer ownership cannot reserve the same identity twice.
At most128 live ownership records bound the maps independently of payload size.

`DataStream::begin_receive(admissions, budget, deadline)` first checks the live
connection and local admissions, reserves the complete ordered payload total,
then fallibly allocates the owned tensor buffers. An overlarge caller budget
cannot increase the authenticated local lease allowance. A poisoned connection
cannot allocate another assembly. The receiver reads incoming chunks directly
into these buffers; it never creates a second frame-sized payload Vec.

These are retained host tensor payload bounds, not measured RSS, allocator
overhead, native state, GPU memory or a numerical/profile qualification. Existing
TLS buffers and bounded metadata remain separate. Native costs, placement and
transactional preparation must account for all actual memory pools independently.

## Driving a transfer

| API | Effect |
| --- | --- |
| `begin_receive` | Exclusively borrow the persistent stream and reserve/allocate ordered inputs; a stage may open the step right after the one it is running |
| `FlowReceiver::grant_credit(bytes)` | Grant positive additional payload bytes only within the already owned total |
| `FlowReceiver::receive_chunk()` | Validate credit, exact current template/order/offset and live input state before reading; acknowledge validated bytes |
| `FlowReceiver::finish()` | Move complete tensors and their reservation once, under a final live input-state check |
| `begin_send` | Exclusively borrow the stream using the corresponding trusted outgoing admissions |
| `FlowSender::receive_control()` | Accept only bounded credit or monotonically advancing receipts for bytes actually sent |
| `FlowSender::send_chunk(bytes)` | Construct the next exact chunk; consume credit without accepting caller-selected wire metadata |
| `FlowSender::finish()` | Require every byte sent and acknowledged; confer no execution or drain permission |

Credits use the existing invocation scope and are additional bytes, not native
memory reservations. A credit record is bounded by the largest trusted tensor
and the existing64MiB control limit; several records may cover a larger ordered
input list. Cumulative grants cannot exceed the approved total. A receipt cannot
replenish credit, acknowledge unsent bytes, execute a step or release buffers.
ExecutionResult is deliberately not accepted as a flow receipt.

The driver interleaves payload writes and control reads. The regression sends a
chunk, reads its receipt and continues; it grants input credit before payloads.
There is no hidden thread/runtime or automatic request retry.

A step's output is sent and its receipt read while the step is pending; the next
step's input is admitted only after it drains. Waiting for that receipt before
granting the next credit would cost each hop three round trips. So `begin_receive`
on a stage (and with it the credit) may open the step right after a pending one:
`validate_data_input_next` checks its sequence, position, phase and invocation
against the session as the pending step's completion will leave it, and refuses
any later step. Every chunk and `finish` still pass `validate_data_input`, which
admits only the step now due: a peer's bytes for the next step wait in the
socket until the running step drains. The daemon pipeline opens the next receive
right after its own output is on the wire, so a token costs one round trip.

## Deadlines, failure and ownership

Every actor has an absolute deadline supplied by the local owner and narrowed
to the existing10-second record cap from construction. Validation, allocation,
credits, payloads, receipts and handoff share that deadline; new traffic cannot
extend it. At most256 total inbound/outbound records and1MiB of JSON header bytes
bound metadata work for one invocation. Each fixed16-byte prefix and header also
retains the existing structural limits. Too many tiny/control frames fail closed.

Duplicate/reordered/gapped chunks, wrong scope/shape/role, insufficient credit,
forged/repeated receipts, malformed/truncated traffic or expired authority poison
the connection. No partial buffer is exposed. Failure or dropping an unfinished
actor also aborts the generation identified by its immutable local admission,
never a generation selected by received metadata. The core checks that generation's
session, plan and authority epoch. This local withdrawal works after peer/control
loss or registry expiry; it does not need to reauthorize through the failed channel.
An unexecuted generation becomes Failed. A pending invocation becomes Cancelling
and retains its original terminal reason until trusted local completion. Existing
terminal states are not rewritten, other generations are untouched, and the
lease is not released by transport failure.

An unfinished receiver drops its unexecuted host assembly and receive-budget
reservation. An unfinished sender closes its connection without freeing
caller-owned native output/state. A local lease release still retains capacity
while execution is pending; only the trusted drain path can finish that release.
An owner must drive bounded I/O or drop an actor when cancelling; no background
collector reclaims inert local objects behind the owner's back.

`ReceivedTensors` owns immutable ordered buffers and the reservation. Moving it
to another local queue retains that ownership; receipt/finish does not release
it. Dropping it releases host inputs, not a core/native reservation. A native
owner must retain inputs through safe device completion and separately recheck
the full current authority incarnation, lease and exactly-once execution state
before dispatch/publication. A retained tensor scope alone is no execution grant.
After successful handoff the local execution owner remains responsible for
disconnect/cancel, input lifetime and trusted native drain. The flow abort does
not run or cancel a native kernel, prove cleanup or retry ambiguous execution.

## Evidence and remaining integration

Model-free controls use enrolled TLS, ordered hidden/residual chunks, all four raw
dtype golden vectors, a16MiB tensor split across two8MiB frames, scoped allocation
limits, malformed/replayed/truncated traffic and pending-step handoff refusal.
Scoped abort checks preserve a second generation and require actual pending-step
drain before releasing its lease. A connected, unread receiver with bounded OS
buffers drives an actual 8 MiB TLS send into the caller's 500 ms deadline;
subsequent driven receive failure releases its host assembly and fails both
local generations. This is a loopback failure check, not a performance result.
Delegated-owner and registry-loss controls prove local abort needs no live peer.
The existing two actual child-process StageWorkers retain their original raw
framing tests and also perform credited reassembly on both endpoints without
authority files, token replication or any execution-state mutation.

GLY-183 remains In Progress for deployed daemon pipeline/owner integration and
its end-to-end ambiguous-execution and native cleanup acceptance. Library
slow-reader/cancellation controls do not qualify a deployed native owner.
GLY-181/185/186/187 and provider235/236 remain required; this does not substitute
for installed native workers, same-type device measurements or milestone review.
See [the evidence record](evidence/GLY-183-data-flow.md).
