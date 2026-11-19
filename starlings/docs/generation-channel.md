# Coordinator-to-worker generation channel

The [explicitly qualified CPU daemon route](production-routing.md) now owns these
shared primitives in production entry points. Native activation requires reviewed
local evidence and full configuration approval; no native qualification record
is supplied here. Earlier library-only boundaries below describe their owning
PRs. Qualification, cleanup/recovery and multi-worker acceptance remain M2 gates.

`GenerationStream` uses TLS 1.3 with the distinct
`starlings-generation/1` ALPN. It is separate from enrollment, heartbeat,
admin and client connections. The worker accepts it through an independently
budgeted `PeerListener`. No new daemon port is enabled by this library change.

The coordinator retains live membership/model policy in its existing protected
database. It checks its own Control permission and the worker's Infer permission
for the immutable manifest, both current credentials, and the worker's fresh,
reconciled registry epoch. Transport connectivity does not make an unavailable
backend schedulable; capability and native qualification remain separate gates.

The worker needs its enrolled device identity, bootstrap CA/coordinator pin,
current `RegistryFence` and locally approved model/profile binding. It receives
neither the CA private key nor the policy database. Mutual TLS checks the chain;
the coordinator must also match the pinned public key and exact certificate name
(node, role and cluster). A valid certificate refresh with the same key works.
A new key/name/CA requires an explicit trust change. The worker's own certificate
and the peer certificate must remain unexpired for the connection lifetime.

Before application records, both endpoints exchange and compare this strict JSON:

```json
{"version":1,"epoch":{"number":"1","incarnation":"<authority UUID>"},"model":"<manifest SHA-256>","profile":"<approved execution profile>"}
```

Each record has a four-byte big-endian length followed by strict JSON, sent as one write
on a `TCP_NODELAY` socket (two small writes waited ~80 ms for a delayed ACK on Linux). Bounds are
1 MiB per record, 32 MiB including prefixes across both directions, 8,192 records
including binding exchange, and ten minutes per connection. This larger record
limit is specific to generation; existing enrollment/admin/control framing stays
16 KiB. Duplicate keys and malformed/deep JSON fail before deserialization;
application records must use strict typed schemas and lifecycle validation.
An entire record has a two-second I/O deadline, not a new deadline per byte.

Policy/fence and certificate checks bracket each record without holding policy
locks across network I/O. Any error permanently closes the connection. Dropping
it also closes the socket even if a supervisor retains a watch. A supervisor must
check `GenerationWatch` while backend execution waits; `cancel` interrupts blocked
I/O. A stalled connection cannot be resumed or joined to another attempt.

This channel does not itself dispatch work, authorize a client token, qualify a
backend, implement admission, free reservations or establish device drain.
The integration must bind validated records to the approved model/profile,
preserve request replay/lease semantics, send bounded progress while inference
waits, propagate client cancellation and retain allocations until trusted local
completion. Backend generation, heartbeat maintenance and client-output gates
must not share a blocking thread. No prompt or output record is logged here.

The initial coordinator record checks still consult local policy. Integration
must batch stream events and measure this cost before claiming the M2/M3
performance gate; this is not the database-free client output gate.

Development builds may opt into `STARLINGS_TEST_GENERATION_GATE_TRACE` to record
`GENERATION_GATE_REFUSED` with only a static side/phase and redacted error enum.
Only a failed authorization check that first closes the channel emits a record;
subsequent checks cannot overwrite it. Explicit cancellation or an earlier I/O
failure may already have closed the channel and supplies no such record. The
trace is silent without opt-in and compiled out when debug assertions are off.
It changes no policy/deadline and does not identify why an OS operation stalled.

## Enrolled worker lifecycle

`EnrolledWorker` binds the existing core lifecycle to a remote worker's registry
fence and one trusted local `WarmReplica` profile. It accepts only whole-request
plans matching the exact manifest, profile, backend build and memory pool, within
the qualified context and memory budget. One lease and one session can occupy
the supervised service. The qualification record is supplied by trusted local
configuration; its evidence digest is not a remote attestation or native proof.

Only the worker side of the matching generation connection can dispatch. A new
accepted Prepare binds that connection to the lease. Another connection cannot
resume its execution or output. Exact request replay on the original connection
returns the existing acknowledgement and must not launch a backend again.
Network completion, token/layer steps and separate Begin operations are refused.

Existing registry maintenance advances lease time, withdraws disconnected work,
and fences stale/offline/revoked or changed epochs. Cancellation keeps in-flight
memory reserved. Enrolled workers sample their own monotonic clock under the
runtime lock, so timestamps calculated by concurrent timer/request threads cannot
arrive out of order. Pure core/coordinator-local tests retain their injected clock.
`withdraw` requires the owning channel and cancels only its leases;
only trusted local `drained`/`failed_and_drained` calls acknowledge completion.
The owner must keep the runtime and native allocations alive through cleanup.
The short `with_output` callback holds the registry fence; it must only enqueue
bounded data, never block on network/backend work or reenter the worker.

## Whole-request relay

`starlings_cli::inference` connects an admitted HTTP request, coordinator generation
connection, enrolled worker and an already approved/loaded `Service`. It opens no
daemon listener and starts no model. The worker profile's build digest must match
the supervised executable SHA-256, with exact manifest/profile identities and a
service timeout at most 60 seconds. Shorter lease, registry, client and record
deadlines still apply.

`admit` accepts only POST `/v1/chat/completions`, the immutable manifest alias and
explicit temperature/top-p/seed. It rejects unsupported fields before dispatch.
`ClientAdmission` holds shared policy without any private key. A background
admission thread uses `ClientListener::try_accept`/`prepare_with`; independent
authority maintenance must continue at least once a second. The existing
`EnrollmentReader::connect_generation` similarly uses only the coordinator leaf
and shared policy, never the CA signing key.

`count_prompt` first asks the selected worker to count the exact admitted chat,
through its approved service's `/v1/chat/completions/input_tokens` extension.
This extension is not part of the official OpenAI API. Unsupported services fail;
there is no byte estimate or tokenizer fallback. The service's two-second count
request uses exactly the same body conversion as generation. The worker checks
count plus requested output against its qualified context before returning a
count bound to the chat digest. The coordinator then builds controls on that same
connection. Client response scope must be Infer for that exact manifest.

`relay` takes exactly Prepare, Commit and StartRequest controls supplied by
qualified placement and this count. It compares the start
payload to the admitted chat. `serve_worker` applies all three to the existing
state machine, requires the exact counted chat and prompt count, and launches only
new accepted starts. The service binds the next generation to the counted backend
body and rejects final usage that reports a different prompt count. Replayed controls close the
attempt; they never invoke the service again. No prompt/output record is logged.

The strict JSON command kinds are `count` (ChatRequest), `apply` (ControlRequest)
and `poll` (generation UUID); replies are `counted` (chat digest and token count),
`applied` (request UUID), `pending`, `fragment` (UTF-8 text and
end-of-event flag), and `finished`. One producer sends through a single-slot queue,
with 16 KiB UTF-8-safe fragments and a two-second queue deadline. The coordinator
polls every 20 ms while idle. Complete events are limited to 64 KiB for SSE and
1 MiB for JSON. Channel record/byte limits also apply. The worker checks its output
fence before dequeuing; neither network nor backend I/O holds that fence.
The complete StartRequest envelope must fit the 1 MiB record limit, so a chat body
at the HTTP maximum can still be refused once lifecycle metadata is included.

SSE uses the documented `X-Compute-Session-Id` header. A failed attempt can send
one redacted `event: error` while client authorization remains valid, then closes
without `[DONE]`. Before headers, existing admission failures close the socket;
the complete documented HTTP error/status mapping remains a daemon integration
gate. JSON is held until the backend reports successful completion. Client loss,
revocation, lease/fence loss and backpressure close the generation channel, cancel
the owned service and join its request thread. There is no failover/replay after
partial output.

The worker relay returns `Ok(None)` only when no Count was accepted, an invalid/
lost input connection is closed, and the runtime has zero reservations. It calls
no backend operation in that path and does not signal the loaded process. The
supervisor must recheck health/fence/empty ownership before reuse. Accepting Count
or observing any old reservation prevents this exception; no cleanup proof or
quarantine reset is inferred.

`Ok(Some(WorkerCompletion))` means the backend request returned successfully, not that its
native allocations have drained. Every exit withdraws output; reservations remain
until trusted local drain. Production activation still needs catalog pins, native
token-count parity and profile/drain qualification, measured placement and daemon
route configuration. Synthetic relay tests establish none of those native gates.

`Dispatch` supplies coordinator controls for one locally qualified worker/epoch.
Keep one serial owner across reconnections; recreate it only after a new durable
coordinator epoch. It counts on the selected channel, builds fresh IDs and a
bounded lease, and relays once. Failed requests consume sequence numbers and
cannot wrap them. The session client is the authenticated controller retained by
the output grant. The relay rejects a different client even when called directly.
Qualified placement, current capacity and native drain remain caller obligations.
