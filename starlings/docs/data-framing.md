# GCP1 framing contracts v1

GLY-182 provides framing contracts and an authenticated transport library over
the existing enrolled peer stream. It adds no daemon listener or native backend
invocation. GLY-183 adds [bounded credit/reassembly APIs](data-flow.md). Deployed
pipeline ordering/cancellation integration and provider conformance remain
separate. A delivered chunk is not a complete tensor or permission to execute it.

The 16-byte big-endian prefix is `GCP1`, major u16=1, kind u16, header length u32,
payload length u32. `DataPrefix::parse` rejects unsupported magic/major/kind,
empty or >16 KiB headers, >8 MiB payloads, empty tensor chunks and any payload on
a control frame, using only its fixed array. Allocate the header only after this
check. `DataHeader::decode` then reuses strict duplicate-aware JSON, embedded
offline schemas and semantic validation. It checks kind/payload agreement before
the transport may allocate the payload. A complete tensor is at most64 MiB.
Connection/session budgets may be smaller and are independent of these maxima.

| Prefix kind | Header kind | Meaning |
| --- | --- | --- |
| 1 | TensorChunk | Existing flat `tensor-header.schema.json`; raw contiguous little-endian bytes |
| 2 | Credit | `bytes` is a positive additional byte credit, capped at64 MiB; never a reservation |
| 3 | ReceiptAck | `tensor_index`, positive cumulative `received_through_bytes` capped at64 MiB; validated receipt only |
| 4 | ExecutionResult | Successful invocation report; no native allocation/drain claim |
| 5 | Cancel | Withdrawal request with a fixed redacted `code`; no cleanup receipt |
| 6 | Error | Failure report with a fixed redacted `code`; never partial output |

Control headers use `data-control.schema.json`: exact `kind`, `protocol_version=1`
and nested `scope`. The scope contains plan digest, nonzero authority epoch,
lease/session/generation/invocation UUIDs, decimal sequence and source/destination
stage indices0–7. It describes the **original forward invocation**, including
when an ACK/credit/result travels in reverse. No free-form error text, addresses,
paths or tensor values are carried in a control header. Codes are `cancelled`,
`expired`, `protocol`, `capacity`, `backend`, `disconnected`. Backend failure uses
Error; ExecutionResult reports only success. Receipt cannot substitute for
execution, and neither report authorizes local reservation release.

The transport must compare every scope to a trusted current admission, validate
the enrolled peer and model permission, and check absolute lease/registry/record
deadlines before allocation and delivery. Control credit/receipt bounds must also
match the actual negotiated budget/tensor. These cross-record checks cannot be
proved by the structural parser. The initial negotiation must bind both peers to
the same protocol and approved direction; metadata from an incoming header can
never create a grant. A malformed/failed frame must poison and close its stream.

`examples/gcp1-golden.json` supplies synthetic fixed prefixes, exact JSON bytes and
little-endian payload vectors for BF16, FP16, FP32 and u32. Floating vectors
include zero, negative zero, infinity and a NaN payload. They must travel as bytes
without numeric conversion. The zero plan digest is an explicit fixture and is
refused by production validation. `examples/gcp1-control-golden.json` supplies all
five control-kind golden inputs. Codec vectors alone do not prove model exactness
or a native backend capability.

## Authenticated persistent connection

`DataStream::connect` / `accept` consume an enrolled `PeerStream` on
`Channel::Data`, ALPN `starlings/1`. TLS 1.3, exact current enrolled leaves,
role, cluster and live credential checks reuse the existing security layer.
No resumption, early data, new cryptography or implicit peer enrollment is added.

Verified data ALPN enables TCP_NODELAY at both endpoints in the shared TLS
boundary, including node-local stage delegation. Other protocols retain the
caller's socket setting. A setting error refuses the connection. Readback is
tested without a timing threshold; physical-link latency, keepalive/power and
one-round-trip token scheduling remain GLY241 qualification work.

Opening negotiation is one bounded security JSON record in each direction
(big-endian u32 length, at most16 KiB), with exact fields: `version=1`, `epoch`
(number and incarnation), `plan`, `source`, `destination`, `source_stage` and
`destination_stage`. The initiator sends first. Each compares the peer's binding
to its independently derived local binding before accepting GCP1 records. A
connection persists across invocations for that plan/direction/authority; it does
not negotiate a broader permission or silently change plans.

The local lifecycle owner calls `SecuredWorker::data_admission(peer, token,
expected_header, clock_origin)`. `clock_origin` is the same monotonic origin used
by its control dispatch and maintenance. The expected boundary template is trusted
local state, never a header copied from an incoming data stream. Factory and every
send/receive gate require the current committed layer plan, ready unexpired local
lease, live session, model permissions on every stage, owning client token, exact
enrolled peer/direction and online registry fence. Authority incarnation cannot
change under an existing binding. The receiver independently proves its own lease;
a sender's declaration is not evidence that the destination lease remains live.

Tensor dtype, shape, boundary, tensor role/count/index, scope, phase, position and
sequence must match the local template; only validated chunk offset/length/final
flag may vary. Adjacent stages carry hidden/residual tensors; only last-to-first
may carry u32 token IDs. Receiving tensor data also checks current local position,
sequence and absence of an already started/reused invocation. Raw values never
undergo floating-point conversion or precision changes. A template/profile
declaration does not qualify native numerical compatibility or token vocabulary.

`receive` validates the fixed prefix, bounded header, local admission and deadline
before allocating the payload (at most8 MiB). Prefix/header/payload and final
dispatch share the existing absolute10-second record cap. The final enqueue runs
under the policy, runtime and registry-fence locks, with a deadline check before
the callback. It must be short, synchronous, nonblocking and must not reenter those
objects. No network/backend I/O runs under these locks. Consumers must use
[the flow APIs](data-flow.md) for aggregate credit/reassembly before backend work;
this raw receive primitive alone provides no aggregate guarantee.

Any malformed record, authorization failure, disconnect or record deadline failure
poisons and shuts down the connection. Buffered plaintext cannot be read by a new
operation after failure. Nothing retries an ambiguous operation, advances worker
state, frees native buffers or treats an ACK/ExecutionResult as local drain proof.

## Integration and evidence boundary

Two local admission paths now feed the same GCP1 parser and persistent stream:
authority-backed `SecuredWorker`, and [node-local `StageWorker`](stage-delegation.md)
using short-lived coordinator route grants. The latter reuses pinned generation
control, enrolled device keys and public CA trust without authority DB, CA signing key or client-token
replication. Both actual stage endpoints exchange all four golden dtypes
across child processes with local lifecycle admission; the earlier raw byte
producer remains a narrower framing regression.

The production daemon's remote `EnrolledWorker` remains whole-request only.
Transactional/native preparation, qualified external stage execution and daemon
pipeline activation still belong to GLY-181/185/186/187 and provider235/236.
The [GLY-183 flow APIs](data-flow.md) add reserved direct reassembly and scoped
credit/receipt tracking; deployed pipeline/owner integration remains open.
No model inference, multi-device performance or native sharding qualification is claimed.
