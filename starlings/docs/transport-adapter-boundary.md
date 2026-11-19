# Transport adapter boundary — source proposal

GLY-241's adapter-seam artifact. This source proposal introduces no runtime
interface or RDMA support. The existing path remains `PeerStream` over enrolled TLS,
`DataStream` GCP1 framing and exclusive `FlowSender`/`FlowReceiver` invocation
ownership. No custom UDP transport is planned.

## Boundary to preserve

| Existing owner | Required invariant for any later adapter |
| --- | --- |
| `starlings_protocol::DataPrefix/DataHeader` | Same bounded prefix/header validation, raw dtype/shape/chunk order and fixture refusal. Parse before allocating payload. |
| `DataAdmission` / stage authorization | Owner-created, exact peer/cluster/plan/lease/epoch/invocation/direction; live credential/grant checks before I/O and before publication. An endpoint or memory key is not an admission. |
| `PeerStream::frame_until` | One absolute deadline for the complete operation, bounded I/O and terminal poisoning on uncertainty. A partial transfer never resumes on a replacement connection. |
| `FlowReceiver` / `ReceiveBudget` | Complete invocation reservations before credit; bounded pool/peer/session/lease accounting; exclusive ordered receive and retained owner handoff. |
| `FlowSender` / local execution owner | Wire receipt is validated transport receipt only. Native state mutation, cancellation and safe device completion remain separate trusted local operations. |

GCP1 currently calls a concrete `PeerStream`; accepting an arbitrary `Read+Write`
object would lose authenticated peer/deadline/revocation/poisoning obligations.
Do not add a generic adapter trait until there is a real second implementation
and approved hardware/qualification plan. Keep frame/flow/admission validation
above the future byte-transfer boundary, rather than duplicating it per backend.

## Proposed operation contract

1. `open`: accept a locally approved endpoint and transport kind together with
   current admission and an absolute deadline. Bind both authenticated nodes,
   cluster, plan, authority incarnation and stage direction. Refuse unproved or
   unavailable transport; no implicit TCP fallback or downgrade after work starts.
2. `transfer`: move exactly the validated prefix/header/payload bytes into/out of
   owner-reserved host buffers, preserving stream order, total byte/metadata limits
   and the same absolute deadline. Report partial progress as failure, not receipt.
3. `finish_transport`: expose validated receipt only after ordered bytes are
   available to the receiver's owner. Never report backend execution/drain or
   release a native reservation from transport completion.
4. `abort`: withdraw publication immediately, poison the operation and retain
   uncertain native ownership. An RDMA implementation must also prove outstanding
   device/queue access is stopped before unpinning/unmapping a registered region.
   If that proof is unavailable, retain/quarantine ownership; do not free/reuse it.

A future registered-buffer capability must be unforgeable and locally owned,
bound to buffer extent, peer/direction, invocation/generation, memory pool,
expiry and outstanding-transfer lifetime. Peer-provided addresses, remote keys,
queue completion messages and wire receipts cannot mint such a capability.
Start with host staging; no unqualified direct GPU-memory registration or claim
of zero-copy. Exact native completion semantics require the delivered backend
contract and independent review.

The adapter must supply its own authenticated/confidential/integrity-protected
transport binding; a control-channel TLS handshake alone does not prove a
separate data channel has those properties. Reuse live admission checks and
version the binding/negotiation explicitly. No raw RDMA listener, discovery hint,
new enrollment authority or remote memory-registration API is enabled here.

## Qualification and decision

Before an implementation is adopted: authenticated positive/negative conformance
for exact bytes/dtypes/order and expired/wrong/revoked peers; malformed/oversize/
replay/partial/slow-reader failures; absolute cancellation deadlines; safe
in-flight memory ownership under completion loss, reconnect and device removal;
all supported platform checks; independent security review. Real approved hardware
must then record exact adapter/driver/runtime, code/build identities, physical
link topology, copy/registration/CPU/power costs and matched TCP baseline. Software
mocks do not qualify physical RDMA or native inference.

Custom UDP decision: **no-go**. The active roadmap records an earlier operator
Wi-Fi baseline with no demonstrated TCP/UDP benefit; it has not been reproduced
by this artifact. A separate UDP implementation would add ordering, loss,
congestion, bounded reassembly, authentication and lifecycle responsibilities.
No demonstrated acceptance need justifies that work. Reopen only after approved,
reproducible matched evidence identifies a material TCP limitation and the full
security/resource/ownership contract can be met. The token-link profile is an
application experiment and supplies neither raw wire throughput nor this proof.

This defines a source-level seam only. It does not complete GLY-241's actual
per-token scheduling, physical-link measurements or keepalive/power acceptance.
