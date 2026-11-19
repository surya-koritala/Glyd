# Listener boundary

GLY-166 supplies synchronous listener primitives in `starlings-security`. GLY-167 integrates
bootstrap/admin into the [daemon](daemon.md); inference and artifact routing
remain unavailable. GLY-167 owns daemon lifecycle and the admin command allowlist; M2 owns
model resolution, actual routes and streaming output. Do not advertise those
services until their integrations pass the same security gates.

## Admission and deadlines

Share one network `ListenerBudget` across peer/bootstrap/client listeners. Give
local admin a separate budget. The coordinator uses `try_accept_registry(&authority)`
before scheduling work. It verifies a fresh device-key proof before reserving one
of 63 node-specific handshake/registry slots. Bootstrap keeps the ordinary budget.
Control/data listeners use `accept` or nonblocking `try_accept`. The
returned object owns its admission; dropping it releases the connection. Do not
queue accepted connections or pending enrollment objects outside this budget.
The daemon uses admission-bounded threads; no thread pool or asynchronous runtime is added.

| Resource | Ceiling |
| --- | --- |
| Active pre-auth/client network connections | 32 |
| Proved registry handshakes plus established peers | 63, at most one per node |
| Pending admission probes, shared across interfaces | 64 sockets, no TLS object or thread |
| Admission probe lifetime | 1 second, no extension for partial bytes |
| Local admin | Separate 32-connection budget, 4 per loopback IP |
| Active connections from one source IP | 4 |
| Concurrent unproved TLS handshakes | 8 |
| Unproved admissions per fixed one-second window | 32 total, 8 per IP |
| Tracked source addresses | 64 |
| TLS handshake traffic, both directions combined | 128 KiB |
| TLS internal output buffering | 64 KiB per connection |
| Accepted TLS/bootstrap or HTTP request | 5 seconds from acceptance |
| Peer control record, including prefix and all partial transfers | 10 seconds |
| Local admin capability and complete record | 5 seconds |
| HTTP headers | 8 KiB, 32 headers |
| HTTP JSON body | 1 MiB, existing depth/node limits also apply |
| Buffered HTTP response | 64 KiB, 2 seconds to enqueue |
| Local admin and peer control JSON record | 16 KiB |

Fixed-window rate limits allow a boundary burst; they are not sliding-window
limits. All local admin connections share the IPv4 loopback accounting bucket.
A retained object still consumes its slot; the owning daemon must promptly serve
or drop it. Invitation expiry continues to bound approval validity. `ListenerUsage.pending_admissions` separately reports proof-stage sockets;
`connections` counts admitted sockets and `handshakes` counts unproved TLS. A kernel TCP
backlog is not an admitted connection and is not included in `ListenerUsage`.

These bounds cover application admission, parsing, transport buffering and I/O
waits. They are not a hard process RSS or CPU quota, kernel SYN-flood protection,
or an inference scheduler budget. Trusted callbacks must be short and bounded;
they must not run a backend synchronously or allocate unbounded output. No claim
of measured production throughput, resident memory or OS-level DoS immunity is made.

## Client HTTP

`ClientListener::bind_default` binds **127.0.0.1:9740**. `bind` accepts only literal
loopback addresses (including IPv6 loopback); wildcard and LAN binding fail.
Clients must send the exact numeric bound address and port in `Host`.
`ClientToken::bearer()` produces the 64-character lowercase hexadecimal secret
for `Authorization: Bearer ...`; never log it or put it in command arguments.

`IncomingClient::serve(authority, scope, handler)` performs these checks:

1. Parse HTTP/1.1 with `httparse` inside the fixed header limit. Only GET and POST
   are accepted. Reject duplicate headers, ambiguous lengths, chunked transfer,
   compressed bodies, upgrades, expectations and non-origin-form paths. Query
   strings, fragments, percent escapes and dot segments are currently unsupported.
2. Reject every `Origin`, including `null`, and every `Sec-Fetch-*` header. There
   is no browser origin allowlist and no CORS response allowance. Reject a Host
   mismatch to prevent browser DNS rebinding. OPTIONS/preflight is unsupported.
3. Verify current controller identity, token expiry/revocation and the exact
   model/action scope before allocating or parsing a body. A socket read can
   coalesce body bytes into the fixed 8 KiB header buffer; these bytes do not reach
   application parsing or dispatch before authorization.
4. GET has no body. POST requires one decimal Content-Length and exactly
   `Content-Type: application/json`. Apply the protocol's duplicate-key-aware,
   depth/node-limited JSON parser. Recheck authorization immediately before dispatch.
5. Send a bounded response with `Connection: close`, no-store and nosniff. Each
   nonblocking output enqueue rechecks scope under the policy lock. Never wait
   for socket writability while holding that lock. Already enqueued kernel bytes
   cannot be recalled by later revocation. Close on error without reflecting input.

The server chooses `scope` from trusted route/catalog state. The handler must
match method/path and resolve every referenced resource to that same scope.
`ClientRequest` is syntax-validated input, not a model-resolution decision. The
library deliberately provides no unrestricted backend or filesystem dispatch.
The handler follows `with_client_access` rules: no blocking I/O or Authority
reentry. Future asynchronous backend work and streaming must preserve the
GLY-165 secured worker/lease ownership and check authorization for every output.

This initial subset closes rejected requests without an HTTP error body. The
OpenAPI error responses, inspection scopes, route allowlists, model lookup,
streaming/cancellation and browser support remain integration work; it is not an
OpenAPI-complete server. One connection handles one request and closes; bytes
already received past the declared body are rejected, and later pipelined
requests are never dispatched.

## Deferred generation responses

`IncomingClient::prepare` validates framing, credentials and trusted route scope,
then returns a request and `ClientResponse` without sending headers. Its validation
callback must be short and bind the body model to the trusted scope. Backend work
runs after this callback, outside the shared policy lock. The unauthenticated
connection permit is released; the response holds one of 32 authority grants.
`ClientResponse::json` consumes one strict JSON result, bounded to 1 MiB;
`stream` selects SSE. Both use the same per-grant nonblocking output gate,
ten-minute lifetime and two-second backpressure deadline.

Create a `ClientWatch` before blocking on backend work. A bounded supervisor must
check it throughout execution, including before the first backend event, and
cancel the backend when it fails. The check uses nonblocking socket peek and the
grant mutex; it performs no policy/database operation and starts no thread.
Socket EOF (including a client write-half close), errors, later pipelined input,
revocation, stale maintenance or the hard deadline permanently close output.
This single-request protocol requires the client to keep its request side open
until the response completes. Drop the watch once the response completes; it
retains a grant and socket handle while alive.

Maintain authority at least once a second. A grant whose maintenance freshness
has expired cannot be revived by a late maintenance pass, even when the backend
has emitted nothing. Cancellation withdraws output but is not native device
drain; reservations still need trusted completion. These APIs are primitives for
GLY-177 integration; the daemon does not yet expose generation routes.

## Peer and bootstrap TLS

`PeerListener::bind` requires an explicit address. `accept().authenticate(...)`
uses the existing TLS 1.3, CA, exact membership and ALPN verification; its permit
lives with the `PeerStream`. `accept().enroll(&authority)` can only receive and
validate an invitation-bound CSR. Its permit lives with `PendingEnrollment`;
`Authority::approve` still requires local fingerprint confirmation.

Use `PeerStream::receive_record` and `send_record` for control messages. Each
whole frame has one deadline, rather than renewing a timeout after each byte.
A failed frame shuts down the socket. Certificate renewal uses these methods.
Raw `PeerStream::accept`, `Read`/`Write` and `Authority::receive_enrollment` remain
low-level compatibility primitives; they do not enforce daemon-wide admission.
A daemon must use the listener wrappers and framed APIs. Tensor framing is M3.

## Local administration

`AdminListener::bind` requires an existing private directory. Unix uses
`admin.sock` in that 0700 directory, socket mode 0600 and a peer effective UID
check against the owner. The protected parent prevents exposure before chmod
on platforms that cannot set the socket mode before bind.

Windows uses a local named pipe with an explicit protected Owner Rights DACL
(`D:P(A;;GA;;;OW)`), local-only clients and exclusive first-instance creation.
It does not use Windows' broad default pipe DACL. The pipe name is derived from
the canonical private directory. A default object owner can be an administrator
group; therefore the ACL is supplemented on every platform with an independent
256-bit capability stored in a protected `admin.cap` file.

The client verifies a server HMAC over a fresh challenge before disclosing the
capability. The capability and strict bounded command are then checked by
`AdminRequest::receive` or `respond`; raw unauthenticated reads stay private.
`admin_call` validates the bounded reply and re-verifies its protected file before
acknowledging receipt within the same deadline. A failed receipt cannot invalidate
a known reply; a missing/invalid reply after possible dispatch is explicitly
unknown. Server I/O and flush-on-drop remain bounded. See [wire details](daemon.md).
The protected file is verified around use. No TCP admin or shell execution exists.

The bind primitive refuses existing endpoints. The GLY-167 daemon owns an exclusive
persistent state lock before validating and removing stale transient files, and
cleans them only after listener/worker shutdown. Unexpected file types or unsafe
permissions are refused. Same-user and
administrator/root compromise are outside this boundary. Cross-user OS denial
is enforced by native permissions; tests currently run as the CI runner user.

## Source references

- [httparse 1.10.1 Request parser](https://docs.rs/httparse/1.10.1/httparse/struct.Request.html)
- [interprocess 2.4.5 listener options](https://docs.rs/interprocess/2.4.5/interprocess/local_socket/struct.ListenerOptions.html)
- [Windows named-pipe access rights and default DACL](https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipe-security-and-access-rights)
- [Windows Owner Rights identifier](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/manage/understand-security-identifiers)

Dependency sources were inspected for local-only pipe creation, first-instance
exclusivity and socket mode behavior. Dependency licenses are recorded in
[dependencies.md](dependencies.md); the project license remains undecided.

GLY-168 multiplexes registry/enrollment on the existing coordinator port. Optional
TLS client certificates serve bootstrap only; registry dispatch requires an exact
enrolled worker/controller certificate before reading a record. Control/data
listeners retain required client authentication. Draining keeps only enrollment
refusal and authenticated registry replies on that port. See [registry](registry.md).

## Registry reconnect admission

Before TLS, a registry client sends the byte `R`. The server returns a fresh
32-byte random challenge; the client returns its 36-byte canonical node UUID and
a 64-byte P-256/SHA-256 fixed-width signature using its existing device key. The
signed bytes are UTF-8 `starlings-reconnect/1:{cluster}:{coordinator}:{node}:`
followed by the challenge. The domain, identities and per-socket challenge prevent
cross-protocol, cross-cluster, cross-node and old-connection proof reuse. No new
credential is persisted. The node UUID and proof are visible on the LAN; neither
is a reusable bearer credential. This unreleased wire change requires all nodes
to upgrade together. There is no registry fallback without a proof.

The nonblocking probe queue services existing progress before accepting up to eight
new sockets per poll. At 64 probes, a new socket replaces the oldest incomplete
probe. Idle connections cannot pin the queue indefinitely. A valid proof reserves
only that node's single slot, without consuming the anonymous per-IP/rate/handshake
budgets. A second connection for the same node is refused until the first releases
its slot. The deadline and signature check do not hold a lock across network I/O.

The proof grants capacity only. TLS 1.3, CA/name/key validation, exact enrolled peer
certificate selection and role checks still apply. The TLS node must equal the
proved node and the ALPN must be registry; proved connections cannot enroll.
An unexpired revoked current identity can reserve only its own slot to receive the
existing authenticated revocation tombstone, never permission to run work.
The ordinary TLS first byte (`22`) selects bootstrap admission and cannot reserve
a registry slot. Invalid, expired, unknown-key and replayed proofs are dropped.

This protects reconnects from held anonymous slots, including other traffic from
the same IP. It does not guarantee availability against link saturation, kernel
backlog exhaustion, CPU exhaustion, an on-path attacker or a compromised enrolled
key. Proof floods still consume bounded parsing/signature work; there is no claim
of general denial-of-service immunity. Independent review of this admission
protocol remains a merge gate.

## Bounded client output buffering

Accepted loopback HTTP client sockets request a 64 KiB send buffer before
admission. This bounds queued output independently of host TCP autotuning and
propagates slow-reader pressure sooner. Kernel accounting may include additional
overhead; this is not a measured RSS or hard system-memory limit. Existing
nonblocking writes, two-second idle cutoff, absolute response deadlines and live
authorization gates remain unchanged. Peer TLS sockets are unaffected.
The integrated [M2 backpressure check](evidence/GLY-177-backpressure.md) exercises
cancellation with a connected but unread client, an Online registry, one
invocation and retained uncertain ownership.
