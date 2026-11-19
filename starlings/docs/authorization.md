# GLY-165 live authorization, renewal and revocation

This is a local security/runtime library contract. It is exercised with real
loopback TLS and the synthetic worker state machine. It is not a deployed service,
remote membership replication, an artifact server, or model inference. GLY-166
through GLY-169 must integrate and review the remaining M1 service boundaries.

## Membership and operation policy

`Authority::trust()` now returns a live handle to the authority's protected
SQLite state. Existing handles observe additions, rotation and revocation; callers
cannot construct an arbitrary network-derived `Trust`. Connections require the
normal CA/name/ALPN checks and an exact currently accepted enrolled certificate.
Both local and remote credentials are checked before each read/write/flush.
Reads recheck after blocking I/O and clear their destination on denial.

One runtime owns the authority directory using the standard library's exclusive
file lock. A second authority handle can load enrollment state but cannot publish
live trust, dispatch work, or mutate authorization while the first runtime owns
that lock. The lock remains held while any trust, stream or secured worker handle
exists. This is local process exclusion, not a distributed coordinator election.
Use one authority handle for live policy administration. Enrollment-only approval
transactions retain their concurrent single-use guarantee.

All model permissions default to denied, including for coordinators. Permissions
refer to immutable manifest SHA-256 digests, never model display names. A scope is
`{"model":"<64 lowercase hex>","action":"artifact|infer|control"}`.

| Identity | Allowed authority |
| --- | --- |
| Coordinator | Explicit per-model control grants; may also receive artifact/infer grants |
| Worker | Explicit per-model artifact/infer grants; never control |
| Controller | Explicit per-model artifact/infer grants; can own scoped client tokens; never control |
| Unknown, revoked or expired member | None |

Roles are enrollment-time certificate identities. Changing a role requires a new
node enrollment. `set_permissions(node, scopes)` replaces the complete allowlist;
an empty list denies all model operations. At most 32 scopes fit one allowlist.
Mutations are privileged local APIs, not unauthenticated administrator endpoints.

## Client credentials

`issue_client_token(controller, scopes, lifetime_seconds)` returns a secret token
with 256 bits of OS randomness. The controller must currently hold every requested
scope; tokens cannot grant control or escalate permissions. Lifetime is 1–3600
seconds. Validation intersects token scope with current controller membership and
permissions on every operation. Removing a grant invalidates existing tokens for
that grant. Revoking the controller invalidates all its tokens.

`ClientToken::save/load` uses protected versioned files. Raw token material exists
only in the returned object and protected file, never in SQLite; the database
stores SHA-256, controller node, scopes and expiry. No Debug implementation or
logging is provided. A token file is a bearer credential and must not be put in
arguments, URLs or diagnostics. HTTP credential extraction/redaction is GLY-166.
There are at most 256 unexpired token records; expired rows are removed in the
successful activation transaction.
`revoke_client_token` deletes the hash and cancels owned live plans.

File-producing commands use
`issue_client_token_saved(controller, scopes, lifetime_seconds, directory, name)`.
It checks admission, publishes the protected file without holding policy/SQLite
locks, then rechecks current permissions, drain, capacity and expiry before
committing the token hash and clock floor together. The lifetime starts before
publication and is not extended by slow storage. Failed file publication never
activates a credential or needs compensating revocation.

A crash, policy change or refused commit after publication can leave an inactive
file. File presence alone is not proof of activation. An uncertain commit storage
failure closes the local authorization runtime; the existing control command
outcome rules still apply. Existing output files are never overwritten. The
in-memory issuance API remains for callers already owning the returned bearer.

`with_client_access(token, scope, dispatch)` and
`PeerStream::with_access(scope, dispatch)` check current policy while serializing a
short synchronous dispatch with revocation. The latter also verifies both stream
credentials. Callbacks must not reenter the runtime, perform long work, or treat a
returned value as a reusable authorization permit. Each application operation and
streamed chunk needs its own check. Actual artifact routes remain a later task.

## Worker admission, cancellation and leases

`secure_worker(Worker)` accepts a fresh worker for an enrolled worker node and
owns it behind `SecuredWorker`. There is no raw mutable worker escape hatch.
The pure `starlings-core::Worker` remains available for trusted simulation.

`SecuredWorker::apply(peer, token, request, monotonic_ms)` requires a control-channel
peer authenticated as the coordinator, a local identity matching the worker, the
coordinator's model control grant, a client's current infer scope, and every
participating stage's current worker membership and infer grant. The model is
resolved from the actual prepare plan or the worker's stored plan. The normal
core checks still enforce request digest, cluster, epoch, lease, order and budget.
Authorization runs before the replay cache. A successful exact replay returns
`true` and must never execute the backend again.

One plan is bound to one token hash and coordinator node. Other tokens cannot
adopt it, even with the same model permission. Sharing a resident plan among
multiple independent clients needs an explicit later design. No model is loaded
by this library.

Revoking a node, a client token or a needed grant immediately reconciles registered
workers and releases affected plans. Any pending invocation enters Cancelling;
memory remains reserved until the **local backend** calls `drained`. Network
`CompleteStep` is forbidden through this wrapper. `with_output` checks the plan,
client scope, lease deadline and generation state for every output publication;
cancelled/failed work cannot publish. Callbacks must be synchronous and bounded;
queued output and real backend interruption must preserve these checks when those
components are implemented. Bytes already delivered cannot be retracted.

`maintain(monotonic_ms)` must run at least once per second even with no traffic.
It closes expired/overlap-ended streams, cancels plans whose credentials or tokens
are no longer usable, and advances every registered worker's original monotonic
lease. Disconnect and replay never extend a lease. Loss of the local authority
runtime is handled by worker lease expiry once remote membership integration
exists; this library does not claim cross-machine revocation propagation yet.
Wall-clock rollback or protected-storage verification failure closes tracked
streams and cancels live plans. Clock correction/recovery needs local inspection.

## Renewal and overlap

`Identity::renew(stream)` sends `{version:1, csr:DER byte array}` through an existing
mutually authenticated control stream and returns a replacement identity. The
coordinator calls `renew_peer(stream)`. The CSR proves possession of a fresh key;
old key reuse, stale credentials, revoked membership and concurrent overlapping
rotations fail closed. Caller-supplied subject/CA/usage extensions are discarded.
The response is the standard bounded `Member` record; the device verifies its
chain, exact identity, validity and new key before accepting it.

There is one current certificate and at most one previous certificate per node.
The previous certificate is accepted for **at most 300 seconds**, never past its
own expiry. Rotation does not renew tokens, plans, leases or permissions. A second
rotation is refused while overlap is active, so it cannot extend old credentials
indefinitely. TLS resumption and 0-RTT remain disabled. `maintain` closes idle old
streams; operation checks reject them immediately when their overlap expires.

Save the returned device identity under a new protected filename before switching
consumers. A lost renewal response leaves the old credential usable only until its
fixed overlap ends. There is no replay/recovery endpoint; after losing the new key,
re-enroll with a new node identity and revoke the orphan locally.

`renew_coordinator()` writes a new protected key file before atomically committing
the current certificate, old overlap record and active identity filename in SQLite.
Restart follows this filename. A crash before commit can leave an unused private
file; a committed rotation has a durable key file. Old private files are retained
for explicit local inspection, not automatically deleted. Outstanding invitations
pin the earlier coordinator leaf and should be replaced after coordinator rotation.
Root-CA replacement is not implemented; never bypass TLS verification to recover.

## Durable state and bounds

The original enrollment database gains `revoked(node)`, `grants(node,scopes)`,
`tokens(hash,node,scopes,expires)`, `previous(node,record,expires)` and
`settings(name,value)`. Existing GLY-164 state is upgraded on open with additive
idempotent table creation. Revocation is a permanent tombstone for an enrolled
node ID and survives restart. Revoked nodes still count toward the 64-member cap.
No automatic membership compaction or stale-backup recovery is supported.

There are at most 128 live TLS stream handles and 64 registered worker handles per
runtime. These limits bound local tracking; listener-wide handshake/IP/frame/queue
budgets remain GLY-166. One policy mutex intentionally serializes bounded local
dispatch and administrative changes. No throughput or revocation-latency SLA is
claimed; measure contention before changing the design.

Dependencies are unchanged apart from the internal `starlings-security` →
`starlings-core` edge. File exclusion uses Rust `File::try_lock` and safely duplicated
native handles; no unsafe code or new third-party dependency was added. The
[enrollment storage boundary](enrollment.md) still applies, including trusted local
filesystems and exclusion of same-user/root compromise and stale backup restoration.

## Registry integration (GLY-168)

`Authority::start_registry` enables current-epoch and fresh-stage checks in the
existing plan authorization gate. `SecuredWorker::bind_registry` adds the matching
remote profile's fence to every dispatch and output callback. Drive
`RegistryFence::maintain` independently of network I/O; its returned reserved bytes
must reach zero after drain before safe shutdown. Epoch updates and dispatch/output
are serialized, while trusted local completion remains independent of connectivity.
See [registry](registry.md). The daemon has no inference worker yet; M2 must bind
real backend workers and install authenticated model/token policy before exposing
remote operations. Heartbeats do not replicate those policy tables.
