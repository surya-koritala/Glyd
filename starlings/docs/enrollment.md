# GLY-164 enrollment and secure stream contract

This library foundation is implemented by `starlings-security`. It does not start
listeners, expose administrator routes or change the existing CLI. Its tests run
actual loopback TLS with temporary keys and SQLite state. This is not evidence
of a deployed LAN cluster, Raspberry Pi inference or a completed M1.

## Local API and flow

1. `Authority::create(path)` exclusively initializes a private directory, cluster
   root, coordinator identity and enrollment database. `Authority::open(path)`
   loads an existing complete authority. Partial initialization fails closed.
2. `Authority::invite(private_dir, filename, socket_address, role)` writes a
   protected invitation for a worker or controller. Coordinator enrollment is
   refused; there is one configured coordinator. Transfer the file privately.
3. The device loads `Invitation`, generates `Identity` for its cluster and saves
   the pending identity before connecting. `request_fingerprint()` is a SHA-256
   hex digest of canonical node UUID text followed by DER SubjectPublicKeyInfo.
   The administrator obtains this fingerprint directly from the device through
   an authenticated channel, not solely from the incoming network request.
4. `Invitation::enroll(&mut identity)` connects to the explicit address, verifies
   the pinned CA, coordinator DNS identity and exact coordinator leaf before
   transmitting the invitation token and CSR. The bootstrap ALPN is separate
   from both authenticated peer protocols.
5. A trusted local caller accepts TCP and calls `receive_enrollment(socket)`.
   It validates the invitation and CSR and returns `PendingEnrollment` with the
   requested node, assigned role and fingerprint. The caller compares the device
   fingerprint, then calls `approve(pending, approved_fingerprint)`. This is a
   privileged API, not an unauthenticated remote approval endpoint. The current
   bounded exchange expects the administrator's fingerprint to be ready before
   joining; a future interactive queue belongs with the local CLI/admin service.
6. Approval rechecks expiry and consumes the invitation in the same transaction
   that inserts the member. The device validates the returned certificate chain,
   identity name and its own public key before installing the certificate. Save
   the resulting identity under a new protected filename.
7. `Authority::trust()` supplies a live local membership handle. `PeerStream::connect`
   and `accept` require a local identity, trust handle and `Channel::Control` or
   `Channel::Data`. Connect also requires the expected peer node. Only a verified
   stream exposes `Read`/`Write`; `peer()` and `role()` expose authenticated identity.
   Secure membership distribution to workers is part of GLY-168, not provided by
   deserializing arbitrary network input into a trust store.

No discovery result supplies trust. No tokens appear in a URL, command argument,
error or normal log. Secret-bearing types have no `Debug` implementation. The
library emits no logs. It does not promise erasure of all allocator copies or
protection against process compromise.

## Wire and certificate records

Bootstrap uses four-byte big-endian JSON length followed by that many UTF-8 bytes.
Each frame is at most 16 KiB. The existing duplicate-aware parser rejects duplicate
keys; typed deserialization rejects unknown fields and unsupported versions.
DER vectors use JSON byte arrays in this initial bounded protocol.

| Record | Fields / invariants |
| --- | --- |
| Invitation file | `version: 1`, `endpoint: SocketAddr` with nonzero port, `coordinator: Member`, `ca: DER bytes`, `token: [u8;32]`, `expires: i64` Unix seconds |
| Request | `version: 1`, `node: canonical UUID v4`, `token: [u8;32]`, `csr: DER bytes` at most 4 KiB |
| Member / approval response | `version: 1`, `cluster: UUID v4`, `node: UUID v4`, `role: coordinator/worker/controller`, `certificate: DER bytes` at most 4 KiB on receipt |
| Local identity file | `version: 1`, `member`, `key: PKCS#8 DER bytes`; file contents are secret |
| Authority file | `version: 1`, `ca: DER bytes`, `key: PKCS#8 DER bytes`; file contents are secret |

CSR signatures are verified; trailing DER is rejected. Requested subject,
identities, validity, usages and CA extensions are discarded. The issuer sets a
non-CA leaf with digital-signature usage, server/client authentication and exactly
one DNS SAN: `<node>.<role>.<cluster>.glyd.invalid`. The generated root is constrained
to directly signed leaves. Defaults are ten years for the root, seven days for
leaves, and one minute of not-before tolerance. Certificate validity still relies
on the system clock. Leaf renewal and bounded overlap are implemented by
[GLY-165 authorization](authorization.md); root replacement remains unimplemented.

All three protocols require TLS 1.3; only the cluster CA is a trust root.
Enrollment ALPN is `starlings-enroll/1`, control is `starlings-control/1`, data
is `starlings/1`. Both peer directions require certificates. A CA-valid leaf
absent from current membership is rejected before application bytes are exposed.
Control and data ALPNs cannot be interchanged. TLS resumption and 0-RTT are disabled.

Bootstrap has a ten-second monotonic budget covering handshake and record I/O;
approval response writing gets a new bounded budget. Peer handshakes have the same
budget; subsequent stream calls get individual budgets. Aggregate message, slow
sender, listener admission, admin socket and connection-count policies remain
GLY-166/167. Do not expose a general network service based on these primitives alone.

## Durable state and failure behavior

SQLite stores `invitations(hash PRIMARY KEY, expires, role)`,
`members(node PRIMARY KEY, record)` and a single maximum-observed wall-clock row.
A 256-bit OS-random token is stored only as SHA-256 in SQLite. Invitations expire
at `now >= expires`, normally five minutes. Observed backward wall time fails
closed; this detects rollback relative to recorded activity, not every possible
clock change. SQLite uses immediate write transactions with a zero busy timeout,
delete journals, EXTRA synchronization and memory temporary storage.

Invitation issuance publishes the protected file outside the policy mutex and
SQLite transaction. Before committing its hash, it rechecks drain, expiry and
capacity. The clock floor and token hash commit in the same transaction; issuance
does not force separate durable clock checkpoints. A clock-write failure rolls
back the token row, and hard storage failures close active grants. Publication failure or refusal after publication leaves no valid token
row; the untracked file is retained for inspection. The daemon runs one owned
invitation publication at a time, outside its control loop; another issuance is
refused with `INVITATION_BUSY`. A caller timeout does not release that slot or
cancel a started mutation. Shutdown drains admission and retains the worker until
filesystem I/O returns; a blocked OS syscall has no claimed hard deadline.

Member insertion and token removal commit together before the response is sent.
Concurrent approvals yield at most one member. A crash before commit rolls back;
a crash/lost response after commit leaves the token consumed. There is no replay
endpoint or KV/session recovery. After an ambiguous response, use a new invitation
and node ID; orphan revocation uses the GLY-165 local authority API. Restoring stale backups is not a
supported recovery procedure: it could restore old invitations. CA loss requires
an explicit recovery/re-enrollment design, never bypassing verification.

Limits are 64 enrolled nodes (including the coordinator) and 128 unexpired
invitations. Expired invitations are removed when a new one is created. Full tables
reject work; membership entries are not evicted to make room.

## Storage and remaining review gates

`private-fs` creates owner-only keys and invitations and rejects insecure existing
objects, symlinks and hard links. New records reserve their name exclusively and
use durable atomic publication. An interrupted write may leave an empty file or
private staging orphan; it is never silently replaced on retry. Choose a new name
or inspect failed initialization locally. Linux/macOS require private modes;
Windows requires a protected user-only DACL. Unsupported permission or durability
semantics fail closed. Use a trusted local filesystem. Access control is not
at-rest encryption. Keep private state outside the repository.

SQLite requires a pathname, so the verified private directory is its namespace
boundary; same-user/root replacement attacks are outside it. Its journals contain
hashes and public membership records, not raw tokens or private keys. This does
not replace protection of the full state directory.

[GLY-165](authorization.md) adds current operation membership/role/model checks,
credential rotation, revocation, connection closure and worker cancellation. GLY-166 hardens all listeners;
GLY-167 integrates daemon/configuration/admin CLI; GLY-168 adds cluster membership,
heartbeats and durable epochs; GLY-169 adds explicit-address reachability and
optional discovery hints. M2 stays gated on review of all M1 work.

## Primary references

- [Rustls](https://docs.rs/rustls/0.23.45/rustls/): TLS configuration, standard certificate verification and stream I/O.
- [rcgen CSR verification](https://docs.rs/rcgen/0.14.10/rcgen/struct.CertificateSigningRequestParams.html): parsed CSR signature validation; issuer policy still belongs to the caller.
- [private-fs security contract](https://github.com/orielhaim/private-fs/blob/main/SECURITY.md): native access controls, atomic publication, durability and excluded adversaries. Version 0.1.0 source was inspected; no independent audit is claimed.
- [SQLite atomic commit](https://sqlite.org/atomiccommit.html) and [synchronous settings](https://sqlite.org/pragma.html#pragma_synchronous): transaction and durability assumptions depend on the filesystem/storage honoring flushes.
