# Local daemon and CLI

GLY-167 makes enrollment and local administration usable in one `starlings`
binary. Coordinator, worker and client roles start the same foreground `daemon`
command. Client is the CLI name for the protocol's `controller` role. The daemon
also exposes an explicitly approved loopback generation route to enrolled workers
and their external CPU services; see [production routing](production-routing.md).
A coordinator can also route one model across two worker stages; see
[daemon pipeline](daemon-pipeline.md).
Init/join alone does not enable inference or qualify a backend. Authenticated
heartbeats, epoch fencing and `cluster nodes` are described in [registry](registry.md);
explicit routes and optional untrusted mDNS hints are in [discovery](discovery.md).

`cluster diagnostics` prints an opt-in [redacted JSON bundle](diagnostics.md) from
three existing local queries. It starts no service or probe and includes no raw
logs, credentials, addresses, model names or node identifiers.

If output fails after a successful admin reply or saved init/prepare/join/routes
operation, the command returns `OUTPUT_UNAVAILABLE` and says it completed but
output failed. The operation is not rolled back. Restore output and inspect the
relevant state before retrying a mutation; a lost acknowledgement is not a reason
to repeat it. This differs from an admin outcome reported as unknown.

## Permission observations

`cluster permissions NODE PRIVATE_SCOPES` replaces that node's allowlist. After
an unknown result, use `cluster permissions-status NODE` on the coordinator to
read the current committed model/action list. The query checks membership and
protected policy state, changes no grants/tokens and returns no credentials or
paths. It is local privileged administration, not a public HTTP endpoint.

The list is a current observation, not an operation receipt or future authorization
guarantee. A negative result/error does not prove a pending mutation cannot finish
and does not authorize retrying it. Three-second control replies and five-second
IPC bounds remain unchanged; started mutations can outlive their reply wait.

## State and configuration

Install/build the binary using the pinned toolchain in the root README. On Linux,
the default state root is `$XDG_STATE_HOME/starlings` (normally
`~/.local/state/starlings`); macOS uses `~/Library/Application Support/starlings`;
Windows uses the user's local application-data directory plus `starlings`.
Cache uses the OS cache directory plus `starlings`; where the OS data/cache
roots coincide (Windows), a separate `cache` child keeps identities out of it. A leading
`--state-dir ABSOLUTE_PATH` selects an isolated profile and `STATE/cache`.
Do not share a profile between machines or concurrent daemons.

The binary creates private final state/cache directories. Existing permissive
or unsafe directories/files fail closed rather than being silently fixed.
`config.json` is generated during init/join and contains only version, role,
cluster/node IDs, coordinator enrollment address and absolute cache path.
Parsing rejects duplicate/unknown fields, wrong versions, invalid role/address
combinations and records over 16 KiB. Startup checks configuration against the
stored identity. No hand editing is needed for the flow below.

A persistent `daemon.lock` excludes competing init/prepare/join/daemon/offline
leave operations. It must never be deleted while a process owns the profile.
After acquiring it, restart removes only validated transient `admin.cap` and
Unix `admin.sock`. Wrong file types, owners, permissions or symlinks are refused.
Identity keys, enrollment database, config and model/cache data are retained.
The supported storage boundary is a trusted local filesystem; network shares,
root/same-user compromise and power-cut recovery are not certified.

## Two-machine enrollment

The coordinator's default enrollment bind is **127.0.0.1:9741**. To enroll a
second computer, choose its concrete reachable LAN address during init; wildcard
and multicast binds are refused. This example uses a private address:
replace `192.168.10.10` with the coordinator's real LAN IP and all absolute paths with
local paths. Do not use the default loopback address for another machine.

On the coordinator, use an unused profile and start the foreground process:

```sh
starlings --state-dir /absolute/coordinator cluster init --listen 192.168.10.10:9741
starlings --state-dir /absolute/coordinator daemon
```

In another coordinator terminal, create a private, single-use invitation:

```sh
starlings --state-dir /absolute/coordinator cluster invite worker /absolute/coordinator/worker-invite.json
```

The output parent must already be private (the initialized state directory is
suitable). Securely transfer the invitation to a private directory on the joining
machine, preserving owner-only access: directory 0700/file 0600 on Unix, protected
current-user-only DACL on Windows. It contains a credential; do not paste its
contents into chat, command arguments or logs. The five-minute expiry starts at
issuance. Transfer is deliberately out-of-band; the CLI does not configure SSH,
firewalls or network routes. Permit the selected TCP port on the trusted LAN.

If issuance reports an unknown outcome, do not repeat the mutation. On the
coordinator, inspect the same protected file with
`cluster invite-status /absolute/coordinator/worker-invite.json`.
`INVITATION_STATUS: active=true` means that the same authority/key/CA has a
committed, unexpired token row matching the file's expiration and a permitted
invitee role, and admission is not draining. File existence alone is insufficient:
publication precedes commit. The query consumes nothing and returns no credential,
certificate or path. It is local coordinator administration, not a public route.

`active=false` covers inactive, consumed, expired or untracked credentials;
malformed/missing/unsafe files and authority mismatches are errors. A query error
or negative observation does not prove an earlier operation cannot still finish,
and does not authorize a retry. Repeating a read-only query is safe within the
existing admission limits; no automatic issuance retry or longer deadline is added.
The result is a current observation, not a guarantee of future enrollment or
network reachability. Generic `cluster status` alone cannot confirm issuance.

On the joining machine:

```sh
starlings --state-dir /absolute/worker cluster prepare /absolute/private-inbox/worker-invite.json
```

This generates a private pending identity and prints its public node ID and
SHA-256 key fingerprint. Compare those values directly with the joining operator.
On the coordinator, approve the exact displayed values, then join on the device:

```sh
# Coordinator terminal
starlings --state-dir /absolute/coordinator cluster approve NODE_UUID FINGERPRINT
# Joining machine
starlings --state-dir /absolute/worker cluster join /absolute/private-inbox/worker-invite.json
starlings --state-dir /absolute/worker daemon
```

Use `cluster invite client ...` for a client profile. Approval is held in memory
for at most five minutes, limited to eight entries, consumed by one matching
request and cleared on drain/restart. A mismatched fingerprint cannot consume the
invitation. TLS authenticates the pinned coordinator before any invitation token
is sent. The invitation transaction atomically issues membership and consumes its
hash. A certificate reply lost after commit requires a fresh invitation/profile;
do not repeatedly retry an ambiguous enrollment. If the certificate was already
saved locally but configuration publication failed, rerunning join with the same
invitation and pending identity completes local publication after verification.

The protected `bootstrap.json` retains the pinned CA and consumed invitation for
startup verification; it is still treated as secret. Pending and enrolled identity
files are private. Worker certificates must still be valid at startup. Online registry exchanges
refresh certificates when fewer than three days remain. An expired coordinator
can recover locally with its protected CA/key and unchanged enrolled public key;
an expired offline worker requires re-enrollment. Registry heartbeats authenticate current membership and
report liveness; they do not replicate model/token allowlists or certify reachability
between arbitrary workers.

## Operation and shutdown

```sh
starlings --state-dir /absolute/coordinator cluster status
starlings --state-dir /absolute/coordinator cluster nodes
starlings --state-dir /absolute/coordinator cluster resources NODE
starlings --state-dir /absolute/coordinator cluster drain
starlings --state-dir /absolute/coordinator cluster stop
# On a joined device, leave disables this local profile and stops its daemon:
starlings --state-dir /absolute/worker cluster leave
# On the coordinator, revoke that device's credential:
starlings --state-dir /absolute/coordinator cluster revoke NODE_UUID
```

Status reports role, IDs, bound enrollment address, admitted connections and pending
approvals, registry state and authority epoch with stable reason codes. `cluster nodes`
reports current remote membership, freshness and schedulability. `ENROLLED_IDLE` means local identity loaded,
not a healthy distributed cluster. Stopped daemons return `DAEMON_UNAVAILABLE`.
`DRAINING` records local request intent. Inspect `admission=closed` to confirm
actual policy withdrawal: coordinator status reads its existing policy flag with
the epoch; worker status uses its registry fence. `admission=open` describes only
that gate, with all other authorization checks still required; `unknown` means
no current fence can establish it. A policy read error remains an error.
After an unknown drain result,
observe status without resending the mutation. Admission closure does not attest
native cancellation, empty reservations or complete device/process cleanup.
Drain refuses enrollment, clears approvals and rejects new secured work/model
callbacks. It cancels owned plans while retaining in-flight reservations until
trusted local backend completion. Admission closure is separate from native cleanup;
`reserved_bytes=0` is not a model-performance result. A drained daemon continues
local status/admin service and authenticated heartbeat replies until stopped;
restart reopens admission under a durably newer epoch.

Stop and Unix SIGINT/SIGTERM/SIGHUP close listeners, drain, finish bounded I/O,
join workers, then remove transient endpoints. Windows uses the `ctrlc` crate's
console-control handling; admin stop is the tested portable shutdown route.
Admin/handshake I/O has a five-second deadline; approved certificate delivery has
a ten-second deadline. Forced process termination can interrupt cleanup; the next
exclusive startup validates and removes stale endpoints. The process lock persists.

Leave retains identity/cache and writes a durable local disable marker. It also
works with the daemon stopped. It does **not** revoke the credential remotely;
run coordinator revoke. The next authenticated heartbeat delivers revocation;
disconnected registry authority expires within six seconds. Re-enrollment uses a fresh profile.
A coordinator cannot leave/revoke itself; stop it instead. No implicit cluster
reset, recursive state deletion, model deletion or recovery of old KV state exists.

## Runtime and local IPC contract

One 20 ms control loop owns config, authority and approval decisions. Each remote
profile adds one outbound heartbeat thread; all inbound work remains admission-bounded. A shared
listener budget bounds all accepted admin/bootstrap connections; network reads
and certificate reply writes run on admitted threads. Threads never receive the
CA signing key. The event queue holds at most 32 items. Maintenance runs at least
every 250 ms under ordinary operation, independently of network traffic; filesystem
operations and OS scheduling are not hard real-time guarantees.

Local admin uses the native protected Unix socket or Windows pipe. The client
sends a fresh 32-byte challenge; the server proves capability possession with
HMAC-SHA256 before the client sends its private capability. Then each side sends
one four-byte big-endian length plus strict JSON (maximum 16 KiB). The client sends
one receipt byte `1`; this bounds Windows pipe flushing on close. Entire server
request/reply/receipt I/O shares five seconds. Connect wait is separately at most
one second; daemon response wait is three seconds. Expiry before dispatch prevents
that queued command from executing later and reports `CONTROL_EXPIRED: not executed`.

Explicit topology observation runs outside the control loop using a snapshot of
listener addresses, the mDNS choice and the live registry fence. Each daemon permits
one pending topology job; another request returns `TOPOLOGY_BUSY` without spawning
another observer. Invitation publication has its own independent pending slot.
Client-token publication also runs in a single owned background job; a second
pending request returns `TOKEN_BUSY`. Its local privileged issuer shares live
policy access without copying CA/device signing keys. Durable private-file
publication still precedes activation, and policy, expiry, capacity and drain are
checked again before committing the token row. The main control loop no longer
waits for the token's private-file publication. This does not eliminate shared
policy/database contention or prove the cause of every Windows deadline failure.
The existing three-second response deadline and unknown-outcome semantics remain;
there is no mutation retry. A timed-out job keeps its permit until it returns,
and shutdown retains the owned job while draining prevents late activation.
Interface enumeration holds no policy lock. Existing reply deadlines and owned
worker cleanup remain: a timed-out diagnostic cannot renew authority, and shutdown
still waits for a host query already running to return.

Expiry after dispatch or a lost reply means unknown
command outcome: inspect status rather than retrying a mutation automatically.
For a token mutation, use `cluster token-status PRIVATE_TOKEN` on the coordinator
with the same protected output file. The authenticated reply is only
`TOKEN_STATUS: active=true` or `active=false`; no credential or scope is returned.
True means a committed token currently has every stored scope permitted under
live membership, certificate, expiry and model policy, with admission not draining.
A token that retains only some scopes reports false. This is a read-only snapshot,
never a grant, renewal, durable acknowledgement for future use or guarantee that
later admission succeeds. With a valid private file, uncommitted/revoked/expired/
draining tokens report false.
The snapshot describes the supplied token, not which particular mutation created
it or whether that mutation's requested controller, scopes or lifetime won a
concurrent publication. An already-active file is not a receipt for a new request.
Missing/unsafe/malformed private files and policy/database errors are refused.
File presence alone proves nothing; an unknown mutation response must not be
resent automatically. A later explicit observation can use the same command, but
there is no automatic poll or retry loop. Older daemons refuse the new command;
that refusal cannot be interpreted as activation.
Client failures after possibly sending the command return the same explicit
`CONTROL_UNAVAILABLE: outcome unknown; check status` refusal; they cannot use the
offline-leave fallback. Serialization/size refusal before sending and failed server
proof remain pre-dispatch failures. Once the bounded reply is decoded and the
protected capability file verified, receipt-byte failure does not invalidate that
known reply. The receipt is best effort within the same deadline; missing/invalid
replies or failed protected-file verification never become presumed success.
On Windows, the native connector enforces the timeout explicitly. The pinned IPC
library reports both an empty nonblocking pipe and EOF as a zero-byte read; either
waits only until the unchanged request deadline. Zero-progress writes are likewise
bounded. This can delay disconnect detection, but cannot renew the deadline.

Commands use `{"command":"status"}` (also `drain`, `stop`, `leave`),
`{"command":"invite","role":"worker","output":"/absolute/private/file"}`,
`{"command":"invite_status","file":"/absolute/private/file"}`,
`{"command":"approve","node":"UUID","fingerprint":"64 lowercase hex"}` and
`{"command":"revoke","node":"UUID"}`. Client invitations use wire role
`controller`. Replies contain exactly `ok`, `code`, `message`; errors omit secret
values. This is an internal local protocol, not an HTTP API. `StateLock`,
`AdminRequest::respond`, `admin_call`, `PeerListener::try_accept`,
`Authority::enrollment_reader`, `approve_for_delivery` and `drain` are the owning
security APIs. See [listener bounds](listeners.md).

## Evidence and limits

Separate-process loopback tests enroll both roles, reject unapproved enrollment,
exercise a stalled TCP sender alongside local status, drain/stop/leave/revoke,
restart after normal and forced termination, and retain identity/cache bytes.
Unix tests send actual SIGTERM. Library tests cover incorrect server proof and
in-flight drain reservations. Four-platform CI is required on the exact PR head.
This is not a physical two-machine LAN, Windows console-event, cross-user attack,
Raspberry Pi hardware or inference benchmark. GLY-168 registry behavior is covered
in [its contract](registry.md); [GLY-169](discovery.md) adds explicit routes and topology.
The [M1 review](evidence/M1.md) records remaining acceptance gates before M2.

Sources: [dirs 6.0.0 platform paths](https://docs.rs/dirs/6.0.0/dirs/),
[ctrlc 3.5.2 signal support](https://docs.rs/ctrlc/3.5.2/ctrlc/),
[interprocess connect options](https://docs.rs/interprocess/2.4.5/interprocess/local_socket/struct.ConnectOptions.html).

Windows polling semantics: [Microsoft pipe wait modes](https://learn.microsoft.com/en-us/windows/win32/ipc/named-pipe-type-read-and-wait-modes).

See [discovery](discovery.md) for multiple --listen addresses, --via selection,
protected routes/source configuration, topology and optional interface-scoped mDNS.

Windows local admin chunks writes at 256 bytes to fit the native nonblocking pipe
buffer while retaining the whole-frame deadline. A 12 KiB bidirectional regression
checks exact framing beyond the pipe buffer size.
