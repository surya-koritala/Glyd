# Registry, authority epochs and freshness

GLY-168 adds authenticated coordinator heartbeats and `cluster nodes` to the
existing daemon. Workers and clients contact their pinned coordinator automatically
after enrollment. Online membership alone supplies no inference capacity: the
current workers announce diagnostic host RAM with no backend, while clients send
`sample: null`. GLY-173's [resource view](resources.md) is available through
`cluster resources NODE`; backend qualification and real link probes remain open.
[GLY-169](discovery.md) adds discovery and directed reachability inventory.
GLY-168 itself added no listener port or dependency.

## Durable and transient state

The existing protected SQLite `members`, `revoked` and `settings` tables retain
membership, revocation tombstones and `registry_epoch`. That setting is a strict
`AuthorityEpoch` JSON object. `registry_boots` retains one row per enrolled device:

| Column | Encoding / meaning |
| --- | --- |
| `node` | Node UUID primary key, bound to authenticated membership |
| `counter` | Decimal u64 worker/client process boot counter |
| `incarnation` | Random non-nil UUID for that boot |
| `sample_counter` | Decimal u64 highest durably reserved resource sequence in that boot |

`Authority::start_registry` holds the existing exclusive runtime lock and commits
an incremented epoch with a new random incarnation after binding and validating its listeners, but before admitting any request. SQLite retains DELETE journaling and EXTRA synchronization. Overflow,
corrupt records and storage failures refuse admission. Opening a second runtime
against the same profile cannot obtain authority. Restoring a stale backup or
cloning a profile to another machine is not a supported failover procedure.

Each remote profile adds protected `registry.lock` and `registry-fence.json`.
The fence record contains `version`, `node`, `cluster`, `coordinator`, `boot`, and
`epoch`. Boot increments durably before network use. A process starts offline,
regardless of previously saved membership or epoch. Registry ownership lasts until
the heartbeat thread exits. Initialization never treats a malformed existing file
as missing. Neither table nor fence file stores prompts, activations, KV state,
inference output or raw invitation tokens.

Freshness lives only in memory. Before publishing a resource report beyond its
persisted reservation, the coordinator durably reserves that sequence and up to
255 successors. Further increasing reports within the range need no database
write. Live lower sequences and mutated repeats are refused; exact repeats keep
their original age. Omission disables capacity without discarding that age.

A coordinator restart restores no online node, capability or link timestamp.
The entire persisted range becomes a replay tombstone, including unused slots.
A report inside that range supplies no capacity and receives an authenticated
`sample_sequence_floor` above the range. Lost replies may be recovered by another
heartbeat; they do not consume another reservation. The worker must wait for a
new probe capture, preserve its actual age and use a sequence at least that high;
it cannot relabel the rejected snapshot as fresh. Reservation failure or sequence
exhaustion refuses publication. These are resource-report counters, not inference
token or request limits. SQLite journaling/synchronization and freshness limits
are unchanged.

## Wire contract

One request and reply use four-byte big-endian JSON byte lengths, each at most
16 KiB, with the existing duplicate-aware/depth-bounded parser, strict field
sets and version `1`. Decimal u64 fields are canonical strings. Resource batches
must fit the complete record; oversized reports are refused, never truncated.
These are internal TLS records, separate from the future HTTP API.

| Record | Fields |
| --- | --- |
| `AuthorityEpoch` | `number: Decimal`, `incarnation: UUID` |
| `Heartbeat` | `version: 1`, `boot: Decimal`, `boot_id: UUID`, `sequence: Decimal`, `epoch: AuthorityEpoch`, `draining: bool`, `sample: ResourceSample or null`, `keep_alive: bool` |
| `HeartbeatReply` | `version: 1`, `boot_id: UUID`, `sequence: Decimal`, `epoch: AuthorityEpoch`, `status: online/offline/draining/revoked`, `reconciled: bool`, `certificate: Member or null`, `reenrollment_required: bool`, optional `sample_sequence_floor: Decimal` |
| `ResourceSample` | `sequence: Decimal`, `capabilities: NodeCapabilities`, `links: LinkSample[]` |
| `LinkSample` | `peer: NodeId`, `age_ms: u32`, `rtt_us: u32`, `bytes_per_second: Decimal` |
| `NodeSummary` | `node`, `role`, `status`, `schedulable: bool`, `fresh_links: count` |
| `ResourceView` | Current local-admin diagnostic: original `sample`, receiver-adjusted `age_ms`; null when stale/unavailable |

Boot and heartbeat sequences start above zero. Epoch zero with a nil incarnation
is only the initial unknown-authority value; replies always carry a nonzero epoch
and non-nil incarnation. Lower boot counters, changed IDs under an existing boot,
non-increasing heartbeat sequences, lower sample sequences and mutated repeated
samples are refused. One current sample and its high-water mark are retained per
node; long-running heartbeats do not exhaust the M0 simulator's snapshot history.

The floor is omitted on normal replies. When present it must exceed the resource
sequence in the matching request; the client accepts it only through the existing
authenticated, boot/sequence/epoch-bound, timely reply path. It grants no capacity
or authority. Coordinator and workers require a coordinated upgrade: older strict
clients reject the new field on restart recovery. An older coordinator interprets
the reserved upper bound as an accepted sequence and may refuse current lower
reports after downgrade. A reviewed fresh worker boot can reset that boot's
counter, but downgrade/rollback operation is not qualified by these unit checks.
Never reset service ownership/quarantine to work around recovery.

Capabilities use the existing production validator: fixture flags and placeholder
digests are forbidden, node/cluster must match the TLS identity, and only workers
may report capacity. At most 64 links are accepted. Targets must be distinct and
not self; age is 0–60,000 ms, RTT is 1–60,000,000 microseconds and bandwidth is a
positive u64 byte count per second. Unenrolled/revoked targets never count as fresh
links. Reported endpoints are inventory data, never automatically dialed or trusted.
The existing 64-member enrollment ceiling includes the coordinator.

The coordinator's existing port accepts `starlings-enroll/1` and
`starlings-registry/1`. TLS 1.3 requests and verifies optional client certificates
because enrollment has none. The registry dispatcher **requires a verified exact
enrolled worker/controller leaf before reading the JSON request**. A CA-valid
unregistered certificate is insufficient. Existing control/data channels still
require client certificates and their distinct ALPNs. Resumption and early data
remain disabled. A revoked current leaf can receive only its own minimal revoked
reply; it cannot update registry state, enroll again or use control/data channels.

## Epoch handoff and admission

1. The coordinator durably advances its epoch before startup admission. A known
   worker returning with a higher boot counter also advances the global epoch,
   atomically with the boot update, invalidating every prior observation.
2. A heartbeat carrying an older or initial epoch receives the current epoch. The device rejects a lower
   epoch or a different incarnation under the same epoch, persists a newer epoch,
   and remains offline. A client restarting with an already current epoch may
   acknowledge it on its first fresh exchange. Replies are bound to its random boot ID and current
   request sequence.
3. A subsequent heartbeat acknowledges that epoch. Only that acknowledged round
   trip can make registry authority fresh. A new process cannot reuse freshness
   or work from its prior boot.
4. `State::plan` requires every stage to be a current authorized worker with fresh
   capabilities under the current registry epoch. Role, manifest permissions,
   controller token, lease and worker lifecycle checks continue to apply.
5. `SecuredWorker::bind_registry` binds a local worker to its matching node/cluster
   fence. Every control dispatch and output callback checks freshness while holding
   the fence through the operation. A concurrent epoch update cannot pass that gate.

The initial policy uses one configured coordinator, not leader election or consensus.
A worker restart conservatively fences the entire cluster (at most 64 members).
Client-role restarts do not fence compute workers. Per-worker plan incarnations are a future optimization only if restart churn is
measured to require it. Compromised coordinators/enrolled reporters, duplicated
private keys, same-user/root access and cloned/stale databases are outside this
single-profile authority guarantee. There is no automatic failover.

## Expiry, partition and drain

Normal heartbeat starts are spread randomly over 2–3 seconds, measured from the
previous exchange's start. Successful exchange latency consumes that interval
instead of adding more idle time to the start-aged freshness window. An overdue
successful exchange starts the next attempt without another idle interval; this
cannot extend the six-second fence or make a late response valid. After a transient failure,
the daemon retries after 250–375 ms of jitter. A returning node proves its existing
key before entering its reserved TLS slot; anonymous per-IP/rate limits do not
charge that slot. The connection, thread and frame limits remain bounded. One outbound
exchange has a three-second absolute deadline including connect, TLS and frames;
server admission/handshake has its existing five-second deadline. No network I/O
holds the policy or fence lock.

Each worker has one owned memory sampler and one snapshot slot. Host RAM/cgroup
collection runs outside the slot lock, at most once per second; the heartbeat
thread reads only the retained capture. A blocked collector cannot prevent a
no-sample heartbeat or local admin contact. Missing or stale captures advertise
no resources, and repeated publication retains the original capture ID and age.
The sampler uses the existing stop/join path. Admission and output drain before
joining; an OS memory query that does not return can still delay final exit.
No inference capability follows from a memory observation.

Heartbeats and samples expire at six seconds, not strictly after six seconds.
Device freshness starts at request send time, so delayed replies cannot extend it.
Sample age is reported probe age plus receiver elapsed time; repeating a batch,
omitting it then restoring it, or merely sending more heartbeats cannot refresh
it. Capability and each link's own reported age are checked independently.
An expired sample may remain as bounded replay history but cannot be scheduled.

Monotonic time drives expiry. A paired wall-clock observation can only shorten
freshness: a forward jump/suspend expires it; a backwards jump refuses it until a
new authenticated exchange. This covers platforms whose monotonic clock pauses
across suspend. These guards do not make OS scheduling or filesystem I/O hard
real-time, and physical suspend/power-loss recovery is not certified by unit tests.

The daemon maintains local fences every 250 ms even when heartbeat I/O is blocked.
Offline, draining and revoked fences cancel owned work and deny output. Expired
nodes receive no new plans. Existing monotonic leases still expire independently.
`Worker::reconcile_epoch` cancels prior work and changes epoch only after its
reserved bytes reach zero. Unstarted reservations are released; an in-flight
invocation retains memory until the trusted local backend acknowledges drain.
A network peer cannot fake that completion. Replay tombstones survive the handoff;
no KV state or ambiguous invocation is resumed.

`cluster nodes` returns enrolled remote nodes, roles, online/offline/draining/revoked
status, schedulability and fresh-link counts. Disconnected nodes become offline;
revocation remains a durable tombstone. Local drain is immediate; coordinator
revocation reaches a connected device on its next heartbeat and otherwise its
old authority expires locally. Coordinator drain refuses enrollment and work but
keeps the shared port open for authenticated drain/revocation replies. Stop closes
all listeners. Remote stop/leave sends a bounded best-effort draining heartbeat;
loss of that message falls back to expiry.

## Integration and remaining limits

The CLI exercises membership, epochs, drain and revocation over real TLS. It does
not create an inference worker, backend, model route, tensor listener or scheduler.
The secured-worker binding and orphan accounting are tested against synthetic
reservations; M2 must bind each real backend and drive trusted completion. The
local authorization database is not a remotely replicated model/token allowlist.
Remote policy synchronization and rotated certificate pin delivery must be supplied
before exposing a remote inference/control route. Registry validity refresh is wired as described below; explicit key rotation and
remote inference policy remain distinct integration requirements.

Hardware/resource probes, real link measurement, physical suspend/multi-NIC validation and Pi inference remain unqualified.
Mac–Ubuntu LAN control acceptance is recorded in `evidence/M1-hardware.md`. Linux ARM64 CI is not
a Raspberry Pi inference result. No model, GPU, paid host, package publication or
license decision is part of this delivery. See the [integrated M1 review](evidence/M1.md) before M2.

## Review remediation

Registry TLS verifies the pinned CA, exact coordinator name and enrolled public
key; automatic validity refresh retains that key. Other control/data channels
retain exact-leaf authorization. Explicit key rotation requires out-of-band pin
recovery. Workers save refreshed certificates atomically before publishing them.
For same-key automatic refresh, the previous leaf remains acceptable until its
original signed expiry, allowing recovery after a lost reply beyond five minutes.
Key-changing rotation retains its separate five-minute overlap. Successful refresh
does not close a persistent registry stream; both ends advance the in-memory member
after the replacement reply. Revocation still wins over either credential.
Expired offline workers require re-enrollment; expired network credentials fail.

There are 63 node-specific handshake/registry slots, separate from 32 ordinary
network admission slots and local admin admission. A bounded nonblocking proof
queue precedes TLS; see [the admission contract](listeners.md#registry-reconnect-admission).
The next heartbeat has a six-second read deadline. IPv6 ordinary admission uses
individual /128 addresses. Filled anonymous handshakes no longer consume a
returning node's slot. TLS and live membership checks remain mandatory; a proof
cannot authorize work or change the node identity.

Stale boot counters receive an authenticated re-enrollment-required response.
Use a new profile and invitation; do not delete coordinator replay history.
Worker-triggered global epoch changes are limited to one per six-second TTL.
Repeated crashes may still disrupt service; per-device epochs remain deferred.
Receive time is captured before policy locking. Fence persistence invalidates
freshness first and performs the durable write outside the fence mutex.

Authorization reads do not checkpoint the clock. Maintenance checkpoints at most
once per minute, with forced checkpoints on durable admission and first drain.
Transient SQLite contention does not revoke credentials; integrity failures remain
fail-closed. Crash rollback detection has a checkpoint-interval window; snapshot
restoration is unsupported. Short, nonblocking authorization callbacks must not
reenter the authority. A callback panic deliberately poisons policy until restart.

Automatic refresh extends certificate validity, not key lifetime. Device keys have
no automatic maximum age in M1. A copied private identity can renew until explicit
revocation; suspected compromise requires revocation and a fresh enrollment. There
is no secure-enclave or key-theft protection claim. A revoked worker intentionally
keeps its local admin endpoint available, stops registry retries and cannot admit
work; the operator can inspect, stop or leave it.

SQLite busy timeouts are zero. Transient contention is surfaced immediately and
maintenance retries on its next tick; it cannot sleep two seconds under the global
policy lock. Integrity/clock failures remain fail-closed. A locked database may
temporarily deny operations or expire remote freshness; it must not exit the daemon.
