# Bounded authenticated directed probe session

GLY-174 adds a real signed-proof/mTLS transfer path with explicit background daemon
activation/status and typed placement-input fixture replay. This does not close M2.
It serves NVIDIA-only/Mac-only placement and the later matched Exo scoreboard.
No mixed hardware, mobile or Pi capability is inferred by the protocol.

Use a separately budgeted `PeerListener` for probes; persistent registry heartbeats
already own the one reserved node slot in their listener class. After native
`try_accept_registry` signed-key proof, `IncomingPeer::probe` accepts only the
distinct `starlings-probe/1` TLS 1.3 ALPN and a current enrolled worker.
CA/leaf identity, coordinator key/name, role, cluster, live authority epoch,
credential/revocation and online status are checked before data and throughout I/O.
No model permission, inference route or worker copy of the CA private key is used.

Worker `probes::measure` uses an invitation, its live RegistryFence and an explicit
numeric local route/source address. It opens its own connection; it does not
borrow heartbeat or admin I/O. Untrusted discovery never supplies the destination.
Certificate refresh may retain the already enrolled coordinator key/name/CA, as
on the existing registry path; arbitrary CA/name/role/key changes are refused.

When `token_profile` is absent, `ProbeLimits` version1 requires payload4096–262144 bytes in4096-byte multiples,
1–8 pings, inter-ping gap0–100ms and minimum interval30000–3600000ms. Gap is
recorded so warm back-to-back RTT is not silently treated as idle token latency.
The existing owner baseline remains the hardware/transport reference; this API
does not re-derive it or qualify a stage path. Both endpoints enforce identical
limits. Share one ProbeBudget across all probe listeners/jobs for that profile.
Authenticated attempts consume cooldown even on subsequent failure; at most64
node cooldown entries are retained. Application I/O has a1MiB combined cap, fixed
4KiB payload buffer and a fixed five-second client attempt deadline including connect,
proof, handshake, gaps and transfers. Server I/O has five seconds from proved admission;
the existing reconnect proof's own bounded deadline precedes it. TLS handshake protection
also applies. No automatic retry or added timeout is introduced.

Measure upload and download separately, with a complete-transfer acknowledgement,
local monotonic clocks and explicit timing basis. These are acknowledged application
goodput measurements including control/ACK and authorization costs, not raw link
bandwidth, tensor throughput or inference speed. Sender-start-to-ack-received and
receiver/start-request-to-receive-and-ack-sent are distinct bases; compare matched
bases. Direction, payload, total application bytes, local/peer address and selected
interface are retained. TLS framing/proof bytes are not application bytes.
Only successful complete transfers publish. Unknown server RTT is null, not zero.

Ping RTT samples, integer mean and variance retain their units. Client observations
have locally retained monotonic/wall timestamps (not deserializable),30-second TTL
and live fence/epoch checks; reading cannot refresh them, and rollback/suspend,
drain, revocation or offline status withdraws them. Server observations retain
daemon-owned capture age and verify live membership/epoch before publication.

Loopback software tests are actual authenticated socket transfers but do not
establish physical LAN performance, a new model/backend qualification or an Exo
scoreboard win. Device/LAN workloads,
downloads and spend require separately reviewed bounded approval.

## Explicit daemon workflow

Coordinator server record: `{"version":1,"listen":"LOCAL_IP:0","limits":{...}}`.
Worker route record: `{"version":1,"destination":"SERVER_IP:PORT","source":null,"limits":{...}}`.
Limits must match exactly. Store records as protected local files. On each profile,
`cluster probe-digest PRIVATE_FILE` validates and prints the canonical SHA-256;
it grants no permission or performs network work. Review the complete record.

Coordinator `cluster probe-listen PRIVATE_SERVER --approve SHA256` activates one
independently budgeted listener and reports its bound port; activation is immutable
for that daemon lifetime. Worker `cluster probe PRIVATE_ROUTE --approve SHA256`
starts one bounded background job and replies immediately. It reloads the current
identity/invitation from private state without another RegistryClient or boot/epoch
increment. Repeated activation must keep the same approved route; the retained
budget enforces cooldown and concurrent jobs are refused. No automatic retry occurs.

`cluster links WORKER_NODE` returns one bounded status view: pending, failed,
measured or unavailable, with original capture age and explicit null measurement
when stale or withdrawn. Coordinator views also verify current membership and
online epoch; worker views use the existing fence. Status cannot revive freshness.
Completed samples are in memory only, bounded by enrolled-node capacity; restart
restores neither observations nor activation. Drain/revocation closes eligibility.
Network waits and payloads are never processed by the admin/control handler.

## Placement inputs and replay

Current status includes `placement_inputs`, two strict `starlings_core::links::LinkInput`
records preserving source/destination, sample ID, authority number/incarnation,
payload, timing basis, RTT availability/variance/gap and original capture age.
Unavailable status has null inputs. A parsed status file is diagnostic evidence,
not live authorization: production consumers must obtain a current authenticated
view and recheck the authority/online gate before using it. No archived input
can qualify a backend or restore a live node's freshness.

`LinkFixtureReplay` accepts only synthetic fixture records. It retains at most128
directed pairs and256 historical sample keys; capacity exhaustion is explicit,
with no silent eviction. Exact repeats cannot extend freshness, earlier replaced
samples are refused, withdrawal retains tombstones, and a backward injected clock
fails. `examples/directed-links.json` supplies two asymmetrical directions and a
congested forward sample. The runnable core test checks the resulting fact values,
age, conflict, expiry and withdrawal. It makes no stage-placement decision or
model-speed prediction; GLY-245 remains first in M3 before planner implementation.

Authenticated socket tests reject limit mismatch, stale epoch, wrong marker,
corrupt payload and partial EOF at their specific error boundaries. A stalled
proved peer hits the existing five-second I/O bound; revocation closes a proved
idle peer before payload. Failed authenticated attempts retain cooldown, release
the probe permit and leave the independent heartbeat usable. Default daemon tests
verify both typed directions, unchanged sample ID/increasing age and null inputs
after drain. These are software acceptance records, not physical link benchmarks.

## Optional token-link diagnostic

[GLY-241 token-link profile](token-link-profile.md) documents the separately
approved16KiB/64B mode, explicit sample/traffic/time caps, bounded keepalive,
failure-preserving private JSONL and interpretation. Its profiles never produce
placement inputs. The optional field is omitted from legacy serialization,
retaining existing configuration digests and behavior. No physical-link or power
result is inferred from protocol regression tests.
