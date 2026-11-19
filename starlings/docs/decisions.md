# Decisions and compatibility gates

Current scope follows the [same-family Refocus](https://linear.app/getglyd/document/glyd-compute-continuous-completion-goal-m0-through-m6-a805bb7bddc0).
Independent software/source work may proceed while external acceptance stays open.
Historical implementation notes below do not restore parked mixed/mobile/Pi work
or grant new workload, independent-review, license or publication approval.

GLY-172 adds [explicit qualified CPU recovery](service-recovery.md). Policy is
reviewed before launch and bound to configuration/target; the owner alone records
completed cleanup. Explicit recovery durably unpins before recovered publication.
Legacy/unqualified work and missing receipts stay quarantined. No forced reset,
PID guess or model retry is added.

M2 daemon routing reuses the existing per-worker Dispatch and supervised
Service, with explicit full-digest approval of strict local qualification/route
records. The route is CPU/host-pool only, one manifest and one service slot per worker.
Optional qualified replicas use bounded queues; admission refuses work beyond
the count/body limits, retains failed runtime reservations and
service quarantine, and supplies no native qualification automatically. See
[the contract and remaining gates](production-routing.md).

Recorded 11 October 2026. Accepted here means a foundation design choice, not
proof of a production backend or physical-device result. Roles are accountable
functions, not newly assigned people. Changes require rationale and affected
contracts/tests in the PR and Linear issue.

| ID | Decision / status | Accountable role | Review trigger and milestone |
| --- | --- | --- | --- |
| D01 | Independent repository; established | Maintainer | Any source boundary change; all milestones |
| D02 | Study Exo; copy no implementation; established | Maintainer | Any requested reuse; all milestones |
| D03 | License undecided; established | Project owner | Before publication/distribution; GLY-205, M6 |
| D04 | Rust orchestration, separate inference processes; accepted | Maintainer | Required backend language/runtime changes; M2/M3 |
| D05 | Whole requests first; independent software may proceed while external qualification stays open | Maintainer | Complete actual milestone acceptance and review before claiming qualification; current Refocus in Linear |
| D06 | One configured coordinator; no election; accepted | Control maintainer | Durable epoch allocation/reconciliation in M1; later failover need |
| D07 | Explicit enrollment and mutual TLS; accepted design | Security maintainer | M1 implementation and M6 review; no M0 network service |
| D08 | Bounded TCP/TLS stage traffic; proposed | Transport maintainer | Complete frame contracts and measured need; M3 |
| D09 | Contiguous layers, one proven model profile first; proposed | Backend maintainer | Real partial loader and safe boundaries; M3/M4 |
| D10 | Bitwise same-backend reference gate; accepted | Backend maintainer | Nondeterministic reference must be investigated, not silently loosened; M3/M4 |
| D11 | Mixed-backend numerical comparisons; parked | Backend maintainer and project owner | GLY-193 is canceled under the same-family refocus; no mixed execution or tolerance relaxation |
| D12 | External MLX Apple adapter; open | Backend maintainer | [Source feasibility](apple-backend-feasibility.md) is unqualified; actual model/profile, dependency and native evidence still required; GLY-191, M5 |
| D13 | Mobile contribution; parked | Mobile maintainer | M7 canceled; restoring scope requires an explicit decision |
| D14 | Hardware inventory, budget and release method; open | Project owner | Before real-device/paid work; M2 onward, release M6 |
| D15 | RFC 8785 + SHA-256 excluding only self-digest; accepted | Protocol maintainer | Independent golden vectors and schema/profile changes; GLY-161, M0 |
| D16 | Physical Pi qualification; parked; retain existing ARM64/CPU software | ARM64/backend maintainer | GLY-210 is canceled under the refocus, not a desktop-release dependency; restoring physical Pi scope requires an explicit decision |
| D17 | Ponytail from initial writing through review; established | Maintainer | Every task; correctness/security review remains separate |
| D18 | Available desktop inventory: M1 Max 32 GiB and Ubuntu x86_64/RTX 4080 SUPER; historical Mac CPU smoke only | Backend maintainer | No same-family native pair is qualified; GLY-175 inventory/profile and explicit device/workload approval remain required |

## Repository boundary

This repository owns coordinator/worker contracts, scheduling, admission,
transport integration and external process adapters. Glyd's CUDA kernels,
compression formats and native layer engine remain separate. A future installed
backend must report exact build/model/profile support; the presence of an API or
source module is not evidence of inference compatibility. No precision is
silently reduced to make a model fit. The project's undecided license does not
alter Glyd's license or grant rights to third-party model weights.

## M0 state policy

Time is injected as monotonically increasing milliseconds. Backward clock values
fail. Leases expire at `now >= deadline`; wall-clock timestamps never extend them.
Recovery after confirmed worker shutdown uses a newer authority epoch, fails
unfinished generations and starts with no reserved memory. It is not a live takeover.
Release/cancel results participate in the same replay window as other commands.
Each authenticated authority serializes requests to a worker using a strictly
increasing `request_sequence`. The digest binds request ID, authority epoch,
sequence and operation. The latest 4,096 results are retained, including failures
and cleanup; older sequences return `ReplayExpired` and never execute again.
A sequence gap is allowed, but out-of-order arrival is rejected. Callers retry the
original serialized request rather than assign a retry a new sequence.

There are at most 256 resident lease records and 256 resident sessions. Terminal
records are reclaimed on admission pressure without changing the authority epoch.
Cached replies retain their authorization plan through a shared reference. IDs
must be fresh for new logical leases/generations; a new sequence is a new command,
not a way to retry an expired result. Recovery requires a newer durable authority
epoch and keeps at most 256 diagnostic generation records. KV state is not restored.
Invocation tracking is bounded by the validated session context/token count, not
an unrelated 256-token ceiling. Active work retains resources until trusted local
completion. Capability tracking keeps one current snapshot for each of 64 nodes.

Each simulated invocation carries one complete tensor within its stage's scratch
budget. Chunk reassembly, multi-tensor boundaries, lease renewal, durable storage
and live epoch handoff remain later work. Recovery records bind node/cluster and
retain identities/status only, never prompts, outputs, activations or KV buffers.
A process must be confirmed stopped before recovery can release its allocations.

## GLY-164 implementation choices

TLS 1.3 uses Rustls with its ring provider; CSR signatures and certificates use
rcgen. Directly signed node leaves have one DNS SAN encoding node, role and
cluster; control/data/enrollment use distinct ALPN values. Only the pinned cluster
root is trusted. Leaf equality against the local enrollment snapshot adds the
unknown-node check beyond certificate-chain validity. Resumption and early data
are disabled. GLY-165 adds live policy and renewal as described below.

SQLite transactions atomically insert a member and delete its invitation hash.
No raw invitation token enters SQLite. An issued response lost after commit
requires fresh enrollment; replay cannot retrieve it. The first implementation
caps members at 64 and outstanding invitations at 128. It fails closed at capacity.

Private key and invitation files use owner-only, handle-oriented `private-fs`
operations on supported local filesystems. This is a new dependency with an
explicit isolation contract, not an independently audited security guarantee.
Its 0.1.x range is permitted after review; Cargo.lock pins the reviewed artifact. No keychain or encryption-at-rest
claim is made. Same-user/root compromise and dishonest filesystems are outside
the boundary. See [enrollment](enrollment.md) and [dependencies](dependencies.md).


## GLY-165 implementation choices

Use the existing SQLite, Rustls, rcgen and worker state machine. No new third-party
dependency. Model permissions are explicit manifest-digest allowlists; only a
coordinator with the matching control grant can dispatch worker operations.
Controller-owned random tokens have narrower scopes, at most one hour of validity,
and only hashes in durable state. One plan belongs to one client token.

A standard-library file lock excludes competing local authorization runtimes.
Live handles share one bounded policy lock. Revocation updates durable state,
closes affected TLS sockets and cancels plans; backend drain remains a trusted
local acknowledgement. The daemon must tick maintenance even without traffic and
gate every output chunk. Remote membership replication remains GLY-168.

New keys rotate over verified control channels with at most five minutes of old
certificate overlap. A second rotation during overlap fails. Coordinator identity
publication uses a durable new private file followed by a transactional pointer.
Role changes, CA replacement and stale-backup recovery require separate designs.
See [authorization](authorization.md) for exact bounds and failure behavior.

## GLY-166 listener choices

Keep synchronous, bounded listener primitives and the existing authorization lock.
Use `httparse` for HTTP grammar and `interprocess` for safe native Unix/Windows IPC;
add no asynchronous runtime or general web framework. Client HTTP binds only to
loopback, uses exact numeric Host matching and rejects all browser origins.
A shared admission object caps connections, TLS work and fixed-window connection
rates across every listener. Absolute deadlines cover whole control records.

Local admin IPC uses native owner restrictions and a per-listener random
capability in a protected file. Do not use Windows' permissive default pipe DACL.
Raw capability-bypassing reads are private. Existing endpoints are not replaced;
GLY-167 must integrate exclusive ownership and restart cleanup. Client handlers
receive an explicit model/action scope selected by trusted routing. Every bounded
response enqueue rechecks that scope; M2 must carry this gate into streaming.
See [listeners](listeners.md) for limits, API contracts and excluded guarantees.


## GLY-167 daemon choices

Retain a direct standard-library control loop with fixed admission-bounded threads;
no async runtime, framework, thread-pool abstraction or service installer. Native
state/cache discovery uses `dirs`; signal notification uses `ctrlc`. Exclusive
profile ownership uses the existing protected filesystem and standard file lock.
Only validated transient admin files are removed, never identity or model data.

Enrollment is prepare, explicit fingerprint approval, then join. Keep approval
state bounded and ephemeral; TLS workers hold only a leaf credential, while the
control loop owns certificate issuance. Separate certificate delivery from issuance
so a stalled peer cannot block administrative commands. No client HTTP listener is
started until M2 supplies trusted routes and model scopes. Worker/client status
reports the authenticated registry state added in GLY-168. Local leave is distinct from
coordinator revocation; propagation remains GLY-168.

Add server capability proof before local clients disclose their capability, and a
bounded reply/receipt exchange for native pipes. Drain rejects model access and
keeps in-flight reservations until trusted local completion. See [daemon contract](daemon.md).
CI reuses the existing OS/architecture cache after lockfile changes; Cargo's
fingerprints still select required rebuilds. All four platforms, default/all-feature
checks and warning failures remain mandatory. GLY-227 changes the cold-build cap
to ten minutes. The feature pass excludes duplicate daemon/core/security tests.
Caches save after test failures; only stale PR runs are cancelled.

## GLY-168 registry decisions

Reuse protected SQLite, Rustls, listener budgets and the existing worker lifecycle;
no new dependency or runtime. Heartbeats use a distinct registry ALPN on the existing
coordinator port. Only that narrowly scoped route may return a node's own revocation
tombstone to its revoked current certificate. Control/data authorization is unchanged.

Persist startup epoch increments and known-worker boot changes before admission.
A worker restart fences the global cluster; per-device plan incarnations are deferred
until measured need. This is one configured coordinator with exclusive local state,
not consensus, automatic failover or protection against cloned profiles.

Retain monotonic boot/sample high-water marks, but never restore freshness after
restart. Repeated/omitted samples cannot freshen themselves. Receiver monotonic age
is augmented with a wall-clock guard that can only shorten freshness across suspend
or clock jumps. Device freshness starts at request send time; normal retries have
bounded jitter. A new epoch cancels old work, waits for trusted drain and preserves
tombstones. Dispatch/output holds the fence lock through the authorized operation.

Keep one protected fence and at most one bound local worker per remote profile.
The daemon reports membership without claiming resource probes or inference. See
[registry](registry.md) for records, limits, state handling and M2 integration gates.

## GLY-169 network decisions

Use at most four explicit local numeric listeners/routes and native source binding;
all listeners share admission and all routes retain one coordinator pin. One route
attempt per heartbeat avoids multiplying timeouts. OS routing remains the default.
Optional DNS-SD is IPv4-interface-scoped, separate from trust, and never edits
configuration. Reuse pinned mdns-sd, if-addrs and socket2 instead of owning DNS or
platform FFI. Async/logging defaults are disabled. No automatic interface selection,
NAT traversal, public destination or remote inference route is added.
See [exact limits](discovery.md) and [M1 acceptance inventory](evidence/M1.md).

## GLY-211 remediation boundaries

The available physical pair is an M1 Max Mac and Ubuntu x86_64 with RTX 4080 SUPER.
M1 acceptance uses CPU/network only. Raspberry Pi remains an ARM64 CPU target;
no physical Pi is available. These checks establish no inference qualification.

The ordered replay window above replaces epoch-driven history reclamation.
The sustained regression runs 1,000 generations in one epoch, including a
1,000-token prefill and 512 decode steps, with exact replay and cleanup checks.
No session reserves future request-cache slots: rolling acknowledgements cannot
exhaust during concurrent sessions. Context, memory and active-session admission
still apply. This is an unreleased wire-contract change; all nodes must upgrade
together. Earlier requests without `request_sequence` are rejected.

GLY-229/230 are M2/M3 gates: streaming admission separate from registry/admin,
a whole-request operation, and measured authorization/chunk/per-token overhead.
No inference implementation is included here. M6 audit/distribution gates remain.


## M1 reconnect admission

Registry capacity is reserved only after a fresh challenge signed by the enrolled
device key. This reuses the existing P-256 key and ring verifier; TLS remains the
application authentication boundary. A shared 64-socket nonblocking queue prevents
idle probes from holding thread/TLS resources, while the single slot per proved
node includes its TLS handshake. Node identity and registry ALPN must match after
TLS. See [the wire contract and limits](listeners.md#registry-reconnect-admission).
No new dependency, persistent credential or public service is introduced. This
unreleased protocol change needs coordinated node upgrades and independent review.

## M2 entry and catalog

PR #9 merged after all five checks passed at d9789a0. Green task PRs may proceed
after correctness/security/complexity review under explicit owner authorization;
independent review is scheduled separately and is not claimed by an automatic merge.
Repository protections remain unchanged. Physical Pi qualification is deferred to
GLY-210 without blocking ARM64 code. The available control pair is Mac M1 Max plus
Ubuntu x86_64; no new GPU/model workload is authorized by that inventory.

The model catalog reuses protected atomic files, StateLock and protocol hashes.
Manifest registration, explicit local approval, artifact readiness and numerical
qualification are separate states. Require a full pinned revision; never silently
resolve `main` during offline use. Rehash protected source assets in constant
memory; retain explicit lossiness, recurrent profiles and excluded components.
Provider resolution and cache lifecycle continue in GLY-171/172. See [catalog](catalog.md).

## M2 whole-request dispatch

Reuse the lease/session reservation lifecycle for one complete generation. A
`start_request` has one backend invocation, resolved sampling and exact manifest
identity. It cannot use a layer-pipeline lease, and tensor steps cannot use its
whole-request lease. Cancellation withdraws output immediately; capacity remains
reserved until trusted local completion/failure drain. Control replay stores no
prompt payload and cannot launch a second backend attempt. See spec/README.md and
GLY-230 evidence for the protocol-cost diagnostic and M3 validation budget.

## M2 streaming admission and output authorization

Keep blocking client accept and bounded synchronous I/O. An authenticated SSE
stream releases its unauthenticated connection permit and holds one of 32 live
authority grants. Production integration uses a separate client-listener budget;
admin and registry listeners retain their independent budgets.

A grant binds the token hash and model/action scope, with token/certificate expiry,
a monotonic lifetime and a two-second maintenance freshness limit. Each output
enqueue takes only that stream's small mutex and clock checks; it performs no
SQLite, filesystem, certificate parsing or global-policy lock operation. Policy
maintenance revalidates grants at least once a second. Revocation and drain close
those gates before returning; failed/expired gates never revive. A failed policy
read closes the affected grant rather than extending its freshness. The gate is
held only around a nonblocking write of at most 4 KiB, ordering revocation against
output without holding a global lock across I/O.

SSE events are bounded to 64 KiB, total output to 16 MiB, and stream lifetime to ten
minutes (shorter authorization/lease limits still apply). Socket backpressure has
a two-second progress timeout. Invalid events, exhausted limits and write failures close the stream permanently;
the adapter must cancel backend work and wait for trusted drain before releasing
resources. The current 1 ms retry runs only under socket backpressure; no idle
stream thread polls for output. No asynchronous runtime or new dependency is added.
The shared generation relay now uses this primitive; production daemon activation
and qualified catalog/backend lifecycle integration remain open.

## M2 daemon service ownership

Use the existing ServiceConfig, catalog lock, durable pins and process supervisor
for explicit worker warm-up. Do not introduce a second launch schema or substitute
model paths after approval. Bind every weights argument to its verified catalog
object; approve the entire resulting configuration. Never infer backend consumption
or native qualification from that argument binding or the model listing.

Persist a service launch quarantine before spawn and preserve it through crash,
drain and restart until native cleanup qualification can support a recovery path.
The quarantine itself protects the manifest, independently of ordinary pin files.
A second launch refuses unresolved ownership. A backend failure leaves registry
and admin operational; no automatic retry or eligible replica is published.

## Resource-report replay reservations

Reserve bounded sequence ranges durably before publishing capacity, using the
existing per-boot SQLite counter. Keep exact replay and age checks in memory.
After coordinator restart, burn the whole reservation and request a fresh probe
above it through the authenticated heartbeat reply. This amortizes durable writes
without weakening crash replay protection or changing SQLite synchronization.
The optional recovery field requires coordinated coordinator/worker upgrades;
older coordinators interpret the counter differently, so downgrade remains an
explicit qualification gate. No inference request/token limit is introduced.

## GLY-188 daemon pipeline (2026-10-22)

The owner directed on 2026-10-22 that Starlings run Glyd models end to end ("Go for
all and fix all the issues"). For the two-stage daemon pipeline this supersedes D05's
whole-requests-first order and the M2 note that no new GPU/model workload is
authorized. Security checks are unchanged: the pipeline uses the existing stage
grants, admission and core state machine; see [daemon pipeline](daemon-pipeline.md).
