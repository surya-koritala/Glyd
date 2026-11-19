# Architecture and implementation boundary

Current scope (13 October 2026): M0–M6 target the [Exo scoreboard](https://linear.app/getglyd/issue/GLY-234)
on NVIDIA-only and Mac-only single machines/same-type clusters. Mixed NVIDIA/Apple,
mobile and Pi work is parked; merged CPU-policy code remains. M5 covers Mac clusters,
and GLY-245 establishes the pinned matched scoreboard first in M3 before cluster
work. Pool only when the approved model does not fit one machine. Missing hardware
is recorded as not measured; software fixtures are not a scoreboard win. Historical
scope/progress sections below do not override this boundary.

Starlings is an independent Rust control plane for local devices. It does not
embed Exo or private Glyd code. The [research architecture](https://linear.app/getglyd/document/glyd-compute-architecture-and-implementation-blueprint-421e922c65bb)
records source observations and later milestones. This document specifies the
foundation implemented here; [decisions](decisions.md) records open gates.

One configured coordinator owns a monotonically increasing authority epoch.
Workers reserve capacity during prepare, execute only after commit, and stop
when their local monotonic lease expires. An epoch change invalidates old work.
GLY-168 persists the epoch before admission and uses authenticated, expiring
heartbeats. M0 still models the lifecycle independently of the M1 database/TLS.

Whole-request distribution runs each independent request on one eligible worker.
Layer pipelines assign ordered contiguous layer ranges to compatible workers.
The first owns embeddings; the last owns final norm/head and sampling. Pipelines
retain state with each layer. Combining devices does not imply faster inference
or interchangeable backend arithmetic. An ambiguous executed step ends the
generation; retrying its input cannot safely reconstruct mutated model state.

The Rust protocol crate owns bounded records, validation and content digests.
The core crate owns pure state transitions and reservations with injected time.
Adapters own external backend/process integration. The CLI owns user entry
points and the opt-in simulation process harness. No dependency may point to a
private Glyd checkout. Actual partial loading, tensor export/import and GPU
completion belong in a separately reviewed backend delivery, not this repository.

M0 simulation uses a coordinator process and two subprocess workers connected by
inherited stdin/stdout pipes. These are deterministic synthetic operations, not
model inference. No unauthenticated network listener is provided. M1 must add
explicit enrollment, mutual TLS, certificate/role/model authorization, durable
membership and receiver-side deadline enforcement before LAN communication.
Discovery alone must never confer trust. Diagnostics omit prompts and secrets.

Linux x64/ARM64, macOS and Windows run core CI. The available first desktop pair
is Mac M1 Max (32 GiB) plus Ubuntu x86_64 with RTX 4080 SUPER. Initial native work
uses the Mac CPU and an external whole-request service. Pooled acceptance needs
separately qualified backend profiles and workload approval for both devices.
The earlier two-Linux-NVIDIA proposal is not the available inventory.
Raspberry Pi 5 with 64-bit Linux is a planned CPU target. F17 software proceeds
in M2; GLY-210 retains physical storage, thermal and standalone/pool acceptance
until hardware is available. ARM64 CI is not physical Pi qualification.
Apple's optional backend and mixed numerical contract remain open in M5. Mobile
clients and bounded contribution are M7; mobile is not an initial pipeline stage.

Contracts in `spec/` use Draft 2020-12 and OpenAPI 3.1. Strict control parsing has
an input-size cap, rejects duplicate/unknown fields, unsupported versions and
out-of-range values before admission. Byte counts and authority epochs are
canonical decimal u64 strings. Numeric tensor extents are checked before any
payload allocation. Plan/manifest digests use RFC 8785 JSON and SHA-256, excluding
only their own digest field. Synthetic fixture acceptance is an explicit mode.
A digest proves content integrity, not publisher trust or model permission.

The JSON schemas and semantic validators do not replace enrollment, artifact
verification, backend conformance or cross-record authorization. Full tensor
transport integration remains M3; the [GCP1 library](data-framing.md) provides
bounded authenticated framing with all control kinds and local lifecycle gates.
[Node-local stage delegation](stage-delegation.md) reuses pinned coordinator
control and short-lived route grants, without replicating authority secrets;
the [flow library](data-flow.md) adds scoped credit and reserved reassembly.
Daemon/native layer execution and qualified cleanup remain separate gates.
M0 supplies records and simulated ordering, not a production transport stack.

Complete and review M0 before M1. M0 requires decisions (GLY-159), development
conventions (GLY-160), strict contracts (GLY-161), lifecycle semantics (GLY-162)
and separate-process fault simulation (GLY-163). Every task links its PR and
checks in Linear. License selection, publication and paid/device workloads
require their own approval. No inference measurements are claimed by M0.

## M1 enrollment foundation

GLY-164 adds `starlings-security`: protected cluster/device keys, invitation-file
bootstrap, explicit public-key approval, SQLite invitation consumption and
TLS 1.3 peer streams for both control and data channels. See
[the enrollment contract](enrollment.md) for its wire records and limits. The M0
simulator still uses local pipes. GLY-167 adds the foreground daemon described below.

The library verifies the certificate chain, exact enrolled leaf, identity name
and channel before exposing peer application bytes. GLY-165 adds live membership,
model allowlists, scoped hashed client tokens, bounded certificate overlap and
authorization around the existing worker lifecycle. Revocation closes registered
streams and cancels affected plans without freeing in-flight buffers prematurely.
See [authorization](authorization.md) for APIs, integration obligations and limits.
Listener-wide budgets and client API hardening (GLY-166), CLI/daemon
integration (GLY-167), durable membership/epochs (GLY-168) and discovery/reachability
(GLY-169) must pass review before M2. Enrollment SQLite state does not implement
worker recovery or coordinator failover.

## GLY-166 implementation boundary

The security crate now provides loopback-only client HTTP, admitted TLS/bootstrap
connections and native local admin IPC. Each listener class has its own bounded admission budget; local admin and
authenticated registry reconnects are isolated from public client handshakes. HTTP authorization precedes body allocation and dispatch;
whole control records have absolute deadlines. GLY-167 wires bootstrap and admin primitives into a running service; HTTP
remains a library primitive. GLY-167 owns lifecycle/admin command routing; GLY-168 and
GLY-169 still own remote membership and reachability. Model resolution, artifact
serving and streaming are M2 integration gates. See [listener contract](listeners.md)
and [GLY-166 evidence](evidence/GLY-166.md). Review all M1 tasks before M2.


## GLY-167 implementation boundary

The CLI now runs coordinator, worker and client profiles with validated private
configuration and exclusive process ownership. A single control loop serves
bounded authenticated local commands and fingerprint-approved bootstrap TLS.
Admitted threads handle network waits without holding the control loop or CA
signing key. Maintenance runs independently of traffic. Drain closes admission,
cancels owned plans and retains reservations until local backend acknowledgement;
stop and signals finish bounded I/O before transient cleanup. Identity/cache data
survive stop, leave and restart. See [daemon operations](daemon.md) and
[review evidence](evidence/GLY-167.md).

No inference or peer control listener is exposed by this daemon yet.
Local identity status alone is not remote reachability. Local leave must be paired
with coordinator revocation. GLY-168 below adds membership heartbeats and durable
epochs; the M1 review and physical control-plane evidence are recorded below.

## GLY-168 registry implementation

The existing coordinator port now multiplexes enrollment and authenticated registry
ALPNs. SQLite stores membership, epochs, device boot incarnations and sample replay
high-water marks. In-memory heartbeats, capabilities and links expire; restart never
restores freshness. Remote profiles persist their highest authority and boot counter.
A new epoch requires acknowledgement before eligibility; local secured workers cancel
old work and wait for trusted drain before epoch handoff. Dispatch and output gates
hold the fence through the operation. See [registry contract](registry.md) and
[review evidence](evidence/GLY-168.md).

The daemon publishes fresh host-memory observations but no inference capability.
Backend qualification, replicated model/token policy and actual worker routes
remain integration gates. GLY-169 below adds discovery/topology; the full M1
review records the M2 gate.

## GLY-169 and integrated M1 review

Concrete peer listeners share their class admission budget; local admin has a
separate budget, and proved registry reconnects reserve per-node capacity. Explicit local
routes retain the same enrolled coordinator pin, with optional source binding and
bounded fallback on subsequent heartbeats. Local topology records only directed
authenticated reachability and expires independently; reverse/peer paths are unknown.
Optional IPv4-interface DNS-SD emits untrusted hints and cannot enroll or select a
route. See [discovery](discovery.md) and the [full M1 review](evidence/M1.md).
Mac–Ubuntu physical LAN control acceptance passed; see `evidence/M1-hardware.md`.
Physical Pi and inference remain unqualified. Pi hardware acceptance is deferred
to GLY-210; ARM64 software does not wait for a device. The available control pair
is Mac M1 Max plus Ubuntu x86_64. M2 begins with the immutable [model catalog](catalog.md),
followed by verified storage, capabilities and external whole-request routing.
The catalog grants no backend capability or inference qualification.

## M2 implementation progress

The catalog and protected artifact cache now enforce immutable source revisions,
approval, hashes, quota, pinning and recovery. Host probes publish explicit
unavailable/stale values and a single host pool on ARM64; probe output alone
never claims a working inference backend. Whole-request control reuses committed
leases and trusted backend drain. SSE admission and per-stream grants separate
authenticated output from incoming connection limits; see [decisions](decisions.md).

The supervised service adapter pins the approved executable/configuration and
owns an authenticated loopback process group/job. Placement requires qualified
build/model/profile identities and measured additional request memory, retains
reservations until trusted drain, and never retries ambiguous execution.
The [generation channel](generation-channel.md) lets enrolled workers authenticate
the coordinator without a copy of its authority database or CA key. It binds a
separate bounded TLS connection to one immutable model/profile and registry epoch;
the library relay and serial coordinator dispatch use it; production daemon activation remains open.
Deferred client responses permit slow work outside the policy lock and monitor
disconnect/revocation before the first backend event. See [listener contracts](listeners.md).

One approved [native CPU API smoke](evidence/native/2026-10-12-m1max-cpu-api/README.md)
on the M1 Max matched direct, adapter JSON and SSE output/usage. Its unregistered
manifest label, sampled RSS and parent-process exit do not qualify a catalog
manifest, native allocation budget or device drain. Production generation daemon/worker
routes, automatic cache pin lifecycle, directed link measurements and real task
benchmarks remain M2 work. No mixed-device inference or physical Pi result exists.

A separately approved [native CPU relay check](evidence/native/2026-10-12-m1max-cpu-relay/README.md)
matched the direct reference through the shared coordinator/worker JSON and SSE
paths, including exact prompt counts and usage. It used one Mac and loopback TLS;
production daemon activation, catalog pin lifecycle, qualified memory/device drain
and task/link benchmarks remain open. M2 is not complete.

Explicit worker daemon warm-up now owns verified catalog paths, a durable service
pin and the approved backend process. Loading and health checks run independently
of registry/admin control. Failed or unqualified cleanup leaves a durable launch
quarantine; no automatic restart or cache release follows a health check or parent
exit. See [service activation and its remaining gates](service-adapter.md#catalog-backed-daemon-warm-up).
Warm-up alone does not activate generation or qualify native memory. The explicit
[qualified CPU route](production-routing.md) now connects daemon HTTP admission,
coordinator controls and worker execution. Its local qualification record must
bind real reviewed evidence. The repository now includes synthetic integration
checks and a [bounded native Mac CPU daemon record](evidence/native/2026-10-12-m1max-cpu-daemon-acceptance/README.md).
General memory qualification, cleanup/recovery, multi-worker scheduling and task/
link benchmarks remain M2 gates.

[Qualified service recovery](service-recovery.md) now has a software path requiring
policy before launch and durable local cleanup proof. Legacy/unqualified records
remain protected; synthetic checks cannot supply native qualification.

## M3 proposed cost profile contract

[Model cost profiles](model-cost-profiles.md) now define strict immutable workload/
backend/device/pool bindings, complete layer/component coverage and checked
per-pool stage budgets with explicit evidence/freshness/confidence. This is a
provider-independent result proposal and pure calculation, not a working external
InspectModel operation or native sharding support. Actual provider delivery and
calibration, transactional admission and real layer execution remain
GLY-175/179/181/185/235/236 gates. M2 remains incomplete.

GLY-180 adds [bounded advisory partition preview](partition-preview.md), using
declared cost profiles, directed links and fresh capability assertions. It checks
contiguous coverage and per-pool fit without reserving or executing. Scores are
uncalibrated ranking estimates; supplied approvals and inventory do not confer
native qualification. Pooling remains conditional on no eligible single-node fit.

GLY-239 adds [offline mode-cost selection](mode-costs.md) over that same search:
explicit physical-tier read rates, separate SSD traffic and directed RTT/tail
limits rank pinned whole-request/offload configurations before capacity splits.
This is advisory supplied-data replay; actual provider mapping, measurements,
prediction calibration and trusted transactional admission remain open.

GLY-181 adds a [coordinator-owned reservation transaction](transactional-admission.md)
over those bounded candidates: all physical pools are held before preparation,
all stages must prepare and commit, and rollback retains ownership until trusted
cleanup. This is an in-process library, not daemon/native activation. Actual
provider mapping, live authority fencing, durable native recovery and measured
qualification remain separate open gates.

## M3 partial-worker contract proposal

[The partial-worker proposal](partial-worker-contract.md) distinguishes the
observed public wrapper/whole-request service from undelivered native range APIs.
Typed boundary layouts and a feature-gated in-process byte-echo fixture support
software conformance work; they do not implement native process transport,
allocation, numerical equivalence or daemon layer activation. Provider delivery
and actual API mapping remain explicit gates.
