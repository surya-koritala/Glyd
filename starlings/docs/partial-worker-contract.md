# Partial-worker contract proposal v1

GLY-185 defines provider-independent metadata, a delivered `glyd-range/1` pipe
adapter and an explicit byte-echo fixture. The adapter does not launch or qualify
an installed worker or enable daemon/native routes. The
[whole-request adapter](service-adapter.md) remains separate.
Actual range execution is the provider's GLY-235 delivery; measured costs are
GLY-236. Both depend on an installable package and its actual API mapping.

## Observed, proposed and still missing

| Interface | Evidence and current disposition |
| --- | --- |
| Whole-request HTTP service | Compute implements authenticated health/model/chat, counting, streaming and owned cancellation for approved qualified configurations; see the existing service/routing contracts. This is not layer-range execution. |
| Public Glyd CLI wrapper | At pinned [public revision fb73c7e](https://github.com/surya-koritala/Glyd/blob/fb73c7eef45218790077535debe5054bab51d02a/bindings/python/glyd/cli.py), run/serve dispatch to the installed GPU package. This wrapper does not establish the package's native range API or present capabilities. |
| Provider range API | The delivered `glyd-range/1` contract is mapped below from pinned provider API documentation. An installed release artifact, actual conformance and multi-machine qualification remain GLY-235/187 gates. |
| BoundaryContract / RunChunk metadata | Rust proposal implemented in `starlings_adapters::partial`, including strict parsing and checked ordered layouts. These are not native capability attestations. |
| SyntheticStage | Available only with `test-backend`; explicit fixture opt-in, no model files or native work. Exercises lifecycle and opaque byte ownership in process. It is not an installed subprocess, transport driver or substitute for GPU drain. |
| Native process launch/wire binding | `partial::glyd::Worker` owns supplied protocol pipe halves and maps the delivered framing/lifecycle. The existing local supervisor must own the approved process, deadlines, assets and quarantine; this module starts no executable. |

The public wrapper was read as source only, not executed; SHA256
`f94650ba7de793228687a6f193392393338c181db7b0b2ae5c18d945ff61ad70`.
No private implementation or installed GPU package was read or copied.

The [GLY-238 delivery checklist](https://linear.app/getglyd/issue/GLY-238) covers
whole-request service compatibility. It does not request layer sharding. The
additional range artifact, boundary/state description and cost delivery belong
to GLY-235/236; a whole-request package cannot satisfy them implicitly.

## Delivered glyd-range/1 adapter

The mapping follows the provider's
[API contract](../../docs/range-worker.md),
read as documentation only. No backend implementation is imported. The public
`starlings_adapters::partial::glyd::Worker<R, W>` accepts exclusive `Read`/`Write`
pipe halves from an approved, separately owned process. Its caller must maintain
the existing live `StageGrant`/lease fence before every begin, dispatch and publication and
retain process/pin/quarantine ownership after any uncertain result. This adapter
confers none of those permissions or cleanup qualifications; its raw methods
cannot activate a daemon/native route.

The adapter writes little-endian header/body lengths and JSON plus ordered raw
body slices without concatenating tensor buffers. It negotiates the documented
1 MiB header and 64 MiB body ceilings down to the provider's declared limits.
The reader validates lengths, duplicate-aware bounded JSON, a locally pending
request ID and the expected complete body length before allocating that body.
Status and cancellation replies may precede the current chunk reply. At most
three requests are pending: one executor request plus status and cancellation.
The process owner must impose bounded I/O; a blocking generic reader is not a
watchdog or process supervisor.

`run_chunk` accepts borrowed byte views, including the `Box<[u8]>` buffers held by
the authenticated receive owner. It validates the existing `RunChunk`, approved
boundary and all lengths before writing, and stores at most eight immutable slice
views instead of copying whole boundary payloads into new vectors. Existing vector
callers and first-stage token JSON encoding remain unchanged. The process/queue
owner must retain the original input receipt through qualified completion;
borrowing bytes does not transfer cleanup ownership or authorize a native route. See the
[pure wire control](evidence/GLY-185-borrowed-range-inputs.md).

Approval binds the exact descriptor digest after excluding only request `id`,
`ok`, process `pid` and `incarnation`, plus the existing `BoundaryContract`,
workload, payload and engine-byte limits. Each descriptor/prepare/status
incarnation is checked locally. Opaque instance/session handles are never
accepted from callers or reused across a broken connection. A new process
requires a new owner/mapping. Descriptor integrity is not package attestation;
engine counters omit device context/code and cannot supply a physical pool budget.

Prepare validates checked engine counter sums/peak against the explicit local
limit; commit precedes begin. Begin binds a provider session to the caller's
complete `DataScope` and checks its fixed context and retained KV/state counters.
Run validates the independently supplied current `StepIdentity` against that
owner, exact position/sequence, unique invocation, approved context/prompt/output
limits and all ordered input layouts. The documented BF16 residual/MLP-output
pair maps to `residual` then generic `hidden`, with rank-two row-major layouts;
the first stage accepts little-endian u32 token bytes, and only the last stage
requests greedy sampling/logits. No other boundary/sampling profile is inferred.
The final prompt chunk already produces the first output token, so an approved
output limit of one permits no later decode chunk. The live authoritative
worker must reject retired generations before binding another provider session;
this adapter does not replace that registry with a second generation history.

Cancellation withdraws output before writing its request. Both a cancelled error
and a racing successful chunk are returned as withdrawn, with no published
tensor/token. End waits until all replies have been consumed. A nonzero unload
retention receipt cannot permit reuse. Neither cancel/status/end nor zero engine
bytes proves GPU/context drain or releases a Compute reservation. Any malformed,
foreign, truncated or ambiguous reply/write permanently disables this mapping;
no chunk is retried, including when the provider retains replay responses.
The provider's 256-entry reply history imposes no Compute token/session ceiling.

The [implementation evidence](evidence/GLY-185-range-wire.md) distinguishes pure
in-memory wire controls from installed/native qualification. Actual process
launch integration, live remote stage activation, numerical reference proof,
physical pool accounting and trusted native cleanup remain open.

## Proposed process and identity ownership

One local Compute owner starts one approved external process from immutable
package/configuration digests and verified pinned assets. Reuse the existing
no-shell/no-PATH-search, explicit environment, owned process group/job, resource
approval, pin and quarantine conventions. Credentials never go in argv/logs.
No process starts merely because discovery, a schema or a fixture advertises it.

The proposed local channel is private owner-created pipes; stdout is protocol
only and diagnostic output is discarded by default. A different supplied local
IPC/API needs an explicit mapping with equivalent ownership and limits. No network
listener, shared-memory transport or global broker is added by this proposal.
Control JSON uses the existing bounded duplicate-aware parser (1 MiB, depth32,
16,384 nodes), length-prefixed before allocation. Raw tensor bodies follow checked
ordered metadata and the admitted total budget; they are never JSON float arrays.
An assembled tensor may be up to64 MiB even though a network frame is at most8 MiB.
The metadata parser alone implements no transport; the delivered adapter above
provides wire mapping while process launch/supervision remains separate.

Describe returns a fresh process incarnation and exact build/runtime, model,
requested/effective level and precision, execution/boundary/state profile, device,
pool inventory, supported operations and workload limits. Compute compares these
to the approved `ProfileKey`, manifest and qualification. Pin the complete
`starlings_protocol::content_hash(boundary_contract)` established by conformance;
a matching profile name/key cannot replace ordered layouts. Unknown features
cannot silently change precision, mode or model. Fixture replies remain ineligible.

Provider-generated instance/session handles are bounded opaque IDs, never pointers,
PIDs, device addresses or network credentials. Every handle is scoped to the
process incarnation, full coordinator epoch/incarnation and immutable model/plan/
stage owner, then session/generation. A live local registry fence must still gate
native dispatch/output; the decimal epoch in DataScope alone is insufficient.
Restart invalidates handles; an old receipt cannot revive them. Requests bind
version, request ID, monotonic sequence, incarnation and operation parameters.
Exact lifecycle repeats return historical results without repeating native work;
conflicting repeats fail, and retired sequences cannot execute. No automatic retry
follows an ambiguous RunChunk/response loss. Bounded replay retirement must not
impose a token/request lifetime limit on a valid long session.

## Proposed operations and completion meaning

| Operation | Required request / result and invariant |
| --- | --- |
| Describe | Version/capability negotiation and immutable runtime identity; availability alone is not qualification. |
| InspectModel | Exact workload/key and complete `ModelCostProfile`, or explicit unavailable/unsupported. Measured basis requires verified native evidence. |
| Prepare | Approved model assets, stage range/first-last ownership, key/plan/lease, per-pool limits and boundary descriptor; return a process-scoped instance and actual allocations/readiness. No execution before commit. |
| Commit | Same prepared owner/plan/lease; whole-plan all-stage readiness is the coordinator's GLY-181 transaction. A logical acknowledgement is not allocation evidence. |
| BeginSession | Instance, client-owned session/generation, exact context/prompt/output/sampling and state profile; report actual local KV/recurrent/convolution allocation or refusal. |
| RunChunk | Session handle and complete ordered raw inputs matching `RunChunk`/trusted StepIdentity; accepted/received is distinct from completed. One invocation may mutate state once. |
| Status/completion | Independently responsive control path reports running/cancelling/completed/drained, exact handles/invocation, bounded output layouts and native timing/allocation availability. Completion events cannot confer network authority. |
| Cancel | Withdraw new output/dispatch promptly; acceptance retains active buffers. Report drain only after all device/CPU/copy work for that owner is safely complete. |
| EndSession | Reject or wait while native work is uncertain; release/reset owned request state after verified completion. Receipt/release is scoped to the exact generation. |
| Unload | After all owned sessions are drained, release the instance and report actual retained pools/caches. Parent/socket exit or an unrelated process's state is not GPU cleanup proof. |

The provider must keep cancellation/status responsive while execution is slow.
Compute's registry, lease and permission maintenance remain independent. Device
loss, crash, OOM or ambiguous disconnect fails the generation and retains native
ownership/quarantine until separately trusted cleanup. Never retry mutated state
or infer cleanup from a successful process stop. Remote receipt/execution frames
are not this local native completion path.

## Implemented boundary and chunk metadata

`BoundaryContract` contains version1, fixture, the existing immutable `ProfileKey`,
ordered inputs/outputs and output_tokens (`same_chunk` or `sampled_one`). Every
`TensorPattern` has role (`hidden`, `residual`, `token_ids`), dtype (BF16/FP16/FP32/u32)
and one-to-four dimensions: fixed1..1,048,576 or tokens. The one variable token
axis follows the existing tensor contract: last for token IDs, penultimate for
hidden/residual. Only token IDs use u32. Sampled-one outputs are token IDs; their
shape uses one output token. This declares layouts, not sampling correctness.

Materialization checks dimension products, dtype widths, each tensor's64 MiB cap
and the sum of input plus output bytes against the caller's admitted combined
payload budget. It allocates only small metadata, not declared tensor bodies.
This host byte count does not replace per-pool native weights/state/scratch/staging
profiles or prove aliasing, allocation overhead, vocabulary validity or arithmetic.

`RunChunk` contains version1, `step: StepIdentity` and ordered input TensorLayouts.
StepIdentity contains the full existing DataScope, phase, position_start and
input token_count. Each layout has role/dtype/shape/decimal bytes. Decode checks
structure, lengths/roles, fixture scopes and overflow before payload allocation;
validation then requires the exact locally owned current step and every layout
from the approved boundary. Incoming metadata cannot select that expected step.
The input list is complete; no partial tensor or missing residual may execute.

Outputs use the prepared descriptor and validated completion metadata, not a
self-selected shape from a peer. The first stage owns embeddings; the last owns
head and sampling. GLY-186 must establish exact positions/RoPE, residual meaning,
state locality and sampling/next-token return mapping for the actual provider.
The current core single-tensor start rule is not silently loosened by these types.

[The synthetic example](../examples/partial-boundary.json) is not a real model or
backend identity. `BoundaryContract::decode(..., FixtureMode::Reject)` refuses it.
RunChunk metadata likewise requires explicit fixture mode for placeholder scopes.
A valid non-fixture record still needs provider/native qualification and live
plan/session admission before actual execution.

## Contract-only fixture and remaining gates

`SyntheticStage` exposes Describe/InspectModel, prepare/commit/begin/run, explicit
synthetic completion, cancel/end/unload. It supports one session at a time,
context<=64 tokens,<=64 KiB payload capacity and<=64 retired generations. These
are declared test-fixture limits; they are not native product token limits.
It accepts only fixture-mode, identical input/output byte-echo boundaries and
always reports unavailable native costs. No kernel, model, file or network runs.

A run retains owned payloads until `finish_synthetic`; duplicate/busy invocations
cannot increment its execution counter twice. Cancel retains busy payloads until
that synthetic completion; end/unload cannot bypass it. Normal completion moves
raw buffers unchanged and advances the session once. Retained bytes are Vec payload
capacities only, not RSS/native allocation. Its Rust object is not a native handle,
GPU completion receipt, durable process owner or production capability source.

Still required: installed native artifact and process launch/negotiation
conformance; live authority integration for provider handles and retired
generations; full GLY-183 flow/
reassembly and native handoff; GLY-181 transactional allocation; GLY-186 bitwise
boundary/sampling proof; GLY-187 worker integration; GLY-235/236 actual provider
artifacts/profiles; approved real-device evidence. Neither this document nor the
fixture completes GLY-185, M2, M3 or any native/performance acceptance.

## Upstream delivery record

| Required handoff | Current boundary |
| --- | --- |
| Installable range-worker build, platform/device support, artifact/dependency identities and license notices | Await GLY-235/provider package; may be the same delivered build as GLY-238 only if range capability is explicit. |
| Actual endpoint/pipe launch, schemas, version/capability negotiation, handle/replay and lifecycle semantics | Delivered glyd-range/1 wire/session mapping implemented above; approved process launch, installed conformance and live authority/retired-generation integration remain open. |
| Complete ordered boundary/state/sampling metadata and digest | Proposed metadata here; actual provider agreement and GLY-186 equivalence remain open. |
| Actual partial preparation, committed allocation and memory per physical pool | GLY-181/187 integration plus GLY-235/236 native evidence; schema/fixture bytes cannot satisfy it. |
| Native cancellation, drain, crash/device-loss and cleanup ownership evidence | Provider completion semantics plus separately approved conformance/device tests; no receipt or process-exit shortcut. |
| Exact matched model/revision/level/backend/hardware/workload and raw numerical/cost records | GLY-186/235/236 and later same-type pipeline/benchmark qualification; use approved 2026-model workloads. |

No new installed package was executed or native API inferred during this audit.
Keep unsupported and undelivered operations explicit in Linear until the actual
bundle is mapped, tested and reviewed; no private source access is required.
