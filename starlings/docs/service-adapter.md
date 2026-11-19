# External whole-request service boundary

`starlings_adapters::service` starts a separately installed local service only from an
operator-approved `ServiceConfig` digest. It checks an absolute canonical executable,
its SHA-256, canonical working directory, arguments, explicit environment, loopback
endpoint, declared backend version, manifest/profile and model alias. It never
uses a shell, searches PATH, installs a runtime or changes precision. The caller
supplies a fresh cryptographically random alphanumeric API key (at least 32 bytes);
the key goes through the named environment variable and HTTP authorization, never
argv, debug output or returned errors. Child stdout/stderr are discarded. Environment
inheritance is disabled, including provider credentials and proxies.

One service owns one request at a time. Readiness requires refusal without a key,
a successful `/health`, and exactly one matching model from authenticated
`/v1/models`. HTTP redirects and
proxies are disabled. Requests use the fixed `/v1/chat/completions` route, strict
text-only input and resolved sampling. Streaming requests require final usage.
Content and `reasoning_content` stay separate; tools/media, mismatched model or
response identity, unsupported finish reasons and invalid usage fail. No request
is retried automatically. The caller's output callback applies live authorization,
lease fences and backpressure. Confirmed Glyd request cancellation may retain the
service; other output failures stop it as described below.

A 503 response to the unauthenticated model probe means temporary unavailability,
never readiness or evidence of authentication. Some backends apply their loading
gate before API-key validation. The same bounded startup owner may probe again;
once ready, 401/403 refusal, authenticated health and the exact model still must
all pass. A 200 response without the key remains a fatal configuration refusal.

Limits: 128 arguments, 32 explicit environment entries, 64 KiB configuration and
SSE events, 16 KiB response headers, 16 MiB streamed output, 1 MiB ordinary response,
an approved 1–600 second request timeout and a shared two-second readiness HTTP budget. SSE requires
a terminal finish reason, usage and `[DONE]`; EOF is failure. A cloneable cancellation
handle signals the owned process group (Unix) or job (Windows), including during a
blocked HTTP read. Stop observes parent exit with a two-second bound. Failure leaves
the caller responsible for retaining reservations; it must never infer backend/GPU
drain solely from a successful stop call.

## Trust and qualification

The daemon integrates these library APIs through [explicitly qualified CPU
request routing](production-routing.md). The external service is operator-approved;
the coordinator exposes a separately authenticated loopback client route. Caller obligations:
authorize and pin the exact catalog assets, enforce measured memory/context limits,
keep permission/lease maintenance running, observe client disconnects, invoke
cancellation and retain pins/reservations until trusted native drain. The adapter
has no model tokenizer or backend attestation. Version/profile are declared in the
approved configuration; model listing proves reachability only. Executable SHA-256
does not attest dynamic libraries, Python packages or loaded weights. Keep executables,
configuration and runtime dependencies in operator-controlled directories; hostile
same-user replacement and escaped daemon descendants are outside this boundary.
No service is automatically advertised as an eligible inference backend.

## Explicit glyd-serve/1 request mapping

The merged provider [PR168](https://github.com/Sigbound/glyd-gpu-private/pull/168)
documents `glyd-serve/1` at revision `8552912684394e49566564f49902997299be15e0`.
An operator-approved `ServiceConfig` may explicitly select `"api":"glyd-serve/1"`.
The adapter maps counting, JSON/SSE reasoning, approved identity, owned request
cancellation and allocator receipts; none grants installed/native qualification.

With no `api`, the existing `/v1/chat/completions/input_tokens` contract remains.
The default is omitted when serializing, preserving old configuration digests.
Selecting Glyd changes the complete digest and requires fresh configuration approval.
It posts the exact generation body to `/v1/glyd/tokenize` and strictly reads
`prompt_tokens` and `max_model_len`. Counts remain bounded; prompt plus requested
output must fit the returned context using checked arithmetic. There is no alternate
endpoint retry or byte estimate. The next generation still binds to the exact
counted body and must report matching prompt usage.

The provider duplicates reasoning in `reasoning` and `reasoning_content`.
Only the explicitly selected Glyd dialect removes that alias, after requiring it
to match the canonical string/null field exactly. Missing/conflicting/ill-typed aliases fail;
the canonical content/reasoning and usage still pass the existing strict checks.
JSON and emitted SSE records contain reasoning once. Legacy parsing stays strict,
and unknown message fields, tools, identity changes and incomplete/error streams
remain refused. A separate final usage event with empty choices remains valid.

The current qualified daemon route accepts only its existing CPU slot-drain
protocol and refuses this new dialect. Physical GPU pools and qualified admission
remain GLY-173/176 work. The existing `/props` option remains the earlier properties
contract. No package, device or model is approved or launched
by selecting an API string; native testing still requires the pinned package,
assets and separate workload approval.

### Approved Glyd identity and request cancellation

The Glyd dialect requires `glyd_identity` in the approved configuration and
refuses legacy `properties`. It compares authenticated `/v1/glyd/info` with the
exact approved build version/commit/development flag, model alias/repositories/
revisions, requested/effective level, lossless flag, all reported precisions,
context and concurrency (one). Nullable identity must be reported explicitly;
unknown values remain unknown. `request_fields` is the existing canonical SHA-256
of the complete reported `fields` object (`read`, `no_effect`, `refused` arrays).
Any change needs configuration review and a new digest. Ready/no-error/zero-active
request status is required at each readiness probe; token counting must report
that same context. Paths/file lists and memory counters are not retained in the
identity projection. The already approved model alias may itself be a local path.
No chat-template capability, package/asset attestation or native allocation is
inferred from this self-reported receipt. Legacy configurations omit the new
field and preserve their existing digests.

Counting creates a fresh UUID for the request; the identical `request_id` is
included in counting and generation. Responses and status must use its exact
`chatcmpl-` ID. After counting, `request_cancellation()` returns a handle bound to
that ID and owned process. A retained handle cannot cancel its successor. A cancel
before dispatch prevents generation. During generation, request cancellation
withdraws output, posts only to that owned ID, and confirms `ended` through status
within a shared two-second HTTP budget. Repeated calls share cancellation intent.
An independently observed ended status cannot settle an outstanding cancellation
operation. Reuse waits until that operation finishes; a late failure withdraws
the owned process before publishing settlement, so it cannot stop a successor.
Missing, malformed, wrong-ID, failed or unconfirmed completion stops the owned
process; no ambiguous execution is retried. Permanent `cancellation()` and `stop()`
keep their existing process-group/job shutdown semantics.

Successful JSON and the at-most-two terminal SSE records are held until status
reports normal completion and fresh info confirms the same ready, idle service
with coherent allocator counters. `how: cancelled` cannot become success because its
stream finished with `stop`. Confirmed request cancellation can keep the loaded
service alive, but is not a native drain receipt and releases no reservations or
catalog quarantine. The CPU-only qualification and legacy `/slots` check refuse
the Glyd dialect. The shared worker relay selects the owned request handle on
request-channel failure; daemon/native GPU eligibility remains unqualified.

### Allocator receipts and retained ownership

Readiness, generation dispatch and completion consume the delivered `memory`
object from authenticated `/v1/glyd/info`. The getter `glyd_memory()` exposes only
a validated sample received within the existing five-second telemetry TTL.
An identical timestamp cannot renew that age, but coherent same-millisecond changes
remain valid because provider times are not unique capture IDs. Backward time/peak
counters, changed process-start time, GPU ordinal or total
memory fail. A new owned service starts a new history. Completion and confirmed
cancellation recheck the approved identity, ready/no-error/zero-active status and
memory before terminal output or service reuse. Each additional info call has a
two-second HTTP bound; an unknown or inconsistent result stops the service.

Allocator components must be present and either unsigned bytes or explicit null.
Known parts cannot exceed `engine_bytes`; a complete breakdown must equal it.
The lifetime peak cannot be below current buffers or above the reported device
total; free device memory cannot exceed that total. Times and declared exclusions
are bounded and checked. Lossless-engine null parts stay unknown, never zero.

These are self-reported process allocator observations, not physical pool IDs,
per-process driver usage, loading/request budgets or native cleanup evidence.
The provider excludes CUDA context/code, cuBLAS workspace and host memory; device
free/total cover the entire GPU, including other processes. KV and recurrent state
remain allocated and reusable after an ended request. No allocation is subtracted
on completion/cancel, and no reservation, catalog pin or quarantine is released.
Qualified GPU admission still requires separately approved physical-device/pool,
package/dependency/assets, overhead, budget and native completion/cleanup evidence.

The public [Glyd CLI](https://github.com/surya-koritala/Glyd/blob/main/bindings/python/glyd/cli.py)
delegates serving to an installed GPU package. Its public wrapper alone does not
prove offline model identity, precision or this exact readiness/stream/drain contract.
Do not claim a qualified Glyd native/lossless path until its installed version and
native direct/proxied records satisfy those gates. Private code was not copied.

## Optional installed-service properties check

An approved configuration may add `"properties":{"build_info":"b1-d812350",
"context_tokens":512}` for the pinned llama.cpp-compatible properties contract.
These example values describe expectations, not a measured deployment. Omission
preserves the existing serialized configuration and its digest. Adding or changing
expectations requires approving the complete configuration and any bound route;
existing qualification records and workload approvals do not change automatically.

When enabled, every readiness attempt additionally requires refusal without a key
and authenticated JSON from `GET /props`. All readiness operations share one
monotonic two-second budget, with only the remaining duration given to each call;
adding metadata checks cannot multiply the attempt's I/O timeout. Expiry refuses
readiness publication. The existing loading retry policy remains unchanged.
Each operation retains its
16 KiB header limit, no redirects/proxy and a 64 KiB body limit. Duplicate JSON
keys are rejected. Build information, model alias and per-slot context must match
the approved configuration; exactly one slot and system/text content support are
required. A route cannot promise more context than the inspected expectation.
Missing, malformed or mismatching properties refuse readiness. Backends without
this contract retain explicitly declared settings and no observed properties.

The [pinned server properties contract](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/README.md#get-props-get-server-global-properties)
describes `build_info`, per-slot `n_ctx`, the model alias and chat-template capability
flags. Diagnostics retain only the matched build/context and three template flags;
paths, templates and unknown fields are discarded. Daemon status distinguishes
`backend_version_declared` and `execution_profile_declared` from
`startup_properties`, a capture taken at startup rather than live attestation.
Later failed probes withdraw readiness; the historical startup capture can remain
visible beside the failed state.

While loading, `startup_phase` names the current static catalog/validation or
spawn/probe step and
`startup_error` retains the last failed probe until a successful readiness attempt
clears it. This exposes a stalled step at the existing fixture deadline without
logging response bodies or widening startup, request or admin timeouts.

After readiness, registry withdrawal is a service failure. `error=REGISTRY_LOST`
and `registry_failure` retain the observed `offline`, `draining` or `revoked`
status before process cleanup. `process_group_absent` and `cleanup_proven` remain
separate observations; unconfirmed Windows cleanup cannot replace the initiating
registry error. Reconnection does not revive or relabel a terminated service.
Explicit local stop/drain does not invent a registry failure. No expiry, output,
reservation or quarantine rule changes.

Before probing, static startup phases distinguish configuration digest/path
checks, executable hashing, key validation/generation and process creation.
Development fixtures additionally expose temporary-directory and endpoint-report
steps. The [Windows investigation disposition](evidence/GLY-244-246-windows-disposition.md)
records the remaining historical cause gaps; these phases do not preempt OS calls
or turn a successful CI run into proof of an earlier cause.

Properties report what the service says. They do not attest loaded weights,
dynamic libraries, numerical precision, allocation budgets, layer execution or
Glyd native/lossless support. Template capabilities do not add tools or media to
the adapter's text-only request contract. Generation qualification still requires
the independently reviewed model/build/profile and real workload evidence.

## Native cancellation smoke

The explicit `service_check CONFIG DIGEST --cancel-after-text` mode performs
one eight-token streaming request and cancels on the first nonempty content or
reasoning event whose finish reason is still null. A role-only event or completed
response cannot satisfy the check. It requires cancellation to propagate and
parent exit within three seconds, and records identities/timing without content.
Use a separately approved external watchdog to bound the whole native run and
verify owned process-group cleanup. This mode does not establish GPU drain,
allocation budgets or the client/coordinator/worker cancellation path. CI runs
only its synthetic predicate check and the existing fixture-process cancellation
regression; it never runs a native backend.

## Raspberry Pi CPU candidate, not a supported hardware claim

Candidate source: [llama.cpp v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0),
resolved annotated tag commit `d81235049384534c167caea52b85a694f6103d14` (MIT).
Build separately for 64-bit Linux CPU with GPU backends disabled; record the compiler,
CPU flags, linked libraries, resulting executable digest and license inventory.
No binary was built or installed here. CPU service execution never implies layers.

Candidate model metadata read on 2026-10-12:

- Base [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), revision `2fc06364715b967f1860aea9cf38778875588b17`, Hub createdAt `2026-02-28T23:57:01.000Z`, declared Apache-2.0.
- [GGUF conversion](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0), revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`, createdAt `2026-03-01T14:38:35.000Z`, declared Apache-2.0.
- `Qwen3.5-0.8B-BF16.gguf`: provider-reported 1,516,744,736 bytes, SHA-256 `cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`. This is artifact metadata, not measured resident memory or throughput.

Use the BF16 artifact explicitly, without silent quantization. Text-only excludes
the separate vision projector; catalog inventory must declare that exclusion. The
embedded tokenizer/template and tensor completeness still require artifact inspection
and hashes before a production manifest can be registered; these metadata alone do
not establish bitwise conversion fidelity or BF16 execution arithmetic. No weights
were downloaded. License texts/transitive notices remain GLY-205's distribution gate.

The upstream [server contract](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/README.md)
documents loopback binding, environment API keys, model paths and reasoning format.
A qualified CPU launch must use an approved local GGUF path, explicit zero GPU layers,
one slot, measured context/memory limits and the embedded pinned template. Disable
web UI, tools, model autoload and remote fetching. The library does not synthesize
or approve that launch configuration for the operator. Hardware RAM/OS, native
readiness/features, usage semantics, cancellation/tree cleanup and direct/proxied
conformance remain unmeasured. GLY-210 retains physical Pi acceptance.

## Explicit native API smoke harness

`cargo run -p starlings-adapters --example service_check --locked -- CONFIG_JSON CONFIG_DIGEST`
starts only the exact approved service configuration. Routine CI compiles this
example but never runs it. Obtain device and workload approval first. The harness
does not install a runtime or download weights. Choose a short request timeout in
the configuration and an external runtime cap for the approved job.

The harness creates a fresh secret, waits at most approximately one minute for
readiness (plus an in-progress bounded probe), then sends one direct, one adapted
nonstreaming and one adapted SSE request. Each uses the same fixed benign prompt,
seed, zero temperature and eight-token output limit. It requires exact content,
separate reasoning, finish reason and usage equality. IDs and timestamps are
excluded from that comparison. A mismatch fails; it does not relax the check or
retry generation. Parent process exit is observed before success is reported.

Output is one redacted JSON smoke record with declared artifact identities,
configuration/executable digests, platform, timings and usage. It contains no
generated content or API key. Timings include request overhead and are diagnostic,
not a throughput benchmark. `device_drain_qualified` and `memory_qualified` remain
false: this check alone cannot qualify a worker, establish tensor fidelity or
prove native device cleanup. Record dependency/build identities and hardware
inventory separately. GLY-175/176/177 and physical Pi acceptance remain open.

## Catalog-backed daemon warm-up

An enrolled worker can now explicitly load an approved service under daemon
ownership. This is a qualification preparation path: it opens no generation
listener and never advertises a qualified replica.

1. Register, approve and import the complete manifest using the model commands.
2. `model paths DIGEST` verifies every asset and prints its canonical cache path.
   Put each weights path in its own complete `ServiceConfig.arguments` entry.
   Other arguments, runtime dependencies and embedded tokenizer/template behavior
   must be audited for the particular backend. Merely mentioning a weights path
   does not attest that the executable consumes it.
3. Save the existing `ServiceConfig` JSON in a private directory. The daemon CLI
   accepts up to 16 KiB. `model service-digest PRIVATE_CONFIG` prints its canonical
   configuration digest without launching or approving it. Inspect the exact
   executable, environment, arguments, precision, device and limits before launch.
4. After explicit device/workload approval, run
   `daemon --service PRIVATE_CONFIG --approve CONFIG_DIGEST` on the worker.
   The configuration is not persisted as permission to autostart on later runs.
5. `cluster service` reports loading, `ready_unqualified`, failure and cleanup.
   `cluster drain`, `stop`, `leave` and registry authority loss stop the service.
   Heartbeats and local admin remain on their own control paths.

Validation requires the approved executable/configuration digest, exact manifest
profile, complete verified cache inventory and every weights asset bound in argv.
After catalog ownership checks, the full configuration digest is approved before
manifest/profile/asset checks. Invalid bindings refuse before executable hashing;
valid bindings still require complete configuration and executable verification
before pinning or spawning. Static phases identify catalog open/ownership,
configuration approval, manifest approval, profile/asset binding and the existing
configuration/path/executable checks. They identify the attempted operation and
publish no path, credential or model content. No status lock spans host I/O.
A service-lifetime pin and `service-start.json` are durable before process launch.
The service thread retains exclusive catalog ownership while running. A failed
load or backend crash leaves the daemon available for inspection; it does not
retry the process, approve a profile, or change precision. One load has a 60-second
readiness deadline plus its in-progress bounded HTTP probe. Stop waits for ongoing
local file hashing before the thread can join; admin acknowledgement remains
asynchronous. There is no hard bound for a stalled filesystem operation.

Unqualified launches retain the pin and launch record after stop, including
observed POSIX group absence. Group absence alone cannot prove escaped-child or
device cleanup. Windows reports group confirmation unavailable. A surviving
launch record refuses another launch, protects its manifest from GC even if an
ordinary pin file is missing, and prevents `model unpin` from removing the service
pin. There is deliberately no automatic quarantine reset based on PID absence,
process-group absence or a completed HTTP response. Preserve these records for
qualification/recovery; do not delete them to force a restart. Qualified cleanup
has a [policy-gated recovery path](service-recovery.md); native qualification remains
an M2 gate, alongside measured memory admission, native catalog conformance,
installed Glyd capability delivery and matched real-worker qualification.
Daemon generation routing is implemented; its synthetic checks do not satisfy
those native gates.

CI uses a feature-gated synthetic child with its own bound ephemeral port; it
executes no model. Production builds have no synthetic launch path. Existing
[native relay evidence](evidence/native/2026-10-12-m1max-cpu-relay/README.md) covers
the shared relay library, not this daemon integration.
