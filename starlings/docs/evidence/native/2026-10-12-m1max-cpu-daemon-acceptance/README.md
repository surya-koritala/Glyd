# Native coordinator/worker daemon acceptance, 12 October 2026

One explicitly approved CPU experiment passed through the actual coordinator and
worker daemons, supervised native runtime and verified catalog. It completed in
52.033 s with four bounded requests, JSON/SSE semantic and usage parity, context
refusal followed by reuse, live CPU request drain and midstream token revocation.
No owned process group remained and external spend was $0. Approval is consumed;
no automatic repeat is authorized.

Implementation: PR [#34](https://github.com/Sigbound/glyd-compute/pull/34), reviewed
head `d329e9ffcc43d3f832f479c61a748c6b051ea6ea`, merge
`bd0e0a3d652be7dde513680869a2f7a2bc55db70`.
[All five CI checks passed](https://github.com/Sigbound/glyd-compute/actions/runs/38325754492)
before the run. The corrected loading probe permits the same service to finish
loading after 503; authentication remains mandatory before readiness.

## Scope and observed results

M1 Max/32 GiB, CPU only, two inference/batch threads, one slot, 512 allocated
context and 128 admitted context. One attempt, five minutes including cleanup,
watchdog interrupt at 280 s, four requests capped at eight output tokens each,
at most 12 count calls and a monitored 4 GiB owned daemon/runtime RSS cap sampled
approximately every 100 ms. No download, runtime rebuild, install, GPU, Ubuntu,
Pi or paid infrastructure. Only fixed benign inputs; no user documents.

| Observation | Actual result |
| --- | --- |
| Entire verification/run/cleanup | 52.033 s |
| Requested generations | 4, with ≤32 total output tokens |
| Count-call bound | ≤9 from bounded code paths; backend counter not separately instrumented |
| Completed JSON / JSON / SSE | 6,420 / 6,373 / 6,396 ms |
| Midstream-revoked SSE | 3,989 ms; no DONE |
| Successful semantic/usage comparison | All three completed responses equal; prompt count 17, output ≤8, total coherent |
| Successful request reservations | Zero before next request |
| Unauthenticated / unsupported fields / output bound / context / revoked token | 401 / 400 / 400 / 400 / 401 |
| Sampled owned group RSS peak | 1,718,403,072 bytes |
| Failed reservation after revocation | 1,140,850,688 bytes retained |
| Original owned group remaining | None observed |
| Temporary token in durable database afterward | Absent |
| Current and prior launch quarantine | Both retained |
| External spend | $0 |

The fourth request is intentionally interrupted after actual content/reasoning
appears. Its failure is the expected acceptance behavior: the worker becomes
`failed_pins_retained`, withdraws eligibility, stops the backend and retains its
reservation, service pin and quarantine. Coordinator/worker administration remains
responsive. No partial result is replayed and no backend restarts. The historical
quarantined catalog remains unchanged; a fresh isolated cluster/catalog was prepared
for this run. No reset or unpin is inferred from observed group absence.

These single-run timings include request overhead, not a latency distribution,
throughput or comparative speed result. Sampled RSS is not an allocation guarantee.
Only the three daemon responses are compared in this run; a direct reference is
not generated under this approval. Earlier direct/adapter and shared relay evidence
retain their separate scopes. This is one physical Mac with loopback networking,
not two-machine pooling, mixed-backend or layer-sharded inference.

## Exact identities and policy

The fixed-input operator policy reuses the prior
[sealed-artifact CPU candidate](../2026-10-12-m1max-cpu-qualification/README.md).
The runtime, model bytes, hardware, precision and context are unchanged; the
verified cache path and enrolled node/cluster differ. Complete new record digests
were reviewed for this bounded experiment. `inference_qualified=true` in ready
status reflects that explicit local policy; it is not a general production profile.

- Default CLI: `6dd1102c98692dd880c8ccd1263479e87c92e676daa6fc6191a38c62b1ca0080`.
- Service configuration: `80229aa172edc4e1975264410c2d10f2e380a3f0fcd08349f0293a0678735846`.
- Worker record: `ef82fae64db7b6464b830e9129d42c490fd578cacd73f2fa8464c516aeb3d38a`.
- Manifest: `8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835`.
- Artifact: `cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`.
- Runtime executable: `66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398`.
- [Runtime source](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14): `d81235049384534c167caea52b85a694f6103d14`.

Asset inventory, source revision, build flags and declared licenses are in the
earlier candidate/API records. Exact GGUF import does not qualify conversion
fidelity to the upstream checkpoint. No weights or credentials are redistributed.
Service cleanup, escaped-child/GPU drain and arbitrary-prompt memory remain
unqualified; the raw record explicitly keeps those flags false.

## Archive, reproduction and remaining gates

Adjacent files contain the actual runner, raw result with redacted service states,
preflight, identities, preparation, post-run durable verification and route records
with local paths replaced. Original digests refer to unredacted paths/ports and
change when those values change. No generated content, prompt payload or key is
logged. The one-shot runner captures terminal service errors promptly; permission,
target and available-loading-memory checks precede its marker. Denied final
observation becomes unknown/failure, never a success.

For a separately approved reproduction, prepare fresh isolated local enrollment
and complete verified cache assets without altering a quarantined catalog. Copy
this runner and preparation/route/state/identity/cache-path records into an isolated
native root, build and freeze the default CLI from the selected reviewed head,
recompute complete digests and review the exact fixed-input policy. `--preflight`
starts no daemon/model and grants no approval. Only after fresh explicit approval
run `--run-approved` with host process-observation/loopback permissions and an
unused marker. Never remove a consumed marker or quarantine to repeat a test.

Native catalogue-backed JSON/SSE/revocation now has this bounded acceptance record.
Service recovery, general memory qualification, discrete/thermal policy, directed
links, actual multi-worker queues/benchmarks and full M2 review remain open. No Pi
or GPU support is inferred. M3 remains unstarted; independent review is deferred,
not approved.
