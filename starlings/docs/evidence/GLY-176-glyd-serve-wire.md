# GLY-176: explicit glyd-serve/1 request mapping

Issue: [GLY-176](https://linear.app/getglyd/issue/GLY-176).
Provider contract: merged [backend PR168](https://github.com/Sigbound/glyd-gpu-private/pull/168),
revision `8552912684394e49566564f49902997299be15e0`, `docs/serve-api.md`.
Only the interface documentation was consumed; no engine implementation was copied.

Reuse the existing approved service configuration, bounded HTTP/JSON/SSE readers,
request-body hash and output validator. An omitted default API preserves legacy
serialization/digests; explicit `glyd-serve/1` selects its count path/strict response
and validates duplicate reasoning aliases before emitting one canonical field.
No inference runtime, dependency, workflow or extra CI job is added.

One pure in-process check covers old digest preservation, opt-in approval changes,
unknown protocols, count/context boundaries and overflow, strict response fields,
matching/mismatching/missing reasoning aliases, JSON/SSE and a usage-only final
event. It uses JSON and non-opened paths only: no socket, child process, device,
model or backend. Existing count/body/usage process-fixture coverage is extended
to both dialects in normal CI, including the exact endpoint and emitted reasoning.
Local native/service-process tests are not run.

Review adds malformed root/choice/message cases to the same pure check. They
reproduce a draft mutable-JSON-indexing panic; checked object access corrects the
shared normalizer without accepting malformed shapes. The corrected check passes
in 0.01s after a cached/offline two-job compile of 1.05s. Both workspace/all-target
warning-denied lint modes pass in 2.17s and 2.41s; formatting/diff checks pass.
Exact-source and separate main CI results are recorded in Linear after execution.
No native result is inferred from these controls.

Separate complexity review keeps one small explicit dialect enum and shares the
existing transport, count binding and strict validator. Correctness review retains
context/usage/identity limits, rejects ambiguous aliases rather than ignoring them,
and leaves legacy output bytes/configuration semantics unchanged. Current CPU-only
route validation rejects the Glyd dialect: no GPU eligibility, metadata attestation
or drain proof is introduced. Provider package/identity, memory/completion mapping,
native qualification and independent milestone/release review remain open.
