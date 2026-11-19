# GLY-176: approved service identity and owned request cancellation

Issue: [GLY-176](https://linear.app/getglyd/issue/GLY-176).
Interface source: `glyd-serve/1`, `docs/serve-api.md` in merged
[backend PR168](https://github.com/Sigbound/glyd-gpu-private/pull/168),
`8552912684394e49566564f49902997299be15e0`. Only the delivered API documentation
is consumed; no backend implementation is copied or changed.

The adapter reuses the existing configuration approval, UUID dependency, bounded
HTTP/strict JSON, canonical hashes, output validator and process supervisor.
An explicit nullable identity projection binds reported build/model/level/precision,
context/concurrency and all request-field lists. It is a checked service receipt,
not package, asset, numerical or native memory attestation. Legacy configuration
serialization remains unchanged; the new dialect requires explicit identity approval.

Each generation owns a fresh non-reused request ID, including its exact count body.
Request cancellation targets only this ID and confirms status within a two-second
HTTP budget; unconfirmed cancellation retains the forced-stop fallback. Stale
handles retain their original target. Terminal JSON/SSE success is withheld until
normal status completion; a cancelled `stop` response is refused. Global shutdown
and all authority/output, lease, reservation and quarantine fences remain intact.
The relay uses the request handle after count; GPU daemon routes remain refused.

Local checks are limited to pure JSON/configuration controls and offline cached
compilation with two Cargo jobs. The new pure check covers identity drift, missing
nullable fields, malformed capability lists, status IDs/states/times and terminal
classification. The existing pure wire check retains legacy digest/parser coverage.
Clippy runs in both workspace/all-target feature modes with warnings denied.
HTTP/process cases run only in ordinary CI: matching/mismatched identity, live
cancel, reuse with a fresh ID, stale/pre-dispatch cancellation, cancelled JSON/SSE
without success output and unconfirmed-cancel fallback. No extra job/workflow,
dependency, local listener/service, native/model/GPU/Ubuntu workload or download.

Separate complexity and correctness reviews precede the exact-source merge gate;
source and separate main CI outcomes are recorded in Linear. Native installed
package, configuration/assets, physical GPU resource/drain qualification and
independent milestone/release review remain open. No simulation qualifies M2.
