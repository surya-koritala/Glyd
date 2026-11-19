# GLY-177 / GLY-175: qualified CPU daemon routing

PR: to be attached in Linear. Base: `aa8032b` (PR #31).

## Outcome

Explicitly approved worker/coordinator records connect the daemon's HTTP admission,
serial Dispatch, authenticated generation channel, EnrolledWorker and supervised
Service. Existing catalog verification and durable pins/quarantine remain owned.
Loading, TLS handshakes, generation and registry/admin maintenance have separate
owners. The first route has one immutable model/profile and one worker/service slot.

Qualification binds the exact service configuration/executable/model/profile,
enrolled node, current OS/host target, CPU/host pool, measured loading and request
memory, context/output envelope, request timing and native evidence reference.
It requires explicit memory/count/request-drain qualification. These are reviewed
operator assertions, not remote attestations or automatically measured proof.
No native qualification record was issued in this change.

The pinned backend source sends final output before `slot.release()` (llama.cpp
`d81235049384534c167caea52b85a694f6103d14`, server-context.cpp). Qualified execution
therefore checks authenticated `/slots` state before terminal success/drain. It
requires exactly one CPU slot with `is_processing=false`, within two seconds.
An always-busy synthetic slot proves that neither JSON success nor resource release
can bypass this acknowledgement. This is not GPU/device or escaped-child proof.

Local admin exposes complete model allowlists, narrow protected token publication/
revocation and live route activation without restarting/fencing a warm cluster.
Failed token-file publication revokes the newly issued credential. The route retains
sequences through reconnects; only a new durable epoch recreates Dispatch.

## Verification and review

Pinned Rust 1.97.1, locked offline dependencies, `-D warnings`:

- `cargo fmt --all -- --check`.
- Default and all-feature workspace/all-target Clippy passed.
- Default workspace tests: 118 passed, 1 ignored native smoke.
- Development library/binary/CLI tests: 25 passed, 1 ignored native relay.
- Filtered development `catalog_service_` process tests: 2 passed in 47.41 s.

The routing process test runs normal, backend-crash, perpetually busy CPU-slot and midstream token-revocation
scenarios. It checks repeated JSON, SSE/session header, exact usage and preserved
content/reasoning; recoverable context rejection; output-limit/unsupported-field/
wrong-model/credential errors; independent admin; slow TLS handshakes; failed token
publication; and retained live runtime/reservation, one backend start and durable
quarantine after uncertain failures. Existing warm-up tests still run. CI's existing
filtered step picks up this test; no duplicate default suites or workflow/cache
changes were added.

The first regression setup restarted the coordinator after warm-up, correctly
fencing the worker. Live authenticated route activation removes that setup error.
A synthetic reasoning declaration mismatch was corrected in the fixture; strict
backend validation was preserved. The CLI help scope assertion was updated for the
new explicitly qualified route. Correctness/security and simplicity review covered
all changed callers; one shared DispatchOwner type alias reduced duplicate syntax.
No new dependency, asynchronous runtime, retry layer or general router was added.

All native tests stayed ignored. No model/GPU workload, model download, Ubuntu
access or paid infrastructure ran. Independent review remains deferred under the
authorized green-merge policy; no independent approval is claimed.

## Open M2 gates

The new [routing contract](../production-routing.md) is implemented software, not
native qualification or a completed M2 milestone. Native catalog/precision/count/
memory/request-drain proof, qualified service cleanup/recovery, directed-link
measurements, multi-worker placement/queues and real task benchmarks remain open.
Service pins/quarantine are not cleared by parent/group absence. Failure reservations
stay observable and cause incomplete drain at shutdown. M3 has not started.
