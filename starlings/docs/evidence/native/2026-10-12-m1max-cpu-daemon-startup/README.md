# Native daemon startup attempts, 12 October 2026

Native production-daemon acceptance did not pass. The original approved attempt
stopped before any daemon/model startup because its execution sandbox denied the
process snapshot required by the watchdog. A separately approved renewed attempt
ran with host permission, reached service launch, then finished without a ready
worker or generation. Both single-use approvals are consumed; their markers remain
retained outside the repository. No repeat, quarantine reset or remote work ran.

Implementation was PR [#33](https://github.com/Sigbound/glyd-compute/pull/33), head
`8d2eb777edcc56ad1eca6f9729b7e06b7da582f0`, merge
`83329931d4ca1a1a184604cba7fefcca0823538f`, with
[all five checks passing](https://github.com/Sigbound/glyd-compute/actions/runs/38317918537).
The frozen default CLI hash was
`66f58631f7544e647f2e614799c2df3478794a3b051d20d6ba95f57778883819`.
The service configuration, executable and sealed artifact were unchanged from the
[CPU qualification candidate](../2026-10-12-m1max-cpu-qualification/README.md).
Native acceptance remains open; independent review is deferred, not approved.

## Actual bounds and outcome

Both approvals allowed one attempt on the M1 Max/32 GiB, CPU only, two inference
threads, one slot, at most four eight-token requests and 12 count calls, five minutes
including cleanup and a monitored 4 GiB RSS cap. The watchdog interrupts at 280 s.
Context remained 512 allocated, 128 logically admitted. Fixed benign inputs only;
no downloads, runtime rebuild, installation, GPU, Ubuntu, Pi or paid infrastructure.

| Observation | Original attempt | Renewed attempt |
| --- | --- | --- |
| Outcome | Preflight stopped | Service launched, no ready generation worker |
| Elapsed | 0.992 s execution command | 71.758 s including verification/cleanup |
| Generation / count calls | 0 / 0 | 0 / 0 |
| Owned sampled group RSS peak | Unavailable | 57,622,528 bytes |
| Additional server process groups observed | No daemon launch reached | 1 |
| Original owned group remaining | No daemon/runtime created | None observed |
| Temporary client token / launch quarantine | Neither created | Token absent from durable database afterward; quarantine retained |
| External spend | $0 | $0 |

The first runner consumed its marker before checking process visibility, then
also failed to write a result when final observation was denied. The adjacent
first-attempt record is reconstructed from stderr and absent daemon logs/token/
quarantine. A log is opened before each daemon spawn, so their absence proves that
launch was not reached. It is not a process-absence claim based on a denied snapshot.
Readonly control CLI subprocess calls were not instrumented as a separate counter.

The renewed runner checks process visibility, loopback binding and immutable inputs
before its marker. Its non-inference preflight passed with host permission. Denied
final observation records unknown/failure instead of discarding the result. The
actual result records the startup timeout and original-group absence; it does not
preserve the service's failure status or an HTTP response. No generation/count call
was reached. The two daemons exited and the durable service pin/quarantine remained. A separate
read-only database check verifies that the temporary token no longer exists.
No cleanup/recovery qualification or general activation profile was issued.

## Readiness defect and evidence boundary

The pinned runtime's [server HTTP middleware](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-http.cpp#L308-L360)
returns 503 while loading before checking API keys. The adapter previously
classified an unauthenticated 503 as `Unapproved`; the catalog daemon treated that
error as fatal and cancelled its owned service. Earlier isolated smoke loops
retried all probe errors, hiding this distinction.

A model-free HTTP fixture reproduces the defect: its first unauthenticated request
returns 503, then ordinary authenticated readiness becomes available. Before the
fix, the targeted assertion fails with `Unapproved` instead of `Unavailable`.
The shared probe now treats only 503 as temporary unavailability, keeping readiness
false. The same startup process may be probed again within its existing deadline;
no generation or process launch is retried. A separate fixture verifies that 200
without a key still fails and cannot grant readiness. The existing daemon route
acceptance now begins with this loading response, so it exercises the supervisor.

This confirmed software defect is consistent with the native startup outcome.
The failed native record does not prove that its actual response was 503; native
requalification is still required after the fix. Do not present this failure as
successful routing, a latency benchmark or a memory qualification.

## Archive and reproduction

Adjacent records preserve both actual runners, the first reconstructed result,
renewed raw result, non-inference preflight, preparation, identities and route
records with local paths replaced by `<NATIVE_ROOT>` / `<STATE_ROOT>`. Original
digests refer to unredacted paths/ports and change when those values change. No
credential, prompt payload, generated output or weight is included.

For a separately approved run, prepare an isolated local cluster/catalog using the
recorded artifact/build identities and complete verified assets. Copy the runner
and proposed route/preparation files into an isolated native root; its sibling
`qualification-check` supplies the manifest/cache paths/identity/state records.
Build the default CLI from the selected reviewed head with locked dependencies,
freeze its hash, recompute exact configuration/record digests and review the new
fixed-input experiment. `--preflight` performs no daemon/model launch or workload
approval. `--run-approved` requires fresh explicit approval and an unused marker.
Existing quarantined catalogs and consumed markers must never be cleared to retry.

The observed first failure also motivates preserving redacted service states and
failing promptly on a terminal startup state in future experiment drivers. Native
route/cancellation acceptance, service recovery, arbitrary-prompt memory, two-worker
benchmarks and directed links remain M2 gates. M3 has not started.
