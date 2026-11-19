# GLY-251: fresh Windows unavailable capability follow-up

PR #56 overload-only head `e5e16653924bd082e04420aa44d43c985e4ac35d`,
run38491919219, Windows backend job115618865479 fails the existing ten-second
qualified-ready capability wait. The sample is fresh (age2492ms, sequence11), but
its state is unavailable, backends empty and control endpoint unavailable. The
record lacks the scenario mode and current worker service/contribution state.
This is not evidence of a queue-overflow failure or the token-revocation incident.

Failure-only diagnostics now identify the scenario and query the existing redacted
worker service report after the acceptance deadline has already failed. The optional
second worker is queried in its own failure branch. No success-path query, retry,
extra acceptance wait or timer change is added. The report omits prompts, keys,
paths and backend response text. Correctness review traces qualify_sample's service
state and contribution guards; those eligibility checks are not loosened.

Raw log: `work/m2-capability-windows-115618865479-failure.log`, SHA-256
`596f4e4db92bd5e61ed539f16c0c3fabc7c115c830a2ba6bb30f0957da177477`.
The preserved failed head remains failed. No specific backend, resource sampler or
OS cause is asserted. A later green run alone cannot establish that cause; GLY-251
stays open until genuine service/phase evidence supports a correction or reviewed
disposition. This is separate from completed GLY-244/246 and historical GLY-248/249.

Ponytail review reuses the existing report and failure branch rather than adding a
new telemetry path or rerun harness. Formatting and all-feature lint remain required
with warnings denied; final-head CI evidence belongs in Linear. Native/provider/
hardware qualification remains open; M3 has not started. No workload, spending,
quarantine reset, dependency or CI job change.

## Diagnostic head and setup ordering

Head `f6d145ce4373d768446fc2abb548bfae3b0538f2`, run38492840799,
Windows backend job115622214182 identifies busy-idle: startup phase complete,
failed_pins_retained, PROCESS_GROUP_UNCONFIRMED and zero request reservation.
The idle owner can leave execution when its stop/registry fence withdraws; cleanup
may obscure the earlier condition. The triggering stop/heartbeat event remains
unknown. Raw diagnostic log SHA-256:
`9fb0050e17cd663360fd9344bbda7782feaafa123674ad3186adf3e0cbc621f5`.

The fixture performed permission/token publication and duplicate-output rollback
after launching lease-fenced services. Initial administrative preparation now
finishes before service launch. This removes unrelated synchronous setup from the
initial ready-lease window, without changing live revocation/expiry tests, runtime
behavior, any deadline or failed-work ownership. Review verifies every member is
already enrolled and the sealed manifest exists before those permissions are set.
The same token publication/rollback and policy read-back assertions remain.

All eleven local lifecycle cases pass in 36.76 seconds, with formatting and
all-feature workspace/all-target lint warning-free. No specific original OS cause
or original heartbeat event is inferred from a passing reordered fixture.

## Controlled registry withdrawal

Starting from main `8806da3b9ad67ab0074d20a26f5e3392f8df1b26`, the existing
synthetic production-route harness stops its real loopback coordinator after a
fresh ready sample. No generation is requested. The unchanged six-second authority
lease expires; the worker service terminates, while the registry daemon remains
operational. The original source reports `stopped_pins_retained` rather than
`failed_pins_retained`. The controlled assertion fails in 18.00 seconds. Raw log
`work/m2-gly251-before-state.log`, SHA-256
`4974c5602347ea14ae276a9b108a45b09a0d4af40e2e0df14317d8a0965642e7`.

The shared cause is that idle execution/warm-up treated a non-Online fence as a
successful loop exit. Cleanup then chose the error using its own outcome and a
later fence state. Windows cannot confirm group absence, so its fallback obscured
the initiating event with PROCESS_GROUP_UNCONFIRMED. On Unix it could classify
registry withdrawal as a successful stop. The source now returns a failure at
the observation and retains `registry_failure` before cleanup. Both owner paths
use the same guard. An explicit local stop remains distinct.

The production test restarts the coordinator and proves authenticated heartbeat
recovery cannot revive/relabel the terminated service. Its fresh capability remains
unavailable, with no backend/control endpoint; request reservation stays zero.
The durable service quarantine and catalog assets survive, and garbage collection
releases nothing. The existing warm-up matrix separately revokes a ready worker
and verifies the initiating revoked status survives cleanup with protected assets.
These cases run within the existing lifecycle CI step, including the serialized
Windows matrix. No dependency, job, retry, timer or native workload is added.

Correctness review traces explicit drain/stop, warm-up, idle execution, backend and
resource errors, scoped handshake join, registry reconnect, cleanup and recovery.
The correction preserves failed-work ownership and does not turn process absence
into native cleanup proof. Complexity review reuses the existing guard/report and
process harness, and removes the later redundant registry-error fallback.

This proves and fixes withdrawal reporting/classification, not the triggering
heartbeat/stop event in archived head f6d145c. Its diagnostic record cannot recover
that missing event retrospectively. GLY-251's original attribution remains open;
later green CI is not substituted for that evidence. Provider/native/performance
M2 gates remain incomplete, and M3 has not started.

Local final validation: formatting; workspace/all-target Clippy in both feature
modes with warnings denied; all twelve `catalog_service_` lifecycle cases pass
in 44.99 seconds. Commands use locked offline dependencies and the pinned toolchain.
The unchanged-source reproduction command is
`cargo test -p compute-cli --all-features --test daemon catalog_service_registry_loss_survives_cleanup_and_reconnect_without_restart --locked --offline -- --exact --nocapture`.
Final regression selection is
`cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture`.
Source-head/platform CI and merge identities are recorded on GLY-251 in Linear.

Initial PR #57 source `4c4090aea8f637fb1c4b01e4df51728cf1fa5490`,
run38497368004/job115636105130 fails the pre-existing twelve-second execution
startup wait in a different phase: waiting_for_registry, with no catalog/pin,
startup phase, endpoint or initiating error. The original post-start busy-idle
condition is not reproduced. Warm-worker revocation passes. Linux x64/ARM, Mac,
Windows default and dependency checks pass; Windows backend/aggregate fail.
Raw log SHA-256:
`3db19c73f9f3b9afc493ea8f3e817f08d59d38f0358060cf765417b9f5ec4d27`.
The failed source is preserved. Startup's failure branch now reports the scenario
and existing registry status/topology outcomes after that same deadline fails.
There is no extra success-path query, admission retry, timer expansion or inferred
OS/network cause. Final-source CI remains required before merge.
