# GLY-172 / GLY-176: qualified service recovery

The [contract](../service-recovery.md) reuses catalog locking, protected atomic
files, exact route approval, process supervision and worker lifecycle. A reviewed
CPU/no-device/no-escape policy is bound before launch. Only the local owner can
persist completed cleanup; explicit recovery durably unpins before publishing
recovered state. Legacy/unqualified work stays quarantined. No new dependency,
runtime, PID signal, retry or force-reset is introduced.

Model-free checks cover wrong lease/config/policy, no-proof and legacy refusal,
interruption after unpin, permissions/write failure, idempotence and old proof
versus a new launch. The old worker closes permanently without epoch/history reset.
The daemon checks add one POSIX qualified revoke → cleanup → admin
recovery → fresh restart → offline recovery/GC scenario. Other modes retain
unqualified reservations/pins; Windows rejects this unqualified cleanup scope.
The same filtered CI step runs it as an independent test so it can overlap the
existing matrix; no duplicate process suite/workflow.

Correctness/security review traces every cleanup, GC, pin, status and lifecycle
caller. Complexity review keeps one current durable record and three necessary
phases. Storage quotas/resume/shared-shard costs are unchanged; peak backend
allocations remain explicitly unknown without separate qualification. Permission/
write fixtures are not actual disk-full or power-cut measurements.

Formatting and both warning-denying lint modes pass. The default suite passes
122 tests (one native test ignored), the development suite passes 26 (one native
relay ignored), and all 10 catalog tests pass. The three filtered daemon checks
pass in 48.07 s locally after splitting the new recovery scenario (the combined
matrix took 67.16 s). Exact commit/five CI URLs will be recorded in
Linear before merge. Native qualification needs source/build/topology review and
a separately approved bounded experiment. No existing Mac quarantine/native record
is changed. M2 In Progress, M3 unstarted; independent review deferred, not approved.

## Native attempt and control correction

PR #36 merged as `e13418f0d397c5ca131f1a87c8ab7159a2a950a8` after all five checks
passed on `419a6fdbb6e26d008d7e407ce3c4121738dbd86f`. The separately approved
[native attempt](native/2026-10-12-m1max-cpu-recovery-control/README.md) FAILED
after readiness, before generation/count calls: 29.030 s, 1,684,930,560-byte
sampled peak, no owned group remaining, $0. The exact control operation is unknown
because the original runner lacked diagnostics. The marker/drained record/pin
remain; no explicit recovery or native request qualification is claimed.

A model-free reproduction hits the eight-per-address admin admission limit on a
burst; paced reads pass. Corrected runner control pacing/diagnostics, non-null
resource readiness and cleanup failure reporting are tested without a model.
The suppressed token withdrawal was completed through a control-only coordinator
restart; final database verification has zero token rows and old ownership hashes
are unchanged. A fresh native attempt requires separate explicit approval. M2
stays open, M3 unstarted. No production admission limit is loosened.

## Verified bounded native recovery

[Renewed native acceptance](native/2026-10-12-m1max-cpu-recovery-acceptance/README.md)
completed two native requests and all revocation/recovery assertions in 43.396 s,
1,716,879,360-byte sampled peak, no owned group remaining and $0. Its unchanged
runner reports failure at the final read-only cache hash's three-second deadline.
Separate 30-second-bounded verification passed in 11.102 s with no additional
model/generation/count, confirming all assets, exact recovered receipt, no pin,
idempotence, zero token rows and six previous ownership files unchanged. The
separate acceptance record preserves the raw failure and qualifies only the
controlled CPU/no-device/no-escape recovery behavior. Future file verification
uses 30 seconds; generation/count deadlines and budgets are unchanged. No model
rerun is needed. General memory/hardware/link/placement gates keep M2 open.
