# Bounded CPU routing comparison: direct stage passes, full attempt fails

[GLY-178](https://linear.app/getglyd/issue/GLY-178) acceptance evidence;
[GLY-261](https://linear.app/getglyd/issue/GLY-261) retains the unidentified
post-Ready control failure. The single-use Mac CPU attempt was explicitly approved
and executed on14 October2026. [Approval](user-approval.json) binds the frozen
[plan snapshot](approved-plan.md); its preapproval wording remains historical.
The [marker](started-once) is consumed. No native retry was performed.

## Exact outcome

The unchanged [raw result](daemon-result.json) is failed, SHA-256
`95a65fb676f0b965e7c5198b20855af522b19258d318654cb2a0882d518af81e`.
The attempt ended in93.235s at `worker_readiness`, `ACCEPTANCE_RuntimeError`.
Ten direct requests completed: five JSON/five SSE, every response equal to the
in-memory reference for content, reasoning, finish and usage. Usage checks require
prompt17, completion≤8 and coherent total counts. No proxied generation occurred.
The worker reached native Ready with architecture `qwen35` and startup complete.
The broad phase label also covers later route/control operations; it does not
identify the failed command. The runner discarded that command's actual error.

[Derived direct-stage evidence](direct-stage-summary.json) records only this
partial success; raw overall JSON/SSE and direct/proxy acceptance remain false.
Direct JSON median6239ms and SSE median6251ms each have five samples. Timings
include generation, decoding and slot-idle confirmation, with a cold first request
and fixed JSON/SSE/stage ordering. No proxy overhead, TTFT, TPOT or throughput is
measured. Counts are not separately instrumented; the code-path upper bound is21.
The160-token record field is the original total cap, not an observed output count.
Generated text and response bodies were compared in memory and not recorded.

Sampled aggregate owned RSS peaked at1,685,716,992bytes (100ms sampling, not an
allocation guarantee). Two server groups were observed; none remained afterward.
Temporary token rows were zero; all protected prior ownership hashes matched.
Direct and worker pins remain; the new worker quarantine remains. Group absence
does not establish allocation cleanup qualification. Spend was $0.

## Immutable identities and limits

Frozen implementation `78b737495ff52fc69e05b928a460f400fcd4af70` has the same
production Rust/manifests/lockfile as merged PR61,
`1094af0a04d491f444b12966cfb49de9211c7517`. Its seven automatic merged-main checks
passed in [run38600862972](https://github.com/Sigbound/glyd-compute/actions/runs/38600862972);
PR source7a13323's seven-check run38599604510 also passed.
CLI SHA-256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`;
[runner](run_profile.py) `6f0f6a62b78dbef11be7f3d21453a37890d6dabe2c548743f0d9faafe47a388c`.
[Run log](approved-run.log)
`91b1c836900c7735c98ec7609973b72a7a0992623b503c9f942694dd734ad3af`.

The plan targets M1 Max/ARM64,32GiB, macOS27.0/26A428; two CPU inference/batch
threads, two sequential starts,≤20 requests,≤8 output tokens each,≤41 count-call
upper bound, five minutes including cleanup,240s watchdog, monitored4GiB RSS,
cached assets and loopback only. Actual native outcome stays within that envelope.
No GPU, Ubuntu, download, runtime rebuild, spending, recovery or old reset.

Manifest8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835;
2026 Qwen3.5-0.8B BF16/F32 GGUF artifact
cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5;
runtime66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398,
source d81235049384534c167caea52b85a694f6103d14, CPU-only build.
F32 KV, one slot, allocated context512/admitted128; caches/offload/repack/warmup
disabled, offline text-only. Conversion fidelity remains unqualified.
Servicecc6d5929c066828006fcdd893ff362a94d54c760d561f6d883884e01d903bba0;
worker recorda46c07b079532ac13e86be2f81efefc93e8564c6a33ea1e4adb0a4fb149ff3fb.
[Preparation](preparation.json), [worker](worker-route.proposed.json),
[route template](coordinator-route.template.json) and
[generated route](generated-coordinator-route.json) preserve bindings.
Preapproval `native_started/approval_granted=false` preparation fields remain
historical; the approval/marker/raw result establish actual execution.

## Model-free diagnosis and remaining uncertainty

[Read-only route diagnostic](readonly-route-diagnostic.json) establishes strict
parsing/content hashing, not route qualification. It succeeds in0.008s.
[Control route diagnostic](control-route-diagnostic.json) reaches the deliberate
foreign-cluster activation refusal after structural validation; no model starts.
[Status pacing comparison](control-rate-diagnostic.json) passes both100ms and220ms
queries. [Full control sequence](control-sequence-diagnostic.json) passes route
activation/status and12 resource queries, initially null then unavailable because
no model is started. These checks do not reproduce the native failure or establish
a rate-limit, SQLite, route or heartbeat cause.
[File-time diagnostic](phase-time-diagnostic.json) records route-write→result1.253s;
filesystem timing cannot replace the missing exact error/operation.

The current readiness lambda dereferences valid null telemetry. That separate
demonstrated harness bug and missing safe control/phase diagnostics are corrected
by [GLY-262](https://linear.app/getglyd/issue/GLY-262), reusing the existing recovery
driver's bounded pacing and null-aware readiness. A fresh one-start,
zero-generation diagnostic must be separately approved after reviewed green merge.
No old runner, result, marker or quarantine is modified to retry.

This is native direct-stage and failure evidence, not a passing direct/proxy test,
installed Glyd integration, general memory/cleanup qualification, two-worker
scaling, GPU/Metal or Exo result. GLY-238's provider contract and approved physical
inventory/profiles remain open. M2 is incomplete; M3 has not started.
Correctness/security and separate complexity review preserve these distinctions;
independent milestone review remains deferred, not approved.
