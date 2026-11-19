# GLY-194 offline latency summaries

Issue: [GLY-194](https://linear.app/getglyd/issue/GLY-194).
Base: main043d52c027963b9836f65bb5c101271ff772278f.

Reuse the validated BenchmarkRun inspector and scoreboard's sorted summary.
Reports group supplied client duration and TTFT by each outcome; timeout/error/
OOM/cancellation observations remain visible separately from success. Empty
distributions are null, not zero. The existing failed IDs/codes, file/hash checks,
fixture refusal, run status and qualification=not_assessed remain unchanged.

Both report surfaces share counts/range/median and observed nearest-rank p95/p99.
Explicit reporting floors leave p95 null below20 samples and p99 null below100;
these are not confidence guarantees. TTFT samples exclude only absent TTFT values,
so its count can differ from the outcome's duration count. There is no full-run
window, aggregate throughput, server-phase timing or baseline/native qualification
inferred from these records. Input schema is unchanged; report fields are additive.

One pure check covers fixed expected ranks at1/19/20/99/100 samples, odd/even
medians and finite extremes. The shared median now uses the standard f64 midpoint
instead of adding two values before halving, preventing finite input overflow.
The existing file-only CLI case now includes both a timeout and success, verifies
outcome separation, missing TTFT/no observations, tail nulls, artifact tampering,
unsafe paths and continued failed status/fixture refusal.

Basic Mac checks, max two build jobs and no overlapping Cargo:

- Five bounded file/in-process scoreboard cases PASS0.04s.
- Exact file-only CLI case PASS0.76s default /0.79s all features.
- Final standard-midpoint pure check PASS0.00s.
- Warning-denied workspace/all-target lints PASS2.70s default /2.67s all features.
- Formatting and diff checks pass.

No socket, daemon/service, model, network probe, instrumentation, device workload,
new dependency/download or spending. Separate complexity review reuses the existing
summary/std midpoint, with no statistics framework or replacement collector.
Correctness/security review verifies rank floors, empty handling, no failure
selection, finite midpoint and unchanged artifact/trust gates. Full exact-source
CI/current-main/review and separate post-merge verification remain required;
independent milestone review is deferred, not approved.

Full GLY-194/245 native metric sources, workload/runtime/topology receipts,
direct/proxy/replica/sharding/capacity recipes, provider delivery, approved hardware
and matched2026-model/five-paired best-working Exo evidence remain open. Neither a
compound acceptance subtask nor the full milestone is completed by this feature.
