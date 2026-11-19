# GLY-262: bounded control observation in native checks

[GLY-261](https://linear.app/getglyd/issue/GLY-261) retains the unidentified
93.235s native control failure and [raw evidence](native/2026-10-14-m1max-routing-comparison/README.md).
This correction fixes demonstrated harness null handling and diagnostic loss;
it does not claim to fix that original RuntimeError.

The full model-free coordinator/worker sequence passes route activation/status,
then returns11 null resource observations before unavailable telemetry. The current
comparison lambda dereferences null and fails with TypeError. Its generic control
exception also discards the exact operation/error, and one broad worker-readiness
label covers four later operations. Direct native JSON/SSE matches all ten samples,
but no proxy generation ran; preserve raw overall failure.

Reuse the existing recovery runner's per-state200ms pacing and null-aware resource
wait. Record only allowlisted operation/role/refusal/timeout codes and return status;
never log stderr/stdout payloads, paths, tokens or arguments. Record the already
queried resource's state/sequence/age, including null. Add exact phases for route
hashing, activation, status and resource readiness. No mutation/generation retry,
deadline increase, product source/API, dependency or workflow change is introduced.
The local100ms polling probes pass; pacing is protocol compliance under variable
call latency, not a demonstrated original native failure cause.

The [comparison patch](GLY-262-run_profile.py.patch) preserves the original20-request
envelope. The [control-only patch](GLY-262-run_control.py.patch) specializes it to
one worker service start, zero generation/count calls and no inference-client token.
It removes HTTP generation/decoding helpers, requires qualified Ready plus ready
telemetry and an empty route, then preserves the same ownership/token/group cleanup
observations and retained pins/quarantine. It never reports JSON/SSE or matched
direct/proxy success. No native repeat is authorized by this software correction.

To reconstruct, copy the frozen run_profile.py to the destination filename first
(run_profile.py or run_control.py), then apply the matching patch with
`git apply --unidiff-zero`. Both reconstruct exact bytes:

- Comparison SHA-25600d867278eb06343daf1506c1b3930ea2068e1ab593146d772b0183d8fda2c4c.
- Control SHA-256f447da3193204f430ba08ba3cb5294481343db89c0faeb012dd7c31ff7b0d8cc.

The first rename-style reconstruction command failed because run_control.py did
not exist; corrected copy-to-destination preparation passes without a native run.

`python3 docs/evidence/GLY-262-check-control.py /path/to/frozen/run_profile.py /path/to/new/run_profile.py /path/to/new/run_control.py`

The standard-library check AST-extracts only call/resource functions, reproduces
old null failure and validates absent/unavailable/ready handling, independent
per-state pacing, unchanged3s subprocess budget, safe error/timeout records, one
call on failure, precise phases and zero-generation/count/token code paths.
Existing GLY-259 stream and GLY-257 pin/immutable manifest checks also pass.
Fresh control-only offline preparation and non-inference preflight pass; its
runner/CLI/configuration and protected old ownership are frozen before approval.
Original raw failures/markers remain immutable, including the newest worker pin
and service-start quarantine. No inference runtime/model is rebuilt or downloaded.

Correctness/security review traces all helpers/callers, safe diagnostics, null
observations, exact budgets/identity, no retries and cleanup uncertainty. Separate
Ponytail review reuses existing helpers/stdlib and removes unneeded generation code
from the diagnostic rather than adding a configurable execution framework.
Independent milestone review remains deferred, not approved; exact-head seven-check
CI and reviewed merge are required. M2 remains open; M3 unstarted.
