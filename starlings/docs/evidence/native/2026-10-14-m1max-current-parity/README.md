# Current CPU direct/proxy semantic comparison passed

[GLY-177](https://linear.app/getglyd/issue/GLY-177) and
[GLY-178](https://linear.app/getglyd/issue/GLY-178) native parity evidence.
The separately approved single-use comparison on 14 October 2026 passed in
158.352 seconds: ten direct and ten coordinator/worker requests, five JSON/five SSE
per path. All content, reasoning, finish reason and usage values matched exactly
in memory. Prompt17, output≤8 and coherent total checks passed; direct idle, proxy
trusted drain and zero retained requests/body bytes passed. No generated text or
response bodies were recorded. [Raw result](daemon-result.json) and
[run log](approved-run.log) remain unchanged.

Two sequential CPU runtime starts used the same cached model/runtime, precision,
two inference/batch threads and fixed input/sampling. The one-host Mac M1 Max/32GiB
inventory is [recorded](machine-inventory.json); the sandbox refused hw.model, then
a read-only host-permission inventory retry succeeded without a native repeat.
Sampled aggregate owned RSS peaked at1,718,009,856bytes at100ms intervals, below the
monitored4GiB cap; sampling is not an allocation guarantee. The20 requests have an
output cap160, not an observed output-token total; ≤41 count calls is the code-path
bound, not independently instrumented. Spend was $0. Two native server groups were
observed, none remained, token rows were zero, all31 protected prior ownership
hashes matched, and all pins/quarantine files remain retained. Group absence is
not allocation cleanup proof.

| Format | Direct median elapsed | Proxy median elapsed | Observed median difference | Samples per path |
| --- | --- | --- | --- | --- |
| JSON | 6245 ms | 6404 ms | 159 ms | 5 |
| SSE | 6234 ms | 6400 ms | 166 ms | 5 |

[Derived summary](derived-elapsed-summary.json) retains all source timings and raw
result digest. These times include generation, decoding and idle/trusted-drain
confirmation, with fixed stage order and a cold first request. They do not isolate
causal overhead or measure throughput, TTFT or TPOT. No Exo or cluster result is claimed.

The [approval](user-approval.json) binds the [plan](approved-plan.md). Its prerequisite
was satisfied by [PR63's actual merge](reviewed-merge.json),
c5d883ffc7bdfdce8b2c1377ed4f80ca0d241dd9, after all seven exact-source da03693
checks in [CI38613425008](https://github.com/Sigbound/glyd-compute/actions/runs/38613425008)
passed. The [marker](started-once) is consumed; no automatic repeat occurred.
[Preparation](preparation.json) keeps historical false preapproval flags;
[worker](worker-route.proposed.json), [template](coordinator-route.template.json),
[activated route](activated-route.json), [UUID identities](identities.json),
[pin receipt](direct-pin.json) and [preflight](preflight-result.json) bind the run.

The unchanged runner is reconstructible from the prior frozen comparison runner
using [GLY262's comparison patch](../../GLY-262-run_profile.py.patch): copy the frozen
runner to run_profile.py first and apply with git apply --unidiff-zero.
SHA25600d867278eb06343daf1506c1b3930ea2068e1ab593146d772b0183d8fda2c4c.
CLI46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c;
implementation78b737495ff52fc69e05b928a460f400fcd4af70 matches production
Rust/manifests/lockfile at mergedc5d883f (only daemon tests changed afterward).
The later GLY263 correction adds development-only startup diagnostics and test-lock
recovery; this native record remains pinned to its actual implementation/binary.
Runtime/model/configuration identities and exact budgets are in the approved plan.
The native resource history safely waits through unavailable sequence11 telemetry
until ready sequence12; initial daemon-status Unavailable observations and a
transient loading probe are retained, with no terminal fault.

All earlier failed raw attempts remain failed and immutable. This successful fresh
sequence does not identify the original discarded control error; GLY261 attribution
remains open. It establishes this bounded CPU semantic parity, not installed Glyd
integration, conversion fidelity, general allocation/cleanup qualification,
two-real-worker scaling, GPU/Metal support or an Exo win. Provider GLY238 and approved
same-type hardware/profile/two-worker gates keep M2 open; M3 has not started.
Correctness/privacy review validates approval/identity binding, hashes, actual
limits, no stored credentials/output and honest acceptance fields. Separate
complexity review reuses the existing runner and patch. The native runner was not
changed or repeated. The accompanying [GLY263 correction](../../GLY-263-windows-init.md)
changes gated diagnostics and test serialization, and applies the existing Windows
serial fixture schedule to the default CLI package. No dependency, deadline or
coverage changes. Independent milestone review is deferred,
not approved.
