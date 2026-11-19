# GLY-259: direct SSE usage contract

[GLY-259](https://linear.app/getglyd/issue/GLY-259) tracks the comparison runner,
not a production adapter change. The corrected single-use Mac CPU comparison
was explicitly approved and executed on 14 October 2026. Implementation
`78b737495ff52fc69e05b928a460f400fcd4af70` has identical Rust/crate/lockfile contents
to merged PR60, `15716a3bf9072d2f1ca3beb7fca4f789001a6959`. All seven automatic
merged-main checks passed in [run38595325161](https://github.com/Sigbound/glyd-compute/actions/runs/38595325161).

## Failed native attempt

The [unchanged raw result](GLY-259-native-attempt.json) remains failed, SHA-256
`26e4c9c02d1c6531df56b290837916867375746819633a1eef76ded8b7afd3e1`.
CLI SHA-256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`;
runner `c9af0a01c2be78ca6aaffe705239a02be87127fec647da59dde76a8ba169ef18`;
run log `511c99ff13f71b59e9521b20a35c9985c7997281e8591661dbcbb6aba90d0364`.
Frozen manifest, runtime, service and profile identities remain in the raw record
and [preparation evidence](GLY-257-native-comparison.md).

The attempt ended after14.630s with `ACCEPTANCE_TypeError` at `direct_comparison`.
Two generations were requested. One direct JSON run completed,6302ms including
idle confirmation; the SSE request did not produce a successful measurement.
No coordinator/worker stage was reached, so no direct/proxy comparison passed.
Count calls were not separately instrumented; the path upper bound is five.
Sampled aggregate owned RSS peaked at1,668,464,640bytes (100ms samples, not an
allocation guarantee). Token rows were zero, all six old ownership hashes matched,
the prepared direct pin remained and no owned group was observed afterward.
No worker service-start record was created. Spend was $0. The consumed marker and
all raw records remain immutable; no automatic native retry is permitted.

The raw `owned_server_groups_observed=0` undercounts this direct stage: its set
subtracts every tracked child PID, including the direct runtime's own group.
The full owned-group absence observation uses the unchanged unfiltered group set.
The correction excludes only coordinator/worker wrappers from the server count;
a synthetic set check proves the direct runtime and worker child remain counted.
Do not rewrite the original observation as a different measurement.

## Source cause and minimal correction

At pinned runtime source `d81235049384534c167caea52b85a694f6103d14`,
[`server-task.h`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-task.h)
defaults `include_usage` to false;
[`server-schema.cpp`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-schema.cpp)
accepts `stream_options.include_usage`, and
[`server-task.cpp`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-task.cpp)
emits the final usage event only when requested. The direct runner omitted that
option and dereferenced its absent usage object. Synthetic SSE follows that
protocol and reproduces the same TypeError without executing the native module.
Generated response text was not retained; this is source/protocol reproduction,
not a captured native response replay.

The [runner patch](GLY-259-run_profile.py.patch) requests usage for direct SSE only.
The strict proxy client schema stays unchanged: the existing production adapter
already injects the backend option in `Service::request_body`. Both stages now
reject missing/noninteger/out-of-range/incoherent usage with the safe
`USAGE_MISSING_OR_INVALID` fault. Exact prompt17/output≤8/total coherence remains;
no token estimation or fabricated usage replaces the missing result.
Corrected script SHA-256:
`6f0f6a62b78dbef11be7f3d21453a37890d6dabe2c548743f0d9faafe47a388c`.

Apply with `git apply --unidiff-zero` to a fresh copy of the frozen PR60 runner.
The consumed approved script is never edited or restarted. The new script is
unprepared and unapproved; it needs a fresh isolated preparation, frozen plan,
preflight, reviewed green merge and explicit single-use approval before inference.

## Model-free checks and review

`python3 docs/evidence/GLY-259-check-stream.py /path/to/frozen/run_profile.py /path/to/corrected/run_profile.py`

The check AST-extracts only request/decoder/generation functions. It reproduces
the old failure, passes direct and proxy usage, rejects six absent/invalid usage
cases, missing DONE and output after DONE, preserves strict proxy fields, and
checks direct/worker group accounting. The existing GLY-257 ownership/immutable
manifest check also passes against the corrected script. Patch reconstruction
must reproduce its exact bytes. No model process, HTTP request or inference retry
is executed by these checks.

Correctness/security review traces both request paths, integer/token limits,
safe fault codes, immutable attempt records, token/ownership observations and
unchanged CPU/request/time/RSS bounds. Separate Ponytail review keeps the existing
runner and standard library; no product abstraction, dependency, workflow or
deadline change. Independent milestone review remains deferred, not approved.

This does not qualify native parity, installed Glyd execution, conversion fidelity,
two-real-worker routing, GPU/Metal, allocation cleanup or any Exo metric. GLY-238's
actual provider contract and approved same-type hardware evidence remain external
M2 gates. Historical GLY-251/253 Windows initiating causes remain unproved.
M2 stays open; M3 has not started.
