# GLY-255: observation overhead measurement

PR58 source `6b8581f26f082959ce3e89abaa0b5bfda5470b49`,
run38503724940/job115656311437, succeeds in 9m45s inside its unchanged ten-minute
cap. Measured Windows stages: cache40s, fetch68s, lint56s, development tests125s,
lifecycle262s. Raw log SHA-256
`ebc146eb63464f5400cf95858b8b2349d2648dcd3cd9c3af09cdb801cea389b4`.
These identify investigation targets, not a controlled improvement across runs.
Automatic main c478297/run38504729284 also passes all seven checks.

Current lifecycle monitoring launches the actual CLI for every query. The caller
audit covers service/readiness, registry/status/resources, queue retention,
recovery and failure diagnostics. Actual CLI startup/exit, enrollment, permissions,
token/revoke/drain/stop/leave, unknown outcomes and production HTTP/worker paths
must remain tested. The first commit measured these paths before replacing polling.

The existing strict three-field Reply and its methods move unchanged to shared
AdminReply; the binary retains its Reply alias. This lets the harness use the same
serializer/unknown-field validation rather than duplicate it. The new model-free
coordinator check records five paired CLI/shared-IPC status latencies with
alternating order. Each path has equal 220ms pacing, excluded from timing. Response
code and role/node/cluster/enrollment/inference/registry/epoch must match; changing
connection counts are not mistaken for stable state. Shared IPC still uses the
production challenge/proof/capability/file-check/record/deadline implementation.

Final local paired record (microseconds), macOS/aarch64:

| Pair | CLI | Shared IPC |
| --- | ---: | ---: |
| 1 | 66091 | 35317 |
| 2 | 61726 | 32749 |
| 3 | 59734 | 49384 |
| 4 | 50233 | 37079 |
| 5 | 55351 | 45947 |

CLI median59.734ms; shared IPC37.079ms. Ranges overlap and the comparison says
nothing about Windows or Exo/native inference. Test passes in3.75s. Raw
`work/m2-gly255-paired-visible.log`, SHA-256
`c0b439cee6d01afe6ac9d867be9a38d7a02ac866114e599f468fc438106e6b20`.
The bounded OS/architecture/raw-timing record goes directly to stdout so successful
ordinary CI preserves it; no paths, IDs, prompts or credentials are logged.

Command:
`cargo test -p compute-cli --all-features --test daemon catalog_service_observation_paths_match_and_record_paired_latency --locked --offline -- --exact`.
Formatting and both warning-denied workspace/all-target lint modes pass. The
existing CI filter selects this check; no job, dependency, flag or timer changes.
Exact source/PR/CI identities belong in Linear. Windows same-fixture measurement
was required before a beneficial monitoring change.

Windows source `c0999caf872e01250b3a292c6dfb7a408cfd1bae`,
run38506079417/job115663782710 passes with these same-fixture pairs (microseconds):

| Pair | CLI | Shared IPC |
| --- | ---: | ---: |
| 1 | 80208 | 28380 |
| 2 | 66035 | 23661 |
| 3 | 65482 | 34255 |
| 4 | 54711 | 26549 |
| 5 | 65263 | 36230 |

CLI median65.482ms versus IPC28.380ms; every pair has matching stable fields/code
and lower IPC query latency. Raw `work/m2-gly255-measurement-windows.log`, SHA-256
`994266b43ecdcf6cb7f15f282347c712826050bb10abf134f3ca5903099e426f`.
The overall source failed Mac because the registry-loss fixture gracefully
announced drain (GLY-256); it remains failed, not rerun or merged unchanged.

The implementation uses shared IPC only for repeat observations in eight backend
waits: execution readiness, primary/secondary resources readiness, registry-loss
service withdrawal, queue fill/disconnection/reservation release, and execution
failure. Each wait's first query remains a real CLI process. Standalone queries,
warm-up/M1 monitoring, startup/exit, mutations, permissions, ambiguous outcomes
and production HTTP/worker execution remain unchanged. The fixture helper accepts
only status/service/resources, strictly decodes AdminReply, and preserves 220ms
pacing and the production authentication/file checks/deadline. No assertion,
timer, workflow, job or dependency changes. Paired query latency does not establish
an overall CI speedup; final platform outcomes belong in Linear.

Correctness/security review traces query serialization, reply alias/strict fields,
native CLI lifecycle and authenticated IPC, stable/dynamic status, pacing and stop.
Ponytail review reuses the existing reply, fixture, SDK and clocks; no profiler,
monitoring framework or adaptive runtime policy. Full M2 qualification remains
open and M3 has not started. Independent review deferred, not approved. No native/
model/GPU/PC action, downloads/spend, settings/license or quarantine reset.
