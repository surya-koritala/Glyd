# Native CPU coordinator/worker relay, 12 October 2026

One approved bounded run passed on the M1 Max (32 GiB), with one native CPU backend
and local authenticated coordinator/worker connections. The adapter-direct
reference, relayed JSON response and relayed SSE response matched semantic content,
reasoning, finish state and usage. Both backend prompt-count calls matched the
reported 17 prompt tokens; each response used eight completion tokens, 25 total,
and zero cached tokens. Comparison occurred in memory; no generated text or
credential is logged.

The implementation was PR #29 at `0535cc14296fdf9ee432d2da9e89acc3c3b4cbfc`, merged
as `8fcbfa057e1d07770c513d535c98d1ccab9ce310` after
[all five CI checks passed](https://github.com/Sigbound/glyd-compute/actions/runs/38301711640).
The frozen test binary SHA-256 was
`20cdbd2c3c5cdaf8195aec7127e9ab4c1283f9546281a97672283ac0d8ee5f9d`.
The retained executable and model were rehashed before launch. Unchanged hardware,
runtime build flags and model provenance are in the
[earlier native API record](../2026-10-12-m1max-cpu-api/README.md).

## Scope and measured outcomes

The approved budget was one attempt, five minutes including cleanup, two CPU
threads, three requests capped at eight output tokens each, two prompt-count calls
and two service starts. Context was 512, one slot, BF16 weights and explicit F32 KV.
No GPU, Ubuntu, Raspberry Pi, download or native runtime rebuild was used.
External spending was $0. Approval is consumed; the single-use marker is retained.

| Measurement | Observed value |
| --- | --- |
| Entire verification/run/cleanup | 23.183 seconds |
| Acceptance body | 21,422 ms |
| JSON relay request | 6,418 ms |
| SSE relay request | 6,402 ms |
| Sampled server RSS peak (approximately 100 ms intervals) | 1,630,976 KiB |
| Owned group exit after each service stop | Confirmed |
| Remaining owned server or group at watchdog exit | None |
| Test outcome / stderr | Passed / empty |

These timings are one-run diagnostics, not throughput, latency-distribution or
comparative performance benchmarks. Sampled RSS is not a peak allocation bound.
Registry heartbeats and authorization maintenance continued during inference.
The coordinator constructed authenticated-client-owned leases/sessions, counted
on the selected worker connection and relayed each request once. The worker
checked exact request/count agreement, invoked the owned service and withdrew
output on exit. The harness released its provisional reservations only after
confirming original process-group absence.

## Evidence and reproduction

Adjacent records preserve the test log (terminal blank line omitted), acceptance result, watchdog
result, exact original configuration digest and harness identity. Only local
paths in the configuration and harness metadata are replaced with `<NATIVE_ROOT>`;
the stored digest refers to the original unredacted configuration. The watchdog
source is the actual runner used. It verifies runtime/model/harness hashes,
creates `relay-check/started-once` exclusively, enforces the time/RSS sampling
limits and checks cleanup. Never remove that marker to silently retry.

For a separately approved run, prepare an isolated native root with the pinned
model, `build/bin/llama-server`, and a frozen test binary plus build metadata in
`relay-check/`. Build the test target from the recorded implementation with
`cargo test --locked -p compute-cli --all-features --lib --no-run`, record its hash,
and run the watchdog from that native root. The runner invokes only
`inference::tests::native_relay_acceptance` with its explicit configuration digest.
Native runtime rebuilds, downloads and additional generations need their own
approval. This test stays ignored in ordinary CI.

## Qualification boundary

This is single-machine loopback request distribution through shared library code.
It is not production daemon activation, two-machine pooling or single-model layer
sharding. The artifact SHA is a smoke identity, not a sealed approved catalog
manifest. The conservative test reservation is not a measured production profile.
Original-group absence does not prove escaped-descendant or GPU device drain.
Catalog, memory, device-drain and daemon-activation flags intentionally remain
false. Native fault/overload tests and real task/link benchmarks remain M2 work.
GLY-176/177 remain In Progress, and M3 has not started.
