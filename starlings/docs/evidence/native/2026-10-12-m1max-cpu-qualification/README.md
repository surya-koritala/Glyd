# Sealed-artifact CPU qualification experiment, 12 October 2026

One explicitly approved experiment passed on the M1 Max/32 GiB, using the exact
cached GGUF artifact and CPU runtime. The four JSON/SSE requests agreed on content,
reasoning, finish state and exact usage. Both long requests reached the admitted
128-token envelope (120 prompt plus eight output). All four completed requests
had an authenticated CPU slot-idle acknowledgement. No generated text or API
credential is recorded. Approval is consumed; no automatic rerun is authorized.

Implementation: PR [#32](https://github.com/Sigbound/glyd-compute/pull/32), reviewed
head `779cf96e9f23d2e377e1546a59e0bcff114abda2`, merge
`6ab27858a36fe70c66b8c7c0e03a40eaecdb5437`.
[All five CI checks passed](https://github.com/Sigbound/glyd-compute/actions/runs/38315560964).
This experiment calls the native backend directly. It does not exercise the new
production daemon routes. Independent review remains deferred, not approved.

## Approved scope and observations

One attempt, five minutes including loading and cleanup, watchdog deadline 280 s,
two inference/batch threads, one slot, four generations of at most eight output
tokens and at most 12 count calls. RSS was monitored at 100 ms with a 4 GiB stop
threshold. This is a sampled watchdog, not an OS allocation limit. CPU only;
no downloads, rebuilds, installations, GPU, Ubuntu, Pi or paid infrastructure.
The logical envelope is 128; the runtime allocates a 512-token context.

| Observation | Result |
| --- | --- |
| Verification, loading, requests and cleanup | 72.698 s |
| Generations / count calls | 4 / 9 |
| Short JSON / SSE | 17 prompt, 8 output each; 6,287 / 6,303 ms |
| Long JSON / SSE | 120 prompt, 8 output each; 29,144 / 29,148 ms |
| JSON/SSE semantic and usage equality | Passed for both pairs |
| Live CPU slot idle after each request | Passed |
| Loaded idle group RSS after short JSON | 1,668,956,160 bytes |
| OS child peak RSS | 1,693,384,704 bytes |
| Sampled group peak RSS | 1,693,335,552 bytes |
| Remaining original process-group member | None |
| External spend | $0 |

Timings are sequential one-run diagnostics, not comparative performance or a
latency distribution. On this Mac `ru_maxrss` reports bytes. The OS child peak
includes the runner's reaped subprocesses; it is distinct from sampled owned-group
RSS. Neither accounts for all possible mappings, allocator states or escaped
descendants. CPU slot-idle acknowledgement is request evidence, not service/GPU
cleanup qualification.

Candidate budgets add 1 GiB and round up to 64 MiB: loading 2,818,572,288 bytes;
additional request memory 1,140,850,688 bytes. Loading uses the observed OS peak;
request memory uses its positive delta over loaded idle. These are explicit
candidate policy reserves, not measured maxima for arbitrary prompts. Four fixed
requests do not establish a general allocation bound or production profile.

## Immutable identities and reproduction

Adjacent files preserve the sealed manifest, result, driver, machine/run identity,
source metadata and configuration with local paths replaced. Configuration digest
`c11fcf2921da6478d1c2ef23e1101c1edb2f4d5c236e013a8a32e961b25ac683`
refers to the original unredacted paths and port. Redaction changes its digest.
The model is imported unchanged; its manifest's `is_lossy=false` means exact import
of this GGUF artifact, not equivalence to the upstream checkpoint conversion.
The separate vision projector and MTP are excluded. No weights are redistributed.

- Manifest: `8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835`.
- Artifact: `cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`, 1,516,744,736 bytes.
- Executable: `66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398`.
- [Runtime source](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14): commit `d81235049384534c167caea52b85a694f6103d14`.
- [Artifact source](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0): revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.

Hardware, compiler, static CPU-only build flags and tensor inventory are retained
in the [earlier API record](../2026-10-12-m1max-cpu-api/README.md). The actual inventory
is 320 tensors: 187 BF16 and 133 F32, across 24 layers. The manifest's resident
weight bytes describe that inventory, not measured resident memory.

For a separately approved reproduction, check out the recorded implementation in
`work/glyd-compute`, build its CLI with the pinned toolchain and locked dependencies,
and prepare verified local assets in `work/native-check/qualification-check`.
Place `manifest.json`, `cache-paths.json` (mapping `weights.gguf` to its verified
cache object), and this driver there. Restore the local paths in the configuration
and place it at `work/native-check/service-config.json`; the driver chooses a new
loopback port, binds the sealed manifest/profile and enables `/slots`. It verifies
the exact executable and artifact hashes before spawn and records the new complete
configuration and local target digests. Changes to build/model/limits require a
new reviewed experiment, not a replay of this evidence.

Only after fresh explicit device/budget approval, run `python3 run_profile.py
--run-approved` from that isolated directory. The exclusive `started-once` marker
prevents repeats. The driver owns a new process group, serializes signalling/reaping
and terminates on every request path. Never remove a consumed marker to retry.

## Remaining qualification

No native activation record was issued and no production daemon was activated.
Base conversion fidelity, arbitrary-prompt memory behavior, service cleanup and
quarantine recovery, daemon fault/cancellation acceptance, two-worker routing,
directed-link measurements and full M2 review remain open. GPU, mixed backends,
Raspberry Pi and layer sharding are untested. GLY-175/176/177 remain In Progress;
M3 has not started.
