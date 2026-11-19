# Native CPU API smoke, 12 October 2026

One approved local run passed in 21.135 seconds with no external infrastructure
spend. This is API smoke evidence for GLY-175/176/177/208, not M2 completion or
native worker qualification. PR #17 supplies the harness; its latest head
`4e1533eb4f90f8958e0309d9cd6aae65caaa504c` passed
[all five CI checks](https://github.com/Sigbound/glyd-compute/actions/runs/38282497543).

## Actual run

MacBookPro18,2, Apple M1 Max, 32 GiB RAM, CPU only, two inference threads and one
slot. Static llama.cpp 0.6.0 at `d81235049384534c167caea52b85a694f6103d14`, built
with Apple Clang 21.0.0 and isolated CMake 3.31.10. Metal, CUDA, Accelerate, BLAS,
OpenMP, RPC, external subprocess tools and runtime HTTPS support were disabled.
The build used two jobs within the approved thirty-minute preparation cap.

The pinned Qwen3.5-0.8B BF16 GGUF passed the provider size/hash check. Upstream's
read-only GGUF parser inspected 320 named tensors across 24 blocks: 187 BF16 and
133 F32. The embedded tokenizer metadata and chat template were hashed. This
checks the artifact's structure, not conversion fidelity against source tensors.
The separate vision projector is excluded. Source/model references are in
[the service contract](../../../service-adapter.md).

Launch used a 512-token context, explicit F32 KV, disabled warmup/prompt caching,
offline local assets, loopback authentication, no web UI/agent tools, and separate
reasoning. Three fixed requests used seed 1, zero temperature and eight output
tokens. Unauthenticated model listing was refused; health/model readiness passed.

Direct, adapted nonstreaming and SSE responses matched exactly for content,
reasoning content, finish reason and usage. Every response reported 17 prompt
tokens, 8 completion tokens and 25 total tokens; cached prompt tokens were zero.
Raw request durations were 6305, 6410 and 6511 ms respectively. These sequential,
single-sample, unoptimized-harness timings are diagnostics, not comparative
performance claims. No generated content, prompt payload or API secret is logged.

The adapter observed parent exit; the watchdog observed the owned server and
found none remaining afterward. Its sampled server RSS maximum was 1,631,520 KiB
at approximately 100 ms intervals. RSS is not a reservation/peak-allocation
profile and cannot qualify routing. CPU process exit does not prove GPU drain.

## Reproduction and limits

Build flags, tool hash, machine identity, raw redacted build logs, artifact/tensor
inventory and smoke/watchdog JSON are adjacent. `service-config.redacted.json`
replaces the local absolute directory with `<NATIVE_ROOT>`; fill it with canonical
local paths and recompute the configuration digest for another approved run.
The recorded digest identifies the original unredacted local configuration.

The configuration's declared manifest identity equals the GGUF artifact hash for
this isolated, unregistered smoke only. It is **not** a sealed production catalog
manifest. No catalog/model permission, warm replica or backend capability was
installed. Production needs a complete approved manifest, qualified tokenizer,
measured request memory, cancellation/drain evidence and daemon integration.

The unmodified upstream build emitted three configuration notices: absent ccache,
automatic ARM CPU flags, and intentionally absent embedded UI assets. There were
no compiler warnings with fatal compiler warnings enabled. The configure log's
transient `-dirty` GGML label was an index-stat observation after checkout;
the source tree is clean and the built server reports commit `d812350`.
These notices are retained, not hidden or treated as project CI warnings.

No Ubuntu, Pi, Apple GPU, NVIDIA GPU, distributed generation or layer sharding
was exercised. All those applicable gates remain open. The verified model and
isolated build were retained for later approved offline checks. No service or
temporary credential remains running.
