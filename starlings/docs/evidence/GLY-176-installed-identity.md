# Installed CPU backend identity and support audit

The cached executable was rehashed before a model-free version query, matching
SHA-25666b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398.
The inspected [pinned argument path](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/common/arg.cpp)
exits on `--version`; the [server entry point](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server.cpp)
performs backend initialization afterward. The existing build disables all GPU
backends. The one query used an empty environment, five-second deadline and16KiB
output bound; it loaded no model, initialized no GPU and spent$0.

[Raw version output](GLY-176-installed-identity.json) reports0.6.0/build1/commit
d812350, AppleClang21.0.0.21000334 and Darwin arm64. Full source identity remains
d81235049384534c167caea52b85a694f6103d14, matching the existing build record.
This confirms the installed binary's version response, not dynamic-library/model
attestation or a numerical/allocation qualification. No source code was copied.

| Property | Evidence and exact scope |
| --- | --- |
| Executable/version | Fresh digest + version-only result above; same binary as archived CPU checks |
| Model/profile | Sealed Qwen3.5-0.8B BF16 GGUF manifest/artifact/tokenizer/template inventories; text-only, vision projector excluded; conversion fidelity remains unqualified |
| JSON/SSE/content/reasoning/usage | Archived direct/adapter/relay and actual-daemon CPU acceptance, same pinned backend and bounded128-token admitted context/eight output tokens |
| Exact count and request idle | Archived context-boundary qualification and qualified live slot-idle checks; not a general memory/precision guarantee |
| Lifecycle | Crash/cancel/revocation/drain/retained ownership software cases plus bounded native CPU recovery; legacy quarantines remain intact |
| CUDA/Metal/Glyd native/lossless/layers | Unqualified; binary/profile declaration and model listing cannot establish these paths |

The default adapter's version/profile fields remain explicit reviewed declarations.
It does not silently run another executable, download a model, switch precision,
or advertise an unsupported native/lossless route. General hardware/profile and
two-real-worker comparisons remain GLY-175/178. Missing installed Glyd capability
hooks remain external work; this CPU result cannot substitute for them. The new
read-only query consumes no prior single-use model workload approval.

Source checks: reviewed version exit before model initialization; exact cached
digest and build identity; process return0/output bound; no prompt/token/API-key
data in the result. Complexity review uses the installed executable and standard
subprocess inspection, with no new runtime/dependency or generalized probe framework.
This is evidence-only. It is not a benchmark, new qualification, publication,
license decision, independent approval or M2 completion.
