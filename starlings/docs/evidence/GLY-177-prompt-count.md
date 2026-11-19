# GLY-177 exact prompt-count admission

Use the selected service's existing chat-count extension before reserving a
generation. The worker returns the count with the exact chat digest and requires
that same prompt/count at StartRequest. The service hashes the actual backend
request body, refuses a changed next request and checks final prompt-token usage
against its count for JSON and SSE. Client response grants must authorize Infer
for the exact manifest; a different grant cannot be reused for a relay.

The pinned llama.cpp source at `d81235049384534c167caea52b85a694f6103d14`
implements `/v1/chat/completions/input_tokens` through the same chat parameter
parser/template as generation and tokenizes with both special-token options
enabled. This permits reuse instead of duplicating tokenization in Rust. Sources:
[endpoint contract](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/README.md),
[count and generation handlers](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-context.cpp).
These were inspected in the retained pinned source checkout. This extension is
not an official OpenAI endpoint. A service that lacks it is refused, not estimated.

Counting has a two-second request deadline and 4 KiB response cap, with strict
fields and an integer count in 1..=1,048,576. Count plus output must fit the locally
qualified context. Counting does not allocate a core lease or start generation;
it still uses the already loaded backend, so native testing needs workload approval.

Synthetic tests cover exact-body binding, mismatched usage in JSON/SSE, invalid
counts, worker rejection of changed prompts/counts and context overflow before
generation, and the existing full relay/cancellation cases. Both feature-mode
Clippy checks deny warnings. Formatting, adapter/relay tests and all required CI
platform/policy checks gate merge. Exact PR/head/CI evidence belongs in GLY-177.

Correctness/security review checks scope, identity, counting, replay and bounded
I/O; Ponytail review reuses the installed backend contract and existing output
validator with no new dependency. Independent review remains deferred, not
approved. No native count/parity run has occurred: installed-runtime acceptance,
catalog/profile qualification, daemon activation and M2 completion remain open.
