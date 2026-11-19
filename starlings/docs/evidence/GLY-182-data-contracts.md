Current node-local grant/endpoint work is recorded separately in
[stage-admission evidence](GLY-182-stage-admission.md). The PR83 authority-backed
library checkpoint below is preserved with its original scope and results.

# GLY-182 — authenticated GCP1 transport library

Issue: https://linear.app/getglyd/issue/GLY-182

Implements strict GCP1 prefixes/all header kinds, bounded opening negotiation,
persistent enrolled TLS, trusted local plan/session/lease/token/peer/fence admission,
checked receiver position/sequence and exact boundary templates. Prefix/header/
payload/final delivery share the existing10-second record cap. Failure poisons and
closes the stream. No control replay, backend execution or native drain is inferred.
The original contract checkpoint is preserved below as history.

Validation (pinned1.97.1, model-free synthetic/loopback only):

```sh
cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo test -p compute-protocol -p compute-core -p compute-security --locked --offline
RUSTFLAGS='-D warnings' cargo test -p compute-cli -p compute-adapters --all-features --lib --bins --test cli --locked --offline
RUSTFLAGS='-D warnings' cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
```

Both warning-denied workspace lint modes and formatting pass. Core22/protocol11
checks pass; security86 library checks and11 catalog checks pass (the existing
63-peer capacity test remains explicitly ignored in fast CI). Backend17, CLI
library21 (one existing native acceptance ignored), bin10 and CLI7 pass. All18
catalog daemon lifecycle checks pass67.99s. These counts include existing
regressions, not newly created tests or native qualification.

Six new transport controls cover persistent/chunked byte preservation and all
control kinds without lifecycle mutation, invalid/unadmitted/truncated frames
before payload delivery, poisoned-stream reuse refusal, token/model/node/lease/
cancel/epoch/fence withdrawal, absolute partial-record deadlines and the exact
control owner/negotiated binding. Static golden inputs cover all six frame kinds.
All BF16/FP16/FP32/u32 payload vectors cross real child-process TLS connections.
The child has only protected enrolled device keys and public trust: no authority
DB, CA private key, client token or worker runtime. Its polled socket is explicitly
restored to blocking mode before TLS, matching the existing production listener.
No deadline or assertion was relaxed to fix this harness configuration.

Raw SHA256:

- Core/protocol/security: `d24bbe7da1c61303c228e08cc0b9fdd6375e44c5fabbb6dd1a10b950e7332465`
- Backend/CLI: `2ab0096be8e1d9d5724edcb6c256f287cb01d3e1880ad0af9c435872cfb306c4`
- Full catalog lifecycle: `6ed27f0172ee8b1580eadcc13559dc8cab2e977fca01590c68cc269ae41494fd`
- Default lint: `c05690e73051c9b54c3309c78df63aea93f0c015710a00793f165b620c603a90`
- Backend lint: `071973725efbe39588d6d6e01e16083730542e2a6c7dd9c57db453635ce6373e`

Correctness/security review checked allocation-before-authorization order, scope/
peer/client ownership, live lease/epoch/incarnation withdrawal, direction, short
locked dispatch, failure poisoning and retained reservations. The final binding
comparison and deadline check occur before the enqueue callback, not after it.
No network/backend I/O is performed under the policy/runtime/fence locks.

Separate complexity review reused the TLS/policy/worker validators and extracted
one shared absolute-deadline helper for JSON and GCP1. Duplicate post-read policy
locking was removed. A node-ID getter avoids building recovery tables per frame.
No new dependency, runtime framework, retry policy, CI job, skip, warning allowance
or timeout expansion is introduced. Source CI/current-main gates remain required;
independent milestone review is deferred, not approved.

This is transport-library evidence. GLY-183 still owns aggregate credits,
metadata-rate budgets, reassembly, ordering, duplicates and ambiguous-execution
cleanup. The current admission factory is authority-backed SecuredWorker; the
production remote EnrolledWorker is still whole-request only. Node-local pipeline
control/admission and qualified partial-worker integration remain GLY-181/185/186/
187/provider235/236 gates before activation. Never copy the authority DB/CA key to
remote workers. No native model, physical device, performance or M3 milestone
qualification is claimed. See [the complete contract](../data-framing.md).

# Historical contract checkpoint

# GLY-182 — framing contract checkpoint

Issue: https://linear.app/getglyd/issue/GLY-182

This partial implementation finalizes bounded GCP1 prefixes, all five control
header variants and exact synthetic BF16/FP16/FP32/u32 golden inputs before
network code. It reuses the existing strict JSON/schema and TensorHeader checks.
It does not complete the issue: persistent mutual TLS, trusted live admission,
absolute frame deadlines, failure poisoning and actual cross-process transport
remain required. Credits/reassembly/backend execution are not implemented here.

Pinned model-free validation:

```sh
RUSTFLAGS='-D warnings' cargo test -p compute-protocol --locked --offline
RUSTFLAGS='-D warnings' cargo clippy -p compute-protocol --all-targets --locked --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

All seven existing contract checks and four new framing checks pass. The fixed
prefix golden, every kind/length limit combination, kinds0–255, schema/type
agreement, duplicate/unknown/missing keys, invalid decimal/UUID/epoch/version,
fixture refusal, tensor extent overflow/mismatch and exact payload vectors are
covered. These are parser tests, not native/device/performance qualification.
An initial test incorrectly required JSON member order preservation; it was
corrected to compare decoded fields because the wire contract does not require
canonical header order. Payload and fixed-prefix comparisons remain byte exact.

Final test log SHA256:
`8c2c9ea74979fa8f35ac86f9169b70deff9f4c47939393ee55d72615d4690132`.
Warning-denied lint log SHA256:
`38b223397c264f3b267acd5dac1fbc69fab709d46e2ee679f1d1ad40d0b5f03f`.
Local raw logs and exact source are recorded in Linear.

Correctness/security review: prefix fields are private and validated before
length-based allocation; header parsing precedes payload allocation; zero/unknown
versions/kinds and non-tensor payloads fail; schemas are embedded offline;
receipt/execution reports cannot release resources. Trusted plan/model/peer,
lease/session/fence checks must still be implemented by the transport.

Separate complexity review: existing validators and scalar types are reused;
there is no new crate, parser framework, dependency or generic transport layer.
The fixed-size prefix API directly uses standard byte conversions. No speculative
runtime, model workload or production listener is added. Full diff review and
seven-check source CI remain gates for the eventual complete transport PR.
