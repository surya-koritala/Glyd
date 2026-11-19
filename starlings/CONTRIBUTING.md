# Development and evidence

Read the selected Linear issue, docs/architecture.md, docs/decisions.md and
spec/README.md before editing. Use Ponytail to guide new code from the first line,
changes and complexity review. Prefer the standard library and direct code;
use established parsers, canonicalizers and cryptography at trust boundaries.
Do not copy Exo or private Glyd implementation. Preserve the undecided license.

Work on a task branch from current main, use signed-off commits, review the full
diff, and open a PR. Require all checks on the latest head and an up-to-date branch.
Independent reviews remain part of milestone and release qualification; a specific
owner authorization may defer the independent review until after a green merge.
Record that disposition explicitly without claiming independent approval. Merge
only when authorized; do not change repository protections. Complete and
review every M0 task before starting M1. Record assumptions, changes, checks,
limitations and the exact commit/CI URL in Linear; never mark simulation as
native inference. Do not change unrelated repositories or grant a license.

Use the pinned rustup toolchain and the commands in README.md. CI runs formatting
and both feature-mode lint checks before tests. Rust warnings are errors. Keep
all four platform checks; avoid duplicate workflows, redundant builds and stale
runs. Add caches only after measured benefit. Tests must exercise meaningful
invariants and failure cases, including the separate-process simulator.

Evidence lives in docs/evidence/<task>.md with issue/PR, commit, commands, outcomes
and limitations. CI logs are the initial build/test artifacts. Fixtures in
examples/ are synthetic inputs; test assertions live beside the owning crate.
Future benchmarks use benchmarks/<run-id>/ with a machine-readable record,
immutable code/backend/model identities, hardware/network inventory and raw
redacted logs. Do not log prompts, tokens, activations or credentials by default.

No development command purchases hardware, provisions infrastructure, downloads
a model, publishes a package or starts GPU work. Obtain explicit device/budget
approval before real workloads; record the target, spending cap, runtime cap,
cleanup and actual spend. Publication needs a concrete release inventory,
license decision and approval. Until then packages remain publish=false.
