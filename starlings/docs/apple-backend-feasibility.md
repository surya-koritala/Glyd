# Apple backend: source feasibility, not native qualification

Issue: [GLY-191](https://linear.app/getglyd/issue/GLY-191); later conformance:
[GLY-192](https://linear.app/getglyd/issue/GLY-192). D12 stays open.
This source-only review selects no runtime, checkpoint or precision and copies no
implementation. No package/model installation, native run or kernel integration
is performed. Current [production routing](production-routing.md) remains CPU-only.

## Immutable review inputs

| Upstream | Inspected source | Commit date (UTC) |
| --- | --- | --- |
| MLX LM | [93e6d4a6447b3080861985947f0820e76fcb957d](https://github.com/ml-explore/mlx-lm/commit/93e6d4a6447b3080861985947f0820e76fcb957d) | 2026-10-16 02:48:11 |
| MLX | [b6642e604843776705b57b1eaecb6f57e519b793](https://github.com/ml-explore/mlx/commit/b6642e604843776705b57b1eaecb6f57e519b793) | 2026-10-16 09:27:21 |

These are source-review candidates, not a mutually tested wheel/runtime lock or
an approved best-working Exo configuration. Keep the separate
[Exo baseline](scoreboard.md#exo-candidate-and-collection-boundary) and its actual
build/patch records when eventually comparing matched approved 2026 workloads.

## Observed seams and Compute implications

**Whole requests.** The pinned [server](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/server.py)
offers health, model listing and text/chat completions, with JSON/SSE usage.
Its route table does not expose Compute's preflight count, qualified profile,
allocation or drain protocol. A completion handler sets a stop flag in its final
cleanup; the non-batched generation path rejects distributed stop. That is not
native completion evidence. Request handling permits model/adapter selection and
includes body logging/image fetching paths. The source itself cautions about
production security. Adoption needs a closed text-only, fixed-local-model,
authenticated/redacted mapping; HTTP similarity alone is insufficient.

**Hybrid state and execution scope.** The
[Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen3_5.py)
has distinct recurrent-array and attention-KV caches per layer and exposes its
inner pipeline. Its sanitizer excludes vision and MTP weights and can transpose
convolution parameters and adjust normalization parameters. Qualification must
record component exclusions and transformations; this cannot establish
whole-model weight exactness. Head/tied-embedding ownership, positions, cache
reset and ordered boundaries need explicit conformance. Presence of a pipeline
method is not proof of native split equivalence or a GCP1 worker contract.
In particular, a full model call is not an intermediate-range hidden-state API.

**Placement and communication.** The pinned
[loader](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/utils.py)
can resolve/download assets and selects shard files through a weight index.
Non-layer asset files are selected on every pipeline rank. File selection is not
Compute's first/last-stage resident-memory proof. Native MLX groups require an
explicit mapping to the [partial-worker contract](partial-worker-contract.md);
they are not authenticated GCP1 leases or local drain acknowledgements.
The [pipeline mixin](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/pipeline.py)
assigns the last layers to rank zero and validates positive complete splits.
Compute's first-stage ordinal must not be equated with that native rank.

**Memory.** Pinned [MLX memory documentation](https://github.com/ml-explore/mlx/blob/b6642e604843776705b57b1eaecb6f57e519b793/docs/src/python/memory_management.rst)
lists active/peak/cache accounting, limits and cache clearing. Those APIs supply
no measured values here and are not per-owner completion receipts. Actual
weights, KV/recurrent state, scratch and retained-cache observations must bind to
the exact workload/owner; shared Apple memory is one physical pool.

**Runtime and notices.** The [package metadata](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/pyproject.toml)
requires Python >=3.11, MLX >=0.32.2 on Darwin and Transformers >=5.7.0, among
other dependencies. These ranges do not pin transitive builds. Both inspected
root licenses ([MLX LM](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/LICENSE),
[MLX](https://github.com/ml-explore/mlx/blob/b6642e604843776705b57b1eaecb6f57e519b793/LICENSE))
are MIT; wheel/transitive/model notices still require review. No dependency or
license is added to Compute; its license stays undecided.

## Disposition and remaining acceptance

Engineering inference: evaluate an external MLX adapter before undertaking
own-Metal implementation; adoption remains unapproved. Keep the Rust coordinator
and existing narrow owned-process boundary. Do not expose a stock server or
translate its collectives into trusted execution/drain by configuration alone.
GPU route integration and an agreed native identity/count/capability/ownership
mapping remain actual work, not supplied by this document.

1. Prepare a pinned, reviewed runtime/environment and approved local 2026 assets,
   declaring exact precision, transformations and text-only exclusions. Bind the
   model/tokenizer/template, platform, budget and cleanup plan before execution.
2. With separate device approval, qualify one Mac task-mode run: exact counting,
   JSON/SSE, cancellation/drain, measured memory, startup and sustained thermal
   behavior. Preserve failed/unsupported results. Source presence is not a pass.
3. Before Mac sharding, agree range/state/first-last semantics and prove bitwise
   same-backend equivalence against a deterministic unsharded Apple reference.
   Investigate nondeterminism without relaxing tolerances. Mixed NVIDIA/Apple
   comparisons remain parked; two-node claims require an approved same-type pair.
4. Review measured feasibility, dependency/model rights and ownership guarantees
   before deciding adoption versus own Metal. Performance comparisons need the
   matched five-paired-run scoreboard; no result or Exo win is inferred here.

GLY-191/192 and M5 acceptance remain open. This document grants no deployment,
new download/device workload, independent review, license or publication approval.
