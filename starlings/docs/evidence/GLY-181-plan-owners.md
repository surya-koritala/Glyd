# GLY-181 distinct worker ownership

The shared PlacementPlan contract previously accepted two disjoint stage ranges
on one node, including different device names. Worker::prepare already requires
exactly one local stage; the bounded planner also selects each node at most once.
This mismatch let an unsupported plan pass shared validation and fail later during
worker preparation, potentially after another worker had prepared its own stage.

A controlled two-stage fixture proves old validation accepts duplicate owners:
the new rejection assertion fails on baseline a8da85c (1failed,0.03s). The shared
semantic validation now rejects a repeated node ID in the existing coverage/owner
loop. No caller-specific patch, new wire field, schema shape, digest rule or
native limit is introduced. The worker retains its own defensive check.

After correction, all8 protocol contract controls pass0.05s (cached two-job offline
build3.19s), warning-denied protocol/all-target Clippy passes2.16s, and formatting/
diff checks pass. Captured before/after completion logs remain in the workspace;
SHA256 respectively c151c06dc5844bb058b9eb96746bbcebdb4bab0ddad214927b981d7b9cba32e9
and1826937555e5db370f684c24516d0d0fbd38c1b699854744e53d7e12a6219a62.

Separate complexity review retains one set insertion in the existing bounded
stage loop and no abstraction/dependency. Correctness review traces plan
decode/encode and control-request validation into Worker::prepare, checks original
positive/golden contracts and node-local device-name semantics, and retains the
worker's defensive cardinality check. This is implementation review, not independent
milestone approval.

The same control keeps distinct-node plans valid when device names have identical
spelling, and rejects same-node plans with either identical or different device
names. This formalizes the current one-stage-per-worker runtime, not a claim about
future multi-device native execution or a historical Windows failure cause.
Only bounded pure protocol controls run locally; source/platform qualification is
recorded in the PR/issue. No network, model, device, instrumentation, dependency,
CI job or timeout change. Independent milestone review remains open.
