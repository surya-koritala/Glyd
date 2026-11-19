# GLY-181 offline calibration software

Issue: https://linear.app/getglyd/issue/GLY-181

The [offline report](../plan-calibration.md) reuses the bounded planner, full
candidate/profile identities, strict JSON/canonicalizer and existing artifact
reader. Supplied observations bind the complete expected placement/candidate,
metric definition and verified attempted-plan/producer receipts. It retains
failed attempts and missing phases, reporting exact error magnitudes and optional
finite relative ratios. It collects no native measurements and grants no admission
or qualification. BenchmarkRun and scoreboard formats are unchanged.

Four bounded pure/file-only controls cover matching/failure retention, altered
bindings/fixture/duplicate/error/source refusals, zero/extreme arithmetic, strict
JSON and artifact tampering. Initial control expected division by a zero fixture
prediction; the implementation correctly returned no ratio. The positive fixture
was corrected and the independent zero case retained. Final four controls pass0.09s,
all-feature CLI library/binary/test Clippy passes2.62s with warnings denied;
formatting/diff and21relative file targets pass. Source/CI identities are recorded
in the PR/issue. A CLI argument/output control runs in the
existing CI feature modes; no local CLI process/model/service/network/host
inspection or instrumented test is needed.

Separate complexity review reuses current planner and file primitives without
new dependencies, frameworks, collectors, jobs or time caps. Correctness/privacy
review checks canonical full-input/candidate/source bindings, bounds, all-outcome
retention, exact unsigned differences, zero denominator, fixture refusal and no
raw evidence/path output. Independent milestone/native/release review remains
deferred, not approved. GLY181 still needs actual provider integration, live fences,
durable native ownership, real collected calibration and physical qualification.
