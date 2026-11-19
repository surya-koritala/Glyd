# GLY-197 per-node advisory refusal context

Issue: [GLY-197](https://linear.app/getglyd/issue/GLY-197).

Existing offline preview refusal codes aggregated every node and stage, preventing
an operator from identifying which supplied node needs corrected evidence. The
preview now projects the same refusal events into a bounded node/code count map.
It reuses every existing decision and aggregate count; memory arithmetic, ranking,
search order/caps, policy, inputs and qualification are unchanged. No second
eligibility evaluator, live query, provider API or reservation path is introduced.

The additive output has at most64 supplied node UUIDs and ten fixed reason
codes each. It includes no arbitrary error text, endpoint, model/profile payload,
prompt or activation. Candidate identities were already output; the explicit
redacted diagnostic bundle remains separate and still excludes node identifiers.
Global/edge/policy counts are not attributed without a node context. Mode selection
continues its existing aggregate report, without claiming these are live reasons.

One exact pure control uses three synthetic nodes/two layers and a bounded set of
profile/capability/family/backend/pool variants. An independent exhaustive oracle
confirms candidate ranking; a full-model memory refusal coexists with that node in
a selected smaller stage. Node attribution, counts within aggregates, no inferred
admission, incomplete-search limits and serialization redaction are asserted.
This is basic in-process Mac testing, max two build jobs/cached offline assets;
no sockets/services/host instrumentation/inspection/models or native workloads.
The exact new check passes0.12s; all five pure planner tests pass0.41s, including
the existing oracle over three nodes/two-to-four layers and both objectives.
Both workspace/all-target Clippy modes pass with warnings denied9.52s/3.49s;
formatting and diff checks pass. No broader process/network suite ran locally.
The existing offline CLI-file case additionally asserts the new empty node map
when inventory is empty. Its exact selection passes in default/development modes
(0.73s/0.76s), including implicit-fixture/unsupported-flag/malformed-input refusal;
both warning-denied workspace lints pass afterward0.54s/0.29s and fmt/diff pass.
This runs only short local CLI file operations, without sockets or a daemon.
Before opening the combined experience PR, signed merged main438e293 (PR #99,
all seven sourceCI38916457383 checks PASS; automaticmain38917868580 pending) is
integrated asdc8db5244057f87d4c555e4f8ec94f7adcd02f0c. All five planner cases pass
again0.41s, both exact offline CLI modes0.84s/0.76s, both warning-denied workspace
lints1.75s/0.96s and fmt/diff pass. No inference, socket or daemon is started.
The guide and this feature share one PR/matrix; final-source CI and current-main
review remain required before authorized merge, automatic main separately.

Correctness/security review traces preview/CLI/mode consumers and all refusal
callers; node keys are validated supplied IDs, codes are static, and global counts
retain their semantics. Separate complexity review uses one direct map/helper,
existing predicates and oracle, without a dependency or workflow change.
Independent milestone/release review remains deferred, not approved. Full GLY-197
stays open for live native/model/profile/readiness/allocation experience and actual
placement/review/packaging gates; no native qualification or milestone closure.

## Prepared integration with resource-counter correction

PR100 remote source6abd235/CI38918028918 remains failed and unmerged; preserve
its distinct token-publication unknown-outcome record. Locally integrate reviewed
PR101 source694aa96 as signed96d5f2452294c123557412a3055524b3b461c475.
The five pure planner cases pass0.38s; the exact offline preview CLI-file case
passes in default/development modes0.79s/0.77s. Both warning-denied workspace/
all-target lint modes pass3.42s/1.57s, formatting/diff pass. These are bounded
basic Mac checks, max two build jobs, cached offline assets, no sockets/services,
host instrumentation/inspection or native/model workload. No passing CI or
integration/milestone acceptance is inferred from them.

Correctness and separate complexity review confirm the experience diff remains
the same bounded per-node projection and navigation guide; no admission/ranking
change or additional runtime dependency. PR101 sourceCI38925737799 is still
pending. Final current-main integration and all seven new-source checks remain
required before authorized merge, automatic main separately. GLY197/199 remain
open for their actual native/live/packaged/support-claims/review gates.

## Qualified source integrated before PR update

PR101 source694aa96 passes all seven checks in CI38925737799 and merges as
a97b9715e139be4f9a7cf7ef6405b3c1d6011e48 at2026-10-16T15:49:51Z after exact-head,
current-main ancestry, clean tree/mergeability and review-state verification.
Windows default raw SHA256
`f4c52ef5aec6593792d013fc7b27a05b83f1f244fce045bf4e3393776f680b01`;
backend `882dd6880679ff9a1e736a353632e27e1eabb9e8fced7eb257da65b6d196841b`.
Default reuses the same priority/remainder security executable and the remainder
build takes0.24s; all ten daemon cases pass66.44s. Backend's eight short controls,
production195.93s and recovery57.89s pass. These are CI software records, not native
qualification or proof of the historical Windows initiating events. Earlier
failed/canceled sources remain preserved. Automatic mainCI38927095681 is pending.

Merge that current main into the prepared experience branch before updating PR100.
The resulting tree is byte-identical to locally checked a718a02; only this evidence
append changes afterward. No redundant local test run is inferred as necessary
from that history-only merge. The final PR100 source still needs all seven checks
and final current-main/clean/review gates; automatic main is checked separately.
No native workload, license/publication decision or milestone closure.
