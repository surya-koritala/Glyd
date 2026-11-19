# GLY-306: bounded transactional initialization

Issue: https://linear.app/getglyd/issue/GLY-306

Base: `aca7d99e1b334ce875b259f593292eca17f75136`. The task PR records
the exact signed source, platform checks and separate automatic-main result.

## Current failure and controlled defect

PR91 source82ded73/CI38887164920 fails Windows backend initialization at the
existing12-second watchdog. The final marker is `INIT_AUTHORITY_PHASE=access_open`;
raw complete job SHA256
`8d93b464e7cef65c15718c49a9b854bd1f615bfd3645ee57f0843395cbbd8e5e`.
The OS/VFS operation responsible for the captured delay is still unproved.

The traced path performs five modifying core schema/member transactions and
five authorization schema transactions. The controlled before check observes
10 transactions and fails an at-most-two budget in0.05s; raw
`f2d03a324a87544db1b409c3f028f83f13fe75da857efc49f845f64ee608ef7c`.

The check independently reads SQLite's documented
[file change counter](https://www.sqlite.org/fileformat.html#file_change_counter)
from a protected database handle after creation. It verifies the SQLite header
and rollback-journal read/write versions before interpreting the counter, which
tracks modifying unlocks in this mode. It does not infer a count from timing,
PRAGMA data_version, or a mocked SQL driver. It measures transactions, not the
number of underlying filesystem synchronization calls.

## Change and safety

Initial core schema, rollback-floor row and coordinator membership now commit in
one transaction. Authorization schema setup and its clock validation use a second
transaction. Existing tables/member/role values, private ownership, readiness
markers and order, journal_mode=DELETE, synchronous=EXTRA and zero busy timeout
remain unchanged. No journal migration, retry, cache, deadline, test selection,
job, dependency or model workload is added.

Real SQL name-conflict and missing-clock cases refuse access and roll back every
new authorization table. Repairing only each isolated test obstruction permits
reopen with the original node/cluster and one member. Errors retain protected
partial initialization; they do not delete or reset a profile. A failed core
transaction cannot publish a ready marker before committing membership.

The same counter check reports `INIT_MODIFYING_TRANSACTIONS=2` and passes with all
tables, membership, original durability/busy pragmas and reopen validation.
Initial focused after log0.06s:
`0a604192ff9dfe853e16a21f7e7c325cd66a2e7276dcc1121fa8df6a1724bfd8`.
Final suites and source CI are recorded in the PR/Linear issue; local model-free
results do not qualify Windows or the historical initiating OS/VFS cause.

Final local security suite:113 passed/one existing ignored in17.02s, raw
`dddf93c7854e7d238b8aeb5405859483a3a6c5e1f14f4fec17803a138a40976b`.
Actual CLI silent initialization and collision/refusal controls:2 passed in0.86s,
raw`b4c3925a587d35e5ff47e48342767d54f3e7c9bdac0985d2938760e6ba8c0170`.
Both workspace/all-target warning-denied lint modes, formatting and diff checks
pass. The initial uncompiled draft and an earlier empty test filter are retained
locally but are not behavioral or CLI acceptance evidence.

Correctness/security review covers every constructor caller, transaction lifetime,
rollback on DDL/clock failure, initial membership/clock publication, marker order,
runtime ownership and unchanged durability. Separate complexity review reuses
native transactions and pairs the fixed fault/repair SQL cases; no new helper or
storage abstraction is required.

GLY-306 closes only after seven exact-source checks, reviewed current-main merge
and separate automatic-main qualification. GLY-263/267 retain their distinct
historical/OS/registry findings; fewer modifying transactions are not a measured
latency win or proof every captured stall is gone. Independent milestone review,
provider delivery and native/device qualification remain open.

## First source qualification failure and schedule correction

Source7e6d7c3/CI38888698371 fails Windows backend before the catalog lifecycle
step: three existing inference-library fixtures fail during bootstrap enrollment,
with Transport at tests.rs180 and tests.rs178;18 of21 library tests pass.
Raw job SHA256`3a8106c253c80fef8f900fbd0842e131424f97413e23ba00f9f0991c60bd0071`.
This is not proof the initialization transaction change caused those failures or
an attribution of their OS/VFS initiation.

The Windows default CLI and catalog lifecycle phases already serialize unrelated
fixtures. The development feature step did not. Apply the same standard Rust
test-harness serial schedule to that Windows step only; all cases and explicit
within-case threads, workers and request concurrency remain. Other platforms,
commands, warning checks, seven checks and ten-minute job caps are unchanged.
YAML and extracted Bash syntax checks pass locally; Windows qualification of
this changed schedule remains an exact-source CI gate. No unchanged retry or
timeout expansion is used. This test-schedule correction is tracked with GLY-263;
GLY-306 retains its concrete transactional initialization scope.
