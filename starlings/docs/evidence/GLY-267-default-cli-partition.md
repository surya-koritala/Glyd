# GLY-267 default CLI validation allocation

The completed [PR121 source run39046952297](https://github.com/Sigbound/glyd-compute/actions/runs/39046952297)
passed all eight checks at ea1bab395e0ae6b3f92b2b074f8f986dba014130.
Windows default job117437466096 took476s, including367s in its default-test
step. After the non-CLI tests, its serialized default CLI command occupied
approximately158s, including a33.37s build; the12 default CLI daemon cases
passed in101.61s. Windows backend job117437466104 completed in238s.
These are observed timings from one run, not guaranteed future costs.

Move the existing default CLI command unchanged into the existing backend job:

| Required check | Current validation group |
| --- | --- |
| Windows default | Default lint and complete default non-CLI tests, including the isolated security generation case |
| Windows backend | Development-feature lint, serialized complete default CLI tests, then exact development-feature production-route case |
| Windows lifecycle | Existing development-feature tests, short catalog controls and separate exact recovery case |

Linux x64/ARM64 and macOS retain the complete workspace default invocation.
No test command, feature mode, case, assertion or serialization flag is removed
or duplicated. Default CLI tests still run without development features and
with one test thread. The backend production case retains its own invocation.
The eight check names and fail-closed aggregate are unchanged, as are action
pins, permissions, caches, warning rules and10m/1m/5m caps. This adds no job.

File-only validation compares the exact Cargo test-command multiset with the
parent workflow, parses every shell block and checks routing for all four
partition values plus unchanged unrelated steps and safety settings. No local
Cargo, CLI process, service, socket, model or host-instrumentation test is needed.
The earlier [partition evidence](GLY-267-windows-partitions.md) remains historical;
this allocation supersedes only its default CLI job ownership.

Separate complexity review retains one direct conditional step and no helper or
dependency. Correctness review checks full default/development coverage, existing
security-case isolation, explicit CLI serialization and fail-closed aggregation.
New exact-source CI must qualify the changed allocation; automatic main is
verified separately after an authorized reviewed merge. No speedup, native
qualification or historical Windows cause is established by this file-only
change. GLY267's original investigations and independent milestone review stay
open.

The completed local check passes in0.07s: native Ruby YAML parsing, all17 shell
blocks through `bash -n`, an unchanged multiset of12 Cargo test command lines,
exact all/default/backend/lifecycle routing and structural equality apart from
the single moved command/step. Local checker and result are retained as
`work/check-default-cli-partition.rb` and
`work/default-cli-partition-validation.json`. Whitespace checks pass. No local
test suite, application or Cargo build is run for this workflow-only change.
