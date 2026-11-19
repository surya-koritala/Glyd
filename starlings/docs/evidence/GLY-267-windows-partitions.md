# GLY-267 disjoint Windows work partitions

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).

PR113 source6e175d6 passed all seven checks in
[CI38965819640](https://github.com/Sigbound/glyd-compute/actions/runs/38965819640).
Its merged main935078f has a separate terminal canceled
[CI38966980573](https://github.com/Sigbound/glyd-compute/actions/runs/38966980573):
Windows backend job117195760030 exceeded the annotated ten-minute cap;
the aggregate failed and the other five checks passed. Unlike main111, recovery
did not finish. Development tests passed in185s, production modes completed,
and recovery started at21:15:24UTC before cancellation at21:16:14UTC.
The completed Windows default job took575s, including362s of default tests
and90s of short catalog controls. Moving recovery into that job would consume
its remaining25s margin. These are observations from this run, not guaranteed
costs or attribution for historical Transport failures.

The raw backend log is retained locally as
`work/m5-pr113-main-windows-backend.log`, SHA256
30cfbec5480cddb21a26eca28783cbe11a007de30f0514a91a2fd76ab9071524.
Job/annotation metadata and the terminal run result remain alongside it.
No unchanged failure rerun is used.

Move existing commands between three disjoint Windows partitions:

| Required check | Validation group |
| --- | --- |
| Windows default | Default lint and complete default tests |
| Windows backend | Development-feature lint and exact production-route case |
| Windows lifecycle | Development-feature tests, short catalog controls, then exact recovery case in its own invocation |

All seven existing check names remain; `Windows lifecycle` is an eighth required
check. The existing aggregate still depends on the complete core matrix, so any
failed, canceled or timed-out partition fails it. Linux x64/ARM64 and macOS keep
their existing complete commands. Windows CLI/development/catalog execution
remains serialized within each job; non-CLI default tests retain normal harness
scheduling. No case, assertion, timeout, warning rule, cache key, action pin,
dependency or permission is changed. Core jobs retain10m caps; aggregate1m and
dependency5m caps are unchanged. The additional job owns moved validation work.

File-only checks on the Mac pass: native Ruby YAML parsing, all16 shell blocks
with `bash -n`, baseline comparison proving the same multiset of12 Cargo test
command lines, explicit routing for all/default/backend/lifecycle, unchanged
non-Windows commands, safety settings, aggregate, caches and caps. Completed
source113 Windows logs contain eight distinct short catalog cases and the two
long cases; their filters and exact invocations are retained. Existing checks
in different feature modes remain intentional. Local validation records are
`work/check-ci-partitions.rb` and `work/ci-partitions-validation.json`.

Separate complexity review keeps the native matrix and direct conditions;
correctness review checks complete disjoint routing, separate recovery process,
serialization and fail-closed aggregation. Diff whitespace checks pass. No local
Cargo build, test workload, socket, instrumentation, model or GPU run is needed.
All eight exact-source checks and fresh-main/clean/review gates are required
before an authorized merge; automatic main must qualify separately. This
corrects work allocation within the fixed CI cap, not native qualification or
unexplained historical liveness causes. GLY267 and native parents remain open;
independent milestone/security review remains deferred, not approved.
