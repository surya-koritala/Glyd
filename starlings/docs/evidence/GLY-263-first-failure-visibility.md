# GLY-263 — Preserve first Windows failure before long matrices

PR71 source `46bdff73da490f452b0f7bf47ae2224883653667`, CI
[38670973158](https://github.com/Sigbound/glyd-compute/actions/runs/38670973158),
Windows backend job116213765420 reached the ten-minute cap. Five other platform
and dependency checks passed. The raw log SHA-256 is
`c71d45dde965dcbef3ca1b2ca3f669e385ac307bbe7fbee4f3676c3e04bec3b0`.

The observation-path test reported FAILED at02:32:01Z. The production matrix
subsequently passed at02:35:53Z; the last lifecycle test was still running at
02:36:31Z cancellation. Libtest had captured the earlier panic for its final
summary, which cancellation prevented. No observation timing or owned-daemon
failure context appears, but that absence does not establish where it failed.
This record cannot identify an initialization, IPC or OS root cause.

The catalog command now uses `--nocapture`, preserving panic details immediately.
On Windows, the same job first runs the shorter controls, then the existing
production matrix and final lifecycle matrix as separate commands against the
same compiled binary. Bash's existing error exit stops before long matrices when
a shorter control fails. All tests still run when preceding checks pass; no case,
assertion, platform, deadline or ten-minute cap is removed or increased. No new
jobs or dependencies are introduced. Expected caught panic controls can appear
in successful logs; their test result determines success.

The workflow selectors are checked with the actual binary's test listing: the
three groups are disjoint and their union equals the complete catalog selection.
This proves selector coverage on the local build, not Windows execution. Final
source CI remains required. It fixes loss of failure evidence and delayed step
failure, not the initiating observation-test defect; GLY263 stays open.
