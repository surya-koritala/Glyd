# GLY-267 token publication and the control loop

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Base: main18fdb20ce055ce45ea28499726af449d03b5e09a.

Automatic main CI38932851682 fails Windows backend/aggregate, five other checks
pass. Eight short lifecycle cases pass43.45s. Production then fails13.58s at
daemon.rs1773: cluster token returns CONTROL_UNAVAILABLE/outcome unknown; daemon
records executing/invitation_step=none, later total command execution3062ms.
Raw work/m6-pr100-main-windows-backend.log SHA256
3de16ceffdfa308acbdcf50ad9f9c956cb7fd33bf75657786e1c9ee6e01fefa7 is retained.
This does not establish which OS/I/O/scheduling operation consumed that time.
No unchanged-source retry is used.

Source tracing finds that saved token publication releases the policy mutex but
still runs synchronously inside Runtime::command on the main control loop. The
correction dispatches it through the existing bounded admin_job mechanism with
one token-publication permit. A narrow local issuer shares policy access, never
authority signing keys. Existing public Authority token APIs delegate to the same
issuance logic. The transaction, durable publication, both admission checks,
scope/expiry/capacity/revocation/drain rules and three-second response deadline
are unchanged. A late/unknown operation is not retried or presumed successful.

The existing basic background-admin check now exercises actual token dispatch,
invalid input refusal without file creation, the occupied token slot, responsive
sibling status and permit release. Restoring synchronous token dispatch fails
that same assertion in0.05s (zero owned token workers); corrected dispatch passes
0.09s. This controlled negative proves the dispatch difference, not the historical
Windows stall's initiating OS cause. The paused-slot check uses in-process bounded
channels; it does not intentionally delay real disk I/O or run network traffic.

The existing file/SQLite publication contract check adds drain to its busy,
permission and capacity refusal variants: all four leave a published candidate
inactive. It passes0.27s. Its first exact selector accidentally selected zero tests
and is not evidence; the corrected review_regressions selector runs one test.
Helpers were traced before local execution: synthetic enrollment signs/records
membership directly, with no listener, socket, daemon or model. The reused
background-admin case also performs its existing read-only interface enumeration
through the ordinary topology function; it does not probe interfaces, inspect
processes or instrument the host/security stack.
Both warning-denied workspace/all-target lint modes pass6.92s/3.32s, at most two
build jobs with no overlapping Cargo. Formatting/diff and separate complexity and
correctness/security review pass on final source before commit. The complexity
review retains the existing single-job helper and one narrow policy handle, with
no worker-pool framework or dependency. The correctness/security review confirms
key ownership, rechecked admission, queued expiry, permit release, no mutation
retry and owned shutdown/drain behavior.

Limits: policy/SQLite transaction work still shares the existing access mutex;
this change does not claim isolation from a slow durable database commit. A slow
token job can still outlive its caller and make shutdown wait for owned I/O.
The synthetic slot check is not a real disk-stall or Windows cause reproduction.
All exact-source platform checks, current-main merge review and independent main
verification remain required. Full GLY-267/M2 native and milestone acceptance stay
open; independent milestone review is deferred, not approved.
