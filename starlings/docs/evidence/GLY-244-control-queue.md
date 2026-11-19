# Queued control expiry and phase diagnostics

Further PR #43 evidence: run38349663885 Windows default daemon tests failed four
cases while seven independent fixtures ran together. Three init subprocesses
exceeded their existing12-second bound, and another invite reported the newly
retained `CONTROL_WAIT_EXPIRED phase=executing`. The new probe check had not run.
The controlled follow-up serializes only the test harness's Windows CLI child
startups with a native mutex; daemon/socket/request concurrency and every scenario
remain. No timeout, assertion, registry limit or production setting is changed.
This addresses hosted process/filesystem/crypto contention as a hypothesis; final
CI/phase evidence must verify it before claiming the intermittent cause resolved.

PR #41 remains open. Corrected head `296e67ca2b0dd5cd3d6fe8dfd63453c2da5e64ad`
[run 38341767929](https://github.com/Sigbound/glyd-compute/actions/runs/38341767929)
passes the Windows production-route matrix but fails an invitation in the warm-up
matrix with `CONTROL_UNAVAILABLE`. That outcome is ambiguous; no mutation is retried.

Source audit finds a separate concrete issue: the response worker times out after
three seconds while its queued command can still start later. A shared atomic
stage now orders dispatch against queued expiry. A timed-out queued command cannot
transition to running; already started work retains unknown-outcome semantics.
The existing three-second response wait and five-second whole IPC deadline stay.

The model-free regression holds the reply sender through actual timeout, waits for
the expiry result and verifies the queued stage cannot become running. Closed
senders cannot make this check pass early. Shared dispatch/wait paths are traced;
no command payload, credentials or prompts are logged. Wait phase and slow
execution duration are redacted diagnostics; test failures now include daemon logs.

This strengthens queued cleanup and supplies evidence for the remaining Windows
phase issue; it does not establish that every earlier failure has the same cause.
Exact local/CI results are recorded in Linear. Five final-head checks and current
main verification are required before merge. Full control/daemon regressions stay.
Independent review deferred under current authorization, not approved. No native
model/device workload, paid infrastructure, secret/quarantine reset or new dependency.
