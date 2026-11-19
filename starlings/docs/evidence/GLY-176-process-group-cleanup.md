# GLY-176 process-group cleanup evidence

`Service::confirm_process_group_exit` stops the owned service and, on Unix, checks
that its original process group no longer exists. It accepts only ESRCH from a
signal-zero group query; a remaining group, permission error or two-second timeout
returns StopUnconfirmed. The query sends no signal and accepts no caller-supplied
PID. Windows returns Unavailable after parent exit: the current process wrapper
does not expose a bounded proof that the entire job is empty.

On Unix the liveness check first uses waitid with WNOWAIT to observe parent exit
without releasing its PID. It signals remaining group members before reaping.
Once reaping can have occurred, later cancellation cannot signal that numeric
group ID again. This avoids targeting an unrelated group after PID reuse.
The existing locked rustix dependency supplies safe system-call wrappers; no
unsafe implementation or new downloaded dependency is introduced.

References: [waitid/WNOWAIT and child reaping](https://man7.org/linux/man-pages/man2/waitpid.2.html),
[signal-zero and process-group queries](https://man7.org/linux/man-pages/man2/kill.2.html).
The implementation was also checked against the locally pinned rustix 1.1.5 and
process-wrap 10.0.1 source, including the latter's parent-exit behavior.

Synthetic subprocess tests cover a live service with an additional child in its
group, a crashing service, repeated cleanup, and refusal to claim group-wide
confirmation on Windows. Existing service, cancellation and full client/worker
relay tests also pass locally. Formatting and both warning-denied Clippy feature
modes are required, followed by all four platform checks and dependency policy.
Exact PR/head/CI evidence is recorded in GLY-176 before merge.

This proves absence of the original process group only. A separately qualified
backend must not escape that group; shared helpers, detached descendants and GPU
completion need their own contracts. This method does not itself release core
reservations, qualify a native model/resource profile or complete M2. No native
model workload, physical Pi, Ubuntu access or paid infrastructure was used.

The prompt-count PR's macOS CI exposed an immediate refusal in the descendant
case, before the observation deadline. XNU's `killpg1` skips zombie group members
and can return EPERM when it finds no signalable member while the group still
exists ([Apple source](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_sig.c)).
Group confirmation now observes every non-ESRCH result until the original deadline;
it never treats a permission error as successful cleanup or sends another signal.
Interrupted child waits similarly retry within the existing parent-exit bound.
The test reports the failing mode and final observation for unresolved failures.
