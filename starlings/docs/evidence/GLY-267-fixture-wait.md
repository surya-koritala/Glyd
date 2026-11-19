# GLY-267 lost-reply fixture wait

Automatic main f581aa15600cdb3096409f15563be87e7dfdb594 / CI38978236823
was canceled at the unchanged10m Windows lifecycle cap. The other six platform/
policy partitions passed; the aggregate failed. Raw lifecycle log SHA-256:
ae3553133ec9fa29f3bfd8a56d0f6eab58e39f08cabbd81c64716ab776297447.
PR116 exact source a3f0b03 had passed all eight checks38977428932; that
source pass does not qualify this failed automatic main.

The lock-poison and observation tests print deliberately caught panics and pass.
The lost-publication-reply fixture reports an expected inactive-invitation refusal,
then an actual token CLI12s timeout after `CONTROL_UNAVAILABLE: outcome unknown`.
The case never reports completion before job cancellation. The initiating Windows
CLI exit/drop cause and exact blocked server phase remain unknown; no OS/security
software or prior historical failure is attributed from this log.

All four mock-server waits in the CLI lost-reply fixtures used unbounded
`AdminListener::accept()`. If a client exits without the next expected request,
they can keep a scoped test from joining or printing its failure. Replace those
fixture calls with the existing nonblocking `try_accept`, a12s accept deadline and
an explicit client-finished check. A queued request is accepted before testing
client completion; an expired deadline accepts nothing. Errors are static labels.
No command is resent; authenticated receive/reply and every mutation/active-status/
negative assertion remain. Existing production admin I/O/CLI/job deadlines and
listener APIs are unchanged.

The shared wait's pure control covers exited client/no request, expired deadline
without calling the receiver, accept error, an already queued request with a
finished client, and one delayed in-memory request. It uses no IPC/socket, daemon,
child process, model, scan or host instrumentation. Fixed case/phase labels in CI
identify await/received mutation, await observation, response sent and joined;
no credential bytes or payload are logged.

This fixes a real unbounded fixture-wait path and improves failure visibility.
It does not claim the initial Windows CLI stall is reproduced or fixed, nor close
historical GLY-263/267. Separate complexity and correctness/security reviews are
required before exact-head all-eight CI, reviewed merge and separate main checks.
No new dependency/job, timeout extension, assertion weakening or unchanged rerun.
