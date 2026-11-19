# Passing Windows steps reached the job cap — source5632b13

[CI38852081298](https://github.com/Sigbound/glyd-compute/actions/runs/38852081298)
is CANCELLED, not green: Windows backend116815313980 exceeds the existing10m cap,
confirmed by its check annotation. Five other platform/policy checks pass and the
aggregate fails. Every backend step reports success, including all21 library
cases,8 short catalog controls46.59s, the full production matrix209.90s and
failure/drain/restart105.21s. Tests complete04:53:28Z, cleanup04:53:31Z and the job
is cancelled04:53:33Z (started04:43:32Z). Raw full-job SHA256
`e002cd68b80dfb5f8b58997e47be5d90c0ea447cbf26c8ab55c0b0831d348d3a`.
This does not qualify the seven-check merge gate or explain historical failures.

The unchanged locked-fetch command takes35.6s after an exact cache hit. CI now
tries `cargo fetch --locked --offline` first, falling back to the same online
locked fetch only when needed. Suppress only the expected first cache-miss error;
the online command still fails the step normally. Local cached offline fetch
passes0.123s; Windows savings remain to be measured on changed-source CI. All
lint/test jobs, platform coverage, locked checksums and10m cap remain. No compiler
profile, dependency, model workload or timeout change is used to obtain green.

# GLY-304 — Pre-request connection loss

Issue: https://linear.app/getglyd/issue/GLY-304
Historical investigation: https://linear.app/getglyd/issue/GLY-267

An enrolled peer can close or send a wrong command before Count. The old worker
relay still called Cancellation::cancel, signalling its healthy loaded process
with zero reservations. The controlled original test fails1.38s at check_alive;
raw SHA256 `f8034878713f170ccd651501a7f6fec275d55c5d7eef478b582dfaa3c50f4366`.

The shared relay now returns an explicit optional completion: None requires no
accepted Count, an input/transport/capacity failure and a positive zero-reservation
snapshot. It closes only the connection and performs no backend operation. The
supervisor rechecks health, empty ownership and live fence before ready on the
same loaded Service. Some(completion) keeps the existing trusted drain obligation.
Accepted Count, invalid configuration, authorization/fence failure, failed state
inspection or any old reservation cannot use the idle exception. No service pin,
quarantine, process launch, deadline or reservation policy is reset.

The serial coordinator Dispatch already records whether any mutating control may
have been attempted. Every error with a false flag now releases only router
bookkeeping, not worker buffers or backend quarantine. There is no retry after
Count, and any possibly sent Prepare/Commit/Start remains conservative.

Final no-Count EOF/wrong-command test passes1.51s and proves a fresh request uses
the same service, exactly one generation starts, and successful completion still
retains its reservation until explicit fixture drain. Raw
`0ec10c6bbdc8b68b1a7853857eaa057790ee82f29236e848e9626c6ebd41820b`.
Accepted-Count loss and pre-existing-lease negatives pass2.76s and still cancel/
retain; raw `d7a6f862c5ba6db1560a408259ae1cd7b7036309f14999ab8f86c6a998a9821e`.
These are model-free fixtures, not native allocation/drain qualification.

Full local model-free library/bin/CLI suites pass21/10/7 tests (one native test
remains ignored), raw77648e028579911728701373d88ea76ab50a2ff94935fcf4f3f4d282370b2fc5.
All18 catalog lifecycle cases pass68.22s, raw
aa87073291ede9a2baef65a013405cbd730865d558b023fcdeb02df4cb464b78.
Formatting and both pinned warning-denied workspace/all-target lints pass.
This proves a concrete failure-policy defect, not the initiator of archived
Windows stalls. Seven exact-source checks and separate post-merge qualification
remain required; GLY267 historical attribution stays open.
Correctness/security review preserves the native drain boundary; separate complexity
review reuses Option, the existing serial mutation flag and supervisor guards.

# Windows controlled TCP outcome and retained source failures

Source800993b / [CI38850287470](https://github.com/Sigbound/glyd-compute/actions/runs/38850287470)
is FAILED Windows backend116809779708 and aggregate; the five other checks pass.
Short8 controls pass47.85s; production matrix fails159.52s at daemon.rs856:
queued0 versus expected8 after capacity refusal. Both workers later retain4MiB
and Transport; post-unwind maintenance-gap flags do not establish initiation.
Raw backend SHA256 `2a1ef836ed38edff68431686f908ade2bfff2a9fd42420585777730fd21eff9c`.
The same source's Windows default job116809779712 passes, including the paired
TCP comparison: four exchanges27.7–34.6ms; medians30.243ms(false) and30.645ms(true).
Raw `c953bee9e229759dcbf606d9f6efbb42df3cfdf10d952a324ca3352a18c0051e`.
This does not reproduce a TCP_NODELAY benefit or the failed process's initiating
condition. Production socket settings remain unchanged. No unchanged-head rerun.

# Historical first-source failure and controlled TCP question — PR82

Source `0fa6291411619846d89b25117d7fdea37df97a96`,
[CI38848860972](https://github.com/Sigbound/glyd-compute/actions/runs/38848860972),
Windows backend116805392885 fails the two-worker start barrier at daemon.rs806:
primary started, secondary not started before the existing three-second bound.
Short8 catalog controls pass; the matrix fails156.33s. Five other CI checks pass.
Raw full-job SHA256
`109987e54c5be0bc533b5efa23dd73b8fa2c9ecdb38748b1bc032727be721098`.
The later secondary Transport/reserved0 and primary serving/reserved4194304
snapshots occur during unwind, after client sockets may close. They do not prove
Transport initiated the stall. Earlier main49dddea's executable-digest startup
wait and main8afe54d's HTTP503 assertion remain distinct failed evidence.

Source inspection finds split prefix/body writes through rustls::Stream, which
attempts underlying I/O after each write, and no explicit TCP_NODELAY policy.
The bounded controlled question is whether the native small-packet policy adds
material delay to four small authenticated record round trips. The existing TLS
fixture now runs five pairs with both socket options, alternating order, checking
exact sequence/payload and actual native option values. Timing starts after a
thread barrier; all existing authorization, record and fence deadlines remain.
Safe numerical measurements go to the raw diagnostic writer even when a test
passes. No performance threshold, new job, workflow change or product socket
policy is introduced. This comparison measures a loaded CI fixture, not native
inference or the initiating event in the failed daemon process.

Mac model-free comparison passes1.40s. Four exchanges per sample take12–16ms with
either option, providing no evidence for a Mac delay correction. Counterbalanced
raw SHA256 `522f0066b42406ece9f84b628a37a443abb4ab28b636b382add65eca06c01dc1`.
The sandbox attempt failed at fixture listener setup before measurement; the host
loopback run is separate. Production networking is unchanged. The later Windows measurement above also
shows no consistent reduction; do not retry a failed unchanged head or claim any
historical cause fixed. GLY267 remains open.

Formatting and both warning-denied lint modes pass. Correctness/security review
keeps the experiment fixture-local, logs no application data, retains timeouts and
uses scoped joins. Separate complexity review reuses the existing fixture and
native socket option rather than another simulator or dependency. Independent
milestone review remains deferred, not approved.

# GLY-267 — Held-request observation control

PR70 source `a83a8968a79fa3db17f5114c8a98ba8af60951bb` failed the Windows backend
job [116206340422](https://github.com/Sigbound/glyd-compute/actions/runs/38668663927/job/116206340422).
Nine catalog cases passed, including memory-collector isolation and lost Stop
reply observation. In the two-worker scenario, requests held while filling and
discarding the queue returned WORKER_LOST/503; both services retained their pins
and 4 MiB reservations with SECURITY_Transport and no registry_failure. The
coordinator logged Expired, Capacity and queue-watch Transport. Raw job SHA-256:
`e78e69e6f3ae3912f6c91b2ba6a8b6d9230f68466e1ccd0fd22b2b2a4bfb113e`.

The held interval still launched two read-only CLI subprocesses. A test-local
thread guard forbidding process starts during that interval fails the original
path at `[cluster, status]` in 21.45 seconds (log SHA-256
`3cd6e8045405b80eeae2465bd3c60d6f2a132affd5ca75f3e17af317e0e60075`).
Replacing those observations with the existing authenticated IPC helper passes
the same controlled two-worker case in 23.25 seconds (log SHA-256
`fdcaffb672ead41b6b6ec1fceb33615238defc7da7ef8a6794f59a086a78eed8`).
The guard is only in the integration test, never in the CLI binary.

This proves and removes a subprocess dependency from the fixture's critical
interval. It does not establish which operation initiated the historical
Windows expiry/transport failure. The guard fails at the dependency itself;
the before/after wall times are not a performance comparison. No unchanged-source
retry was used, no production behavior changed and no failure was reclassified
as a successful request.

All request/socket/queue deadlines, admission pacing, two active/eight queued
checks, refusal behavior, disconnect discard, subsequent completion, cancellation
and pin/quarantine assertions remain. Full current-source CI and post-merge
verification are still required; original GLY267 investigation remains open.
