# M0/M1 review remediation (GLY-211)

Base: PR #8 merge `1fd3b295a6b637e5f06e97df6d6a68aab1c37e8d`.
This branch addresses the review findings; independent review and physical
acceptance are separate gates. M2 is not started. No model or GPU work occurs.

| Issue | Change / evidence | Remaining acceptance |
| --- | --- | --- |
| GLY-212 | Coordinator local expiry recovery; registry certificate refresh; four repeated thresholds and lost reply on real TLS with worker-only profile | Independent review of key pin/renewal behavior; wall-clock multi-day soak not run |
| GLY-213 | Listener transient errors and foreign IPC peers are per-connection failures; stable nonblocking accept | Cross-user Linux and reset-loop process regressions included; final-head CI required |
| GLY-214 | Separate admin admission and 63 persistent authenticated registry slots; per-host IPv6 admission; bounded 128-event queue | Signed per-node reconnect admission and a bounded nonblocking probe queue; real reconnect/queue saturation regressions included |
| GLY-215 | Read-only authorization, one SQLite connection, periodic high-water checkpoint, transient busy handling | Physical Pi storage benchmark unavailable; checkpoint crash-window documented |
| GLY-216 | Ordered 4,096-result replay window, explicit expired-replay error, terminal-state reclamation and cached cleanup; 1,000 generations in one epoch | No token/lifetime request cap; all peers require updated sequence contract |
| GLY-217 | Stop stays latched through subsequent drain | Covered by runtime regression |
| GLY-218 | Bound ephemeral ports persisted for restart | Passed physical coordinator restart with the same bound port |
| GLY-219 | Committed rotation remains success despite reconciliation failure; CSR-key idempotency and reusable renewal candidate | Regression tests cover both failure paths |
| GLY-220 | Bind before epoch publication; nonpanicking daemon output; revoked heartbeat loop exits; preserve first error; specific CLI failures/global flags | Closed stdout, inherited ignored signals, SIGTERM/SIGINT/SIGHUP, offline leave and specific errors covered |
| GLY-221 | Receive timestamp before lock; authenticated stale-counter error; restart hysteresis; fsync outside fence mutex; corrected first-heartbeat docs | Failed epoch transaction, failed fence persistence and draining-reply regressions covered |
| GLY-222 | Library drain blocks issuance; explicit fail-closed panic policy; randomized Windows pipe name; Unix handle-relative cleanup | Windows native qualification in CI; no malicious same-user guarantee |
| GLY-223 | Core drain guard, explicit withdrawn output, epoch-scoped cache reset, token extent/stage checks, reply IDs/write watchdog, bounded telemetry history | Stalled writer and mismatched reply injection regressions passed |
| GLY-224 | Real plan digests and negative binding check; meaningful long sequence replaces weak generated trace; corrected rollback claims | Synthetic execution only |
| GLY-225 | Expired coordinator/peer, repeated worker renewal, lost reply, contention, invitation cap and settings-tamper regressions | Expired peer, invitation pin/cluster mismatch, both caps and data revocation negatives covered |
| GLY-226 | Existing process/network regression suite retained; portable bound port-0 listeners | Native Windows elevated-owner/CTRL_C/CTRL_BREAK and deterministic HTTP WouldBlock regressions included; final-head Windows CI required |
| GLY-227 | No duplicate daemon/core/security feature tests; save caches on test failure; no main cancellation; ten-minute cold ceiling | Forced-cold four-platform run 38258131720 passed at 3006669; longest job Windows 384 s |
| GLY-228 | cargo-deny advisory/license/bans/source policy; warning failure; private-fs review record; lockfile pin with compatible range | Disallowed-license negative check fails as expected; M6 audit/notices are not claimed |
| GLY-229 | Streaming admission and per-chunk cost remain explicit M2 integration gates | No HTTP streaming implementation in M1 |
| GLY-230 | Whole-request operation/per-token overhead remain explicit M2/M3 integration gates | No tensor/backend implementation in M1 |
| GLY-231 | Removed duplicate DB, repeated invitation lookup and repeated compiled control schemas; persistent heartbeats | Security-sensitive lock/HMAC checks retained; streaming costs require measurement |
| GLY-232 | Independent review required in CONTRIBUTING; fix PR stays open | Repository setting changes explicitly declined; no CODEOWNERS identity supplied |
| GLY-233 | First available control pair is Mac M1 Max + Ubuntu/4080 SUPER; Pi remains later ARM64 CPU qualification | First inference backend/hardware qualification remains M2 |

## Local validation

Formatting, default/all-feature warning-denying lint, complete default workspace
suite and feature-specific adapter/CLI suite. The security suite includes negative TLS, durable storage, admission, HTTP, IPC and registry regressions.
The 63-peer timing test is explicitly run outside routine CI to keep CI short. Dependency policy reports advisories/bans/licenses/sources clean.
CI must validate all four operating-system/architecture targets on the final head.

## Review boundaries

No remote model policy replication, real inference backend, scheduler, public
release, license selection, paid device or GPU work is included. The implementation
review is not the independent approval required before merge. Remaining test and
hardware criteria stay open in Linear; this document does not close M1 by assertion.

## Remaining platform acceptance

GLY-226 includes native Windows elevated-owner and console-event qualification
in Windows CI, and deterministic HTTP WouldBlock coverage through the same
authorized output loop used by sockets. The new IPv6 Host and cleanup-refusal checks pass; a buffer-size-based
WouldBlock test was not retained because macOS auto-tuning made it nondeterministic. Raspberry Pi storage timing
remains unavailable. These are explicitly open; passing Linux ARM64 CI does not
qualify a physical Raspberry Pi. GLY-229/230/233 are M2 design/integration gates.
No repository protections or CODEOWNERS configuration were changed.

## Complexity dispositions (GLY-231)

Persistent registry connections remove repeated heartbeat TLS setup. The three
lock files represent distinct daemon, authority and registry-client ownership and
remain. The HMAC primitive supplies constant-time comparison; a custom comparator
would add security risk. Trust member snapshots serve the existing library API.
Typed request-size validation remains necessary for callers that bypass JSON
parsing. Manifest path checks remain trust-boundary validation. No speculative
streaming or backend abstraction is added to this remediation.

## Independent review follow-up

The initial bounded-history approach was insufficient. It is replaced with an
explicit ordered replay window, terminal record reclamation and full cleanup
acknowledgements; no 256-token or lifetime request limit remains. SQLite contention
is nonblocking and retried by daemon maintenance. Same-key refresh retains the
old leaf through its original expiry and keeps the registry stream open. IPv6
hosts have separate admission buckets. Reset/abort/network errors are recoverable;
new real-process regressions exercise resets, exclusive SQLite locking and a
broken stdout pipe. The stale descriptor-only test is not the evidence for stdout.

Windows lint fixes were already included; a subsequent cold run exposed a
DELETE-sharing conflict, fixed by retaining read-only directory handles.
Reconnect admission now verifies a per-connection device-key proof before reserving
a node-specific TLS slot. Manual key rotation and simulator-driven rollback remain
explicit boundaries; link/kernel/CPU exhaustion is outside the admission guarantee. No autonomous scheduler is claimed.
