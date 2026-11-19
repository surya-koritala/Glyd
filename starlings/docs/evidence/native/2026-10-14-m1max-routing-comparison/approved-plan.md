# One corrected SSE usage comparison — unapproved

Prepared candidate: `/Users/suryakoritala/Documents/Codex/2026-10-11/referenced-chatgpt-conversation-this-is-an/work/native-check/routing-profile-sse-corrected`. Fresh isolated state: `/private/tmp/glyd-m2-daemon-3-35936fd9`.
Implementation `78b737495ff52fc69e05b928a460f400fcd4af70`; CLI SHA256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`.
Production Rust source, crate manifests and lockfile match implementation78b7374 and merged PR60; PR61 changes only tests/evidence.
Runner SHA256 `6f0f6a62b78dbef11be7f3d21453a37890d6dabe2c548743f0d9faafe47a388c`; preparation-script SHA256 `5c3968138889713f929ae731c158d63c7a5d81fd9af13def2506bc94f2015d15`.
Service configuration `cc6d5929c066828006fcdd893ff362a94d54c760d561f6d883884e01d903bba0`;
worker record `a46c07b079532ac13e86be2f81efefc93e8564c6a33ea1e4adb0a4fb149ff3fb`.
Prepared pin receipt SHA256 `cc0f292b64806e8e5b82a88778127402dacda1dd5524a1ddfe16cd511340a71a`;
durable pin SHA256 `df47c2f90a0f7fe5f332c7046b865f6bcb73152b392db3d7217e59ad3b4e9989`.
Reviewed fix/evidence: [PR61](https://github.com/Sigbound/glyd-compute/pull/61),
[GLY-259](https://linear.app/getglyd/issue/GLY-259).
Original CI38597196561 failed in the Windows invitation fixture and remains failed. GLY260 fixes its strict-success contract mismatch with one committed-state observation; Reviewed source7a13323b7f0eddb82f4f6bde98853d05303e57ed passed all seven CI38599604510 checks; PR61 merged as1094af0a04d491f444b12966cfb49de9211c7517. No native start or attempt marker exists.

The prior approved run is consumed and frozen:14.630s, two generations requested,
one successful direct JSON observation, TypeError when direct SSE usage was absent;
no coordinator/worker stage. Spend$0, sampled peak RSS1,668,464,640bytes, token rows0,
no observed owned group remaining, old six ownership hashes intact. This is failure
and partial direct evidence, not a passing comparison. Raw result SHA256
26e4c9c02d1c6531df56b290837916867375746819633a1eef76ded8b7afd3e1.

The fresh script requests backend SSE usage on the direct path only, matching the
existing production adapter. Missing/invalid usage gets a safe fault; exact token
checks remain. It corrects server-group count bookkeeping without changing the full
owned-group absence check. Synthetic SSE reproduces the old failure and validates
six usage negatives, DONE boundaries, strict proxy fields and group counting;
prepared-pin checks and byte-identical patch reconstruction pass.

## Exact requested single-use envelope

- M1 Max ARM64 Mac,32GiB, macOS27.0/26A428; CPU only, two inference/batch threads.
- One attempt, no automatic request or experiment retry.
- Two sequential runtime starts: direct, then worker-owned service, never concurrent.
- 20 total requests:10 direct and10 proxy, five JSON/five SSE each.
- ≤8 output tokens each,≤160 total;≤41 count-call code upper bound (not instrumented).
- Same fixed benign17-token input, seed42,temperature0,top-p1; no user documents.
- Five minutes including hashes/loading/cleanup; watchdog240s leaves one minute cleanup.
- Monitored4GiB aggregate owned RSS,100ms samples; not an OS allocation guarantee.
- Isolated loopback HTTP/coordinator-worker TLS and temporary scoped token.
- Cached assets only,$0; no download,install,runtime rebuild,GPU,Ubuntu,remote or paid work.

Same cached CPU-only runtime SHA256
66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398,
source d81235049384534c167caea52b85a694f6103d14.
Manifest8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835,
2026 Qwen3.5-0.8B BF16/F32 GGUF artifact
cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5.
F32 KV; one slot, allocated context512/admitted128, caches/offload/repack/warmup
disabled, offline text-only. Conversion fidelity remains unqualified.

Compare content/reasoning/finish/usage in memory; never log generated text. Require
exact prompt/output/total coherence, direct idle proof, proxy trusted drain and zero
retained request/body counts. Preserve immutable implementation/profile identities
and per-request timing. Fixed stage order cannot isolate causal overhead, and this
one-host comparison cannot qualify installed Glyd, two workers, GPU/Metal or Exo.

## Cleanup and approval boundary

Revoke temporary token, stop owned daemons/runtime, observe groups, verify zero token
rows and unchanged protected ownership. Retain direct pin and any new worker pin/
quarantine; no recovery/reset/unpin is permitted. Old six ownership files plus the
consumed current runner/result/marker/approval/pin records are protected by hashes.
Observation failure is failure/unknown, never cleanup proof. Any failure ends the
attempt; preserve its marker and raw result. All older attempts remain immutable.

Native-free preparation/preflight passes; no inference has started. Correctness/
security and separate Ponytail review completed for the minimal script correction.
Independent milestone review remains deferred, not approved. Reviewed green merge PR61 is complete; fresh native approval is still required.

CONTRIBUTING.md requires “Obtain explicit device/budget approval before real
workloads.” The previous approved plan allowed one attempt with no automatic repeat;
that attempt is consumed. This prepared candidate requires fresh explicit approval.
Until then only `--preflight` may execute; no `--run-approved`.
Command after green merge and approval:
`python3 work/native-check/routing-profile-sse-corrected/run_profile.py --run-approved`.
