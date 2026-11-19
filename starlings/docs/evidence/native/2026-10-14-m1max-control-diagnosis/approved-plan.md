# One CPU control diagnosis — not approved

Prepared script: `work/native-check/routing-profile-observed/run_control.py`.
Fresh isolated state: `/private/tmp/glyd-m2-daemon-3-7d370384`. Control runner SHA256 `f447da3193204f430ba08ba3cb5294481343db89c0faeb012dd7c31ff7b0d8cc`;
preparation SHA256 `5c3968138889713f929ae731c158d63c7a5d81fd9af13def2506bc94f2015d15`; cached CLI SHA256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`.
Implementation78b737495ff52fc69e05b928a460f400fcd4af70 has the same production
Rust/manifests/lockfile as merged1094af0 and this evidence-only source.
Service configuration `92bf8c262e0029274797490ea8732de3f0afd3224d34533d06e479692dcca219`;
worker record `8667b20cbd783c837ad0ed07eb31371db4ee80c52870bf47b53ddc19cb5b5d5b`;
prepared pin receipt `5aa632095c41930d5f5601451331907cec81de46035ba685d931591f7acb7d62`;
durable pin `30874f4b628db004446e45098d8f698c6ce6a9525a8712c57dcb856f7b493814`.
Tracking: [GLY-261](https://linear.app/getglyd/issue/GLY-261) original cause still
unknown, [GLY-262](https://linear.app/getglyd/issue/GLY-262) fixes control evidence/null handling.
Reviewed PR62 merged as55989bde757be6e4839e2038e50678f102eb9466 after all seven exact-sourceacf3ae6 CI38609367481 checks passed. Fresh device approval is still required.

The prior approved20-request comparison is consumed and frozen. It failed after
93.235s at a broad worker_readiness phase: all10 direct JSON/SSE requests matched,
worker reached Ready, no proxy generation; original control reply was discarded.
RSS peak1,685,716,992bytes, token rows0, protected ownership intact, no observed
group remaining,$0. Raw result95a65fb676f0b965e7c5198b20855af522b19258d318654cb2a0882d518af81e.
Model-free route/control sequence passes; it does not reproduce the original cause.
Null telemetry is valid and now waited for safely. Per-state200ms pacing and safe
control events reuse the existing recovery helper; precise failure phases distinguish
route hashing, activation, status and resource readiness. No production change,
deadline increase, inference retry or assumption that pacing fixes the old failure.

## Requested single-use bounds

- One M1 Max ARM64 Mac,32GiB, macOS27.0/26A428; cached CPU-only runtime, two threads.
- One attempt with one worker-owned service start; no direct baseline start.
- Zero generation requests, zero tokenizer/count API calls, zero output tokens.
- No inference client token; only the isolated coordinator/worker control plane.
- Five minutes including asset hashes/loading/cleanup; watchdog240s leaves cleanup time.
- Monitored4GiB aggregate owned RSS,100ms sampling; not an OS allocation guarantee.
- Loopback only,$0,cached assets; no download,install,rebuild,GPU,Ubuntu or remote host.
- No mutation/native experiment retry; read-only readiness observations stay within bounds.

Cached runtime66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398,
source d81235049384534c167caea52b85a694f6103d14, GPU backends disabled.
Manifest8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835;
2026 Qwen3.5-0.8B BF16/F32 GGUF artifact
cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5.
F32 KV,one slot,allocated context512/admitted128; caches/offload/repack/warmup
 disabled,offline text-only. Conversion fidelity remains unqualified.

Start only the prepared isolated worker. Require native Ready; attempt exact route
hash/activation/status/ready-capability checks; observe empty retained request/body
state. Record safe fixed operation/role/error codes and state/sequence/age, never
payloads, private paths, keys, prompts or generated text. A control failure stops
the attempt; no mutation is reissued. Success qualifies only this diagnostic sequence,
not JSON/SSE parity, direct/proxy performance, installed Glyd, two workers or Exo.

## Cleanup and review

Stop owned daemons/runtime, observe groups, verify zero token rows and protected
ownership hashes. Retain all new/old pins and worker quarantine. No recovery/reset/
unpin; group absence is not allocation cleanup proof. Observation failure stays
failed/unknown. Preserve the single-use marker and raw result on every outcome.
No temporary inference token is created; unexpected retained token rows fail.

Fresh native-free preparation/preflight passes, no attempt marker/model start.
All older ownership files and the newest consumed runner/result/approval/marker/
direct and worker pins/service-start quarantine are protected by frozen hashes.
AST checks reproduce old null failure and validate new readiness/pacing/privacy/
timeout/no-retry/phases and zero-generation/count/token paths. Existing stream/pin
checks and byte reconstruction pass. Correctness/security and separate Ponytail
review complete; independent milestone review deferred, not approved.

[CONTRIBUTING.md](/Users/suryakoritala/Documents/Codex/2026-10-11/referenced-chatgpt-conversation-this-is-an/work/glyd-compute-m2-acceptance/CONTRIBUTING.md) requires
“Obtain explicit device/budget approval before real workloads.” The previous plan
allowed one attempt with no automatic repeat and is consumed. This smaller native
startup still requires fresh explicit approval after the reviewed green merge.
Until then only `--preflight` may run. Approved command:
`python3 work/native-check/routing-profile-observed/run_control.py --run-approved`.
