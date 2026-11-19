# Prepared direct/proxy CPU comparison after passing control diagnosis — unapproved

Candidate: `/Users/suryakoritala/Documents/Codex/2026-10-11/referenced-chatgpt-conversation-this-is-an/work/native-check/routing-profile-control-verified`. Fresh isolated state: `/private/tmp/glyd-m2-daemon-3-ba359143`.
Runner SHA256 `00d867278eb06343daf1506c1b3930ea2068e1ab593146d772b0183d8fda2c4c`; preparation SHA256 `09dd6368274fb91865cb9f3b0f3d86f4cb226a5f1a1340eb970c2044579d9ad7`.
CLI SHA256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`; implementation `78b737495ff52fc69e05b928a460f400fcd4af70`.
Service configuration `a435d9fc94579291bd444c24ddebe07a41ea3a801c01772ed624a21e2cbbac48`; worker record `d32e0e9155e221684d1b849a3c0453dbc1a01afcd3c574a2d016778f26ad0328`.
Worker file `a8478894ad047797ef5e201e6e02e3c4136255bca866d733878de29b5befce39`; route template `83d0b6ca165c44958205387272dd6faa967bfc14f71e62e07250f163a3bb83d0`.
Pin receipt `171c90c2b5cd998d977bb2b35008f7034a481381fb4e995cacfe41a8f775e1b6`; durable pin `7642798ae85f5d1b0ff6cac04423fa077166e6f6c30eba1d3b1dc0983f903676`.
Production Rust/manifests/lockfile match merged PR62/55989bde; PR63 contains evidence only.
[PR63](https://github.com/Sigbound/glyd-compute/pull/63) records the separate approved control diagnosis:
30.023s PASS, route activation and ready telemetry, zero generation/count calls, $0,
no owned group remaining observed, token rows0, ownership intact. Its approval is consumed.
The earlier93.235s comparison remains failed; original discarded control error unproved.
GLY261 remains open. Passing this new comparison will not retroactively identify that cause.

## Requested single-use envelope

- One M1 Max ARM64 Mac,32GiB,macOS27.0/26A428; two CPU inference/batch threads.
- One attempt; two sequential runtime starts (direct then worker-owned), never concurrent.
- 20 requests total:10 direct/10 proxy; five JSON/five SSE per path; output≤8 each/≤160 total.
- At most41 tokenizer/count calls by code-path bound, not separately instrumented.
- Fixed benign17-token prompt, seed42,temperature0,top-p1; no user data.
- Five minutes including verification/loading/cleanup,240s watchdog leaves cleanup time.
- Monitored4GiB aggregate owned RSS at100ms; sampling is not an OS allocation guarantee.
- Loopback HTTP/TLS only; one temporary inference token scoped to this isolated route.
- Cached assets,$0; no download,install,rebuild,GPU,Ubuntu,remote or paid work.
- No automatic mutation/generation/native retry; any failure ends the attempt.

Runtime66b94f533a684c97b8954b80a1bda9eb952f8e924f15c5eb16e71a36209e1398,
source d81235049384534c167caea52b85a694f6103d14; GPU backends compiled out.
Manifest8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835;
2026 Qwen3.5-0.8B BF16/F32 GGUF
cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5.
F32 KV,one slot,allocated context512/admitted128,caches/offload/repack/warmup disabled,
offline text-only. Conversion fidelity remains unqualified.

Use the reviewed GLY262 full-comparison runner unchanged: safe allowlisted control
codes, exact hash/activation/status/resource phases, existing200ms per-state pacing,
null-aware readiness and direct-only stream_options.include_usage. Compare response
content/reasoning/finish/usage in memory, never record prompts/generated text or tokens.
Require coherent prompt17/output≤8/total usage, direct idle, proxy trusted drain and
zero retained request/body bytes. Retain per-request elapsed times and raw overall
success/failure. These times include generation/decode/drain and fixed stage order;
no causal overhead, TTFT/TPOT/throughput, installed Glyd, two-worker, GPU/Metal or Exo claim.

## Cleanup, checks and approval

Revoke the temporary token; stop owned daemons/runtime; observe process groups;
verify zero token rows and all31 protected prior ownership hashes, including the
now-consumed control result/approval/marker/pins/quarantine. Retain new direct/worker
pins and launch quarantine; no reset,recovery,unpin or old-file mutation. Unknown
cleanup observations remain failures. Preserve new raw result and single-use marker.

Fresh native-free catalog/control preparation and preflight pass; no started-once
marker or user approval exists. Existing GLY262 null/pacing/privacy/timeout/no-retry/
phase checks, GLY259 stream usage checks and GLY257 prepared-pin checks pass against
these exact bytes. Correctness/security and separate Ponytail complexity review
reuse the existing runner and stdlib; no new production code or abstraction.
Independent milestone review remains deferred, not approved. M2 stays open; M3 unstarted.
Execute only after PR63 passes all seven exact-head checks and merges, and after
fresh explicit approval. Until then only --preflight may run.

[CONTRIBUTING.md](/Users/suryakoritala/Documents/Codex/2026-10-11/referenced-chatgpt-conversation-this-is-an/work/glyd-compute-m2-acceptance/CONTRIBUTING.md)
requires “Obtain explicit device/budget approval before real workloads.” The completed
control approval allowed zero generation and one start; it cannot authorize these
20 generation requests or a second service start. Every earlier single-use approval
is also consumed. This full comparison needs separate explicit device/budget approval.
Command once approved: `python3 work/native-check/routing-profile-control-verified/run_profile.py --run-approved`.
