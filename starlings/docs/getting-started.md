# Operator entry guide

Starlings is a development preview. It coordinates trusted local devices and
routes complete requests to explicitly qualified external CPU services. Real
single-model layer sharding, general installed Glyd backend support and packaged
desktop installation are not qualified. There is no automatic model/runtime
installation or production activation profile in this repository.

## Start without a service or model

From the repository root, with the pinned Rust toolchain and locked dependencies already
installed, build the default CLI and inspect its scope:

```sh
cargo build -p starlings-cli --locked --offline --jobs 2
./target/debug/starlings --version
./target/debug/starlings doctor
```

On Windows use `target\debug\starlings.exe`. `doctor` prints the compiled
protocol, OS and architecture with `inference=unavailable`; it does not inspect a
backend or qualify hardware. Missing toolchain/dependencies require the explicit
[development setup](../README.md#build-and-check), not an implicit download.
The development `test-backend` feature and simulation are not a native backend.

Choose the next guide for the operation you actually need:

| Operation | Guide and boundary |
| --- | --- |
| Create a cluster and enroll a device | [Daemon guide](daemon.md#two-machine-enrollment): explicit reachable local address, private invitation transfer and fingerprint comparison; init/join alone enables no inference. |
| Register existing model files | [Catalog](catalog.md): inspect/seal/register an exact manifest, review its source/license, approve locally, then import existing files and check readiness. Content hashes do not prove model support. |
| Use an approved whole-request service | [CPU routing](production-routing.md): obtain a reviewed exact service/worker/coordinator qualification first. Do not turn example/fixture records or historical experiments into current approval. |
| Understand HTTP behavior | [Client HTTP](listeners.md#client-http), [chat fields](../spec/chat-request.schema.json) and [CPU routing](production-routing.md): scoped credentials, exact supported text fields, context/output bounds, JSON/SSE and cancellation; unsupported fields fail. |
| Preview placement offline | [Partition preview](partition-preview.md) or [mode selection](mode-costs.md): one bounded supplied-data file; advisory results reserve nothing and never qualify execution. |
| Read a comparison | [Scoreboard](scoreboard.md): exact matched records and evidence; missing measurements remain missing, including absent second same-type machines. |
| Stop or recover a service | [Daemon lifecycle](daemon.md) and [qualified recovery](service-recovery.md): stop/drain retain uncertain ownership; recovery requires the exact durable cleanup proof. |

The artifact guide distinguishes offline `import`/`ready` from explicit network
`resolve`/`fetch`. Read the target manifest and storage budget before a download;
no model is fetched by the commands above. Request routing sends each request to
one complete replica. A placement preview or tensor library is not a working
single-model pipeline across machines.

## Evidence and support limits

Task-only means one complete request executes on one eligible service; it does not
split a model. Experimental describes development software or evidence limited to
an exact configuration. Layer-ready would require a delivered partial worker,
conformant boundaries, real allocations and trusted native drain; no configuration
is currently qualified as layer-ready. Unsupported below names an absent current
execution path, while unqualified means the required evidence is missing.

| Surface | Current classification | Evidence | What is still required |
| --- | --- | --- | --- |
| Linux x64/ARM64, macOS and Windows Rust software | Experimental control plane | [Configured CI coverage](../../.github/workflows/starlings-ci.yml), both feature modes; inspect the exact source/main run for its actual result. | CI is not native model/device qualification; ARM64 does not certify a Raspberry Pi. |
| Mac–Ubuntu enrollment/control | Experimental control plane | [M1 physical evidence](evidence/M1-hardware.md), with exact historical source and rerun scope. | It does not establish current inference, GPU, mixed-family or two-node same-family support. |
| One Mac CPU JSON/SSE daemon route | Experimental, task-only | [Bounded native acceptance](evidence/native/2026-10-12-m1max-cpu-daemon-acceptance/README.md), exact historical source and fixed inputs. | No general arbitrary-prompt memory bound, installed Glyd qualification or current benchmark win. Its one-shot approval is consumed. |
| CPU service recovery | Experimental, task-only | [Recovery contract](service-recovery.md) and [historical bounded evidence](evidence/native/2026-10-12-m1max-cpu-recovery-acceptance/README.md). | Only the exact preapproved cleanup contract/launch can recover; other quarantines remain protected. |
| General installed Glyd service | Unqualified | [External service contract](service-adapter.md); no delivered package/API mapping is established by this table. | Pinned provider delivery and actual capability, memory, cancellation and recovery conformance. |
| NVIDIA or Apple GPU production routing | Unsupported by the current route | [Production routing](production-routing.md) currently accepts CPU/host-pool configurations only. | Delivered backend/device profiles, GPU route integration and approved native qualification. |
| Directed probes and persistent link profiles | Experimental diagnostics; no native admission | [Directed probes](directed-probes.md) and [bounded token-link profiles](token-link-profile.md) distinguish the original transfer probe from the opt-in 16 KiB/64 B experiment. | An approved device/link experiment, physical-link and power records; no model speed or stage readiness follows from a profile. |
| One model split across NVIDIA or Mac nodes | Unsupported; no layer-ready configuration | [Partial-worker proposal](partial-worker-contract.md) and [transport library](data-framing.md) are not a deployed native pipeline. | Actual range execution, whole-plan allocation/readiness, boundary equivalence, drain and same-type hardware evidence. |
| Private Linux GNU x64/ARM64 candidate archives | Experimental native build/archive verification | [Exact-source Linux batch](evidence/GLY-198-native-candidate-verifier.md#actual-linux-native-builds--17-october-2026): both jobs passed; no candidate execution or retained download. | Installed lifecycle, minimum OS/libc/ABI, WSL2 and actual backend/device qualification; license and release review. |
| Mac candidates and inactive per-user service descriptors | Experimental development/review files | [Historical Mac build/static CLI checks](evidence/GLY-198-candidate-packaging.md#actual-mac-candidate-check) and [descriptor export](packaging.md#inactive-service-descriptor-export); exporting files installs or starts nothing. | Current-revision executable and installed-service acceptance, stable binary installation, effective arguments, lifecycle, upgrade/rollback/uninstall and release review. |
| WSL2 installation and GPU/network behavior | Unqualified | [Packaging](packaging.md) distinguishes Linux-format output from WSL2 lifecycle/device qualification. | Actual WSL2 installation, networking, device and lifecycle evidence on an approved configuration. |

Mixed NVIDIA/Apple, mobile and physical Raspberry Pi work are parked under the
current scope. No throughput, speed, memory-saving percentage or production-readiness
claim follows from this table. Historical evidence remains tied to its source,
runtime, device and workload; a later green CI run does not extend it. The
[matched scoreboard](scoreboard.md) retains missing, failed and unsupported
comparisons; no best-working Exo win follows from a source pin or synthetic run.

## Choose the diagnostic output

The [redacted diagnostic bundle](diagnostics.md) reports state and counts without
node IDs, addresses, paths or raw errors. Inspect it before sharing.
A [link snapshot](diagnostics.md#private-link-snapshots-are-a-separate-output)
retains identifiers, LAN addresses, executable digests and profile samples/failures;
keep its JSONL file private. Both read the selected existing daemon. Neither
starts or retries an experiment, and a saved pending/failed/stale snapshot grants
no freshness, placement or inference permission. New profile attempts need a
separately approved configuration and workload plan; these diagnostics are not
required for the static build/version/doctor steps above.

## Interpret failures before changing state

Use `cluster status`, `cluster nodes` or `cluster service` only against your
already-running approved local daemon. [`cluster diagnostics`](diagnostics.md)
produces a redacted JSON report from those existing local queries; it starts no
service or probe. It excludes raw logs, IDs, addresses, paths and credentials,
but its state/count information is still yours to inspect before sharing.

| Observation | Action |
| --- | --- |
| `STATE_DIR_NOT_FOUND` | Check the intended existing profile path; do not initialize over another profile. |
| `CONTROL_EXPIRED: not executed` | The queued command did not start; inspect current state before deciding the next action. |
| `CONTROL_UNAVAILABLE: outcome unknown; check status` | Do not automatically retry a mutation. Consult the matching observation command in the [daemon guide](daemon.md); absence/error is not proof that a pending write cannot finish. |
| `SERVICE_CLEANUP_REQUIRED` | Keep the prior launch and pins intact. Follow [qualified recovery](service-recovery.md); never delete markers or infer drain from a PID/health result. |
| Unavailable/stale registry or resource information | Treat the node as unavailable; do not edit timestamps, freshness limits or qualification flags to make it eligible. |
| `qualification=not_assessed` / missing benchmark cells | Supply reviewed native evidence when available. These values are not success or permission to run a workload. |

Keep identities, invitations, credentials, configuration and raw model/prompt data
private. This guide requires no scan, traffic flood, process inspection or security
control change. If testing raises an alert, stop that activity and report it.
Do not disable alerts to continue. Real device/model workloads need their explicit
target and budget approval. The Mesh license remains undecided; packages remain
`publish=false`. Clean-machine packaged acceptance and release approval are open.
