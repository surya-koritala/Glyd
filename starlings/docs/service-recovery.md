# Qualified CPU service recovery

Recovery releases cache ownership only after the trusted local supervisor records
completed cleanup for the exact launch. There is no force reset, imported receipt,
PID lookup or automatic restart. Legacy/unqualified quarantines stay protected.
Request slot idle does not qualify whole-service cleanup.

On daemon service startup, the protected catalog is opened and existing ownership
checked before waiting for registry contact. An unresolved or invalid local record
reports `SERVICE_CLEANUP_REQUIRED` even while the coordinator is unavailable.
A clear or recovered catalog still requires the unchanged online registry gate
before configuration validation and launch. No offline launch or automatic recovery
is permitted. The catalog lock remains held through registry wait and supervision;
stop or failed registry admission releases it without creating a launch record.

## Policy before launch

An approved `WorkerRoute` may contain `cleanup`:

| Field | Requirement |
| --- | --- |
| `version` | 1 |
| `configuration`, `target` | Exact service configuration and local qualification target |
| `evidence` | Nonzero digest identifying reviewed native cleanup evidence |
| `scope` | `posix-cpu-group-v1` |
| `no_device_work_reviewed` | true; reviewed executable/build/arguments/dependencies use no device allocations |
| `no_escaped_children_reviewed` | true; reviewed process topology leaves no allocations outside the owned group |

This is trusted operator policy, not remote attestation or a worker claim. Its
complete content participates in the existing worker-record approval. Only CPU
Linux/macOS profiles are accepted; GPU/device work, escaped/external helpers and
unqualified Windows job completion are excluded. Missing policy retains prior
behavior. Policy cannot be retrofitted to an old launch or an unconfirmed crashed
supervisor. Audit the exact installed executable, dependencies, model/template,
environment and startup/request paths before issuing it. CPU declaration or group
absence alone is insufficient. The repository supplies synthetic fixtures, not
native cleanup policy; historical native records remain unqualified for recovery.

## Durable ownership

`model service-state` reads protected `service-start.json` under the catalog lock.
Version 2 binds `lease`, `manifest`, `configuration`, optional
`cleanup_qualification` (policy digest) and `phase`:

1. `quarantined`: durable pin/record before spawn; protects objects independently
   of ordinary pin loss and refuses another launch.
2. `drained`: the same local owner joins generation/handshake work, confirms the
   qualified original process group stopped and applies its no-device/no-escape
   contract. It acknowledges full backend drain before persisting cleanup. Objects stay
   pinned; the old worker refuses execution/output permanently and preserves its
   epoch/history instead of resetting or reviving work.
3. `recovered`: explicit local recovery removes the matching pin, synchronizes its
   directory, then atomically publishes the terminal record with full durability.
   Only then can GC or a fresh launch proceed. Other pins still protect objects.

Version 1 means unqualified quarantine. Invalid records, wrong identities,
conflicting pins and missing qualification fail closed. An interruption after
unpin leaves the drained marker protecting objects until explicit retry. Pin
absence is made durable before recovery publication. Uncertain atomic publication
is safe to re-read/retry for that same launch. Recovery is idempotent for the last
launch; an older receipt cannot recover a fresh replacement. Launch IDs must be fresh.

## Commands and limits

- `cluster recover-service LEASE CONFIG_DIGEST`: protected admin after the service
  thread finishes and its reservation is zero.
- `model recover-service LEASE CONFIG_DIGEST`: catalog recovery after durable proof
  is already recorded, including after supervisor restart.
- `model service-state`: exact lease/configuration/phase; an active catalog owner
  refuses an independent reader.
- `cluster service`: `cleanup_proven`, pin/reservation and
  `stopped_cleanup_proven`, `failed_cleanup_proven` or `recovered` state.

Failure stays failure; recovery cannot produce success/DONE or revive its stream.
Commands start no backend, approve no new profile and change no permissions. Fresh
daemon activation still needs full record/workload approval; no same-process
reactivation is added. Unqualified failures retain incomplete drain/quarantine.
Missing proof after a crash, failed spawn or interrupted publication stays
quarantined; no offline guesses or device/escaped-work recovery are supplied.
Filesystem stalls have no hard bound; durability follows the supported protected
local-filesystem contract. Windows retains quarantine until job-wide completion
has its own qualified contract. Fault fixtures are not power-cut/native proof.
