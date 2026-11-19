# Redacted local diagnostic bundle

Run this explicitly against the profile whose foreground daemon is already running:

```sh
starlings cluster diagnostics
```

Use the normal `--state-dir ABSOLUTE_PATH` option when selecting another profile.
The command prints one JSON object to stdout. It creates and uploads no bundle
file. Save or share the output only when you choose to do so. There is no automatic
collection, telemetry, service start, host/process inspection, discovery, probe,
network connection, model execution or backend command.

The CLI makes exactly three existing authenticated private-IPC queries, once each:
`status`, `nodes`, `service`. It reuses the protected local endpoint and capability
verification; it accepts no remote address or new daemon operation. No mutation
is retried and there is no per-node fan-out. Each query retains the existing local
connect/request deadlines. Suspension and local filesystem stalls are not an OS
watchdog guarantee. The three sections are independent observations, not an atomic
registry snapshot or authority/lease renewal.

## Contents

| Field | Exported values |
| --- | --- |
| version, program_version, os, architecture | Bundle v1 and this CLI's build/platform labels; no host name or device identity |
| qualification | Always `not_assessed`; diagnostics grant no approval or admission |
| routing, layer_sharding | `whole_request` and `not_qualified`; this build's daemon route distributes independent requests, while offline layer planning is advisory |
| daemon.data | Fixed role and registry labels, draining flag, connection/handshake/pending-admission/pending-approval counts |
| members.data | Counts of enrolled workers/clients and online/offline/draining/revoked members, plus schedulable workers; no per-node identifiers |
| service.data | Fixed lifecycle state, fixture/inference-qualified/process-group-absent/cleanup-proven flags and nullable reported reserved bytes |

Every section has `status: available`, `unavailable` or `invalid_report`.
`service` available with `data: null` means no service is configured. Unknown
reserved bytes stay null, not zero. Service flags/reservations are the existing
daemon's reports, not new device measurements or a cleanup/qualification proof.
Member scheduling counts do not prove native model support. The bundle omits model
aliases/profiles; use the existing privileged local model/resource views for those
details rather than attaching them to a support message automatically.

Each raw reply is capped at16 KiB before projection and a membership list at64
records. Duplicate JSON keys and invalid typed values fail the affected section.
Known textual status fields are checked and copied only as fixed labels/numbers;
unrecognized roles/states, repeated fields and overflowing counts fail closed.
Unknown service fields are discarded rather than exported. No addresses, node/
cluster IDs, paths, pins, raw errors, logs, runtime arguments, environment values,
credentials, prompts, output text, tokens or tensors enter the output schema.
The exported platform, role and counts are still operational metadata.

## Private link snapshots are a separate output

The bundle above intentionally omits per-node link evidence. After a separately
approved profile attempt, `cluster links-record` can retain its original samples
and failures in a **new private JSONL file**. On the intended already-running
profile:

```sh
starlings --state-dir /absolute/private/profile cluster links-record WORKER_NODE /absolute/private/review/new-link.jsonl
```

`WORKER_NODE` is the enrolled worker UUID. The selected profile and private
output parent directory must already exist; existing output files are refused.
This command performs one read-only owner-IPC query. It never starts,
waits for or retries a probe, and saving a pending/failed/stale snapshot is not a
successful experiment. See the [profile contract](token-link-profile.md) for
configuration, sampling, state and interpretation.

Unlike the redacted bundle, a link snapshot retains node IDs, LAN addresses and
producer/collector executable digests. It is private raw diagnostic evidence,
not automatically suitable for a support message. Neither output qualifies a
backend, revives freshness, restores a revoked node or supplies a benchmark win.
No snapshot or bundle is uploaded automatically.

## Reading an unavailable result

`unavailable` means the existing local query was refused or could not complete.
Check that the selected daemon is running, the role supports that query and private
endpoint permissions are intact. A worker/client cannot list the coordinator's
members; its member section may therefore be unavailable. No failed section is
replaced by successful-looking zeroes, and successful sections remain present.

`invalid_report` means the reply code, size or projected fields did not match this
CLI's supported contract. Check CLI/daemon build compatibility. Neither case
includes the raw error or authorizes retrying an ambiguous mutation. Output errors
produce `DIAGNOSTIC_WRITE_FAILED` and a nonzero exit without panic. Exit zero means
a valid diagnostic bundle was written, not that every section or the cluster is
healthy.

This implements GLY-197's bundle software. Native readiness/allocation and placement
explanations still depend on their actual backend/planner integration. There is no
native sharding, benchmark, hardware-support, release or independent-review claim.
