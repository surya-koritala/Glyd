# Opt-in diagnostic token-link profile

GLY-241 adds a bounded persistent-connection experiment using the existing
[explicit probe workflow](directed-probes.md). This exchanges synthetic bytes;
it does not execute a model, qualify GCP1/native token scheduling, admit a worker
or reproduce physical-link/power results. Running it on devices requires a
separate approved target, traffic/time budget and cleanup plan.

## Limits and protocol

Leave `token_profile` absent for the original probe. Its canonical digest,
1–8 small-marker pings, 1 MiB combined application limit and five-second attempt
remain unchanged. An older peer rejects the new field; there is no downgrade.
The new mode is included in both exact approved configuration digests:

```json
{"version":1,"payload_bytes":16384,"ping_samples":100,"inter_ping_gap_ms":20,"minimum_interval_ms":30000,"token_profile":{"gap_basis":"start_to_start","keepalive_ms":null}}
```

`ping_samples` is 5–100, payload exactly 16384 bytes, gap20–100ms. Keepalive is
null or5–20ms, strictly shorter than the gap. `start_to_start` schedules the next
request no earlier than the previous start plus gap; it never overlaps requests
or catches up missed ticks. `after_reply` adds the gap after the preceding reply,
for explicit idle-wake comparisons such as100ms idle with/without5ms keepalive.
These are separate configurations, immutable for each daemon lifetime; restarting
an isolated experiment requires its own approval. Keepalive exists only during
planned inter-request pauses and sends no burst after delayed scheduling.

Signed reconnect proof, mTLS/ALPN, enrollment/role/cluster, live epoch/revocation,
matching Hello and the old five-second setup deadline must pass before profile
activation. Both ends set/read TCP_NODELAY and hash their running executable as
file data (maximum128MiB), checking the old deadline again before extending it.
Client's absolute attempt cap is30s including setup; server has at most30s from
matching authenticated Hello. These deadlines never reset per frame. The body
has a2MiB combined application cap, fixed4KiB buffers, at most100 exchanges and
2000 keepalive frames. Existing cooldown/node/concurrent-job limits apply even
on failure. There is no automatic retry or general background keepalive.

One persistent authenticated stream carries each16KiB request (opcode1 included,
remaining bytes0x5a) and one64B reply (session UUID, little-endian sequence,
remaining bytes0x5a). Opcode0 is a one-byte keepalive with no reply and is refused
when disabled. Opcode2/final receipt2 closes a complete sample sequence. Unknown
opcodes, excess samples/wakes, early finish, invalid bytes, EOF and failed live
checks fail the attempt. Control/TLS packets are not inference tokens.

## Records and interpretation

Client samples retain sequence, actual monotonic start offset, RTT, successfully
flushed keepalive count and static error code. Root failure phase/error and prior
samples survive an exchange or final-receipt failure. Initial setup failures retain
a static `setup_error`; they have no fabricated sample report. Server records have
completed reply flushes and no invented RTT samples; a successful server flush
alone does not prove client receipt. Client and server windows remain distinct.

Integer mean, average squared deviation from that truncated integer mean and
nearest-rank p50 use successful client RTTs.
p99 is null below100 successful RTTs; at100 it is the observed99th sorted sample,
not a population-tail confidence claim. First exchange is retained, not silently
removed as warmup. Actual offsets reveal missed cadence. Failed attempts can have
partial statistics but are always failed, including a lost final receipt.

Directional payload totals use only completed exchanges. Total application bytes
also include Hello/control/keepalive and partial I/O, but exclude TLS/proof/wire
framing. Rates divide directional completed bytes by the entire profile window,
including deliberate pauses. They are application-window rates, not raw link
bandwidth, model throughput or a stage-cost calibration.

`cluster links NODE` reports pending, profile_complete, failed or unavailable;
`measurement`/`placement_inputs` are always null for this mode. The explicit
`profile_attempt` diagnostic survives age/epoch/revocation withdrawal until
replacement/restart, with original age and `admission=false`. Neither a historical
snapshot nor a current profile can become a placement input. Both producer digest
and `native_qualified=false` are retained.

After one approved attempt, on the enrolled worker's protected profile:

```sh
starlings cluster links WORKER_NODE
starlings cluster links-record WORKER_NODE NEW_PRIVATE_JSONL
```

The second command makes exactly one existing read-only owner-IPC call and writes
one JSONL snapshot to a new private file; it does not start, wait for or retry an
experiment. Check `state`, configuration, sample/attempt ID and failures before
interpreting it. A pending snapshot is not completion. The record includes UTC
capture time and collector/producer executable SHA-256, so a later collector
cannot silently relabel producer code. Exporter drops unrelated root error text,
interface names and unknown fields; nested errors are fieldless codes. It retains
LAN addresses/node IDs, so keep it private. Existing files are refused, and write
failure may leave a partial new file that must not be interpreted as valid JSONL.

Keep separate client/server snapshots, exact source commit/build and approved
physical hardware/link/OS/power inventory in a run directory. Record failed trials
as well as successes. No physical-link type, power/radio result, source security
approval, native readiness or Exo win is established by software tests. RDMA is
not implemented; custom UDP remains no-go pending contrary measured evidence.

The [transport adapter boundary](transport-adapter-boundary.md) defines the future
RDMA seam and current custom-UDP no-go without an unimplemented runtime trait.
