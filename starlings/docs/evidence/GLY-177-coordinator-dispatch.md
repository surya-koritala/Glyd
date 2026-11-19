# GLY-177 coordinator controls

`inference::Dispatch` owns serial control sequences for one worker and full
authority epoch. It validates a locally qualified whole-request plan/profile,
counts the exact admitted request on the same authenticated channel, then builds
fresh plan/lease/session/invocation/request identities using the existing UUID
library. Existing protocol validators check every record before dispatch.
Reconnections retain the owner and sequence; only a new durable authority epoch
may construct a replacement owner. Failed construction consumes sequence numbers;
exhaustion refuses further dispatch rather than wrapping.

Client grants now retain the authenticated controller identity from the same token
validation. Session ownership comes from that grant, never the request body. The
relay independently checks session ownership. Maintenance still validates the
same client, expiry and scope without adding work to the output enqueue path.

Synthetic authenticated HTTP/TLS/service tests use the coordinator builder for
JSON, SSE and large Unicode output. A focused check covers count/context limits,
client binding, unique identities, gaps after failure and sequence exhaustion.
Both warning-denied lint modes and the security/relay checks pass locally; all
platform and dependency CI checks gate merge. Exact PR/head/CI evidence is tracked
in GLY-177. Correctness/security and Ponytail reviews are separate; independent
review is deferred, not approved.

The caller still owns qualified placement and live resources. This is not native
qualification or daemon activation. Catalog pin lifetimes, measured profiles, daemon configuration and broader native
acceptance remain M2 work. No model or device workload ran for the software checks
described above. A subsequent separately approved native relay run is linked below.

An ignored native relay acceptance check reuses the same enrollment, client,
worker and service test harness. It runs three requests of at most eight output
tokens: an adapter-direct reference, coordinator/worker JSON and coordinator/
worker SSE. Two count calls precede the relayed requests. Registry heartbeats and
policy maintenance continue independently during backend waits. Each service
instance must report original-group absence before the harness releases its
provisional reservation. The same harness runs with a synthetic child in CI.

Native execution requires an explicit approved configuration digest and workload
approval. It targets the retained 24-block Qwen3.5-0.8B BF16 artifact only. Its
4 GiB plus existing safety budgets are a conservative test allowance, not measured
production capacity; the artifact hash is a declared smoke identity, not a sealed
catalog manifest. Output records explicitly leave catalog/memory/device-drain/
daemon qualification false. No native acceptance is claimed by compiling this test.

Client failures before SSE headers now use a bounded, redacted HTTP error through
the original live authorization gate. Nonstreaming failures return one JSON error;
streaming failures after headers retain one terminal SSE error without DONE.
Neither path retries generation. Tests cover invalid backend counts before any
invocation, JSON/SSE backend crashes, and suppression after grant revocation.

Run 38300047106 failed in Linux synthetic service readiness. Review found the
fixture selected and released an ephemeral port before its child bound it. The
fixture now binds port zero in the child and reports the actual bound address;
all adapter and relay synthetic callers share that test-only launcher. Production
configuration approval and native runtime launch remain unchanged. Startup and
readiness each remain bounded, and startup checks child liveness for early failure.

Run 38301085456 exposed a separate ARM64 harness shutdown race: the registry
server could retire while its heartbeat client was still connecting, leaving a
closed socket queued for the next acceptance phase. The server now remains live
until the heartbeat thread has finished, including scope teardown. Repeated
short maintenance phases exercise this ordering. Active heartbeat failures still
fail the test; no authorization deadline or production timeout was relaxed.

The separately approved [M1 Max CPU relay acceptance](native/2026-10-12-m1max-cpu-relay/README.md)
passed on the merged head: direct/JSON/SSE semantic parity, exact prompt count/usage
and original process-group cleanup. This does not activate production daemon
inference or qualify catalog assets, memory reservations or device drain.
