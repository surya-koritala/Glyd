# Authenticated probe session evidence

Starting main022ed1e2e5da204d41119bdeafd7970aee995dfe. GLY-174's library sends
bounded real TLS ping/upload/download traffic, separately admitted from persistent
registry traffic. Original code reuses signed reconnect proof, TLS configuration,
live authorization/epoch/fence checks and fixed-buffer I/O; no Exo implementation,
private engine code, networking framework or dependency is copied/added.

The model-free test enrolls a real worker credential, maintains a persistent
heartbeat slot while a separately budgeted listener transfers payload in both
directions, verifies exact application byte accounting and non-invented RTT fields,
then checks cooldown, age/rollback withdrawal, revocation and new-attempt refusal.
Another deterministic check rejects malformed limits/unknown fields/public routes,
caps cooldown table size and preserves distinct synthetic asymmetric/congested
direction values. Synthetic values are labeled fixtures, not hardware measurements.

This is actual authentication/transfer integration rather than only a parser.
It is still incomplete GLY-174 acceptance: daemon activation/status/retained server
freshness, a larger adverse-peer/fault matrix and calibrated planner-input replay
remain open. Nothing here establishes native model execution, physical LAN goodput
or a five-run matched Exo win. The owner baseline is read-only reference material;
non-2026/mixed/patched old rows are not promoted to target claims.

Validation outcomes, exact tested head/PR and all five CI checks are linked in
Linear. Full default/development lint denies warnings; model-free regressions
preserve accepted M0/M1 and merged PR42 code. Review separates correctness/security
(identity, limits, records, credential/epoch checks and publication) from complexity
(existing crypto/transport, native collections/math/buffers). Independent review
deferred under current authorization, not approved. No model/device workload,
hardware purchase, download, spend, settings change or old quarantine reset.
See [contract and explicit limitations](../directed-probes.md).
