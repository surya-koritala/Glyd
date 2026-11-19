# Per-worker resource view collection boundaries

[GLY-292](https://linear.app/getglyd/issue/GLY-292) corrects a resource-age
accounting defect found during GLY-267 investigation. On main
`f05d5280f87e12948c0a2a2383ae4212320f2a40`, the daemon captures one timestamp
before reading every worker. Each returned `ResourceView.age_ms` already includes
age at its individual authority read. Adding the entire batch interval to a later
view can withdraw a still-fresh worker because an earlier read was slow.

Capture an instant immediately before each worker's registry read and keep that
view/instant pair through `Pool::refresh`. Its age still includes its own read,
router-lock wait and processing time; no instant is moved after its read, and
the five-second limit is unchanged. Identical snapshots cannot renew original
freshness. Epoch fencing, record/endpoint/build validation, queue/authorization
limits and trusted drain ownership are unchanged. No clock or policy is relaxed.
The per-read boundary remains conservative about time spent inside that read.

The deterministic control first assigns both workers. During refresh the first
is either unavailable or stale through its own elapsed capture time; the later
worker reports age2500ms at its own current capture,
and the earlier batch boundary is3000ms old. Old production incorrectly cancels
the later worker; the corrected path keeps it active. A subsequent genuinely
stale5000ms view still cancels it. Both reservations remain until explicit trusted
drain in the fixture; no sample publication frees ambiguous work.

Qualified old-production baseline: one selected case FAIL0.03s, log SHA-256
`e2cbfa2d21c24203bb9e175fefd1bb48e26834f36f65169c5652a3aa04c4a440`.
Corrected production: all three routing controls PASS1.13s, log SHA-256
`8d90e047b4b05c57a4f46346404f372198f207bd9569b20c8bb62e6b77938fb6`.
The initial fixture changed snapshot ID without advancing its timestamp and
also hit the existing timestamp-order guard. Those initial logs are excluded
from causal proof. The qualified baseline restores both original production
files, adds the corrected test only, then restores the candidate source.
The experiment uses controlled capture instants rather than real delays.

Validation uses pinned Rust, `--locked --offline`, warnings denied:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test -p compute-cli --lib --bins --test cli --locked --offline
cargo test -p compute-cli --all-features --lib --bins --test cli --locked --offline
cargo test -p compute-cli --test daemon catalog_service_initialization_ --locked --offline -- --nocapture
cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
```

Exact source, PR, final outcomes and source/merged-main CI receipts are recorded
in Linear. Local formatting and both warning-denied workspace/all-target lint
modes pass, as do both CLI feature-mode suites, two real initialization boundary
controls in0.76s and all18 synthetic lifecycle cases in60.49s. Captured validation
log SHA-256: `c3893c13ff15797f84c9a2e9a50ee0f7abd38ac7bae203e9ec44712b231dc063`.
Final missing/elapsed-stale variants and all three routing controls pass1.13s
after the final lint/format pass (log
`5b1b10763742713a267d5a813718589a83e6e4d81f4e76d81f6eba5a48fa57ee`).
These controls prove an accounting defect, not which operation caused
an archived Windows stall or queue/registry/transport loss. GLY-263 and GLY-267
remain investigations; new green runs do not establish historical attribution.

Correctness/security review traces every `Pool::refresh` caller and per-read
timestamp, checked/saturating age, strict stale/missing withdrawal, repeated sample
and epoch handling, and reservation ownership. Separate complexity review uses
standard tuples and the existing registry/router validators; no new abstraction,
dependency, timer, CI job/filter/cap or backend integration is introduced.
Independent milestone review remains deferred, not approved. No native/model/GPU,
Ubuntu/SSH, download/spend, old-state recovery or quarantine reset is involved.
M2 provider/hardware/native benchmark qualification remains open; no new M3 work.
