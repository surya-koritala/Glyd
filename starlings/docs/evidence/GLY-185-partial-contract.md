# GLY-185 — partial-worker contract software

Issue: https://linear.app/getglyd/issue/GLY-185

The [contract proposal](../partial-worker-contract.md) distinguishes current
whole-request integration, a pinned public CLI source audit, proposed native
operations, implemented metadata/fixture code and undelivered provider work.
`compute_adapters::partial` reuses strict bounded JSON, profile keys, typed scopes,
scalars and checked layout arithmetic; no new dependency or private source.

Implemented: ordered input/output tensor patterns; checked full-tensor layouts;
complete RunChunk metadata validated against a trusted current step; explicit
fixture rejection; and a test-backend-only in-process byte-echo lifecycle.
The fixture cannot advertise native support/costs, free busy buffers on cancel,
execute an invocation twice or bypass context/capacity limits. Its limits and
lack of native process/handle/replay support are stated explicitly. The public
wrapper is audited at fb73c7eef45218790077535debe5054bab51d02a; raw SHA256
f94650ba7de793228687a6f193392393338c181db7b0b2ae5c18d945ff61ad70.
It dispatches run/serve to the installed GPU package; no package API is inferred
from that wrapper and no private implementation was accessed/copied.

Validation on pinned Rust1.97.1, model-free only:

```sh
cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
RUSTFLAGS='-D warnings' cargo test -p compute-adapters --lib --locked --offline
RUSTFLAGS='-D warnings' cargo test -p compute-cli -p compute-adapters --all-features --lib --bins --test cli --locked --offline
```

All pass: default adapter19; feature adapter23; CLI library21 (one existing native
check ignored), bin10 and CLI7. Six new controls cover strict fixture/unknown/
duplicate/version fields, role/dtype/rank/token axes, true u64 overflow, combined
payload budgets, four dtypes and sampled-one layouts, complete ordered local
steps, reassembled tensors larger than one8MiB network frame, duplicate execution,
raw hidden/residual bytes, cancellation/retained ownership, Vec capacity and
prefill/decode/output/context limits. These are API/fixture results, not native
inference, layer arithmetic, model exactness, RSS/GPU memory or performance proof.

Raw SHA256:

| Check | SHA256 |
| --- | --- |
| Default lint | 0287931b46ae289ab87c92f74d35e6e0e1ee7b9653e23e896e6895462e449cc1 |
| All-feature lint | 5fb3b4b4745994880d67d1303966d3edc3dcbc4e9aaab7519c6905a2133a51e8 |
| Default adapter regressions | d826c86a21659fee4950719e918baeec73167dca0039325698fd9d976203ccb5 |
| Feature adapters/CLI regressions | c7a9ba561924eff526708e2189e34bc0f1c25c9f1ab164a5984961da87bcd79f |

Correctness/security review checked that incoming metadata cannot choose trusted
step state; placeholder scopes and fixtures fail in production mode; layouts are
fully ordered/checked before payload use; duplicate/cancelled/over-budget runs do
not execute; synthetic completion alone releases pending fixture ownership; and
public source/schema validity never proves native compatibility or authority.
Full coordinator incarnation/live fences and native per-pool allocation remain
required at the real owner; DataScope's decimal epoch alone cannot satisfy them.

Separate complexity review kept direct typed records and one bounded fixture,
reused existing validators and one shared JSON example, and added no native
runner, generic backend trait, broker, transport, retry framework, CI job or
new dependency. No existing test, deadline, platform, precision or warning gate
was weakened. Production core/security/daemon behavior is unchanged; final source
CI still exercises the normal complete matrix. The additional fixture correction
and current Windows failures are recorded below. Independent milestone review
remains deferred, not approved.

Keep GLY-185 In Progress: actual installed-provider API mapping/probe, agreed native
process/envelope/handle/replay/ownership implementation and qualified external
worker remain open. GLY-235/236 delivery, GLY-186 numerical/state boundaries,
GLY-187 integration and approved native/device evidence are not supplied here.
GLY-183 may reuse metadata, but flow/reassembly and backend dispatch remain to
implement; no Native StartStep check is removed or bypassed. This does not finish
M2/M3 acceptance. Record exact source/CI/merge and automatic-main evidence in
Linear after their actual outcomes; no model, GPU, SSH, download or spend occurred.

## PR85 first-source CI and controlled fixture correction

Source592b578d90a52b298846ba64e803ded1dc3332e7,
[CI38865648402](https://github.com/Sigbound/glyd-compute/actions/runs/38865648402),
FAILS Windows default/backend and aggregate; Linux x64/ARM64, macOS and dependency
policy pass. Windows default fails the stage ownership test during Fixture issuer
construction with Expired (94 passed,1 failed,1 existing ignored). Windows backend
fails the existing production-route queue-admission8 assertion: active2/queued7/
retained9; later worker registry-online/Transport/pins-retained observations do
not establish the initiating condition. Combined failed log SHA256
09ff17b607e045fed35cf2cc9b693a60cd1ec9d472ea17f28b4b21e431ec5c95.

Controlled fixture reproduction delays preparation by the existing6s registry
TTL: original setup fails at the same issuer unwrap with Expired in6.39s,
SHA25617b915d1885740f0567e82de36fbe04e1b7f7ff2caf50a19888f91f8b4506702.
It published both worker snapshots before filesystem/policy/plan preparation,
then never refreshed them. Move those real authenticated heartbeat exchanges
after preparation and immediately before issuer construction. The identical
regression passes6.40s, SHA2569889e429fb5fe9721d3049ab09383a311f7d39e61b80574fb5fb62dbff184b40.
This proves that setup ordering defect; the old Windows log lacks sample-age
telemetry and cannot distinguish every possible Expired source. Production
freshness, request/grant deadlines and credentials are unchanged. The runnable
regression retains the bounded6s delay; no timeout is enlarged.

Queue-admission failure now also records elapsed time since the last submission,
total queue phase and overrun of the existing observation budget. These three
failure-only numbers distinguish a late last submission from a genuinely stalled
admission on a matching future failure; they do not assert either cause now.
No extra observation, retry, request, process, dependency or CI job is introduced.
GLY267 remains open; final source and automatic-main qualification are required.

Corrected local source: full security library98 PASS plus one unchanged ignored
capacity test (16.98s), SHA25604c602e7e93f9702f95b3ae1b82ec4c12659145fc9dd067bdf7cece287bcf7d5.
Both warning-denied workspace/all-target lint modes pass; raw default
SHA256a265a5ed974ca77731d09877190ee2a98effc9ce56fbb8b14f478b3f12fd15d8,
all-feature3ef02a31205355a4c47c904689ab28cc7855a3a35358729e5584ce09e776b2ef.
The real two-daemon/one-coordinator byte-fixture queue/drain test passes25.92s;
its exact raw hash is recorded in Linear. This local pass is not Windows
qualification or an explanation of its failure. Formatting/diff checks pass.
Correctness/security review confirms readiness is a fresh real heartbeat, not a
clock rewrite or bypass. Separate complexity review keeps the existing fixture,
one setup callback for the bounded regression and three failure-only durations;
no background heartbeat scheduler or production abstraction is introduced.
