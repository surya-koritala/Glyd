# GLY-257: verified catalog preparation before native comparison

The one-shot Mac CPU comparison was explicitly approved on 14 October 2026.
Frozen implementation: `83c92adbdf9d30fa30ee483e4d94a18943fd29da`;
CLI SHA-256: `92b5a254b36e3d803d86b695f55d8ee5995bac1e5e283a47938662ef6a22ec2a`;
runner SHA-256: `e741c59c3b0168ab07a19b6843c1a8e2823753d206ec77d2a8ac0c059ce9566a`.
The fixed envelope allowed two sequential CPU starts, twenty requests of at most
eight output tokens, two threads, five minutes including cleanup, a 240-second
watchdog, monitored 4 GiB owned RSS, cached assets and zero external spend.

## Failed attempt and preserved identity

The attempt ended in 4.053 seconds with `ACCEPTANCE_TimeoutExpired`, before any
runtime/model process or generation/count call. No direct log, direct-pin receipt,
worker/coordinator log, isolated pin or service-start record exists. Final token
rows are zero, old ownership hashes match, no owned server group remains and
external spend is $0. Sampled owned-server RSS is zero because no server started;
this is not an allocation measurement. The single-use marker remains consumed.

The unchanged raw result is [GLY-257-native-attempt.json](GLY-257-native-attempt.json),
SHA-256 `bde745bdbd8f6b8a3f63335591f2820462b31e188691cd73af38b9035f63f177`.
Run log SHA-256:
`f43638d5cc09448041929ee9fbb0b2ad1807dfdc84e0ee3076c8e3489f13a5c1`.
The raw result's `manifest` is the direct-stage backend alias because that mutable
variable had already changed. The authoritative manifest remains qualification
`8f993f2242910f9f535611e0e82e62e6054bcaccc96dc28297f09bc6731e0835`.
Do not rewrite the old raw result or infer successful semantic/timing acceptance.

## Controlled source cause and correction

Runner ordering isolates the timeout to its pre-start `model pin` subprocess.
`Catalog::pin` invokes `ready()` before publication, rehashing every protected
asset. The runner incorrectly puts that preparation work through the three-second
control wrapper. Read-only `model ready` uses the same validation path and passes
in 11.190 seconds on the same cached catalog, without a pin retry or model start.
This demonstrates an operation/budget mismatch; no OS/storage deadlock is claimed.

The corrected fresh candidate creates the direct pin during existing offline
preparation, with its unchanged 30-second command bound and full verification.
Native preflight rechecks the exact receipt/hash, manifest/lease binding and pin
bytes. The native runner no longer mutates the catalog to create that pin. Its
three-second control deadline, 240-second watchdog, five-minute envelope and all
request/RSS/ownership limits remain. Future failure records keep the immutable
qualification manifest and a safe stage identifier. No native repeat is authorized.

Reproducible patches against the frozen original scripts:
[preparation patch](GLY-257-prepare.py.patch) and
[runner patch](GLY-257-run_profile.py.patch). Apply only to a new candidate copy;
the original runner, result, marker and old quarantine records are immutable.
Apply with `git apply --unidiff-zero`; controlled reconstruction matches each corrected script byte for byte.
Preparation pins are retained; no reset, recovery or automatic unpin is introduced.

## Verification and limits

Corrected control/catalog preparation and native-free preflight pass. The
[model-free check](GLY-257-check-pin.py) rejects the original runtime catalog pin,
then validates good ownership and rejects wrong manifest/lease, changed bytes and
missing pins. It checks the result's manifest expression using an early direct-stage
alias without executing the native module body:

`python3 docs/evidence/GLY-257-check-pin.py /path/to/corrected/run_profile.py`

Correctness/security review traces full asset verification, durable pin publication,
receipt/hash checks, original ownership, native budgets, no mutation retry and
failure identity. Separate Ponytail review reuses the existing scripts, SDK and
standard library; no new product API, framework, dependency or CI job. Exact-head
seven-check CI and current-main ancestry are required before authorized merge;
their identities belong in Linear. Independent milestone review deferred, not approved.

This is failure evidence and a model-free runner correction, not a passing native
comparison, current-main execution, installed Glyd qualification, conversion fidelity,
two-worker scaling or Exo result. M2 still requires the provider contract and
approved physical/native evidence. M3 has not started.

## Superseding patched-current-source candidate

The original approved attempt remains frozen at83c92ad. After the new Rustls
advisories were detected, the next unapproved candidate was prepared from tested
implementation `78b737495ff52fc69e05b928a460f400fcd4af70`, including Rustls0.23.46.
Only the Rust CLI was rebuilt; the cached CPU inference runtime/model are unchanged.
CLI SHA-256 `46f702186fb5d22c1d27ffa93bb215f9abb130f8799267e5f0a0788389a6521c`;
runner SHA-256 `c9af0a01c2be78ca6aaffe705239a02be87127fec647da59dde76a8ba169ef18`;
configuration `becb51af2af8a843ae1c1dfd6c390a271accf37b0dc7d8e193cd1492e1c4ba8c`;
worker record `03f1275fec7f9001d035c1d13057ac5c46bdc158264f6218874b77fdf1cd9fed`.
Fresh full-verification pin preparation and model-free preflight pass. The patches
now reconstruct this current candidate; old candidates/markers remain intact.
No native attempt has run here and no previous approval is reused. Final CI source
may add evidence only; Rust/crate/lockfile identity must remain the tested implementation.
