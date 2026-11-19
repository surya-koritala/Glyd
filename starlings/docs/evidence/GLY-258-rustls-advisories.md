# GLY-258: patched Rustls dependency

PR60 source `a2de5d20b7b870cbb8da0a5c50d6daf9754d9ed1`,
run38591271115/dependency job115946493584 fails on locked Rustls0.23.45.
RustSec identifies RUSTSEC-2026-0346, 0347, 0348 and 0349: session Debug
secrets, ECH rejection ordering, Brotli decompression allocation and unauthenticated
key-exchange prediction. The checker requires >=0.23.46 for all four. Raw failure
log SHA-256 `2d3a7fee8ae44b61f1fe1ca095fe049cf105e7019f432be1f23388873744ccd5`.

The [official 0.23.46 release](https://github.com/rustls/rustls/releases/tag/v/0.23.46)
and [RustSec record](https://github.com/RustSec/advisory-db/blob/main/crates/rustls/RUSTSEC-2026-0346.md)
confirm the patch. Production peer transport directly uses Rustls TLS1.3/ring,
with resumption and early data disabled; ureq also uses Rustls. Default Brotli,
ECH and hybrid prediction applicability is not assumed, and no Compute exploit
is asserted. Development TLS1.2 rejection tests keep their existing feature.

Raise both direct/development minimums to0.23.46 and update only that lockfile
package/checksum. Existing TLS/authentication policy, enabled features, dependency
source/license checks, deny-warnings and empty advisory-ignore list remain.
No API migration, dependency exception, new workflow/job or timeout expansion.

Validation uses formatting, both warning-denied workspace/all-target lint modes,
existing transport limits/permit tests, real mutual-TLS registry membership/revocation
and expired-peer tests. Final-source all seven CI checks/current-main ancestry and
review remain required before authorized merge; exact identities belong in Linear.
Correctness/security review traces production, HTTPS and development users; separate
Ponytail review keeps the smallest compatible patch. Independent milestone review
deferred, not approved. No native repeat, model/GPU/PC workload, quarantine reset,
spend or release. The original advisory failure remains preserved.
