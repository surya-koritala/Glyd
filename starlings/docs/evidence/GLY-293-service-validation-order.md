# Invalid service bindings before executable verification

[GLY-293](https://linear.app/getglyd/issue/GLY-293) follows a genuine failure on
main `481e4836d53136ab423fede903ffcd125561d519`, automatic
[CI38818550601](https://github.com/Sigbound/glyd-compute/actions/runs/38818550601),
Windows backend job116709199586. The final lifecycle case fails in wrong-path
mode at the existing10-second observation wait, state=verifying_catalog,
startup_phase/pin/reservation=null and registryonline/epoch3. Production routing
passes144.68s and GLY-292's new routing/init controls pass. Later Transport during
unwind is not established as initiation. Raw job-API log SHA-256:
`81ae17498392c9ea97b91e06350a1e0ee74a5cd703a4d3ffa4418f5c55c561fa`.
Five other checks pass; Windows backend and aggregate fail. Preserve this run as
failed, separately from PR76 sourceCI38817444781's seven-check pass.

Source trace finds full executable hashing before model/profile/weights binding
refusal. A controlled approved wrong-path configuration also carries a deliberately
mismatched executable digest. Old production reports backendUnapproved before the
existing SERVICE_WEIGHTS_NOT_BOUND assertion: FAIL30.09s, log SHA-256
`8e1238ce2169a406706781bcf0c0d288ca151bfcc5ba20ee7089a251146019c0`.
This proves avoidable executable verification precedes a known-invalid binding;
it does not identify the exact host operation delayed in the genuine Windows run.

Retain catalog ownership/quarantine checks first, then full canonical configuration
digest approval before consulting its approved model. Check manifest approval,
profile and verified cache-asset binding next. Valid bindings still require the
existing complete config/path/executable validation before any pin or spawn.
Spawn retains its own complete validation. There is no cached approval or skipped
executable check. The same trusted-local-account file replacement boundary applies;
exclusive catalog ownership remains held through preparation and service lifetime.

Reuse the existing static validation observer and startup_phase field for
catalog_open, catalog_ownership, configuration_approval, manifest_approval,
profile_binding, asset_binding and the existing configuration_digest,
configuration_paths and executable_digest steps. Status locks end before host I/O.
Phases describe attempted operations, not completion or native qualification.
Spawn/probe phases replace preparation phases normally; startup probe error
retention and error codes are unchanged.

The same lifecycle case retains normal, startup-crash, unapproved, wrong-path,
wrong-profile and registry-loss coverage. A valid-binding/bad-executable mode
requires executable refusal/no pin; invalid profile/path modes require their
binding refusal first despite a bad executable digest. Every refusal checks its
exact static phase. Candidate case PASS49.65s, log SHA-256
`4824e9d6f302af572d8682f536671791287acaf369e3e1f4aeddc368e7e25ea4`.
The baseline stops at its first failure, so these durations are not a speed
comparison. One invalid mode is added within the existing case; no job or filter
is added and no deadline/TTL/CI cap is increased.

Final fmt, both warning-denied workspace/all-target lint modes, both CLI feature
suites and all18 synthetic lifecycle cases pass (64.94s). Captured validation log
SHA-256: `3f4a1c184da25b1682fbfba14c5f3ace971b6e4add2c30bfba98f7e5c180db37`.
Exact source/PR/source-main CI receipts are recorded in Linear. The first local lint caught an unnecessary mut;
it was removed without an allowance before qualification. A green diagnostic run
cannot retrospectively establish the archived OS cause. GLY-263/267 historical
attribution and M2 provider/hardware/native benchmark gates remain open.

Correctness/security review traces every validation caller, approval before model
lookup, complete valid-path hashing, strict catalog paths, pre-pin refusal,
unchanged ownership/drain, static redaction and mutex lifetimes. Separate complexity
review reuses the existing adapter observer, fields, errors and fixture: no fault
framework, dependency or alternate validation implementation. Independent milestone
review deferred, not approved. No native/model/GPU/Ubuntu/SSH/download/spend,
old-state recovery, quarantine reset, license/settings/publication or M3 work.
