# GLY-204 boundary inventory preparation

Base: mainf8714b14550448aa33f6461553a95e9104c38d6e (PR105).
The [inventory](../security-boundaries.md) maps assets, actors, local admin,
enrollment/renewal, admission/registry/TLS, HTTP/streaming, artifacts, external
services, library-only stage transport, discovery and evidence/packaging to source.

This is document preparation, not completed independent/security/native release
qualification. At this initial inventory revision all GLY-204 compound subtasks
remain open. Device keys, worker
honesty, same-user backend privileges, ambiguous execution, irreversible enqueued
output and missing provider/native/OS evidence are explicit boundaries.

Separate complexity review reuses existing contracts and one source-linked table.
Correctness review distinguishes deployed whole-request routing from library-only
layer transport and byte identity from trust, authority, native completion and
publication approval. Local verification is bounded relative-link/file validation
and diff review only; no test suite, network, daemon, scan, instrumentation,
model/device workload, dependencies or permissions/control change. Final source
CI/current-main/review gates still apply before merge. Independent review deferred.

PR107 also includes the separately reproduced [GLY-311 ownership correction](GLY-311-profile-lock-lifetime.md)
after a genuine Linux main failure. Its file-only checks and code review are
recorded there; they do not turn this inventory into independent audit clearance.

## Source threat model expansion

The model now covers main935078fba7f05ad0e5812e6d8b22d8880eb5e740, including
explicit attacker capabilities and a scenario/invariant for each of14 entry-point
groups. Existing sources remain linked; standalone link diagnostics, benchmark
collection, native-library observations, advisory planning and build/descriptor
tools are now explicit. These are source-review scenarios, not executed attacks,
new vulnerability findings or native control-validation results.

The source walk checks the collector's literal loopback/no-proxy/no-redirect and
credential/output boundaries; probe route approval, ALPN/roles/freshness and charged
cooldown; the NVIDIA loader's explicit primary-library/content approval and
conservative pool reports; advisory admission=false; trusted build code/raw
receipts; and inactive native descriptor rendering. It preserves same-user/root,
local storage, native dependency and build-host trust assumptions instead of
claiming a sandbox or full dependency attestation.

The artifact models all currently implemented Compute entry points plus explicit
assumptions/gates for proposed native channels. Delivered backend mapping,
allocation/drain/profile proof, physical/OS/fault qualification, complete
SBOM/license/provenance and independent review remain required. Refresh the model
on the listed surface/authority/lifetime/native/packaging changes. Full GLY-204
and release parents remain open; the source-model artifact subtask alone can be
completed after actual source/main/PR qualification. No native acceptance or
independent approval is supplied by the document.

Separate complexity review reuses the existing diagram and one expanded table;
no framework, policy engine, dependency, runtime code, tests or CI changes.
Correctness review traces the additional tools and distinguishes source controls,
caller trust, runtime observations and remaining release evidence. Local checking
is file/link/diff only; no local Cargo, service/socket, scan, traffic/attack test,
model/GPU, host instrumentation, runtime/model download or spend. All required exact-source CI checks,
fresh-main/clean/review gate and separate automatic-main verification still apply.
