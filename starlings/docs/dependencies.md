# Rust dependency inventory

Cargo.lock fixes the exact graph. No dependency implements model inference.
The project license remains undecided; dependency licenses do not license this project.

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [serde](https://github.com/serde-rs/serde) | 1.0.229 | MIT OR Apache-2.0 | Typed records and bounded deserialization visitor |
| [serde_json](https://github.com/serde-rs/json) | 1.0.154 | MIT OR Apache-2.0 | JSON syntax and values |
| [sha2](https://github.com/RustCrypto/hashes) | 0.11.0 | MIT OR Apache-2.0 | SHA-256 integrity digests |
| [serde_json_canonicalizer](https://github.com/evik42/serde-json-canonicalizer) | 0.3.2 | MIT | RFC 8785 canonical bytes |
| [jsonschema](https://github.com/Stranger6667/jsonschema) | 0.58.6 | MIT | Draft 2020-12 validation with asserted formats |
| [uuid](https://github.com/uuid-rs/uuid) | 1.28.0 | Apache-2.0 OR MIT | Canonical UUID v4 identities |
| [time](https://github.com/time-rs/time) | 0.3.55 | MIT OR Apache-2.0 | Checked RFC 3339 timestamps |
| [base64](https://github.com/marshallpierce/rust-base64) | 0.23.1 (already in the graph through ureq) | MIT OR Apache-2.0 | `starlings up`/`join` codes pasted between machines |

The standard library supplies process management, pipes, watchdog channels,
collections and the fake clock. JSON/schema/crypto implementations are reused to
avoid owning a parser, canonical-number formatter or hash implementation.
`jsonschema` default features are disabled, including external HTTP/file retrieval.
M0 selected no TLS or GPU library. GLY-164 adds the security dependencies below;
there is still no GPU runtime.

Reproduce the complete transitive inventory with
`cargo metadata --locked --offline --format-version 1`; each package includes its
license expression and repository. Before distribution, review the complete graph
and retain its required notices under GLY-205. This inventory is package metadata,
not a legal opinion or a project license selection.

## GLY-164 security dependencies

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [rustls](https://github.com/rustls/rustls) | 0.23.46 | Apache-2.0 OR ISC OR MIT | TLS 1.3 and standard peer verification |
| [rcgen](https://github.com/rustls/rcgen) | 0.14.10 | MIT OR Apache-2.0 | Key, CSR and certificate generation; CSR signature verification |
| [ring](https://github.com/briansmith/ring) | 0.17.14 | Apache-2.0 AND ISC | Shared crypto provider and OS randomness |
| [x509-parser](https://github.com/rusticata/x509-parser) | 0.18.1 | MIT OR Apache-2.0 | Complete DER consumption and certificate/public-key inspection |
| [private-fs](https://github.com/orielhaim/private-fs) | 0.1.0 (0.1.x range; lockfile pinned) | MIT OR Apache-2.0 | Native protected file creation, validation and durable publication |
| [rusqlite](https://github.com/rusqlite/rusqlite) | 0.40.2 | MIT | Atomic invitation/member transactions |
| libsqlite3-sys | 0.38.2 | MIT wrapper; bundled SQLite public domain | Reproducible SQLite build on four CI platforms |
| [tempfile](https://github.com/Stebalien/tempfile) | 3.27.0, test only | MIT OR Apache-2.0 | Isolated temporary test directories |

Rustls and rcgen share ring; AWS-LC is not selected. The production TLS config
explicitly enables only TLS 1.3. The dev dependency also enables TLS 1.2 so a real
negative handshake can prove rejection. No custom cryptographic primitive, CSR
parser or certificate-chain verifier is implemented.

`private-fs` is new rather than an established audited component. Its published
security contract and Unix/Windows source were inspected; four-platform tests
exercise creation, enrollment persistence and refusal of broadened file access.
Its retained-handle approach avoids application-owned Unix ACL and Windows DACL
code. A future upgrade requires the same platform/security review. No independent
audit, power-cut test, network-filesystem guarantee or secure-enclave storage is
claimed. GLY-205 must include the complete transitive notices before distribution.

## GLY-166 listener dependencies

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [httparse](https://github.com/seanmonstar/httparse) | 1.10.1 | MIT OR Apache-2.0 | Established bounded HTTP syntax parser |
| [interprocess](https://github.com/kotauskas/interprocess) | 2.4.5 | 0BSD OR Apache-2.0 | Safe Unix socket / Windows named-pipe APIs; async features disabled |
| [widestring](https://github.com/starkat99/widestring-rs) | 1.2.1 | MIT OR Apache-2.0 | Windows-only native security-descriptor string |

The added locked transitive packages are `recvmsg` 1.0.0 (0BSD)
and `doctest-file` 1.1.1 (0BSD). No application-owned unsafe code,
HTTP grammar, OS ACL FFI, asynchronous runtime or inference dependency is added.


## GLY-167 daemon dependencies

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [dirs](https://github.com/soc/dirs-rs) | 6.0.0 | MIT OR Apache-2.0 | Native state/data/cache directories |
| [ctrlc](https://github.com/Detegr/rust-ctrlc) | 3.5.2 | MIT/Apache-2.0 | Signal notification; termination feature enabled |

Added transitive packages from locked Cargo metadata: `dirs-sys` 0.5.0
(MIT OR Apache-2.0), `option-ext` 0.2.0 (MPL-2.0), `nix` 0.31.3 (MIT),
`cfg_aliases` 0.2.2 (MIT), `block2` 0.6.2 (MIT), `dispatch2` 0.3.1
(Zlib OR Apache-2.0 OR MIT), `objc2` 0.6.5 (MIT), `objc2-encode` 4.1.0 (MIT),
`libredox` 0.1.25 (MIT) and `redox_users` 0.5.3 (MIT). Platform selection means
not every locked package is built on every target. The MPL-2.0 dependency is
unmodified; distribution/license obligations remain part of GLY-205. No inference
runtime, application-owned unsafe code or project license grant is introduced.


## GLY-169 networking dependencies

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [mdns-sd](https://github.com/keepsimple1/mdns-sd) | 0.21.5, exact pin | Apache-2.0 OR MIT | Optional DNS-SD protocol; default async/logging features disabled |
| [if-addrs](https://github.com/messense/if-addrs) | 0.15.0 | MIT OR BSD-3-Clause | Native interface/address inventory, including link-local |
| [socket2](https://github.com/rust-lang/socket2) | 0.6.5 | MIT OR Apache-2.0 | Safe native source binding and bounded connect |

New locked transitive packages: flume 0.12.0 (Apache-2.0/MIT), spin 0.9.9 (MIT),
mio 1.2.4 (MIT), socket-pktinfo 0.4.1 (MIT), log 0.4.34 (MIT OR Apache-2.0).
Mio is an internal DNS-SD dependency, not a new application runtime. Source/API
review verified interface selection, bounded event channels, unsolicited-cache
policy and shutdown. Internal DNS/cache allocations do not provide a hard RSS
quota. GLY-205 retains full transitive notice review before distribution.
Project license remains undecided; no upstream implementation is copied.

## Review remediation policy

`deny.toml` checks current RustSec advisories, declared licenses, bans and sources
across all features. CI uses upstream cargo-deny 0.20.2 with fixed archive SHA-256.
Warnings fail. Unpublished workspace crates are excluded from dependency license
classification; the project license remains undecided. Locked duplicate transitive
versions are allowed; wildcard registry and unknown git/registry sources fail.

Private-fs 0.1.0 source review covers retained handles, owner-only creation and
verification, no-follow access and atomic durable publication; Windows uses owner
DACLs/native handles. Cleanup now uses rustix 1.1.5 (MIT OR Apache-2.0), already in
the graph, to unlink relative to the Unix directory handle. Windows retains a
directory handle without delete sharing while owned. Tests cover permission,
symlink and Unix profile-rename refusal/cleanup. Same-user/root compromise, network
filesystems and physical power cuts remain outside the tested contract. This is
not an independent audit or M6 distribution-notice completion.

Validation: the CI checker and locked graph were also run with a temporary policy
allowing only 0BSD. It failed with `error[rejected]: failed to satisfy license
requirements`. The committed allowlist was unchanged. This exercises a denied
license; it does not inject a vulnerable dependency into the repository.

## M2 artifact access

| Dependency | Locked version | Declared license | Purpose |
| --- | --- | --- | --- |
| [ureq](https://docs.rs/ureq/3.4.2/ureq/) | 3.4.2 | MIT OR Apache-2.0 | Bounded explicit provider HTTP, reusing Rustls; default features disabled |
| [cap-std](https://docs.rs/cap-std/4.0.3/cap_std/fs/struct.Dir.html) | 4.0.3 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | Contained read-only source access and cache entry removal |

The existing protected file dependency requires owner-only writable modes and is
unsuitable for read-only preloaded trees. Cap-std provides native handle-relative
containment without application-owned unsafe OS code. Private-fs continues to own
secret/cache publication. Ureq provides HTTP/TLS parsing and timeouts; application
code only restricts approved hosts, redirects, ranges and credential forwarding.
There is no asynchronous runtime, custom HTTP client/parser, inference library or
model download during builds/tests. Full transitive notice review remains GLY-205.

The locked trust-root data `webpki-roots` 1.0.9 uses CDLA-Permissive-2.0;
its LICENSE requires the agreement text when sharing that data. The Windows
capability helper `winx` 0.36.4 uses Apache-2.0 WITH LLVM-exception; its LICENSE
contains the standard Apache text plus LLVM exceptions. Their published license
files were inspected and `deny.toml` permits only those exact package versions.
No advisory/source checks are disabled. GLY-205 must retain applicable license
texts/notices before any distribution; this does not select the project license.

## M2 host probes

[sysinfo 0.39.6](https://docs.rs/sysinfo/0.39.6/sysinfo/) (MIT) supplies read-only
OS memory, disk and temperature APIs. Only system/disk/component features are
enabled; default multithreading, network and user enumeration are disabled. The
code never refreshes process data. This avoids separate application-owned unsafe
Windows/Apple/Linux implementations. Its native platform wrappers are locked in
Cargo.lock and remain under the dependency-policy and GLY-205 notice gates.

## M2 external process supervision

[process-wrap 10.0.1](https://docs.rs/process-wrap/10.0.1/process_wrap/)
(MIT OR Apache-2.0) exposes safe process-group/Windows-job lifecycle control.
Only std, process-group and job-object features are enabled; no async runtime.
Its one new locked support package is indexmap 2.14.2 (MIT OR Apache-2.0); native
nix/windows dependencies already existed. Ureq and SHA-256 reuse the existing
versions. The application adds no unsafe process APIs and copies no upstream
inference implementation. Parent exit remains distinct from qualified device drain.

## M2 explicit NVIDIA diagnostics

[nvml-wrapper 0.13.0](https://github.com/rust-nvml/nvml-wrapper) and its generated
nvml-wrapper-sys 0.10.0 bindings (MIT OR Apache-2.0) provide safe read-only NVML
calls without application-owned FFI. Both published crates have no build script;
the locked bindings dynamically load an explicitly approved installed library.
No driver, SDK, header generation or inference runtime is fetched during builds.
NVML is separate trusted installed software, not redistributed by this repository.

The new locked graph also contains libloading 0.8.9, static_assertions 1.1.0,
thiserror/thiserror-impl 1.0.69, wrapcenum-derive 0.4.1, darling/core/macro 0.20.11,
fnv 1.0.7, ident_case 1.0.1 and strsim 0.11.1. Published metadata/source and
runtime loading/query/error behavior were reviewed; the all-feature advisory,
license/source policy continues to deny warnings. Full distribution notice review
remains GLY-205; this does not select a project license or constitute independent audit.
