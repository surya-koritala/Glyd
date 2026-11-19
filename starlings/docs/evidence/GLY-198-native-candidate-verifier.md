# GLY-198 native candidate verification

Issue: [GLY-198](https://linear.app/getglyd/issue/GLY-198).

The existing builder already supports native Linux GNU x64/ARM64 and Mac x64/ARM64,
but synthetic archive checks do not establish real Linux release builds. Reuse
the builder, Python's standard-library gzip/tar/JSON/SHA readers and the existing
Unix packaging test. No product command, dependency or installer is added.

The verifier bounds decompression and logical archive entries without extraction,
requires the fixed seven members/modes/checksums and binds the manifest, compiler,
lock, optimized default CLI artifact and executable bytes to the trusted clean
checkout. It never follows receipt-supplied paths. Duplicate JSON keys, fixture
manifests, failed/incomplete receipts, changed files, links, extra/duplicate members
and path escapes are refused. Normal output contains no raw build metadata.
The synthetic check uses non-executable fake bytes and runs inside the existing
packaging test, adding no CI job/test binary. It tests rejection independently of
the builder's archive-writing fixture.

A file-only check against the unchanged private62a0d6a Mac candidate passes the
new reusable verifier. That candidate remains bound to its original source;
it is not relabeled as this change, rebuilt, executed, installed or published.
Detailed compiled/build/dependency metadata stays private locally. Synthetic
verifier controls pass. The exact Rust packaging check passes0.18s (cached
recompile0.15s); both warning-denied workspace/all-target Clippy modes pass8.99s
and2.83s, formatting/diff checks and YAML/embedded-shell parsing pass. These
are bounded cached/offline Mac file/compile controls, max two Cargo jobs.
Source/main CI results are recorded in Linear after execution. These initial
file-only checks do not establish native Linux success; the later batch below
provides separate build/archive evidence.

A final controlled malformed-deflate check found that Python's `zlib.error` is
outside the original sanitized exception classes. A valid gzip header with a
reserved deflate block type and a matching outer hash reproduces the actual
verifier traceback before correction. Catch that decoder error at the shared
decompression read and return `ARCHIVE_COMPRESSION`. The same synthetic
archive regression must now be refused; bounds and valid candidate behavior are
unchanged. No existing private candidate was mutated or executed.
The existing Rust packaging check passes0.19s after the correction (cached
compile0.32s). This follow-up changes no Rust code, dependencies or workflow.

The opt-in workflow has two native Linux jobs, ten minutes each, two Cargo jobs,
no cache action or artifact upload. Source SHA/ref equality is checked before
compiler setup and the locked native dependency fetch; the build then runs
offline. The initial cache-restore design was removed in review: GitHub cannot
restore a sibling main-branch cache here because main is not the repository
default. No settings change or guaranteed cache miss is needed. Build output
is kept private on the ephemeral runner; warning output refuses qualification. Ordinary eight-check CI
and all deadlines are unchanged. The extra batch requires explicit bounded
approval before its qualification ref is pushed; none had run when the workflow
was introduced. The ref mechanism avoids a repository default-branch/settings
change. See [the workflow and developer commands](../packaging.md).

Separate complexity review: reuse one builder and one existing integration case,
stdlib parsing/hashing and native Cargo messages, with no archive extraction,
publisher framework, new parser dependency or speculative runtime feature.
Correctness review: strict identities, independent trusted paths, private bounded
reads and explicit refusal preserve the distinction between unsigned file facts
and deployment qualification. Independent milestone/release review remains
deferred, not approved.

The [Cargo JSON contract](https://doc.rust-lang.org/cargo/reference/external-tools.html#json-messages)
describes artifact/profile/features and build-finished records. Python's
[tarfile reader](https://docs.python.org/3/library/tarfile.html) supplies member
types and file streams without extraction. GitHub's
[push event/ref behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push)
allows this opt-in workflow without changing the default branch. No compiler
receipt is publisher authentication, a complete linked/native/system/SDK/backend
SBOM, legal clearance or a signature. Retained review artifacts, untested target
builds, installed lifecycle, WSL2, native workload/hardware, independent review,
license/release decisions and publication remain open.

## Actual Linux native builds — 17 October 2026

The separately approved one-shot batch
[39088336507](https://github.com/Sigbound/glyd-compute/actions/runs/39088336507)
builds exact main `2a71bbae5c12c6508069bad4e3877adc240e5e58`, integrating
[PR128](https://github.com/Sigbound/glyd-compute/pull/128). Its signed reviewed
source is `8dd13260592103266071d363013488ca10d596c7`.
All eight [source checks](https://github.com/Sigbound/glyd-compute/actions/runs/39076071862)
and the separate eight-check
[main run](https://github.com/Sigbound/glyd-compute/actions/runs/39076657332)
passed before dispatch; source and main trees matched.

| Native build | Hosted-runner label | Exact job | Outcome | Duration |
| --- | --- | --- | --- | --- |
| Linux GNU x64 | `ubuntu-latest` | [117559518706](https://github.com/Sigbound/glyd-compute/actions/runs/39088336507/job/117559518706) | Build/archive verification PASS | 4m10s |
| Linux GNU ARM64 | `ubuntu-24.04-arm` | [117559518861](https://github.com/Sigbound/glyd-compute/actions/runs/39088336507/job/117559518861) | Build/archive verification PASS | 3m43s |

Each job checks the exact SHA/ref, installs pinned Rust1.97.1, fetches native
dependencies from the reviewed lockfile, then runs the existing offline release
builder with two Cargo jobs and denied warnings. The existing verifier confirms
the fixed archive contents, checksums, permissions, source/compiler/lock identity
and native build receipt without extracting or executing the candidate. Both
verifier pass messages are confirmed in the job logs. The ten-minute job caps
were retained. No compiler/CI warning or error annotation was reported; GitHub
added one informational x64 notice about the upcoming `ubuntu-latest` image
migration. The runner label is recorded, not a minimum-OS/libc support claim.

There was one run and no retry or artifact upload. Candidates and raw build
receipts were private ephemeral-runner files; none is retained for download or
independent binary inspection. No candidate executable, service, inference model,
GPU, user computer or physical Pi was run. No repository settings, license or
publication decision changed.

This completes the scoped native Linux build/archive check only. The original
GLY-198 platform/service-template, installed lifecycle and WSL2 acceptance remain
open, as do actual backend/device evidence, complete dependency/notice review
and independent milestone/release approval. This document reports the exact
revision above; a later documentation-only revision does not relabel these
candidates or establish byte-identical rebuilds.
