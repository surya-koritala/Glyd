# GLY-198 private candidate packaging

Issue: [GLY-198](https://linear.app/getglyd/issue/GLY-198).

Use the existing pinned Cargo toolchain and native tar/checksum tools to stage a
private default CLI candidate. The builder accepts one new output directory,
requires clean committed source before/after compilation, uses cached locked
assets/two build jobs and denies compiler warnings. It packages only the executable,
unsigned build facts, lockfile and compiler facts, with inner/outer SHA256SUMS.
Target allowlisting is native Linux GNU/macOS x64/ARM64; it is not deployment
qualification. No installer, service setup, model download or publisher is added.

One Unix test exercises the actual archive helper with small synthetic files in
paths containing spaces. It checks byte preservation, executable mode, inner and
outer hashes, tamper rejection, excluded identity data, no overwrite, symlink
input/output refusal and invalid archive-name refusal. No fixture binary is
executed and no Cargo build is invoked by that self-test. Existing default CI
automatically selects this test; no workflow, job or runtime dependency is added.

The initial control fails because a function invoked as an if-condition ignores
Bash errexit: failed mkdir did not stop existing-output processing. Explicit
mkdir failure exits before any write. Corrected exact packaging check passes
0.11s, then0.10s with space-containing paths. Both warning-denied workspace/
all-target lints pass4.70s/2.36s, shell syntax, fmt and diff checks pass. These are
basic Mac file/compile checks only, max two build jobs, cached/offline assets.
No sockets, services, host instrumentation/inspection, model or native inference.

Correctness/security review checks quoted paths, fixed file selection, new-output
mkdir, symlink refusal, source revision stability and unsigned/unqualified facts.
Separate Ponytail review reuses native Cargo/tar/SHA tooling, one helper and one
bounded check; no custom archive/crypto/parser, installer framework or dependency.
Incomplete output remains unqualified and is never overwritten automatically.
Checksums are integrity evidence, not publisher authentication. No byte-identical
archive, minimum OS/libc or external backend support claim is made.

Remaining target candidate builds, service templates, all-platform installed flows,
upgrade/rollback/uninstall, WSL2 and native qualification, dependency notices,
release provenance, D14 release method, the license decision and publication
approval remain gates. No compound acceptance checkbox is completed by fixtures.
Independent milestone/release review remains deferred, not approved.

## Actual Mac candidate check

An ordinary Rust CLI release build from source2311eb1 completes in152s with
two build jobs, denied warnings and cached offline assets, below its300s cap.
The first real archive fails fixed-member inspection: native Mac tar serializes
AppleDouble metadata, yielding12 entries instead of the intended6. No candidate
is executed at that point. The initial synthetic check alone does not reproduce
those platform metadata entries; it must not be counted as that negative control.

The archiver now excludes extended metadata using native tar's no-xattrs option
and its scoped copyfile setting. Source attributes and OS security controls are
untouched. The runnable check now asserts the exact member set as well. A fresh
candidate from def88bd57f23b898c6cf1ded70c5fcdca93d6c1e reuses the unchanged
compiled binary (cached Cargo0.25s) and passes outer/inner checksums, exact6-member
inspection, manifest/source/target/flags checks and static --version/--help/doctor.
Development-only help is absent and doctor reports inference=unavailable.
No model or service starts; no installation/publication or hardware qualification.

The19,013,888-byte executable has SHA256
`d593428fe380f0940ebc309d4c06da887b0125a762e144caa77c715634285d97`,
identical to the first archive's executable. Corrected archive SHA256
`bd3c9d5a938cc540402557ebef4b6d1e2b164bebea4714438928e98200f38119`.
Both private candidate directories remain in the checkout's ignored target tree;
the rejected archive is preserved. The local verification receipt is
work/m6-candidate-def88bd-verification.json in the parent workspace. These hashes
identify a local development artifact, not publisher authentication or a release.
