# Verifying a release

Every release's files are signed and attested by the workflow that built them, with no key of ours involved: GitHub's OIDC identity for the workflow run is
what Sigstore certifies. A file that did not come from `surya-koritala/Glyd`'s release workflows at a `v*` tag does not verify.

## What is published

For a release `vX.Y.Z`, on its GitHub release page:

| File | What it is |
|---|---|
| `glyd-vX.Y.Z-PLATFORM.tar.gz`, `glyd-engine-vX.Y.Z-PLATFORM.tar.gz`, the wheels | the files themselves (`.sha256` beside each tarball, as before) |
| `SHA256SUMS` and `SHA256SUMS-engine` | the SHA-256 of every tarball and wheel of `release.yml` and of `release-gpu.yml` |
| `FILE.sigstore.json` | a keyless [cosign](https://docs.sigstore.dev/) signature bundle for each tarball and for each `SHA256SUMS*` list |
| `glyd-vX.Y.Z.spdx.json`, `glyd-engine-vX.Y.Z.spdx.json` | an SBOM (SPDX 2.3) of the source the files were built from (the Rust crates in `Cargo.lock`) |

Also, for every file listed in the `SHA256SUMS*`: a GitHub [build provenance attestation](https://docs.github.com/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds)
and an SBOM attestation, stored by GitHub. The wheels on PyPI carry PyPI's own attestations (trusted publishing), shown on each file's page there.

## Check a file

With the [GitHub CLI](https://cli.github.com/) (the simplest). Always name the workflow and the tag: an attestation by itself only says that some workflow of this repository, on some branch or pull request, attested a file with that digest.

```
gh attestation verify glyd-vX.Y.Z-linux-x86_64.tar.gz --repo surya-koritala/Glyd \
  --signer-workflow surya-koritala/Glyd/.github/workflows/release.yml --source-ref refs/tags/vX.Y.Z
gh attestation verify glyd-vX.Y.Z-linux-x86_64.tar.gz --repo surya-koritala/Glyd \
  --signer-workflow surya-koritala/Glyd/.github/workflows/release.yml --source-ref refs/tags/vX.Y.Z --predicate-type https://spdx.dev/Document/v2.3
```

The first checks where and how the file was built (the workflow, the tag, the commit); the second, its SBOM. For the engine bundles (`glyd-engine-...`) the workflow is `release-gpu.yml`.

With `cosign` alone, no GitHub account needed:

```
cosign verify-blob --bundle glyd-vX.Y.Z-linux-x86_64.tar.gz.sigstore.json \
  --certificate-identity-regexp '^https://github.com/surya-koritala/Glyd/\.github/workflows/release(-gpu)?\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  glyd-vX.Y.Z-linux-x86_64.tar.gz
```

Or verify the list once and check the files against it: `cosign verify-blob` the same way on `SHA256SUMS` (and `SHA256SUMS-engine`), then `sha256sum -c SHA256SUMS --ignore-missing`.

A file that was changed, or built by anything else, fails both: `.github/workflows/sign-check.yml` runs these commands on a test file, including a changed one that must fail.

## install.sh checks it for you

From release 0.34.0, `scripts/install.sh` downloads the signature beside each tarball it installs (the compression program's and the engine's) and checks it with cosign
(yours if there is one on PATH, else cosign v3.1.4, downloaded and checked against its sha256) against Glyd's release workflows at a `v*` tag. A tarball that is unsigned, altered
or signed by anything else is not installed. A release before 0.34.0 has no signature: the installer says so and goes on with the sha256 alone.

- `GLYD_SKIP_VERIFY=1` installs a tarball without the signature check (the sha256 is still checked), for a mirror or a build of your own that has none.
- `GLYD_VERIFY_IDENTITY` replaces the expression of who may have signed, to try a release built from a branch (a dry run of the release workflow).
- `GLYD_ENGINE_TARBALL` (your own build of the engine) is not checked: it is yours.

`scripts/check_install_verify.sh` (run in CI) checks these cases with a stand-in cosign; `install-check.yml` runs the installer on each published release.

## What is not covered yet

- Platform code signing (Windows Authenticode, Apple notarization) needs certificates that only the owner can get: [#223](https://github.com/surya-koritala/Glyd/issues/223).
- The later backend downloaders (P7b) and `glyd update` (P21) will use the same check.
- Reproducible builds and a SLSA level beyond GitHub's provenance: D6b ([#259](https://github.com/surya-koritala/Glyd/issues/259)).
