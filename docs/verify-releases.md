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

With the [GitHub CLI](https://cli.github.com/) (the simplest):

```
gh attestation verify glyd-vX.Y.Z-linux-x86_64.tar.gz --repo surya-koritala/Glyd
gh attestation verify glyd-vX.Y.Z-linux-x86_64.tar.gz --repo surya-koritala/Glyd --predicate-type https://spdx.dev/Document/v2.3
```

The first checks where and how the file was built (the workflow, the tag, the commit); the second, its SBOM. Add `--signer-workflow surya-koritala/Glyd/.github/workflows/release.yml`
(or `release-gpu.yml` for the engine bundles) to require that workflow.

With `cosign` alone, no GitHub account needed:

```
cosign verify-blob --bundle glyd-vX.Y.Z-linux-x86_64.tar.gz.sigstore.json \
  --certificate-identity-regexp '^https://github.com/surya-koritala/Glyd/\.github/workflows/release(-gpu)?\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  glyd-vX.Y.Z-linux-x86_64.tar.gz
```

Or verify the list once and check the files against it: `cosign verify-blob` the same way on `SHA256SUMS` (and `SHA256SUMS-engine`), then `sha256sum -c SHA256SUMS --ignore-missing`.

A file that was changed, or built by anything else, fails both: `.github/workflows/sign-check.yml` runs these commands on a test file, including a changed one that must fail.

## What is not covered yet

- Platform code signing (Windows Authenticode, Apple notarization) needs certificates that only the owner can get: [#223](https://github.com/surya-koritala/Glyd/issues/223).
- `install.sh` and the later backend downloaders do not check these signatures yet; they check the SHA-256 of the release (#207's second task).
- Reproducible builds and a SLSA level beyond GitHub's provenance: D6b ([#259](https://github.com/surya-koritala/Glyd/issues/259)).
