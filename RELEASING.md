# Releasing Glyd

One version number, `X.Y.Z`, for everything a release ships. No other numbering anywhere: no per-model repos with their own
versions, no separate model versions.

| Where | What carries X.Y.Z |
|---|---|
| GitHub, surya-koritala/Glyd | tag `vX.Y.Z` and a release page (notes below, binaries attached) |
| PyPI | `glyd` and `glyd-gpu` X.Y.Z |
| Homebrew | surya-koritala/homebrew-glyd formula X.Y.Z |
| Hugging Face | every `glyd/*` model repo added or changed in this release gets tag `vX.Y.Z` (the Glyd that packed it); its card says "Glyd X.Y.Z or newer" |
| Site | version badge, the models table, install.sh from the tag byte for byte |

New models ship through `glyd run` only (`glyd run glyd/gemma-4-31B-wren`). Model repos are `glyd/<model>-<level>`, the level a
bird: penguin, kestrel, swift, wren, hummingbird.

## Before the release

- [ ] Every model in the release is on the Hub (private until release day), with a complete card (research/*/CARD-*.md, the HF card
      template) and passes `glyd verify`.
- [ ] Each model card's first screen: what the quant is (bits per weight, lossless or not, against what), quality against the
      original (KLD + one non-saturated eval), speed on 2-3 common cards with the machine named, known limits, how-it-works link.
- [ ] Numbers in the notes and cards match research/*/results/ (no number without a results file).
- [ ] Binaries built on Ubuntu 22.04 (glibc 2.35 or older), sha256 written down.
- [ ] Clean-install check on a fresh machine: `pip install glyd==X.Y.Z`, `glyd run` each new model, `glyd verify` it.
- [ ] Plain style everywhere (first person, raw numbers, no adjectives, no comparisons with other projects).

## Release day (one repository: surya-koritala/Glyd)

1. `scripts/bump_version.py X.Y.Z` (every Cargo.toml, both pyproject.toml, the installer's pin), CHANGELOG `## Unreleased` → `## vX.Y.Z`.
   PR to main, CI green.
2. `scripts/release_ready.sh X.Y.Z` on the merge commit, then tag `vX.Y.Z` there, push the tag, and right away
   `gh release create vX.Y.Z --verify-tag --notes-file notes.md` (notes below): both workflows upload to that release.
3. The tag runs both release workflows from this source:
   - `release.yml`: the `glyd` CLI tarballs and wheels attached to the release, `glyd` to PyPI and crates.io.
   - `release-gpu.yml`: the GPU library, the engine, the `glyd-gpu` wheels to PyPI, and the engine bundles
     (`glyd-engine-vX.Y.Z-linux-*.tar.gz` and their `.sha256`: `glyd` with the engine linked, `glyd-engine` as a link to it, the GPU
     library) attached to the same GitHub release (`attach-engine`), which `install.sh` and `condor/tools/install.sh` download.
   PyPI's trusted publisher for `glyd-gpu` must name this repository and `release-gpu.yml` (environment `pypi`).
4. Check the release page has the CLI tarballs, the wheels and both engine bundles.
5. (reserved)
6. Hugging Face: flip the release's model repos to public, tag each `vX.Y.Z`.
7. Homebrew formula PR + tap commit.
8. Site PR (owner approves).
9. Post (r/LocalLLaMA etc.) only after 1-8 are live and a clean install works.

## Release notes template

```
Glyd X.Y.Z

- <Model family> at `:<level>`: `glyd run <command>` downloads `glyd/<repo>` (<size> GB). <One line: what the level is, e.g.
  "Google's QAT weights kept bit for bit, 4.5 bits per weight">. Measured, one user:
  - <card>: <plain> tokens a second, <with drafting> with the draft model (same output); <prompt> tokens read in <s> s.
  - <eval> <score> (<n> questions), KLD <x> against <reference>.
- <Feature>: <what it does for the user, one or two lines>.
- <Fix>: <what was wrong, what happens now>.

Known limits
- <flat statements>

Install: pip install glyd==X.Y.Z (or brew upgrade glyd)
```
