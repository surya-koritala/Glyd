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
- [ ] Release gate clean: `check_wheel.py` on the engine and the library (no dev strings, no unlisted GLYD_* variables, no codec words).
- [ ] Binaries built on Ubuntu 22.04 (glibc 2.35 or older), sha256 written down.
- [ ] Clean-install check on a fresh machine: `pip install glyd==X.Y.Z`, `glyd run` each new model, `glyd verify` it.
- [ ] Plain style everywhere (first person, raw numbers, no adjectives, no comparisons with other projects).

## Release day (order matters: see memory glyd-release-steps)

1. Private: `scripts/bump_version.py X.Y.Z`, CHANGELOG `## Unreleased` → `## X.Y.Z`.
2. Public: the same bump + CHANGELOG (what, not how: no formats or kernels).
3. Fast-forward private `glyd-shim` to the public release commit.
4. Merge private, tag private `vX.Y.Z` → `glyd-gpu` to PyPI.
5. Merge public, tag public `vX.Y.Z`, `gh release create vX.Y.Z --verify-tag --notes-file notes.md` right away.
   Then attach Glyd's engine for install.sh's Rust-only install (GLY-353): download the private release run's `engine-bundle-linux-x86_64` and
   `engine-bundle-linux-aarch64` artifacts (`gh run download RUN -R Sigbound/glyd-gpu-private -p 'engine-bundle-*'`) and
   `gh release upload vX.Y.Z -R surya-koritala/Glyd glyd-engine-vX.Y.Z-linux-*.tar.gz glyd-engine-vX.Y.Z-linux-*.tar.gz.sha256`. install.sh
   refuses a release without them on Linux with an NVIDIA GPU (GLYD_WITH_VLLM=1 still installs the Python tool).
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
