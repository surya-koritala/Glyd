<!-- What changed, and why. Link the issue: "Closes #N" or "Part of #N". -->

<!-- For a change to speed, size or quality: the numbers before and after, from the same machine, with the command
(`glyd-engine bench`, `costs`, `rates`; `examples/bench_suite.rs`, `scripts/bench_versions.sh` for the codec). -->

## How it was tested

<!-- The commands you ran and where: the machine, and the GPU model and driver if a GPU was involved.
Say what you could not test (another GPU, Windows, macOS, a model you do not have). -->

- [ ] `cargo test --release --workspace` passes.
- [ ] Anything written by an earlier version still decodes (a format change comes with a fixture in `tests/data/legacy/` and a line in `docs/spec.md`).
- [ ] A new model or kernel has a logits test against the CPU reference.
- [ ] One change in this pull request, in the style of the code around it.
- [ ] Every commit carries `Signed-off-by:` (`git commit -s`), certifying the [Developer Certificate of Origin](../DCO).
