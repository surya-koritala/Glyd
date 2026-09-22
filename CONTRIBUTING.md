# Contributing

Thank you for working on Glyd. Two things matter here more than
anything else: nothing may ever fail to read back what an earlier
version wrote, and every claim about speed or size is a measurement
someone can repeat.

## Before you start

Open an issue for anything beyond a small fix, so the approach can be
agreed first. Bugs and speed or ratio reports have templates; a
security problem goes by email to suryakoritala1324@gmail.com, not to
a public issue.

## Building and testing

    cargo build --release --workspace
    cargo test --release --workspace

The container tests need `gzip` and `python3` on the path. Continuous
integration also runs the regression floors, the C ABI, the CLI round
trips with corrupted input, and a fuzz run; a pull request has to pass
all of it.

## What a change must show

- **Correctness.** Every round trip byte-exact. A change to a format
  keeps everything written before readable: add files written by the
  released version to `tests/data/legacy/` and a test that reads them,
  and describe the format in `docs/spec.md`.
- **Speed or ratio.** The numbers before and after, from the same
  machine and thread count, with the command that made them:
  `scripts/landscape.py` (every codec, one thread),
  `examples/bench_suite.rs` (the suite, all threads),
  `scripts/bench_versions.sh` (versions and base mode). A change that
  gains in one place and loses in another says both.
- **Scope.** One change per pull request; code that reads like the
  code around it.

## Commits and licensing

Every commit carries a `Signed-off-by:` line (`git commit -s`), which
accepts the [contributor license agreement](CLA.md): you keep your
copyright and grant the project the right to use and relicense your
contribution. A check on every pull request requires it.

The codec (the `glyd` crate, CLI, C ABI and bindings) is under the BSD
3-Clause License or the GPL version 2, at the user's option; the store
(`glyd-store`) is under the Business Source License 1.1. See
[README.md](README.md#license).

## Conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).
