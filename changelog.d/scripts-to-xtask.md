- The release's own tools are Rust: `cargo xtask bump-version X.Y.Z` (and `--check`) sets the version in every file that carries it and puts
  `changelog.d/`'s entries under the release's heading, and `cargo xtask bump-formula X.Y.Z` points the Homebrew formula at a release. They write the
  same files as `scripts/bump_version.py` and `scripts/bump_formula.py`, which are gone; `cargo test -p xtask` holds what those scripts wrote.
