# changelog.d

One file per pull request holds that pull request's CHANGELOG entry, so two pull requests never edit the same lines.

- Name it after your branch or what it does: `changelog.d/<slug>.md` (for example `changelog.d/wren-release.md`).
- Write the entry as it will read under the release's heading in CHANGELOG.md: one or more `- ` lines, user-facing,
  what changed and the numbers where there are some.
- Don't edit CHANGELOG.md's `## Unreleased` section in a pull request.

At release, `cargo xtask bump-version X.Y.Z` puts every entry here under the release's heading in CHANGELOG.md, newest
first (by the commit that added each file), and removes the files. This README stays.
