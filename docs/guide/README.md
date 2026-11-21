# The user guide

The source of the guide at https://surya-koritala.github.io/Glyd/guide/. mdBook, plain Markdown in `src/`.

```bash
cargo install mdbook --version 0.4.52 --locked
scripts/build_site.sh /tmp/glyd-guide       # from the repository root; open /tmp/glyd-guide/index.html
```

- `src/api.md` and `src/service.md` are not in git: `scripts/gen_reference.py` writes them from `docs/serve-api.md` and `docs/service.md`. Edit those.
- `scripts/check_gate_docs.py` fails the build if `benchmarks/gates/m1-gate.sh` has a step that `src/gate.md` does not name.
- `src/models.md` is not in git. `scripts/gen_models.py` writes it from `glyd-engine/src/support.rs`, `levels.rs`,
  `machine.rs` and `kvcal.rs` on every build. If one of those changes shape, the script stops with the reason: fix it in
  the same change.
- `.github/workflows/pages.yml` builds main to `/guide/` and each `v*` tag that has this folder to `/guide/vX.Y.Z/`, and
  deploys on a push to main or a release. A pull request that touches the site is built, not deployed.
- Rules: a command goes in only after it was checked against the code on main; a number goes in with its machine and a
  link to its log; a feature in an open pull request is not described until it merges. Short plain sentences.
- When a change makes a page wrong (a flag, a default, a level), fix the page in that change.
