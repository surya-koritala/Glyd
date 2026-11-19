# Cargo cache packing follow-up

PR #21's Windows job passed, but took 7m04. Its cache step took 142 seconds:
GNU tar/zstd started at 17:52:23.524 UTC and network progress began at 17:54:38.163.
The 709,816,823-byte archive then finished uploading at 17:54:45.424. Thus roughly
135 seconds preceded upload; the bottleneck was mostly archive preparation, not
network transfer. The default test step took 172 seconds, including 29 seconds
of compilation. These are CI diagnostics, not inference benchmarks.
[Source job](https://github.com/Sigbound/glyd-compute/actions/runs/38286584818/job/114936304325).

The previous cache included both compressed crate archives and unpacked copies
under `registry/src`. [Cargo recommends caching the index and compressed archives](https://doc.rust-lang.org/cargo/guide/cargo-home.html#caching-the-cargo-home-in-ci)
and recreating unpacked sources as needed. Cache version v5 follows that guidance
while retaining `target` artifacts. This workspace has no Git-source dependencies.
No credentials, global configuration or installed personal binaries are cached.

Keep the original archive path list (`registry` and `target`): the cache action
includes that list in its cache version, so changing it would make legacy entries
unusable despite matching fallback keys. Prune only `registry/src` immediately
before saving. Fallback can therefore restore existing builds rather than
imposing a cold dependency compile. A fallback may contain old unpacked sources, so the workflow removes
that generated source directory before its existing locked fetch. Every platform
therefore tests recreation from archives. Exact v5 hits contain no source copies.
Cargo validates its normal fingerprints before reusing compiled artifacts.

All four platforms, formatting, default/all-feature warning-denying lint, tests,
dependency policy, failure summaries and the ten-minute job cap are unchanged.
Caches still save after failed tests, but cancelled obsolete runs skip pruning
and saving so post-test cache work cannot delay the replacement run. No tests or checks are skipped. The actual
speed improvement is not assumed: attach latest-head CI timings to GLY-227 after
validation. Later exact-cache hits can avoid saving, as before.

PR107 source82d8cd9/CI38948219733 reaches the unchanged ten-minute Windows
backend cap after its production matrix passes241.86s; recovery has just started.
Its locked fetch alone takes55.6s despite an exact v5 cache hit. The native jobs
now use `cargo fetch --target host-tuple` in both offline and existing online
paths. [Cargo's target selection](https://doc.rust-lang.org/cargo/commands/cargo-fetch.html)
avoids fetching other architectures; every job still builds/tests its native host.
Pinned offline metadata comparison finds235 all-target packages versus195 for
Windows (40 absent there). This is graph evidence, not a measured fetch-time win.
No test, lint, target platform, warning policy, serialization or job cap changes.
Existing Windows production modes additionally print bounded start/done elapsed
markers so future cancellation identifies the unfinished mode. No host/process
instrumentation or local network test is required.
