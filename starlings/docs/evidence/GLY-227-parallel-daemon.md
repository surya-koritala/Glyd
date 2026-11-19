# Parallel existing daemon acceptance scenarios

The completed Windows job for [PR #38](https://github.com/Sigbound/glyd-compute/actions/runs/38332772205)
took 8m22. Its filtered daemon step compiled in 15.70 s and tested in 74.71 s;
the combined four-mode test crossed Rust's 60-second progress notice. Default
tests took 3m16 including 42.73 s compilation; security tests took 59.96 s and
catalog tests 43.63 s. Cache packing was not the dominant cost on this run.

On Linux/macOS, normal, crash, revoked stream and busy-idle scenarios each get an ordinary
Rust test entry point and run concurrently with the existing qualified recovery
and warm-up checks. Every scenario calls the same owning helper and assertions.
Each helper owns its temp directory, enrollment, catalog, process handles,
listener budgets and ephemeral ports; no environment or filesystem namespace is
shared. The existing `catalog_service_` filter still selects all checks. No new
workflow/dependency, removed test, shortened timeout, workload or weakened gate.
Failures name the particular scenario immediately.

Local all-feature Clippy denies warnings; six model-free process tests pass in
36.86 s versus the prior 48.07 s filtered step. This is a local diagnostic, not a
controlled cross-platform speed guarantee. Native tests stay excluded. All four
platform checks plus dependency policy on the latest head remain required before
merge. Exact head, CI timings and review disposition are recorded in Linear.

The first exact-head CI run passed Linux x64, ARM64, macOS and dependency policy,
but Windows failed readiness and an admin IPC command under five concurrent
scenarios. The independent data/ports did not prevent host filesystem/IPC
contention. Windows retains its previously validated two-test layout (four modes
inside the original matrix plus warm-up), with unchanged assertions/deadlines.
No global lock, admission change, failure retry or broader timeout is introduced.
Only Linux/macOS use the concurrent mode entry points; full latest-head CI remains
required. Cross-run total timings do not establish a general speed guarantee.
