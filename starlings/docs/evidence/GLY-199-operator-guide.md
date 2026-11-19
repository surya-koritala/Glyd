# GLY-199 scoped operator guide

Issue: [GLY-199](https://linear.app/getglyd/issue/GLY-199).

The [operator entry guide](../getting-started.md) connects the existing setup,
enrollment, artifact, HTTP, placement, benchmark and recovery guides without
inventing another install or activation flow. It begins with default CLI build,
version and doctor commands; it does not initialize a profile, start a service,
fetch a model or run a network check. The support table links historical records
and separates configured software coverage from qualified native/device behavior.
Troubleshooting preserves unknown mutation outcomes and durable quarantine.

Review traced the commands to main.rs, cluster.rs and models.rs; read the owning
daemon/catalog/production-routing/recovery/diagnostics contracts and historical
M1 physical and Mac CPU acceptance records. An initially drafted client-api.md
link did not exist and was corrected to the real listener contract and strict
chat schema before validation. Relative files/headings are checked locally.
Diff checks and ordinary --version/doctor checks are basic Mac checks;
no network, instrumentation, process inspection, native workload or new download.
This documentation-only change needs no new test suite or CI job.

Initial local review verifies22 README,20 guide and one evidence relative links,
including headings. The documented default CLI build with cached locked assets
and two jobs passes (9.03s); that built executable prints version0.1.0 and
protocol1/macos/aarch64/inference=unavailable. Explicit installed SDKROOT/PATH were
used for this development environment. This is not clean-machine or installed
package acceptance, and no daemon/profile was initialized or launched.
The guide is delivered with the related GLY-197 node-refusal experience in one PR
to avoid duplicate CI matrices. Fresh438e293 main is integrated; all five pure
planner tests, both exact offline CLI-file modes, both warning-denied workspace
lints and formatting/diff pass afterward. Final-source/current-main review and
all seven CI checks remain required; this is not native or packaged acceptance.

Ponytail keeps a single navigation/limits guide and reuses existing detailed
contracts. Separate complexity review removes duplicate command flows; correctness
review checks claims against their exact evidence. Independent milestone/release
review remains deferred, not approved. GLY-199 stays open for packaged clean-machine
acceptance and final supported-model/API/claims review; GLY-198/196 dependencies
remain. No publication, license decision or native benchmark qualification.
