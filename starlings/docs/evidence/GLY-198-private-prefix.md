# GLY-198 private source-prefix staging

Issue: [GLY-198](https://linear.app/getglyd/issue/GLY-198).
Checked source: main `a66eb9dc9d0899d20871798aae6ce1615719ecbb`.

Reuse native Cargo install/list/uninstall instead of introducing an installer,
activation command or global PATH change. The reproducible source workflow is in
[packaging](../packaging.md#private-source-staging-with-cargo); upstream
[install](https://doc.rust-lang.org/cargo/commands/cargo-install.html) and
[uninstall](https://doc.rust-lang.org/cargo/commands/cargo-uninstall.html)
define the prefix and tracking behavior. Pinned Rust 1.97.1 help and the actual
bounded Mac check confirm the selected options, including host-tuple.

The private check uses one clean source checkout and a fresh private workspace
prefix. Locked/offline resolution, denied compiler warnings, two build jobs and
a five-minute total cap apply. Child PATH includes only this new bin directory
in addition to the existing toolchain paths. Cargo's installed-package listing
identifies Compute, and the installed binary matches the build's CLI artifact
with no default features. Its absolute-path --version and doctor outputs match
the compiled version/protocol/OS/architecture and inference=unavailable.

Uninstall then removes the tracked binary; a second package listing is empty.
The source/lock remain unchanged and an adjacent synthetic file remains intact.
That sentinel is not a real profile or evidence of native cache preservation.
No general Cargo installation, shell configuration, service, backend, model,
old profile, ownership record or quarantine changes. No network/scanner/process
inspection or device workload runs; only this check's compiler child group may
be stopped on its timeout. Private logs and detailed build metadata stay local.

The staging/static-output/removal checks pass without warnings. A configured
development Mac is not a clean machine, and this is not version migration,
rollback, cluster join/import/run/restart, service-manager or WSL2/GPU acceptance.
GLY-198 and its parent stay In Progress. Source platform/lifecycle/native evidence,
independent review, license and release approval remain required.

Separate complexity review keeps Cargo's existing prefix and tracking behavior;
no installer wrapper, dependency, new test job or runtime code. Correctness review
covers refusal of an existing prefix, child-only environment, explicit local
source/default feature selection, unchanged source/lock, exact static output,
tracked removal, retained synthetic neighbor and the stated qualification limits.
Documentation link/shell/diff checks and exact-source/current-main/CI/review/merge
gates remain required; automatic main is verified separately. Independent milestone
and release review is deferred, not approved. No publication or license is granted.
