# GLY-198 native Linux unit-file validation

Issue: [GLY-198](https://linear.app/getglyd/issue/GLY-198).
Base: main043d52c027963b9836f65bb5c101271ff772278f (PR104).

The existing inactive descriptor check gains Linux native verification rather
than a second custom parser or installer. It passes the actual generated user
unit to systemd-analyze with man-page execution and unit generators disabled,
and syntax warnings made fatal. Success must have no output/warnings. A second
fixed input adds one unknown directive and must fail with that directive in its
diagnostic; an unrelated environment failure cannot satisfy the positive check.

Upstream v255 source shows that verification also enables environment generators
internally even when unit generators are disabled. The child therefore points
SYSTEMD_ENVIRONMENT_GENERATOR_PATH at a newly created empty private directory,
and uses a private XDG_RUNTIME_DIR. These are test-child settings, not changes to
OS services, logging/security controls or the parent environment. The upstream
test mode does not open an executor. [Packaging](../packaging.md) links the exact
primary sources/options. No source implementation is copied.

Only the existing bounded Linux default/backend CLI test selections execute the
native parser; no extra CI job, case or dependency download is added. The Linux
development test prerequisite is systemd-analyze>=250. It is not required to run
the foreground CLI and is not part of the shipped binary's dependency graph.
Mac checks remain basic file/native-plist round-trip and warning-denied compile
checks with two jobs; this does not run systemd, sockets or models on the Mac.
The selected Mac check passes0.06s; default/all-feature warning-denied workspace
lints pass5.09s/2.51s. Formatting/diff pass. Separate complexity review keeps one
native closure for two inputs and adds no parser framework; correctness/security
review verifies child-only paths, no executor/generator commands, fatal warnings
and a specific negative diagnostic. Initial source a49e313 / CI38941880394
passed both Linux partitions, including the native positive/negative checks.
Windows default failed the unrelated frame-deadline assertion; the aggregate
also failed, so the complete source is not qualified. The correction and retained
failure are recorded in [GLY-184 controls](GLY-184-frame-deadline-controls.md).

This checks syntax/executable/dependencies, not exact effective argv, installation,
login/logout, stop/drain, crash recovery, upgrade/rollback/uninstall, WSL2 or native
inference. Full GLY-198 remains In Progress. All platform source checks, review and
merge gates remain required; independent milestone/release review is deferred,
not approved. There is no license, publication or workload authorization change.
