# GLY-184 absolute frame-deadline controls

PR: [105](https://github.com/Sigbound/glyd-compute/pull/105).
Base: main043d52c027963b9836f65bb5c101271ff772278f.

Source a49e313 / CI38941880394 failed Windows default at
data/tests.rs:1267: the 60ms test expected Transport but received Expired.
Dependency, Linux x64/ARM64, macOS and Windows backend passed; the Windows
aggregate failed. The retained raw Windows default log SHA256 is
be52ea74c6127926ab5e2c2e8a9ea6ac020fc16e97456a22675ddc27140e7027.
No unchanged-source rerun is used.

The test started its deadline before creating/synchronizing the writer. The
existing production contract returns Expired before an expired operation starts
and after a successful operation exceeds its deadline; underlying I/O errors
remain Transport. A buffered read can also reach an explicit receive deadline
guard. The observed error alone does not distinguish those valid expiry paths
or establish any historical Windows routing/registry cause.

Extract the existing before/after clock guard into one private function without
changing runtime semantics. A socket-free controlled-clock check strictly proves
no operation starts at expiry, late success becomes Expired, timely success is
preserved, and the original Transport/Denied errors retain precedence.

The enrolled TLS case now waits for writer readiness before starting the same
60ms budget. Three small inputs trickle the prefix, header and eight-byte payload
separately in four chunks with 30ms spacing. No callback may receive a late frame;
the next read must return Expired. A test-only getter observes the actual socket
deadline: after I/O it must equal the original absolute deadline, not a renewed
per-read budget. If I/O never starts, only Expired is permitted. After I/O, only
Transport/Expired at or after the deadline is permitted. Other errors, early
failure, successful delivery, renewed deadlines and reusable failed streams fail.

Basic Mac verification runs only the controlled-clock unit with an exact
selector: one case passes in0.00s after7.96s cached compilation, max two jobs.
Default/all-feature warning-denied workspace/all-target lints pass3.07s/1.61s;
formatting and diff checks pass. The socket case is compiled/linted locally and
executes only in existing CI.
No local sockets/services, instrumentation, probes, models or device workloads.
No dependency, CI job, timeout, lease, admission or error-mapping change.

Separate complexity review: one direct clock helper and a read-only test getter;
reuse the existing native case and fixtures without a clock framework or added
runtime state. Separate correctness/security review checks first-error precedence,
boundary equality, immutable I/O deadline, no delivery and fail-closed reuse.
Independent milestone review remains deferred, not approved. Complete corrected
source CI and post-merge main qualification are still required. Full GLY-184
physical-link/instrumentation/native acceptance remains open.
