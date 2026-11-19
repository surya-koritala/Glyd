# GLY-311 logical ownership and native file-lock lifetime

Issue: [GLY-311](https://linear.app/getglyd/issue/GLY-311).
PR: [107](https://github.com/Sigbound/glyd-compute/pull/107).
Base: main48c18b32e15aa5985a7867de3fb34d6c9f6a964e (PR106).

Automatic main CI38946764093 failed Linux x64 default at service.rs926:
durable_ownership_refuses_before_registry_without_mutating_records received
PROFILE_IN_USE after the false-registry preflight. Other five partitions passed;
the aggregate failed. Raw work/m6-pr106-main-linux-x64.log SHA256
9eaaa6f5651ddd005c8b854298717397c8c13f52a1043a45df5041ed563d51ad.
Source106 had passed all seven; that does not qualify this failed main.

The file-only control duplicates the owner's native file handle, confirms two
independent competing acquisitions are denied, drops the logical owner while the
alias lives, acquires the new owner, closes the old alias, and checks the new owner
still excludes competition. After the new owner drops, acquisition succeeds.
Original code fails exactly at post-owner acquisition with ProfileInUse in0.00s
(3.52s compilation); work/m2-owner-alias-control.log SHA256
e944d0b3b8096f0bd564b71fae15b3dd67a28ba6310d01f51da0a43a209f0e85.
No process is created or inspected to produce this control.

[Rust File locking](https://doc.rust-lang.org/std/fs/struct.File.html#method.try_lock)
documents release after all duplicated/inherited handles close or an explicit
unlock. The [Linux flock interface](https://man7.org/linux/man-pages/man2/flock.2.html)
associates locks with the shared open-file description. These contracts explain
the reproduced lifetime defect. A concurrent fork/exec can produce such an alias,
but the failed CI log does not identify an actual inherited descriptor or child.
No archived Windows cause is attributed by this proof.

The shared lock_file helper now returns a small owned FileLock guard that invokes
standard File::unlock before native close. All three callers use it: StateLock
(daemon/catalog), lazy authorization State and RegistryClient. Native file flags,
private-file verification, try_lock denial/error behavior and persistent lock
files are unchanged. No lock deletion, acquisition retry/wait or policy relaxation.
Unlock failure cannot grant ownership; native close remains the conservative
fallback, and aliases may then retain the lock rather than forcing entry.

Guards are last fields of owners; Catalog likewise retains StateLock after its
other fields. [Rust field destruction order](https://doc.rust-lang.org/reference/destructors.html)
therefore keeps exclusivity until connections/databases and other owned handles
have closed. Logical Drop occurs only after the existing owner/thread lifecycle;
this does not authorize backend drain, release reservations/pins or reset quarantine.

Basic bounded Mac checks, at most two jobs, sequential Cargo:

- Corrected alias control PASS0.00s, including repeated live-owner refusal and
  no loss of the new lock when the old alias closes. The case is enabled on every
  supported OS; Windows execution remains a CI gate.
- Original durable-ownership/preflight file case PASS0.11s, unchanged assertions.
- Final default/all-feature workspace/all-target warning-denied lints2.65s/1.59s,
  formatting/diff PASS. An initial fixed control caught a Mac unused import;
  gating that Windows-only File import fixes it before source publication.
- Earlier preliminary corrected log12e3d485bc56680360583a08cb1744a86490abc98b4f93e743c8ffba85fd5390
  is retained, including that warning; it is not the final warning-free gate.

Separate complexity review: one standard-library guard for three existing users,
no lock framework/global mutex, dependency, sleep or retry. Correctness/security
review checks failed-acquisition isolation, native identity/permissions, final-field
drop order, lazy-authority acquisition and continued fail-closed cleanup ownership.
Full latest-source CI/current-main/review/merge and separate main qualification
remain required. Independent milestone/security review is deferred, not approved.
No local sockets, services, scan, host/process instrumentation, models or workload.
