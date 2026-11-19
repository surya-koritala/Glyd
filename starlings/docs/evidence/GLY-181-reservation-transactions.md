# GLY-181 software reservation transaction

Issue: https://linear.app/getglyd/issue/GLY-181

## Irreversible local owner withdrawal

The reservation book now has one local owner-withdrawal operation. `withdraw` refuses
future admission permanently and puts every live transaction into `Releasing`,
preserving its handle, reservations and partial cleanup. Delayed readiness and
commit observations cannot reopen it; draining the last record does not resume
admission. This is a local lifecycle operation, not a live authorization check,
native cancellation or daemon integration.

One bounded pure control covers empty books, multiple transactions on separate
synthetic nodes, preparing/committing/ready states, repeated withdrawal and partial
cleanup. The first test attempted two full plans on one capacity-limited fixture
and correctly received `Capacity`; the corrected fixture uses two nodes. All seven
admission controls pass in 0.10 s after a 0.46 s cached offline build. Core
all-target/all-feature Clippy with warnings denied passes in 1.50 s; formatting
and diff checks pass. No network/model/device workload or new dependency/job/cap.

Separate complexity review reuses the existing state and bounded transaction map,
adding one accepting latch and no resume policy or generic controller. Correctness
review checks irreversible closure, all-record withdrawal, retained owner IDs and
partial cleanup, stale reply rejection and unchanged late-allocation safeguards.
The caller still must guard real dispatch/output with live registry authority and
map trusted provider cleanup before production activation. Independent milestone/
security review and full native/durable qualification remain open. Exact source,
PR and CI identities are recorded in the linked issue after qualification.

## Late allocation after partial cleanup

On baseline8d80aa4, a pure two-stage control aborts a transaction, marks stage zero
drained, supplies a late over-budget allocation for that stage, then drains stage
one. The transaction incorrectly disappears (`NotFound`); the regression fails
in0.06s after a6.48s cached offline build. The node fence alone does not invalidate
the earlier per-stage cleanup flag.

The shared invalid-allocation branch now clears that flag. Missing, extra,
over-budget and conflicting observations use the same branch; fresh trusted
cleanup is required before the stage's hold and transaction can disappear.
The new control covers both over-budget and changed previously prepared
allocations. All six admission controls pass0.09s after a0.92s build;
all-target/all-feature core Clippy with warnings denied passes2.94s. Formatting
and diff checks pass. Existing CI owns broader regression coverage; no job,
dependency, deadline or networking fixture is added.

Separate complexity review: one assignment in the existing shared error path,
using the existing stage flags and regression helpers. Correctness review:
the stage index is validated before access; uncertain-node fencing and Releasing
state remain; other stage cleanup is retained; the record survives until the
affected stage is freshly drained. No native lifetime or historical CI cause is
inferred. Independent milestone/security approval remains open.

Source/PR/CI identities are recorded in the linked issue after qualification.
The full provider/live-fence/durable/native acceptance remains unfinished.

## Initial reservation transaction

The Compute-owned [transaction](../transactional-admission.md) recomputes the
existing bounded candidate preview and reserves all stage/pool budgets before
preparation. It serializes competing reservations, refuses replenishment from
larger telemetry, checks complete actual allocation observations and requires
all-stage prepare then commit barriers. Abort/expiry retain ownership until
trusted per-stage cleanup. A fixed owner incarnation/epoch and monotonic serial
prevent stale handle reuse. The module launches no process and performs no I/O.

Local five-case pure synthetic control:5 passed,0 failed,0.09s after a5.33s cached
two-job offline build. No model, socket, device/host inspection, instrumentation
or native workload ran. Warning-denied core/all-target Clippy passes1.63s after
collapsing one nested conditional; formatting/diff checks pass. Source/CI identities
are recorded in the issue/PR after qualification. Existing platform jobs and warning denial
are retained without a new job, dependency or time cap.

Separate complexity review uses the existing planner, profile types, CoreError
and standard BTreeMap/Set; no generic backend interface or guessed wire API is
introduced. Correctness review covers all-pool atomic insertion, late conflicting
allocation fencing, commit barrier, retained rollback ownership, stale handles,
clock bounds and fixture refusal. These reviews are implementation review, not
independent milestone/security approval.

Remaining: delivered provider235/236 mapping and actual allocation/drain evidence,
live authorization fencing, durable native ownership/recovery, daemon integration,
calibrated timing/failure records and real-device/independent qualification. Full
GLY-181 remains In Progress; software observations are not native measurements.
