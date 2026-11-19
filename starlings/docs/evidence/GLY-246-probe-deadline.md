# Shared readiness I/O deadline

Readiness formerly supplied a fresh two-second timeout to each HTTP operation:
three operations without properties, five with properties. A slow local service
could therefore accumulate six/ten seconds of I/O without any individual timeout.
The corrected shared probe uses one monotonic deadline, refuses zero/expired
remaining budgets and checks expiry before setting ready. Authentication, model/
properties validation and response bounds remain unchanged. No new dependency,
fallback, concurrency abstraction or expanded deadline is introduced.

All callers were traced: worker startup/health/serving, the direct service example
and synthetic/native relay helpers use the same Service probe. Existing native
approvals and frozen experiment binaries are unchanged; no native test was run.
This bounds I/O waits, not OS scheduling or synchronous process checks, and cannot
preempt a caller that violates the nonblocking progress-observer contract.

A model-free fixture delays each response by 650 milliseconds. On unchanged main
33d74fa1, the new regression failed with `slow cumulative readiness incorrectly
succeeded` (3.69 seconds for the test). After the correction it rejects readiness
within the shared budget, exposes no properties and stops the owned fixture.
All 14 adapter tests and eight daemon lifecycle cases pass locally with warnings
denied; the lifecycle run took 36.78 seconds. These are software test records, not
native inference, hardware or Exo performance measurements.

GLY-246's original Windows loading timeout at 488022ad/run38380035001 remains
preserved. Later main run38382295428 passed all seven checks and gave no new failing
phase. The additive-timeout defect is demonstrated independently; this correction
does not establish the sole cause of that earlier Windows failure. GLY-244's
executing-invite finding is separate. Both original root-cause/disposition gates
remain open unless stronger exact-head evidence resolves them. Final-head platform
CI and correctness/security plus separate complexity review are required for merge.
