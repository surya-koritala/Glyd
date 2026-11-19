# Installed-service properties software gate

GLY-176 now has an optional, configuration-digest-approved `GET /props` inspection
for compatible installed services. The probe requires unauthenticated refusal,
then bounded authenticated metadata matching build, model alias, per-slot context,
one slot and system/text template support before granting API readiness. This
does not grant inference qualification. Route validation refuses a context promise
larger than the inspected expectation. Existing omitted options retain their
serialization/digests and trigger no additional HTTP operation.

Daemon diagnostics separate declared version/profile from the historical startup
capture. The report does not retain backend paths, templates or unknown fields;
errors remain static and do not expose the API key or raw metadata. Existing
periodic readiness checks repeat the optional inspection and refuse mismatches.

The synthetic-process regression checks approval binding, absent-field digest
stability, matching properties, build mismatch refusing readiness, stop withdrawing
properties, model/context/slot/capability rejection, typed text support and duplicate/
oversized JSON. The existing actual daemon lifecycle regression now uses this
option for its normal fixture, verifies status separation and remains unqualified.
No native/model/GPU workload is run by these checks.

Local validation passed with warnings denied: default and all-feature workspace
Clippy, all 13 development adapter tests (2.27 seconds), the existing service
failure/pin/restart daemon case (36.23 seconds), and the production count/JSON/SSE/
slot-reuse/revocation route case (13.95 seconds). Those elapsed values identify the
local software runs only, not native inference or controlled CI speed comparisons.
The first focused test could not start its child listener in the sandbox; its
host-permitted loopback rerun passed. No application timer or assertion was widened.

Initial Windows CI at `488022ad` / run38380035001 failed the production-route
fixture's readiness deadline while status remained `loading`, with no reported
startup error or properties capture. The separate warm-up case passed. Static
spawn/probe phases and the last probe error are now visible while loading; success
clears the transient error. This supplies diagnostics, not a demonstrated root
cause or evidence that an unknown failure is fixed. The properties mismatch
regression verifies the failed step, and the production route still passes locally.
Final-head Windows and all other checks remain required before merge.

The pinned upstream contract was inspected locally at
`d81235049384534c167caea52b85a694f6103d14`; the implementation uses public field
semantics and existing Rust HTTP/JSON primitives, without copied source or new
dependencies. Correctness/security and separate complexity review are required,
followed by all seven latest-head CI checks before merge.

This gate does not prove loaded model/profile identity, tensor conversion,
precision, native allocation/drain, GPU execution or an installed Glyd service's
missing capability hooks. A model listing or properties claim cannot replace
native qualification. GLY-176 and integrated M2 remain incomplete until their full
acceptance evidence exists. No prepared approval, quarantine, license, repository
setting or parked CPU-policy code is changed.
