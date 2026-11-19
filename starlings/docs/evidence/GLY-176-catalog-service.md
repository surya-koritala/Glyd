# GLY-176 / GLY-172: catalog-owned daemon service

Production worker daemon warm-up now binds approved configuration and executable
to exact verified catalog weights paths. Service ownership, durable pins, loading,
health checks and shutdown run outside registry/admin control. Launch quarantine
prevents uncertain restart and cache release. CLI status distinguishes readiness
from inference qualification. This continues both issues; neither is complete.

Software checks:

- Catalog tests cover exact verified paths, corruption, traversal, directory
  replacement, quarantine restart, denied unpin and GC protection after revocation.
- The feature-gated daemon test enrolls real separate processes and covers warm-up,
  crash, wrong digest, non-cache weights, wrong profile, drain, responsive admin,
  retained pins and refused restart. Its child is synthetic, not an LLM.
- Formatting, default/all-feature Clippy and tests run with warnings denied.
- CI adds only the filtered daemon lifecycle case to the development test pass;
  existing default daemon tests are not repeated. All five CI gates remain required.

Correctness/security review: no shell/path lookup, implicit download, precision
fallback, automatic process restart, eligible replica or inferred native drain.
The local operator/runtime files remain trusted. API readiness and argv binding
cannot attest tensor inventory, actual loaded weights, arithmetic or dependencies.
File hashing can delay shutdown; admin acknowledgement remains asynchronous.

Ponytail review: existing configuration, private files, catalog pins, process
supervisor and standard thread primitives; no dependency or alternate launch
schema added. Independent milestone review is deferred by explicit merge
authorization, not claimed as approved. Exact final commit and CI links are
recorded on the linked PR and Linear before merge.

No native model workload, download, GPU, Ubuntu, Raspberry Pi or paid
infrastructure was used. M2 remains open for production generation routing,
qualified memory/cleanup recovery, catalog-native conformance and link/task
benchmarks. M3 has not begun.

The first Windows CI run exposed costly capacity-test setup: the security suite
passed in 182.18 seconds, ending with the invitation-cap test. That test previously
created 128 durable invitation files. Its fixture now seeds occupied database
slots in one transaction from one genuine invitation, then calls the production
API for the 128th success and the 129th refusal. No production limit or durability
behavior changes. The focused regression and lint pass locally; final-head CI
rechecks the complete suite. This is fewer setup operations, not a promised CI
duration across different runners.

Windows also rejected a new fixture's source import because its directory creator
still held DELETE access, conflicting with the read-only import directory's
rename guard. Both new source fixtures now release that creator after sealing,
before importing; production permissions and sharing protections are unchanged.
The focused catalog regression and all-feature lint pass locally. Windows CI is
the native-platform confirmation; no failed head is eligible for merge.
