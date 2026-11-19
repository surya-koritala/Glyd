# CPU contribution policy integration

Tracks GLY-173/209, based on merged PR #41. Uses existing route approval, host
observations, service stop/cancellation, exact count/control and cleanup ownership.
No dependency, allocator estimate substituted for measurement, new backend or
layer capability is added. Old absent optional fields are omitted from serialization.

Policy checks cover missing/oversized qualified thread declarations, checked memory
budget/floor overflow, insufficient free RAM/affinity, stale clocks and missing/hot/
non-finite required temperature. Two independent gates verify permanent withdrawal
does not withdraw another owner's eligibility, and a real five-second-old snapshot
expires without reads refreshing it.

The actual daemon fixture starts one qualified synthetic service, receives partial
SSE, then changes its approved development-only pressure marker. The sampler closes
admission, cancels the owned backend, closes without DONE and retains unqualified
pin/reservations; admin remains operational and no automatic restart occurs. The
same helper retains normal/crash/busy/revocation/recovery assertions. Windows includes
pressure in its native serial matrix, while POSIX keeps the distinct scenario.

Local unit/pressure checks and all seven daemon scenarios pass; exact full checks,
commit, PR and five-platform/dependency CI are recorded in Linear. Full diff has
correctness/security and separate complexity review before merge. No lock is held
across sensor reads/first-sample waiting; the sampler retains only weak backend
cancellation ownership. Original cleanup qualification/quarantines stay unchanged.

These are software checks, not hardware/native inference. General profiles,
same-cluster multi-worker production routing/benchmarks and physical Pi remain open.
The software test of independent gates does not substitute for those acceptance
checks. Independent review deferred under current authorization, not approved.
No new model/GPU/Ubuntu workload, download, paid infrastructure or native approval.
See [the exact contract](../contribution-policy.md).
