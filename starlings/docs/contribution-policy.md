# CPU contribution policy

`WorkerRoute.contribution` is optional strict local policy covered by the entire
approved route digest. When absent, old records and qualification digests remain
unchanged. With policy present, `Qualification.cpu_threads` must specify the
reviewed maximum threads of the exact approved backend configuration. The program
does not infer thread settings from arguments, apply OS affinity or claim CPU-time
quotas; configuration/build/profile evidence must establish that declaration.

| Policy field | Admission meaning |
| --- | --- |
| version | 1 |
| max_host_bytes | Positive decimal u64; checked loading peak plus additional request budget must fit |
| min_free_host_bytes | Decimal u64 free-memory floor beyond the phase's loading/request demand |
| max_cpu_threads | 1–4096; qualified backend thread count cannot exceed it or available affinity threads |
| max_temperature_c | Null means no temperature ceiling; 1–100 requires a fresh finite measured sensor value at or below it |

Host availability remains a conservative OS estimate, not an allocation promise
or OS RSS limit. Maximum sensor temperature is not a CPU-specific throttle signal.
Missing required temperature/CPU availability, low RAM, observed throttling,
stale samples or unsafe arithmetic refuse work. Physical Pi/general workload
memory and thermal qualification remain GLY-175/210; these fields cannot supply it.

The local sampler reads host/cgroup RAM and optionally OS sensors once a second.
It never enumerates disks/processes, calls NVML, opens model/cache/credential files
or runs on registry/admin handling. Samples retain original five-second monotonic
and wall timestamps. Initial acquisition waits at most that TTL on the service
thread, with no status lock held. Heartbeats and diagnostics read short cached
eligibility and cannot refresh it.

Counting, each accepted control phase and output polling recheck the gate.
Admission demands the additional request budget; monitoring and active output
check the free-memory floor without demanding an already owned allocation twice.
Withdrawal is permanent for that owner: a later sample cannot reopen it. Pressure
withdraws readiness and cancels only the owner's backend through existing stop/
cancellation, closes its current generation watch and retains in-flight allocations;
other owners have independent gates. Idle service maintenance also drains a stale
gate even if its sampler is blocked. Service/runtime pins and uncertain
reservations remain until the existing trusted cleanup receipt. No cluster drain,
automatic restart, profile swap, silent precision change or quarantine reset occurs.

The sampler has only a weak cancellation reference. If an OS sensor call hangs,
cached eligibility expires; stop/admin does not join a blocked read. Process exit
ends that read-only thread. The already reviewed backend cleanup contract remains
necessary; sensor-thread absence is never treated as device drain.

Development-only pressure fixtures require fixture=true and an approved synthetic
service environment marker. Native/default routes do not use this input. Fixture
checks prove policy/lifecycle behavior, not a physical temperature or native budget.
