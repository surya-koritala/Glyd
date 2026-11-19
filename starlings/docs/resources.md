# Host resource observations

`starlings probe CACHE_DIRECTORY` reads OS memory, applicable cgroup limits,
CPU parallelism, the cache filesystem's free space and available temperature
sensors. It changes no device settings, starts no backend, and uses no network.
The cross-platform sysinfo library supplies OS APIs; network-mounted Linux disks
are excluded by its default policy. Unreadable/unavailable values are explicit.

On Apple ARM64, `host` is one unified physical memory pool shared by CPU and Apple
GPU identities. It is never counted twice. Other hosts expose one system-memory
pool. Discrete GPU memory is not inferred from a device name or added to host RAM;
the explicit NVIDIA diagnostic below can supply separate observed pools. Pool reservations
subtract once by pool ID and fail for unknown pools or insufficient availability.

Total memory is OS-reported; available memory is an estimate of reclaimable
capacity, not a promise that an allocation will succeed. Linux cgroup limits cap
host totals and availability. CPU thread availability is not CPU utilization.
The maximum reported sensor temperature is not a CPU-specific throttle signal;
throttling remains unavailable until a supported device probe supplies it.
No serial number, process list, username, prompt or model content is collected.

Each locally retained snapshot has a five-second TTL. Monotonic and wall elapsed
time can shorten freshness; backward clocks or suspend expire it. Stale diagnostic
reports show zero available pool memory. Reservation queries refuse stale samples.
The full CLI probe stays outside the daemon's heartbeat/admin loop. Worker
heartbeats now use a memory-only collection path: RAM and cgroup APIs, without
disk enumeration, sensors, backend execution or device setting changes. The
coordinator/admin loop does not perform those worker-side reads. Client-role
nodes send no resource sample.

Each new worker sample has a boot-scoped monotonic sequence, fresh snapshot ID,
capture time and age. The authenticated registry validates its enrolled node and
cluster. `cluster resources NODE` on the coordinator reports a single current
sample with receiver-adjusted `age_ms`; status reads never refresh it. The view
disappears at five seconds, on drain, missing/invalid credentials, revocation,
offline status or epoch mismatch. It is a diagnostic view, not a new sample to
replay into a scheduler. Querying one node keeps cluster size from multiplying the
administrative response size.

Memory is still an estimate, and the announcement has state `unavailable`, empty
backends and explicit `unavailable` control/tensor endpoints. It cannot make a
node schedulable. Missing/expired local memory observations send no sample.
Installed backend/build/profile support is supplied only by exact approved service
qualification, never inferred by the diagnostic probe. General measured budgets and
thermal policy remain service-adapter integration. Raspberry Pi 5/64-bit Linux shares
this CPU host path; ARM64 CI is not physical Pi validation (GLY-210).

## Explicit NVIDIA diagnostics

`starlings probe CACHE_DIRECTORY --nvidia ABSOLUTE_NVML_LIBRARY SHA256` loads
one explicitly trusted installed driver library after canonical absolute-path and
content-digest checks. It never searches PATH/current-directory loader defaults or
fetches a driver; Windows UNC/device prefixes are refused. The approved library and
its OS-loaded dependencies are trusted native code. Same-user/root replacement
and library dependency compromise are outside this approval boundary.

The [safe NVML wrapper](https://github.com/rust-nvml/nvml-wrapper) loads installed
NVML at runtime; builds need no NVIDIA SDK. Only read queries are used: driver,
canonical GPU UUID/name, memory, bus/NUMA/MIG/virtualization and GPU temperature/
thermal clock reasons. No process list, reset, clock, power or device-setting
operation is used. More than fifteen devices, duplicate/invalid IDs or incomplete
identity inventory fail loudly rather than truncating the report. Missing optional
measurements/symbols remain unavailable.

Each known unshared Linux bare-metal PCIe device has one `nvidia:GPU-UUID` pool.
MIG current/pending mode, virtualization and GPU-NUMA/shared or unknown topology
cannot grant a separate reservable pool. MIG instances are not enumerated alongside
parent memory. The wrapper does not expose the virtualization query on Windows,
so Windows retains device diagnostics with unknown physical pool eligibility.
macOS refuses the NVIDIA path. These are explicit remaining qualification limits.

Memory availability is estimated, not an allocation promise. [NVIDIA's memory
contract](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
notes OS accounting and shared-memory cases; v2 includes reserved bytes in used.
Checked validation therefore tests used plus free against total without adding
reserved twice. Idle/power-cap reasons are not thermal throttle signals.

The entire report retains its original five-second timestamps, including all GPU
calls; slow collection produces a stale report with zero allocatable pools. NVML
calls have no independent driver-side deadline, so this diagnostic remains an
explicit CLI operation, never a heartbeat, admin handler or serving admission call.
Hardware/driver compatibility and native allocation qualification still need actual
approved device checks. This report advertises no inference backend or layer support.
