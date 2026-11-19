# Data-protocol TCP_NODELAY

The explicit GLY241 socket requirement was absent: both authenticated data
endpoints retained false when the caller disabled TCP_NODELAY. A controlled
before-fix regression fails0.10s, raw SHA256
9dedd4db974e00ab4990d2741e6ebf2ed6caf02bf1db502926352bad4ff5f5f1.
The same test first verifies Control sockets preserve both caller-selected states.

Three production lines at Wire::handshake_protocols enable the native option only
after the negotiated data ALPN and existing TLS/deadline checks pass. This shared
path covers connecting/accepting authority-backed peers and delegated stage data
peers. Errors propagate through existing redacted transport handling; no fallback,
new API, configuration, dependency, timer or alternate wire path is added.
Control, registry, probe and bootstrap socket policy is unchanged.

The regression reads the actual OS option at both TLS endpoints for each caller
state and transfers records successfully. It passes0.10s after the fix, raw SHA256
a356a57d6232d4235044f91c1e42dbc4f4a622d2a9cb2379618fd3e342b74240.
The full local security suite passes110 tests/one existing ignored capacity case
in17.77s, rawcfb5d7bb9adbf4e2165dfbc815568bbf0a3ed503fac3cb64ee24108519db847b.
Both workspace/all-target warning-denied lint modes, formatting and diff checks
pass. Cross-platform readback requires the final-source CI; it is not inferred
from a Mac run.

After signed integration of main `aca7d99`, the combined local security suite
passes112 tests/one existing ignored case in18.16s; raw SHA256
a097d0fac163a03b58ea466f63639b23ef195d1bf219f8529a7758705a011a33.
Both workspace lint modes, formatting and diff checks pass. The task PR records
the final source and cross-platform CI; these local results do not qualify it.

Correctness/security review traces the common handshake and preserves ALPN,
identity, trust, admission, deadlines and error handling. Separate complexity
review uses TcpStream::set_nodelay directly and one existing-fixture regression.
No custom packet logic, tuning flag, job or timing assertion.

This proves a socket property, not speed, power, physical-link or pipeline
qualification. Historical GLY267 registry failures remain unproved. Persistent
token scheduling, keepalive/power comparison and approved hardware profiles remain
open; GLY241 stays In Progress. Independent milestone review remains deferred,
not approved. No native/model/GPU/Ubuntu/SSH/download/spend/quarantine reset or
publication.
