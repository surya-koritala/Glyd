# Interface labels outside heartbeat completion

[GLY-269](https://linear.app/getglyd/issue/GLY-269) removes an avoidable OS query
from the registry heartbeat. Main `760a850ebc12e118c9a14ccff22beb769cd13c50`
calls `selected_interface` after each exchange; it enumerates OS adapters solely
to label the source socket. No authority or routing decision needs that label.

The existing real-TLS route fallback test now prohibits interface enumeration
on its client thread during heartbeat completion. The guard is a test-only
thread-local boolean, restored after a caught unwind before joining the server.
Against the original ordering the test fails in 0.23 seconds; raw SHA-256
`6e7342966b152de700f7cc61171a2b6bb63558b123bca10b88c7ac95ba0da632`.
It performs no extra timeout or real device workload.

Heartbeat completion now records the actual local socket and leaves its interface
label unresolved. Explicit topology observation resolves labels after copying
the bounded route records and dropping the live fence lock. It computes age and
authentication freshness after every lookup has finished. The existing test
still requires a real interface label, checks the mutex is available inside
lookup, and verifies a controlled six-second lookup delay returns an expired
route. This preserves source binding, fallback, coordinator trust and expiry.

The full local security suite passes: 77 unit tests, one existing ignored test,
11 catalog integration tests and no doc tests. Raw SHA-256
`2d10fe8616e5795eb27c9c7c3089d8f8f63eb75e1e27807b07b23feafa2bd28a`.
The full 18-case catalog lifecycle passes in 56.75 seconds; raw SHA-256
`0a3a89040ce3b4c60d66cbfaaf31a2e2c4e18de77a551900d6bd01c3fed72477`.
The default real-process multi-listener/source-binding/topology flow passes in
5.97 seconds. Formatting and both workspace/all-target Clippy modes pass with
warnings denied; locked/offline commands use the pinned toolchain.
The guard exists only under cfg(test); no new feature, environment flag,
dependency, wire field, policy, timeout, retry or cleanup exception is added.
Other probe uses of selected_interface remain unchanged. Correctness/security
and separate complexity review trace all callers and keep diagnostic lookup
outside the live fence. Exact-source CI/merge records are attached in Linear.

This control proves a source dependency, not the initiating cause of the retained
Windows failure in [CI38665249219](https://github.com/Sigbound/glyd-compute/actions/runs/38665249219).
That merged-main run failed Windows backend job116195606831 in production mode
disconnect before a ready capability: service REGISTRY_LOST/offline, startup
complete, reservation zero and quarantine retained, with later online status and
no heartbeat Error logged. Raw SHA-256
`09b533a950b9f6a194a85cea048b612395f643c67bb7e4cf07e36ac4420d5e28`.
Five other checks passed; backend and aggregate failed. GLY-268's paused-memory
regression passed and its proved fix is complete. No failed source is rerun or
reclassified as passed; GLY-251/253/263/267 attribution stays open.

The provider contract is still being prepared. Provider-dependent M2 acceptance
remains open; independent work may proceed under the updated project direction.
No native/model/GPU/PC/Ubuntu workload, download, spend or old quarantine recovery
is performed. Independent milestone review remains deferred, not approved.
