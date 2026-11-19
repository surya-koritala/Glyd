# Explicit topology observation outside the control loop

[GLY-291](https://linear.app/getglyd/issue/GLY-291) isolates a demonstrated
dependency found while investigating GLY-267. On base main
`66885f1b999716769dafbd7c1f4b6b4438a6c107`, `Runtime::admin` dispatches topology
through the synchronous command handler. That path resolves route interface labels
and enumerates host interfaces on the same thread that dispatches coordinator
heartbeats and maintains client policy. GLY-269 removed these operations from
heartbeat completion; an explicit diagnostic still had this dependency.

A test-only thread-local guard forbids topology host queries on the caller and
invokes the actual admin dispatcher. With only the topology expression extracted
unchanged into a helper plus that guard, the existing unit case fails immediately
at `topology OS query on control thread`: one selected test, 0.00s body. The
before log SHA-256 is
`25e3ce57e817e78ae8a3307aab173fccb4c9ed2b69503063c0d53f284a497026`.
Moving observation to the existing background admin mechanism passes the same
guard in 0.01s (log SHA-256
`8f4ec1bfc1abcf7d8fe10ebeb8ad8e38dbbdf3304ff11b0e3c857f354baee9c2`).
This proves thread placement, not a reproduced slow OS call or attribution of any
archived Windows queue, registry, transport or initialization incident.

The invitation worker helper is reused for topology with separate one-job pending
slots. Only read-only listener/mDNS values and the registry fence are captured;
no authority, signing key or profile directory enters the topology worker.
Queued expiry still prevents late dispatch; reply stages, deadlines, protected
admin authentication and owned worker cleanup are unchanged. Topology retains
the existing route freshness calculation and `discovery_confers_trust=false`.
Read-only listener values reflect dispatch time; live fence values are observed
by the worker. A blocked OS query can still delay final process shutdown while
the owned worker is joined; this change keeps it off the running control loop.

The extended existing unit case requires a real topology reply from the actual
dispatcher with host queries forbidden on its caller. Paused invitation/topology
jobs preserve sibling status; a paused topology job also permits Stop to latch.
A second topology request through the real dispatcher returns `TOPOLOGY_BUSY`
and creates no worker. Shared busy refusal, successful completion and failure
after reply-receiver loss release the pending slots. No production fault hook,
sleep injection or extra CI case/job is introduced.

Validation commands (pinned toolchain, locked/offline, model-free):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test -p compute-cli --lib --bins --test cli --locked --offline
cargo test -p compute-cli --all-features --lib --bins --test cli --locked --offline
cargo test -p compute-cli --test daemon multiple_listeners_explicit_routes_and_topology_work_without_multicast --locked --offline -- --exact --nocapture
cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture
```

Final outcomes, exact source, PR and seven-check CI receipts are recorded in Linear.
Local formatting and both warning-denied workspace/all-target lint modes pass.
Both feature-mode CLI library, binary and command suites pass; the real
coordinator/worker topology flow passes in 6.24s and all 18 synthetic catalog
lifecycle cases pass in 59.11s. The captured validation log SHA-256 is
`e59c22517c6964a7085ee0b35dade4ff953a9f5ef1427407d6718e68bba2a9db`.
Native-only tests remain ignored. No native/model/GPU/Ubuntu/SSH workload, download,
spending, old-state recovery, quarantine reset, backend implementation, dependency,
workflow or deadline change is part of this correction. GLY-267's historical
initiating causes and M2 provider/hardware/benchmark acceptance remain open.

Correctness/security review traces admin admission and all command callers,
snapshot/fence ownership, separate pending budgets, unchanged freshness/trust,
timeout stages and shutdown joining. Separate complexity review reuses the existing
worker, standard-library thread/atomic ownership and reply finishing function;
there is no new scheduler, abstraction layer or dependency. Independent milestone
review remains deferred, not approved.
