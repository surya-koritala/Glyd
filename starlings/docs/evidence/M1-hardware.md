# M1 physical control-service and capacity evidence

Date: 2026-10-12. Physical source commit:
`fea2fe4b3c013f1daa34f5151a7125c10e1c5daa` (PR #9).
Rust 1.97.1; warnings denied. The physical test used isolated temporary profiles
and explicit LAN addresses with fingerprint approval and private SSH invitation
transfer. No model, GPU workload, router change or paid infrastructure was used.

| Device | Role | Platform |
| --- | --- | --- |
| Apple M1 Max laptop | Worker | macOS 27.0, build 26A428, ARM64 |
| Ubuntu PC | Coordinator | Ubuntu 26.04.1 LTS, Linux x86_64 |

The Ubuntu PC contains an RTX 4080 SUPER; this test did not access or qualify it.
LAN addresses, interface inventory, credentials and node IDs are omitted.

| Check | Observed outcome |
| --- | --- |
| Explicit enrollment and authenticated route | Passed |
| Worker becomes online | 2.82 s |
| Coordinator loss expires authority | 5.04 s |
| Coordinator restart keeps bound ephemeral port | Passed; epoch advanced to 2 |
| Worker reconciles after coordinator restart | 3.33 s |
| Worker drain reaches coordinator | 3.07 s |
| Orderly stop and worker restart | Passed; online after 2.26 s, epoch 3 |
| Revocation reaches worker | 1.75 s |
| Leave disables restart | Passed |
| Reported inference availability | Unavailable; worker not schedulable |

Times include status polling and process scheduling; they are observations from
one run, not latency guarantees or an inference benchmark. Hardware acceptance
covers this source commit. Later review fixes change replay bookkeeping,
certificate-refresh recovery, database contention and listener handling as well
as regression coverage and Windows initialization. Those later changes have
automated CI evidence; this first physical run does not qualify them. A later
physical rerun is recorded below.

## 63-peer capacity test

Run separately from routine CI on the Mac, using actual TLS, SQLite and isolated
identities. All 63 persistent registry peers connected, then idle IPv4/IPv6
connections occupied all eight handshake slots. Twenty local admin round trips
ran alongside heartbeats. Result: 1,796 heartbeat samples, heartbeat p95 11,549 µs,
admin p95 29,753 µs, zero lost heartbeat authority; test wall time 25.38 s.
These are local synthetic load results, not 63 physical computers or an RSS bound.

Reproduce with the pinned toolchain:

```sh
cargo test -p compute-security sixty_three_persistent --locked -- --ignored --nocapture
```

The ignored marker keeps this timed acceptance out of each routine CI pass; it
was explicitly executed successfully. Physical Pi, Windows elevated-owner and
console events, multi-NIC/VPN/IPv6 LAN and inference remain unqualified.

A follow-up capacity run at `300666909666532812a3e93552369840b594c99b` after
the independent review passed: 1,612 heartbeat samples, heartbeat p95 10,623 µs,
admin p95 29,624 µs, zero lost authority, 22.91 s wall time. Same Mac, actual TLS
and SQLite, all 63 established peers and eight occupied handshake slots.


## Reconnect admission follow-up

Source: `31408630068b219619ae6741c54ed663a99408b7`, 2026-10-12.
The separate 63-peer test uses the daemon's 20 ms polling cadence and batches
at most eight ready admissions per listener. After all ordinary handshake slots
were occupied, all 63 clients discarded their existing streams and reconnected.
Observed: 957 heartbeat samples, heartbeat p95 44,724 µs, admin p95 29,154 µs,
zero lost heartbeat authority, 63 successful reconnects; 13.08 s test time.
The same Mac, real TLS/SQLite and synthetic clients were used. These are not
physical Pi, 63-machine or model-inference results.

Routine tests additionally cover a full idle probe queue, four repeated fresh
TLS reconnects with all eight ordinary handshake slots occupied, cleanup,
invalid/replayed/context-switched proofs and TLS identity/ALPN mismatch. The
real-process lifecycle test maintains IPv4/IPv6 handshake pressure through
restart, status, drain, revoke and stop. HTTP WouldBlock is injected through the
production authorized output loop rather than inferred from socket buffer size.

Windows CI includes private-console CTRL_C/CTRL_BREAK termination and
an explicitly Administrators-owned named-pipe fixture. The helper uses the native
[GenerateConsoleCtrlEvent](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent)
and [SetConsoleCtrlHandler](https://learn.microsoft.com/en-us/windows/console/setconsolectrlhandler)
APIs, attaches only to the test daemon's console, and checks explicit user-owned
protected state/capability ACLs. The group-owned pipe fixture accepts a valid
capability and refuses a false one, independent of the runner token default owner.
Exact final-head CI outcomes are recorded in Linear.

Pi-class storage remains unmeasured. To run the existing timed harness on a real
64-bit Pi, set `TMPDIR` to a private directory on the target SD/SSD filesystem
(not a memory filesystem), then run the ignored capacity test above. Record the
commit, Pi/RAM/OS, filesystem/storage, output and wall time. Linux ARM64 CI alone
does not satisfy this physical storage gate.


## Physical rerun with reconnect admission

Source: `ae90339dafb5e8c7451da0b291dab2315ba988cf`, 2026-10-12.
The same M1 Max Mac worker and Ubuntu coordinator passed explicit enrollment,
authenticated LAN route, loss/expiry, coordinator restart on the saved port,
drain, worker restart, revocation and leave. One-run observations: online 2.82 s;
loss expiry 5.04 s; coordinator restart reconciliation 3.41 s; drain 2.17 s;
worker restart reconciliation 2.80 s; revocation 2.26 s. These include status
polling and process scheduling and are not latency guarantees. Both daemons
stopped, and their isolated temporary build/profile directories were removed.
No GPU access, model, paid infrastructure or repository setting change occurred.
