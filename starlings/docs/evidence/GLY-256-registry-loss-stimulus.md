# GLY-256: abrupt registry loss fixture

PR59 measurement source `c0999caf872e01250b3a292c6dfb7a408cfd1bae`,
run38506079417/Mac job115663782683 fails with REGISTRY_LOST and
failed_pins_retained, reserved0 and registry_failure=draining instead of offline.
All other thirteen lifecycle cases pass. Completed job raw log SHA-256
`23eed6813117201965e53e3895981c412e2c436f3e14a7c0166cbdd3fa42a2b5`.

The registry-loss fixture called coordinator cluster stop. That command invokes
Authority.drain and can deliver an authenticated Draining heartbeat before
shutdown. Service correctly records the first withdrawal reason; later lease
expiry does not relabel that failure. Orderly shutdown is the wrong stimulus for
the intended authority-loss/Offline check.

Kill and reap only the freshly owned model-free fixture coordinator, then observe
real worker lease expiry. Keep exact Offline classification, no revival after
coordinator reconnect, fresh unavailable capability, zero reservations, retained
ownership/assets and GC refusal. All original deadlines remain. Graceful drain,
revocation and CLI shutdown keep their separate existing coverage. No product
behavior or accept-either assertion changes.

Controlled corrected case passes in21.54s:
`cargo test -p compute-cli --all-features --test daemon catalog_service_registry_loss_survives_cleanup_and_reconnect_without_restart --locked --offline -- --exact --nocapture`.
Raw `work/m2-gly256-controlled-crash.log`, SHA-256
`cd6db6252ffa6dff80cfa7f1ca1d94751d7dbb9fffc4533ba5532c4fb61a30a4`.
Full final-source platform gates and merge identities belong in Linear. This
isolates this fixture cause, not the historical GLY-251/253 Windows OS trigger.

Correctness/security review traces Stop/Authority.drain, the first service
withdrawal, lease expiry, owned Child cleanup and coordinator restart. Ponytail
review replaces two incorrect fixture calls with the standard Child kill/wait;
no process controller, injected clock or dependency. Independent review deferred,
not approved. No real model/GPU/PC workload, spending or quarantine reset.
