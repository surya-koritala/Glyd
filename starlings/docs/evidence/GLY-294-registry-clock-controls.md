# GLY-294 — Registry clock discontinuity control

Investigation: [GLY-267](https://linear.app/getglyd/issue/GLY-267), with
[GLY-263](https://linear.app/getglyd/issue/GLY-263) initialization counterchecks.
Base: `1607c129858d543148f9f45d35ea61e557c3c43f`.

## Question and result

Can wall-clock discontinuity alone produce an offline resource view despite
fresh monotonic age, followed by online membership after another heartbeat?
Yes, under the existing fail-closed clock policy. The new registry unit test
changes only retained observation timestamps, separately modeling wall time
moving forward beyond the registry TTL and wall time moving backward. It does
not change the machine clock, wait for expiry, or start a backend.

Both cases assert the public membership and resource APIs: initial capacity is
available; the discontinuity makes membership offline and resources unavailable;
a newer heartbeat restores online membership but cannot revive its replayed
resource batch; a new sample sequence and immutable snapshot restore capacity
without advancing the authority epoch.

The counterfactual implementation temporarily ignored wall time in
`Observed::elapsed`. The focused test failed with Online versus expected Offline
(0.11 seconds). Restoring the unchanged production implementation passed both
cases (0.19 seconds). This mutation validates the clock-policy check; it is not
a before/after production bug fix or a speed comparison. No mutation is committed.

## Runnable checks

```sh
rustup run 1.97.1 cargo test -p compute-security --lib wall_clock_discontinuity_withdraws_fresh_resources_until_a_new_sample --locked --offline -- --nocapture
rustup run 1.97.1 cargo test -p compute-cli --test daemon catalog_service_initialization_ --locked --offline -- --nocapture
rustup run 1.97.1 cargo test -p compute-cli --all-features --test daemon catalog_service_initialization_ --locked --offline -- --nocapture
```

The existing two real-process initialization controls pass in each feature mode:
default 0.85 seconds, development backend 1.99 seconds. They verify protected
lock/cache/database/authority collision refusals, retained bytes and ownership,
explicit successful cache-refusal recovery, and untraced initialization silence.
These checks do not reproduce the archived Windows empty-output or exit stall.

Pinned Rust/Clippy 1.97.1 formatting and both workspace/all-target warning-denied
lint modes pass. The complete security library suite passes: 79 passed, one existing
ignored capacity acceptance, 11.70 seconds. Loopback tests require host socket
permission; the sandbox-only full-suite attempt was not a qualification pass.
An initial Homebrew toolchain-shadowed run is likewise excluded. Final commands
put the rustup shims first on PATH, as documented in the development guide.

## Limits and review

This is a test-only addition using the existing registry fixture and private
clock representation. Production code, deadlines, authorization, resource
freshness, replay rules, workflows, test selections and dependencies are unchanged.
The new case runs in the existing security unit suite, before CLI lifecycle tests.
It adds no sleeps, daemon processes or CI jobs.

The archived Windows jobs do not contain clock-discontinuity evidence. This
experiment establishes one possible mechanism, not historical attribution;
GLY-263 and GLY-267 remain open. GLY-238 and real provider/hardware/native
qualification remain separate M2 gates. No native workload or spending occurred.
Independent milestone review is deferred under the authorized green-merge policy,
not recorded as independent approval. Exact source/merge CI receipts are tracked
in GLY-294; successful runs do not reclassify archived failed runs.
