# GLY-179: received Glyd cost data

Issue: [GLY-179](https://linear.app/getglyd/issue/GLY-179), with calibration and
transaction acceptance retained in [GLY-181](https://linear.app/getglyd/issue/GLY-181).

The provider contract is `glyd-costs/1`, reviewed from backend PR171 source
`fb56c3d0be10f9300736bffb133cd1aff1f3bdc2`, merged as
`319a1e2782d764a0b93410067216fb8c54f88cd7`. Only API/evidence documentation was
read; no private implementation, raw measured record or executable is copied.
The delivered measurement command is `glyd-engine rates`; this change runs none.

## Software increment

`plan inspect-glyd-costs PRIVATE_PROFILE_JSON` uses the existing bounded file
reader, duplicate-aware JSON parser, SHA-256 helper and report writer. It emits
selected typed provider data and explicit missing Compute fields, with admission
always unsupported. It creates no `ModelCostProfile`, approval, inventory or
placement. Unprojected provider fields confer no support and are not validated.

The pure control preserves exact input digest, estimated/file distinctions,
nullable allocations, zero timing samples, combined state and the provider's
chunk maximum even when larger than the prompt. It refuses malformed schemas,
noncontiguous/unknown layers, invalid byte types, inconsistent workload/context,
unsupported concurrency, unavailable measured working memory, invalid timing,
duplicate keys and over-budget input. Each projected nullable field must be
present; the control separately retains explicit null and rejects omission.
Data is synthetic and labels no real model.

Validation uses pinned Rust 1.97.1, offline cache and at most two Cargo jobs:

```
cargo test -p compute-cli --lib glyd_costs::tests::received_costs_preserve_unknowns_and_never_become_admission -- --exact
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
git diff --check
```

Results and exact signed commit/source/main CI are recorded in the issue/PR.
No dependency, CI job, test timeout or workflow is added.

## Remaining acceptance

Installed artifact identity and approved runtime/device/pool mapping remain
required. The provider's aggregate working allocation cannot establish separate
per-phase scratch, staging, overhead and reserves, and a combined state number
does not establish its recurrent/convolution split. File-derived weights and
predictions do not qualify native allocations or timing. Calibrated measurements,
freshness/confidence/raw-evidence review, physical acceptance and independent
milestone review remain open; no real workload or source of approval is added.
