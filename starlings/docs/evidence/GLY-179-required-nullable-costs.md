# GLY-179 required nullable profile fields

Preview, mode and calibration file readers deserialize embedded cost profiles
before the planner revalidates their serialized protocol records. Serde's default
for an absent `Option` field erased the difference between a missing required
field and explicit null. Reserialization then supplied null to schema validation.

Shared cost types now require presence for measurement `relative_error_bps`,
layer/component `timing` and `per_stage_timing` during deserialization. Existing
schema, serialized output and legitimately optional fields are unchanged.
Explicit null remains unknown and cannot create a measured value or admission.

The pure file control
`planning::tests::cost_profile_nullable_fields_must_be_present_in_all_file_consumers`
removes each field from the supplied synthetic profile and checks all three
actual Rust file APIs reject it before planning. The corresponding explicit-null
inputs remain valid; missing timings may leave no calibration candidate.
No process, backend, socket, model or hardware is started by this control.

Removing only the four field annotations reproduced acceptance of a missing
`relative_error_bps` and failed the control. With the fix restored, the exact
control passed in 0.34 s using pinned Rust 1.97.1. Both workspace/all-target
Clippy modes passed with warnings denied (10.70 s default, 4.37 s all features);
formatting and diff checks passed. Cached offline compilation used at most two
Cargo jobs. Commands:

```
cargo test -p compute-cli --lib planning::tests::cost_profile_nullable_fields_must_be_present_in_all_file_consumers --locked --offline -- --exact
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
```

This repairs strict input handling. Native profiles, complete physical-pool
accounting, approved measurements and calibration remain open in GLY-179/181/239.
