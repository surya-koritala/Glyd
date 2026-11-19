# Native CPU cancellation smoke, 12 October 2026

One separately approved bounded run passed. The full run, including artifact hash
verification and runtime startup, took 6.987 seconds. Cancellation after unfinished
stream text propagated as `Cancelled`; parent exit was observed 11 ms after the
cancellation trigger. The watchdog observed the owned server and no remaining
member of its process group. External infrastructure spend was $0.

The harness is PR #21 at `9b1dede4906e2f340c9b539de8e5be92b7a53b7f`, which passed
[all five CI checks](https://github.com/Sigbound/glyd-compute/actions/runs/38286584818).
The previous [API smoke](../2026-10-12-m1max-cpu-api/README.md) contains the unchanged
M1 Max/32 GiB inventory, pinned runtime/model identities, build flags and artifact
inventory. This run verified the retained executable/model hashes before launch;
no download or native runtime rebuild occurred. Only the Rust harness changed.

## Run and reproduction

Use the existing `service_check CONFIG DIGEST --cancel-after-text` example. This
run used two CPU threads, one slot, context 512, explicit BF16 weights/F32 KV,
fixed sampling and at most eight output tokens. GPU/offloading, prompt caching,
network model fetching, UI, agents and runtime subprocess services were disabled.
The first content or reasoning event with text and a null finish reason triggered
cancellation. Role-only or already-finished events cannot satisfy the check.
No generated text or credential was logged. The error log was empty.

The adjacent JSON records preserve the exact result, original configuration
identity and harness build identity. The configuration replaces the local root
with `<NATIVE_ROOT>`; its recorded digest is for the original unredacted bytes.
The source `watchdog.py` is the actual external watchdog used. For another
separately approved run, place it in a prepared native root containing the model
and `build/bin/llama-server`, next to the checked-out `glyd-compute` directory with
its compiled example. It checks the pinned hashes, creates a one-shot marker,
reserves cleanup time within a five-minute total cap, and verifies owned process
group cleanup. Do not remove the marker to silently retry a failed native run.

## Limits

This proves cancellation at an unfinished API stream and process cleanup for
this CPU configuration. It does not prove which native computation was in flight
when the event arrived; backend buffering can precede adapter observation. It is
not a GPU drain proof, client-disconnect propagation test, memory reservation
profile, qualified catalog manifest or distributed generation result.
The sampled server RSS peak was 1,626,048 KiB at approximately 100 ms intervals;
this is a diagnostic, not a peak-allocation bound. Device-drain and memory
qualification fields deliberately remain false. GLY-176/177/208 stay open for
those remaining gates. No Ubuntu or Raspberry Pi workload was run.
