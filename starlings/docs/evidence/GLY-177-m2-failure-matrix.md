# GLY-177: M2 software failure acceptance

Base: `b73f9c2ce51006e08bd2b3946fe5ae6e3b4d79b8`. This audit maps existing
checks and adds the missing active client-disconnect case in the production
daemon path. These are model-free software checks, not native qualification.

## Production disconnect check

The existing partial-output fixture emits an SSE event and stalls. The new
`disconnect` scenario uses the actual daemon HTTP admission, TaskRouter,
coordinator Dispatch, enrolled worker and supervised Service. After receiving
visible partial content and a session header, the client closes both directions.

The worker must report failure with retained runtime/reservations within the
existing eight-second observation bound. For this new scenario only, the fixture
request timeout is thirty seconds, matching its stall; neither can satisfy the
earlier cancellation bound. All pre-existing scenario/request/admin/readiness
deadlines are unchanged.

Both daemons must remain available through local admin. The coordinator reports
one retained active assignment, one retained request, nonzero body bytes and no
queued request. There is exactly one backend invocation. Worker shutdown explicitly
reports incomplete drain, and the service launch quarantine survives. No cleanup
qualification or restart is inferred from a socket closing.

Unix runs the new scenario as a separate test alongside the existing parallel
scenarios. Windows includes it in the existing serialized production matrix.
The existing CI filter selects it without a new job or workflow.

## Failure matrix

| Requirement | Runnable check and scope | Remaining evidence |
| --- | --- | --- |
| JSON/SSE, exact usage, content/reasoning and repeated slot reuse | `tests/daemon.rs::catalog_service_production_route_counts_relays_reuses_slot_and_revokes_client_tokens`; relay `enrolled_relay_preserves_json_sse_and_fragmented_output` | Current-route native direct/proxy qualification |
| Exact count/context and refusal before generation | Same production case; relay `counting_failure_returns_http_error_before_any_generation_or_sse_headers` and `worker_rejects_changed_prompt_count_or_context_before_generation` | Installed provider tokenizer/template/count contract |
| Model, credentials, unsupported fields and explicit sampling | Same production case; relay `admission_rejects_model_mismatch_unresolved_sampling_and_unsupported_tools` | Installed provider feature qualification |
| Crash: terminal failure, one invocation and retained ownership | `catalog_service_crashed_backend_retains_ownership_without_restart`; relay `backend_failure_emits_one_redacted_terminal_error_without_retry` | Native backend crash/device drain |
| Active partial-stream disconnect: cancellation, no replay and retained ownership | New `catalog_service_disconnected_stream_cancels_without_replay_or_releasing_ownership`; Windows production matrix | Native daemon disconnect and device completion |
| Revocation and registry drain withdraw output | `catalog_service_revoked_stream_retains_unqualified_ownership`; relay `client_disconnect_revocation_and_registry_drain_cancel_before_first_output` | Current-route native failure qualification |
| Busy slot cannot publish successful drain | `catalog_service_busy_slot_cannot_claim_request_drain` | Provider CPU/GPU completion contract and memory evidence |
| Bounded output/deadlines/backpressure | `catalog_service_slow_reader_cancels_without_replay_or_releasing_ownership` (Windows production matrix); security `streaming::tests::invalid_events_capacity_and_deadlines_close_output_permanently`, `client::tests::would_block_retries_partial_writes_and_stops_at_the_absolute_deadline`, generation `generation_deadlines_and_byte_budget_cannot_be_extended_by_a_new_record` | Integrated synthetic slow-reader check is described in [backpressure evidence](GLY-177-backpressure.md); native/provider qualification remains open |
| Queue overload/refusal, queued disconnect, concurrent selection and trusted slot release | `catalog_service_two_workers_run_concurrently_through_the_coordinator` fills eight queued requests with two active slots, rejects the next request without retaining it, and discards disconnected queued bodies; routing `queues_enforce_bytes_fairness_abandonment_expiry_and_retry_exclusions` | Two same-type physical workers and matched performance/memory records; integrated details in [overload evidence](GLY-178-overload.md) |
| Unknown cleanup remains quarantined; approved recovery fences old work | `catalog_service_survives_failure_and_retains_pins_on_drain_and_restart`, `catalog_service_qualified_recovery_fences_old_work_and_allows_fresh_restart` | Actual provider cleanup evidence; preserve old quarantines |
| Readiness metadata and shared I/O deadline fail closed | adapter `installed_properties_are_approved_bounded_and_cannot_grant_readiness_on_mismatch`, `readiness_attempt_cannot_multiply_its_io_deadline` | GLY-238 installed Glyd identity/precision/capabilities |

Check identifiers above are in `crates/compute-cli` unless qualified as security
(`compute-security`) or adapter (`compute-adapters`). Windows folds the crash,
revocation, busy-slot, disconnect and two-worker scenarios into its production
matrix; it does not omit those cases.

## Review and verification

Correctness/security review traced admission, live client watch, dispatch,
worker cancellation, TaskRouter failure retention and durable service ownership.
The added scenario changes tests only and cannot grant production qualification.
Ponytail review reuses the existing fixture/setup and platform scheduling; no new
dependency, runtime abstraction, backend implementation or CI job is required.

Pinned Rust 1.97.1, locked offline dependencies and warnings denied:

- Formatting and all-feature workspace/all-target Clippy passed.
- The final model-free daemon lifecycle selection passed all nine tests in 34.98 s:
  `cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline`.
- Exact-head seven-check CI remains required before merge; attach final commit and
  CI evidence in Linear. Independent review remains deferred under the existing
  authorization, not claimed approved.

GLY-177 and M2 remain In Progress. GLY-238 delivery, approved native experiments,
hardware/profile and two-real-worker acceptance remain separate open gates.
GLY-244/246's verified work is complete; historical attribution is separately
tracked in GLY-248/249 and does not block M2. M3 has not started.

