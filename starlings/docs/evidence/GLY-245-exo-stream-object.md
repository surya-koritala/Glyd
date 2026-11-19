# GLY-245 pinned Exo stream object compatibility

Issue: [GLY-245](https://linear.app/getglyd/issue/GLY-245).
Base before PR: main `3537c52c5e46697039249bac899cc8b0db9d3255` (PR #96).
The initial reproduction ran against `a0145b7` before that integration.

## Cause and correction

The pinned Exo response type declares `chat.completion`; its stream adapter uses
that same response object for delta chunks and final usage. Compute's collector
required `chat.completion.chunk` for every stream and therefore rejected this
audited candidate before any measurement could complete. Official upstream main
was re-read on16October2026 and still pointed to the same candidate. Source links
and exact file hashes are in [the upstream pin record](../../benchmarks/upstream/exo-bc412e2.json).
No upstream implementation was copied or executed.

The collector now derives the required stream object from the existing matrix
system and exact source revision. Only Exo at this audited revision uses
`chat.completion`; Glyd and every other Exo revision retain the original chunk
spelling. JSON behavior is unchanged. There is no dialect auto-detection or flag,
and no weakening of model/ID, choices, finish, usage, bounds or failure-stop checks.
Collector/program hashes continue to bind this behavior in every raw trial.

## Controlled validation and review

An in-process synthetic event control initially fails with the actual production
parser's `INVALID_RESPONSE`, without opening a socket or starting a service. The
same check passes after the correction. It also checks both spellings and an
unknown spelling under the audited Exo revision, another Exo revision and Glyd;
wrong model/ID, missing choices, incoherent usage and unsupported tool output
remain refused. Finish without usage cannot manufacture token counts. Serialized
request timing/count records retain neither response text nor response ID.

Only basic Mac checks are permitted: at most two build jobs, cached offline
dependencies and this exact socket-free test in both feature modes. Broader
collector/network/lifecycle coverage remains in the unchanged bounded CI jobs.
Before integration, both exact feature-mode controls pass (each0.00s) and both
warning-denied workspace/all-target lints pass (1.90s/1.45s). After merging current
main, both controls pass again (0.00s/0.01s), both warning-denied lints pass
(5.82s/3.00s), and formatting/diff checks pass. Initial SDK auto-discovery
printed local Xcode cache-path warnings; selecting the installed Command Line
Tools SDK explicitly through SDKROOT corrected discovery, without disabling any
warning or security control. Current-main review and all seven exact-source CI
checks remain required before merge. Independent milestone review remains
deferred, not approved.

Correctness/security review traces the selected matrix program through the
single shared Attempt and response parser; endpoint input cannot choose the
response spelling independently. Separate complexity review retains one small
pure policy function and the existing records/parser, with no dependency, wire
schema, framework, additional job or native API.

GLY-245 remains open: source inspection and synthetic compatibility do not qualify
an Exo executable, real2026 model, actual device, native metric source, matched
baseline or benchmark win. GLY-238/235/236 delivery and hardware/review gates remain.

## CI artifact reuse correction

Initial source `65a7efdd9853feae4569d76f1f840caff54079d4`,
[CI38908234740](https://github.com/Sigbound/glyd-compute/actions/runs/38908234740),
is not qualified: five checks passed, Windows backend was cancelled at its10m
boundary during the final ownership/restart case, aggregate failed. The collector
control, all preceding groups and full production matrix passed; the final case
did not finish, so no complete-suite pass is claimed. Raw job116997172817 SHA256
`4c5df982db5f47eef6603788c34f487df7c62b1492b543b181e3e5b222e7f43d`.

The no-run graph built compute_cli-d9ba709/daemon-bc56cf6, but subsequent
CLI-only calls rebuilt compute_cli-72fdd6a/daemon-4233117 (7.90s/25.16s).
The workspace adapter dependency disables defaults. Selecting the adapter itself
with all-features enabled its unused empty default marker, so the supposedly
prepared graph and later invocations selected different dependency features.
This is a proved source of redundant compilation; runtime/host variance remains
separate from its contribution to the overall job cap.

Controlled Mac compilation reproduces adapter features default+test-backend and
different CLI executable identities versus test-backend alone. Deleting the empty
default feature removes the mismatch; no code references that marker and it has
no dependency edges. Combined preparation and later CLI-only library/daemon
compilations now report the same executable/features and fresh artifacts; their
post-preparation compile checks take0.11s each. No test/service executes during
these controls. Existing package selection, execution filters, feature-backed
fixture code, jobs, time caps, warning policy and assertions are unchanged.

After the deletion, both exact collector controls pass0.00s, both warning-denied
workspace lints pass3.15s/3.10s, formatting/diff pass and Cargo.lock is unchanged.
Only bounded offline Mac compilation/basic controls ran, max2jobs. The corrected
source still needs all seven actual platform checks; no unchanged-source rerun,
guaranteed total-CI speedup or historical Windows initiation claim is made.
Separate complexity review prefers deleting the empty marker over introducing
a custom executable runner, feature-normalization flags or another job.
