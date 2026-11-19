# Publish client credentials before activation

PR88/f3950d7's
[CI38879830879](https://github.com/Sigbound/glyd-compute/actions/runs/38879830879)
fails Windows backend/aggregate; the other five checks pass, including the new
test-only transport harness smoke in Windows default. Backend shortcatalog8PASS
45.05s, then production lifecycle fails98.85s at daemon.rs602: duplicate token-file
publication is refused, yet tokens count2 instead of1. Raw job116903615342 SHA256
2f186778ca168870e249f06b6941b2a1f4524373b670386601ff26f0ee5c0352.
The control command is still executing at teardown; the log cannot distinguish
unfinished compensation from a failed compensation.

The production order was issue/commit, save private file, and ignore the result
of compensating revoke after publication failure. A temporary shared helper
preserving that order reproduces the unsafe invariant with actual protected
output collision and SQLite BEGIN IMMEDIATE writer contention. The candidate is
already authorized during publication, and the failed revoke leaves its committed
row. The control fails0.08s with (active=true,rows=1) instead of(false,0), raw SHA256
2b06fa1b25fe241b5400c5c8313f187fe72805c21934955e1a4f141e768ef4fe.
This demonstrates a real publication/compensation defect independently of the
unidentified initiating Windows delay.

The saved-token API now validates admission, generates the bearer, publishes
durably outside policy/SQLite locks, then rechecks scope/role/drain/capacity/expiry.
Only then does one SQLite transaction reap expired rows, insert the token hash
and advance the clock floor. Publication failure has no valid row to compensate.
Busy/policy/capacity refusal after a successful publication leaves an inactive
file. Uncertain commit storage errors fail the runtime closed. No claim of an
atomic filesystem/database transaction; the safe crash window is an inactive file.
The daemon uses the saved API; direct in-memory token callers keep their existing
return type. Secrets, roles, scopes, token format, limits and existing files retain
their security boundaries.

Two focused controls cover output collision plus blocked cleanup, and successful
publication followed by SQLite Busy, removed permissions or filled256-token
capacity. They check no active candidate row, denied client access, preserved
existing output and no policy mutex held during publication. Both pass0.26s,
rawa56ea5805dad8e9a1f3b551cfa9268a1493149f0d298a660a05141e15f4f6489.
Full local security111PASS/one unchanged ignore18.09s,
rawb75645311e4abb847d1e2f0874788979fe7817d937207ec965b85dbc7bc5cc06.
The exact production-routing lifecycle case passes17.08s unchanged,
raw6f4620b2f66ff66e1fdefe7b0e6b2415751d134b7a06eac4570f6063232566a3.
Both warning-denied workspace/all-target lint modes, formatting and diff pass.
These Mac/model-free results do not establish final Windows qualification.

Correctness/security review covers all issuance callers, current policy on both
sides of publication, no lock across private I/O, exclusive output ownership,
transaction rollback/clock floor, inactive files and unchanged client validation.
Separate complexity review reuses the invitation publication pattern, existing
SQLite transaction and one local admission closure; no retry, cleanup scheduler,
new dependency/job, or deadline/assertion relaxation. GLY267 stays open for the
other historical unknowns; final-source all-seven CI, reviewed merge and separate
automatic main qualification remain required. No native/model/GPU/Ubuntu/SSH/
download/spend/quarantine reset/settings/license/publication action.
