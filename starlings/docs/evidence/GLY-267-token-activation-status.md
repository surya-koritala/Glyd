# GLY-267 token activation observation

Issue: [GLY-267](https://linear.app/getglyd/issue/GLY-267).
Initial base: PR102 merge1024f1c855f505afde27960ce852f3ccf9899062.

Main18fdb20/CI38932851682 records a token mutation returning the established
started-unknown outcome. The production lifecycle test calls Daemon::admin/ok,
which requires immediate CLI success. Unlike invitation publication, token
publication has no operation-specific authenticated status observation. The file
can exist before the token row commits, so file presence cannot resolve this gap.
This source/control mismatch is separate from the initiating Windows I/O delay.

The new protected coordinator command `cluster token-status PRIVATE_TOKEN`
returns only TOKEN_STATUS/active=bool. Its read-only policy method uses the existing
token/membership/certificate/scope validator, bounded stored scopes and live drain.
No signing key, token value, scope or model data is returned. A true snapshot is
not an admission grant and does not survive later revocation/expiry. SQLite errors
remain errors; no policy/database writes or token renewals are performed.

The lifecycle helper now sends one mutation, accepts only success or the existing
exact started-unknown classification, and makes one authenticated activation
observation. The helper requires an absent output in its exclusively owned
fixture; it cannot mistake a pre-existing active file for this request's result.
The product snapshot is not a transaction receipt or a claim about requested
parameters winning a concurrent publication. Inactive/error/missing/unsafe results
still fail. No mutation retry,
automatic polling, longer reply/lease/freshness deadline or inferred success is
added. The duplicate-output refusal and later real authorization assertions stay.
The existing reply-loss short case is expanded to invitation/token times
active/inactive: one mutation, one observation, no second mutation. Its mock reply
tests the helper protocol; the real policy gate is tested separately.

One basic local file/SQLite check covers an uncommitted private file, a committed
token, unchanged database change count, partial permission withdrawal, bounded
oversized/empty/excessive and non-BLOB stored scopes, expiry, revocation and drain.
The SQL filter requires a BLOB and a byte-length cap, so SQLite's TEXT
character count cannot be mistaken for a byte bound. The initial check passes
0.12s. Default/all-feature denied-warning workspace/all-target lints pass4.66s/
2.59s with two build jobs. Initial compilation rejected usize as a SQL parameter;
the record bound now uses checked i64 conversion. No guard or assertion relaxed.
After adding the storage-type guard, the check passes0.12s and final lints
pass2.67s/1.61s. Formatting and diff checks pass.
No local sockets, daemon, interface query, process/security instrumentation or
model workload is run. The authenticated reply-loss case is CI-only.

Separate complexity review reuses the existing validators/mutation classifier and
one shared observation helper; no new dependency, worker framework or retry loop.
Correctness/security review verifies the protected admin boundary, coordinator
role, private-file checks, bounded BLOB/scope reads, unchanged admission and no
secret output or durable writes. Full platform CI, current-main integration and
authorized merge gates remain before integration. Historical Windows OS/I/O
causality, shared durable-commit contention, native/provider/device acceptance and
independent milestone review remain open. The API supplies an honest observation;
it does not claim all unknown outcomes have succeeded or every Windows stall is
fixed. No provider behavior, release license or publication decision is inferred.
