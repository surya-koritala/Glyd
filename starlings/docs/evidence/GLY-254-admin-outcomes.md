# GLY-254: local administration outcomes

Automatic main `3a06ad8f4668889e159657e2a3ed9b3eb5e6ee37`,
run38501223090/job115648398830 fails warm-worker revoke with Transport and a
coordinator DAEMON_READY log. Production matrix and five other required checks
pass; Windows backend/aggregate fail. Raw
`work/m2-main-115648398830-failure.log`, SHA-256
`2979829705cbf03714a38ca25e00e355b6e0be6b729426ef3948c756e26464db`.
The original transport phase is unknown; no mutation is retried or failed run
relabeled. GLY-251/253's different event attribution remains open.

Shared client source propagated receipt-byte failure after already decoding a
reply and verifying its protected capability file. It also classified a lost
reply after possible dispatch as generic Transport/unavailability. Two controlled
checks fail against unchanged behavior: an expired receipt deadline loses a
decoded reply, and authenticated dispatch followed by reply loss looks like
unavailability. Raw before log SHA-256
`cc2a03185bd143b20006049a1c71f69788fe841b20a6bb3e86b79a747e998eba`.
The completion check exercises the existing primitive after bounded decoding;
the lost-reply check uses public admin_call and real capability authentication,
observing exactly one handler execution. It does not identify the old main phase.

The completion primitive still verifies the protected file before returning data.
Receipt is best effort within the unchanged deadline, preserving a known reply.
Errors after possible command send, lost/invalid reply or failed final file
verification return AdminOutcomeUnknown. Encoding/size errors occur before send;
server-proof/authentication failures remain closed before command disclosure.
The CLI maps unknown outcomes to its existing explicit refusal and cannot run the
offline-leave fallback. No native IPC wire bytes, rights or deadlines change.

Seven local admin checks pass in 0.13 seconds, including wrong server proof,
protected-file rejection, frame bounds and byte preservation. Raw after SHA-256
`1831ac4a43fbddda496acf121c39adbac97ce3861724d03393b8675f59274993`.
The actual CLI sends one authenticated leave command to an isolated endpoint,
loses the reply, emits the exact unknown-outcome refusal, opens no second connection
and creates no offline state lock. It passes in 0.58 seconds; raw SHA-256
`7e5c58da9fa956ec76a41b3ae40bb1763d6243fa7770526e193979b7332b3d6a`.
Warm revocation sends one mutation and requires the actual authenticated revoked
registry/service observation. All twelve existing lifecycle cases pass in 44.44
seconds; raw SHA-256
`671471d8c8879107b0c6d44da680937c4d0ab5793bd8f5c0c14459ee68929cd8`.

Commands use pinned Rust, locked offline dependencies and warnings denied:
`cargo test -p compute-security --lib admin::tests:: --locked --offline -- --nocapture`;
`cargo test -p compute-cli --all-features --test daemon catalog_service_ --locked --offline -- --nocapture`;
the new CLI case is in that existing CI filter, also checked individually.
Formatting and both workspace/all-target lint modes pass. Exact-source CI/PR/merge
identities belong in Linear. Independent review remains deferred, not approved.

Correctness/security review traces challenge/proof, validation before send,
possible dispatch, bounded reply decode, protected-file verification, receipt,
server deadline/drop, CLI leave handling and actual revocation observation.
Ponytail review uses the existing completion, framing, test endpoint and refusal;
no profiler, retry, dependency, job or runtime framework is added. Original OS/IPC
timing attribution is unproved. No native/model/PC/GPU action, spend, download or
quarantine reset. M2 qualification is incomplete; M3 has not started.
