# GLY-250: verify a started-unknown token revocation

Main head `290e7b31afca651b900c5ceaa7fe4840ad13f6b7`, run38486823800,
Windows backend job115602359371 fails at the production matrix's revoke-token
caller. The command reports CONTROL_UNAVAILABLE/outcome unknown; the daemon records
phase=executing, invitation_step=none. This is fresh evidence, separate from the
archived GLY-248/249 investigations. The nearby publish_private_file timeout belongs
to the deliberately injected binary unit and is not the production failure.

The caller incorrectly required an acknowledgement before checking revocation.
The existing three-second response wait deliberately reports an unknown outcome
once a mutation starts; missing acknowledgement does not establish mutation failure.
The protected token file still exists after revocation and cannot prove database
authorization state. No specific slow OS/database operation is diagnosed here.

## Correction and controlled checks

Reuse the permission test's one-write outcome handling in a single mutation helper.
It accepts successful exit or exactly the documented started-unknown refusal, never
queued-not-executed or unrelated errors. A small deterministic check covers those
reply classes without waiting for CI timing. It does not prove a real OS stall.

Every production revoke-token caller invokes the mutation once and then requires
HTTP 401 for the same token and valid request. The normal scenario already checked
that refusal; midstream scenarios now check it before accepting old-stream EOF and
absence of DONE. Existing worker terminal-state, no-replay, retained ownership and
drain checks remain. Observation errors or a non-401 response fail; no mutation is
retried and no timer is widened. The existing daemon unit separately exercises the
started response expiry with an intentionally withheld reply.

This changes software test callers only. No new runtime API, dependency, fixture,
job, device workload or authentication shortcut. Correctness review traces both
revoke-token callers, the CLI's exact redacted refusal, client admission and token
revocation/output gates. Ponytail review reuses current protocol observation instead
of adding a token-status endpoint. Independent review is deferred, not approved.

Raw log: `work/m2-revocation-main-115602359371-failure.log`, SHA-256
`2805de0913495cb67bb110d339b55fd1da204ce03db0e519d2b7de555930e28a`.
Final source head, PR, local checks, all seven exact-head CI results and merge
evidence are recorded in Linear. No failed original run is relabeled successful.
Formatting and all-feature workspace/all-target Clippy pass with warnings denied;
the final catalog-service selection passes all eleven cases in 36.83 seconds.
Native/provider/hardware M2 qualification remains open; M3 has not started.
